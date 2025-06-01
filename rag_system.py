"""rag_system.py
~~~~~~~~~~~~~~~~~
Core implementation of a Retrieval‑Augmented Generation (RAG) pipeline built on

* **LangChain Expression Language (LCEL)**
* **pgvector** on PostgreSQL / Amazon RDS
* **OpenAI GPT / Embeddings**

The module exposes two public classes:

* `Config` – environment‑driven configuration holder (DB, OpenAI, model params…)
* `RAGSystem` – end‑to‑end object that handles ingestion, retrieval, and QA

`streamlit_rag_ui.py` (or any other UI) can simply::

    from rag_system import Config, RAGSystem

    cfg = Config()
    rag = RAGSystem(cfg)
    rag.ingest_documents([...])
    answer = rag.query("質問…")

The CLI loop from the original script is intentionally omitted; Streamlit/Gradio
or your own runner should orchestrate ingestion & queries.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from operator import itemgetter

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect as sqlalchemy_inspect

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)  # TextractLoader is optional – imported below

# TextractLoader (optional – not in older langchain versions)
try:
    from langchain_community.document_loaders import TextractLoader  # type: ignore
except ImportError:  # pragma: no cover
    TextractLoader = None
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback

# Load .env early
load_dotenv()

try:
    import psycopg
    _PG_DIALECT = "psycopg"
except ModuleNotFoundError:
    try:
        import psycopg2  # type: ignore
        _PG_DIALECT = "psycopg2"
    except ModuleNotFoundError:
        _PG_DIALECT = None

###############################################################################
# Config dataclass                                                            #
###############################################################################

@dataclass
class Config:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_name: str = os.getenv("DB_NAME", "postgres")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "your-password")

    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    vector_search_k: int = int(os.getenv("VECTOR_SEARCH_K", 10))
    keyword_search_k: int = int(os.getenv("KEYWORD_SEARCH_K", 10))
    final_k: int = int(os.getenv("FINAL_K", 5))

    collection_name: str = os.getenv("COLLECTION_NAME", "documents") 
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))
    fts_language: str = os.getenv("FTS_LANGUAGE", "english")

###############################################################################
# Hybrid Retriever                                                            #
###############################################################################

class HybridRetriever(BaseRetriever):
    vector_store: PGVector
    connection_string: str
    config_params: Config

    def _vector_search(self, q: str) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            return []
        try:
            return self.vector_store.similarity_search_with_score(q, k=self.config_params.vector_search_k)
        except Exception as exc:
            print(f"[HybridRetriever] vector search error: {exc}")
            return []

    def _keyword_search(self, q: str) -> List[Tuple[Document, float]]:
        engine = create_engine(self.connection_string)
        res: List[Tuple[Document, float]] = []
        sql = f"""
            SELECT chunk_id, content, metadata,
                   ts_rank(to_tsvector('{self.config_params.fts_language}',content),
                           plainto_tsquery('{self.config_params.fts_language}',:q)) AS score
            FROM document_chunks
            WHERE to_tsvector('{self.config_params.fts_language}',content) @@
                  plainto_tsquery('{self.config_params.fts_language}',:q)
                  AND collection_name = :collection_name
            ORDER BY score DESC
            LIMIT :k;
        """
        try:
            with engine.connect() as conn:
                for row in conn.execute(text(sql), {"q": q, "k": self.config_params.keyword_search_k, "collection_name": self.config_params.collection_name}):
                    md = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}")
                    res.append((Document(page_content=row.content, metadata=md), float(row.score)))
        except Exception as exc:
            print(f"[HybridRetriever] keyword search error: {exc}")
        return res

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def _reciprocal_rank_fusion(self, vres: List[Tuple[Document, float]], kres: List[Tuple[Document, float]]) -> List[Document]:
        score_map: Dict[str, Dict[str, Any]] = {}
        def _id(d: Document) -> str:
            return d.metadata.get("chunk_id", d.page_content[:100])

        for r, (d, _) in enumerate(vres, 1):
            doc_id_val = _id(d)
            score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf(r)
        for r, (d, _) in enumerate(kres, 1):
            doc_id_val = _id(d)
            score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf(r)
        
        ranked = sorted(score_map.values(), key=lambda x: x["s"], reverse=True)
        return [x["doc"] for x in ranked[: self.config_params.final_k]]

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun]=None, **kw: Any) -> List[Document]:
        return self._reciprocal_rank_fusion(self._vector_search(query), self._keyword_search(query))

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun]=None, **kw: Any) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager, **kw)

###############################################################################
# RAG System                                                                  #
###############################################################################

class RAGSystem:
    def __init__(self, cfg: Config):
        if _PG_DIALECT is None:
            raise RuntimeError("PostgreSQL driver (psycopg/psycopg2) not installed.")
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing.")

        self.config = cfg
        self.embeddings = OpenAIEmbeddings(openai_api_key=cfg.openai_api_key, model=cfg.embedding_model)
        self.llm = ChatOpenAI(openai_api_key=cfg.openai_api_key, model_name=cfg.llm_model, temperature=0.7)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        self.connection_string = self._conn_str()
        self._init_db() 
        
        self.vector_store: PGVector = PGVector(
            connection_string=self.connection_string,
            collection_name=cfg.collection_name, 
            embedding_function=self.embeddings,
            use_jsonb=True, 
            distance_strategy=DistanceStrategy.COSINE,
        )
        
        self.retriever: HybridRetriever = HybridRetriever(
            vector_store=self.vector_store,
            connection_string=self.connection_string,
            config_params=cfg,
        )
        self.chain = self._build_chain()

    def _conn_str(self) -> str:
        c = self.config
        return f"postgresql+{_PG_DIALECT}://{c.db_user}:{c.db_password}@{c.db_host}:{c.db_port}/{c.db_name}"

    def _init_db(self):
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        collection_name TEXT,
                        document_id TEXT,
                        chunk_id TEXT UNIQUE,
                        content TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );"""
                )
            )
            
            inspector = sqlalchemy_inspect(engine)
            columns = [col['name'] for col in inspector.get_columns('document_chunks')]
            if 'collection_name' not in columns:
                print("[_init_db] 'collection_name' column not found in 'document_chunks'. Adding column...")
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN collection_name TEXT;"))
                print("[_init_db] 'collection_name' column added.")
            
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_collection_doc_id_chunks ON document_chunks (collection_name, document_id);"
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_fts_content ON document_chunks USING GIN(to_tsvector('{self.config.fts_language}',content));"
                )
            )
            conn.commit()
            print("[_init_db] Database initialization and schema check complete for 'document_chunks'.")

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """あなたは親切で知識豊富なアシスタントです。以下のコンテキストを参考に質問に答えてください。\n\nコンテキスト:\n{context}\n\n質問: {question}\n\n回答:"""
        )
        def _format(docs: List[Document]) -> str:
            if not docs:
                return "(コンテキスト無し)"
            return "\n\n".join(
                [f"[ソース {i+1} ChunkID: {d.metadata.get('chunk_id','N/A')}]\n{d.page_content}" for i, d in enumerate(docs)]
            )
        chain = {
            "context": itemgetter("question") | self.retriever | _format,
            "question": itemgetter("question"),
        } | prompt | self.llm | StrOutputParser()
        return chain

    def load_documents(self, paths: List[str]) -> List[Document]:
        docs: List[Document] = []
        for p_str in paths:
            path = Path(p_str)
            if not path.exists():
                print(f"[load_documents] File not found: {p_str}")
                continue
            suf = path.suffix.lower()
            try:
                if suf == ".pdf":
                    docs.extend(PyPDFLoader(str(path)).load())
                elif suf in {".txt", ".md"}:
                    docs.extend(TextLoader(str(path), encoding="utf-8").load())
                elif suf == ".docx":
                    try:
                        docs.extend(
                            UnstructuredFileLoader(str(path), mode="single", strategy="fast").load()
                        )
                    except Exception as ue:
                        print(f"[load_documents] .docx UnstructuredFileLoader failed for {p_str}, trying Docx2txtLoader. Error: {ue}")
                        docs.extend(Docx2txtLoader(str(path)).load())
                elif suf == ".doc": 
                    if TextractLoader:
                        try:
                            docs.extend(TextractLoader(str(path)).load()) 
                        except Exception as te:
                            print(f"[load_documents] .doc TextractLoader failed for {p_str}. Error: {te}")
                    else:
                        print(f"[load_documents] .doc skipped, TextractLoader not available for {p_str}.")
                else:
                    print(f"[load_documents] Unsupported file type: {suf} for {p_str}")
            except Exception as e:
                print(f"[load_documents] Error loading {p_str}: {e}")
        if not docs:
            print("[load_documents] No documents were successfully loaded.")
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        out: List[Document] = []
        for i, d in enumerate(docs):
            src = d.metadata.get("source", f"doc_source_{i}") 
            doc_id = Path(src).name 
            
            for j, c in enumerate(self.text_splitter.split_documents([d])):
                current_chunk_id = f"{doc_id}_{i}_{j}"
                c.metadata = {
                    **d.metadata, 
                    **c.metadata, 
                    "chunk_id": current_chunk_id, 
                    "document_id": doc_id, 
                    "original_document_source": src, 
                    "collection_name": self.config.collection_name 
                }
                out.append(c)
        return out

    def ingest_documents(self, paths: List[str]):
        docs = self.load_documents(paths)
        if not docs:
            return 
        
        chunks = self.chunk_documents(docs)
        if not chunks:
            print("[ingest_documents] No chunks were produced from the documents.")
            return
        
        chunk_ids_for_vectorstore = [c.metadata['chunk_id'] for c in chunks if 'chunk_id' in c.metadata]
        
        if len(chunk_ids_for_vectorstore) != len(chunks):
            print("[ingest_documents] Warning: Some chunks are missing 'chunk_id' in metadata.")
            valid_chunks = [c for c in chunks if 'chunk_id' in c.metadata]
            chunk_ids_for_vectorstore = [c.metadata['chunk_id'] for c in valid_chunks]
        else:
            valid_chunks = chunks

        if not valid_chunks:
            print("[ingest_documents] No valid chunks with chunk_id to add to vector store.")
            return

        self.vector_store.add_documents(valid_chunks, ids=chunk_ids_for_vectorstore)
        
        self._store_chunks_for_keyword_search(valid_chunks)
        print(f"[ingest_documents] {len(valid_chunks)} chunks processed. Attempted to store in vector store (collection: '{self.config.collection_name}') and keyword table.")

    def _store_chunks_for_keyword_search(self, chunks: List[Document]):
        eng = create_engine(self.connection_string)
        sql = text(
            """
            INSERT INTO document_chunks(collection_name, document_id, chunk_id, content, metadata) 
            VALUES(:coll_name, :doc_id, :cid, :cont, :meta)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                document_id = EXCLUDED.document_id,
                collection_name = EXCLUDED.collection_name; 
            """
        )
        with eng.connect() as conn:
            for c in chunks:
                if 'chunk_id' not in c.metadata or 'document_id' not in c.metadata:
                    print(f"[store_chunks] Skipping chunk due to missing 'chunk_id' or 'document_id': {c.page_content[:50]}...")
                    continue
                
                conn.execute(sql, {
                    "coll_name": self.config.collection_name, 
                    "doc_id": c.metadata["document_id"],
                    "cid": c.metadata["chunk_id"],
                    "cont": c.page_content,
                    "meta": json.dumps(c.metadata or {}), 
                })
            conn.commit()
    
    def delete_document_by_id(self, document_id_to_delete: str) -> tuple[bool, str]:
        if not document_id_to_delete:
            msg = "Document ID cannot be empty for deletion."
            print(f"[delete_document_by_id] {msg}")
            return False, msg

        current_collection_name = self.config.collection_name
        print(f"[delete_document_by_id] Attempting to delete document: '{document_id_to_delete}' from collection: '{current_collection_name}'")
        engine = create_engine(self.connection_string)
        
        chunk_ids_to_delete_from_vector_store: List[str] = []
        deleted_rows_count_table = 0
        
        try:
            with engine.connect() as conn:
                with conn.begin(): 
                    select_chunk_ids_stmt = text(
                        "SELECT chunk_id FROM document_chunks WHERE document_id = :doc_id AND collection_name = :coll_name"
                    )
                    chunk_id_rows = conn.execute(
                        select_chunk_ids_stmt, 
                        {"doc_id": document_id_to_delete, "coll_name": current_collection_name}
                    ).fetchall()
                    
                    if not chunk_id_rows:
                        msg = f"No chunks found in 'document_chunks' table for document_id: '{document_id_to_delete}' in collection '{current_collection_name}'. No deletion performed."
                        print(f"[delete_document_by_id] {msg}")
                        return True, msg 

                    chunk_ids_to_delete_from_vector_store = [row[0] for row in chunk_id_rows if row and row[0]]
                    print(f"[delete_document_by_id] Found {len(chunk_ids_to_delete_from_vector_store)} chunk(s) to target for document_id: '{document_id_to_delete}' in collection '{current_collection_name}'.")

                    delete_chunks_stmt = text(
                        "DELETE FROM document_chunks WHERE document_id = :doc_id AND collection_name = :coll_name"
                    )
                    result_proxy_chunks = conn.execute(
                        delete_chunks_stmt, 
                        {"doc_id": document_id_to_delete, "coll_name": current_collection_name}
                    )
                    deleted_rows_count_table = result_proxy_chunks.rowcount
                    print(f"[delete_document_by_id] Deleted {deleted_rows_count_table} entries from 'document_chunks' table for document_id: '{document_id_to_delete}' in collection '{current_collection_name}'.")

                    if self.vector_store and hasattr(self.vector_store, 'delete'):
                        if chunk_ids_to_delete_from_vector_store:
                            delete_op_result = self.vector_store.delete(ids=chunk_ids_to_delete_from_vector_store)
                            print(f"[delete_document_by_id] Attempted to delete {len(chunk_ids_to_delete_from_vector_store)} vectors from vector store (collection: '{current_collection_name}'). PGVector delete result: {delete_op_result}")
                        else:
                            print(f"[delete_document_by_id] No chunk_ids were found to delete from vector store for document_id: '{document_id_to_delete}' in collection '{current_collection_name}'.")
                    else:
                        print("[delete_document_by_id] Vector store does not support delete operation or is not properly initialized.")

                msg = (
                    f"Successfully processed deletion for document '{document_id_to_delete}' in collection '{current_collection_name}'. "
                    f"{deleted_rows_count_table} chunks deleted from keyword table. "
                    f"{len(chunk_ids_to_delete_from_vector_store)} vectors targeted for deletion from vector store."
                )
                print(f"[delete_document_by_id] {msg}")
                return True, msg

        except Exception as e:
            msg = f"Error during deletion of document '{document_id_to_delete}' from collection '{current_collection_name}': {e}"
            print(f"[delete_document_by_id] {msg}")
            return False, msg

    def query(self, question: str, *, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        with get_openai_callback() as cb:
            answer = self.chain.invoke({"question": question}, config=config)
            usage = {"total_tokens": cb.total_tokens, "cost": cb.total_cost} 
        
        sources: List[Document] = self.retriever.get_relevant_documents(question) # Explicitly type sources
        
        # Prepare sources data for the UI, including excerpt and full_content
        sources_data = []
        for s_doc in sources: # Renamed s to s_doc for clarity
            # Ensure page_content is a string before slicing or calculating length
            page_content_str = str(s_doc.page_content) if s_doc.page_content is not None else ""
            
            sources_data.append({
                "excerpt": page_content_str[:200] + ("…" if len(page_content_str) > 200 else ""), # 抜粋
                "full_content": page_content_str,  # 全文
                "metadata": s_doc.metadata or {} # Ensure metadata is a dict
            })
            
        return {
            "question": question,
            "answer": answer,
            "sources": sources_data, # Use the new sources_data list
            "usage": usage,
        }

__all__ = ["Config", "RAGSystem"]
