
from __future__ import annotations

import os, json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from operator import itemgetter

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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

    # internal helpers
    def _vector_search(self, q: str):
        return self.vector_store.similarity_search_with_score(q, k=self.config_params.vector_search_k)

    def _keyword_search(self, q: str):
        engine = create_engine(self.connection_string)
        res: List[Tuple[Document, float]] = []
        sql = f"""
            SELECT chunk_id, content, metadata,
                   ts_rank(to_tsvector('{self.config_params.fts_language}',content),
                           plainto_tsquery('{self.config_params.fts_language}',:q)) AS score
            FROM document_chunks
            WHERE to_tsvector('{self.config_params.fts_language}',content) @@
                  plainto_tsquery('{self.config_params.fts_language}',:q)
            ORDER BY score DESC
            LIMIT :k;
        """
        with engine.connect() as conn:
            for row in conn.execute(text(sql), {"q": q, "k": self.config_params.keyword_search_k}):  # type: ignore
                md = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}")
                res.append((Document(page_content=row.content, metadata=md), float(row.score)))  # type: ignore
        return res

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def _reciprocal_rank_fusion(self, vres, kres):
        score_map: Dict[str, Dict[str, Any]] = {}
        def _id(d: Document):
            return d.metadata.get("chunk_id", d.page_content[:100])
        for r, (d, _) in enumerate(vres, 1):
            score_map.setdefault(_id(d), {"doc": d, "s": 0})["s"] += self._rrf(r)
        for r, (d, _) in enumerate(kres, 1):
            score_map.setdefault(_id(d), {"doc": d, "s": 0})["s"] += self._rrf(r)
        ranked = sorted(score_map.values(), key=lambda x: x["s"], reverse=True)
        return [x["doc"] for x in ranked[: self.config_params.final_k]]

    # BaseRetriever API
    def _get_relevant_documents(self, query: str, *, run_manager=None, **kw):
        return self._reciprocal_rank_fusion(self._vector_search(query), self._keyword_search(query))

    async def _aget_relevant_documents(self, query: str, *, run_manager=None, **kw):
        return self._get_relevant_documents(query)

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
        self.vector_store = PGVector(
            connection_string=self.connection_string,
            collection_name=cfg.collection_name,
            embedding_function=self.embeddings,
            use_jsonb=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            connection_string=self.connection_string,
            config_params=cfg,
        )
        self.chain = self._build_chain()

    def _conn_str(self):
        c = self.config
        return f"postgresql+{_PG_DIALECT}://{c.db_user}:{c.db_password}@{c.db_host}:{c.db_port}/{c.db_name}"

    # ------------------------ DB init ---------------------------------------
    def _init_db(self):
        eng = create_engine(self.connection_string)
        with eng.connect() as conn:
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
                        embedding VECTOR({self.config.embedding_dimensions}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );"""
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_fts ON document_chunks USING GIN(to_tsvector('{self.config.fts_language}',content));"
                )
            )
            conn.commit()

    # ------------------------ LCEL chain ------------------------------------
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """あなたは親切で知識豊富なアシスタントです。以下のコンテキストを参考に質問に答えてください。\n\nコンテキスト:\n{context}\n\n質問: {question}\n\n回答:"""
        )
        def _format(docs: List[Document]):
            if not docs:
                return "(コンテキスト無し)"
            return "\n\n".join(
                [f"[ソース {i+1} {d.metadata.get('chunk_id','N/A')}]\n{d.page_content}" for i, d in enumerate(docs)]
            )
        chain = {
            "context": itemgetter("question") | self.retriever | _format,  # type: ignore
            "question": itemgetter("question"),
        } | prompt | self.llm | StrOutputParser()
        return chain

    # ------------------------ Ingestion -------------------------------------
    def load_documents(self, paths: List[str]) -> List[Document]:
        docs: List[Document] = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                print("[load] not found", p)
                continue
            suf = path.suffix.lower()
            try:
                if suf == ".pdf":
                    docs.extend(PyPDFLoader(str(path)).load())
                elif suf in {".txt", ".md"}:
                    docs.extend(TextLoader(str(path), encoding="utf-8").load())
                elif suf == ".docx":
                    # unstructured first
                    try:
                        docs.extend(
                            UnstructuredFileLoader(str(path), mode="single", strategy="fast").load()
                        )
                    except Exception as ue:
                        print("[docx] Unstructured failed → Docx2txt", ue)
                        docs.extend(Docx2txtLoader(str(path)).load())
                elif suf == ".doc":
                    try:
                        docs.extend(TextractLoader(str(path)).load())
                    except Exception as te:
                        print("[.doc] Textract failed", te)
                else:
                    print("[load] unsupported", suf)
            except Exception as e:
                print("[load] error", e)
        if not docs:
            print("[load] no documents loaded")
        return docs

    def chunk_documents(self, docs: List[Document]):
        out: List[Document] = []
        for i, d in enumerate(docs):
            src = d.metadata.get("source", f"doc_{i}")
            name = Path(src).name
            for j, c in enumerate(self.text_splitter.split_documents([d])):
                c.metadata = {**d.metadata, **c.metadata, "chunk_id": f"{name}_{i}_{j}", "document_id": name}
                out.append(c)
        return out

    def ingest_documents(self, paths: List[str]):
        docs = self.load_documents(paths)
        if not docs:
            return
        chunks = self.chunk_documents(docs)
        if not chunks:
            return
        self.vector_store.add_documents(chunks)
        self._store_chunks_for_keyword_search(chunks)
        print(f"[ingest] {len(chunks)} chunks stored")

    def _store_chunks_for_keyword_search(self, chunks: List[Document]):
        eng = create_engine(self.connection_string)
        sql = text(
            """
            INSERT INTO document_chunks
                (collection_name, document_id, chunk_id, content, metadata)
            VALUES (:col, :doc, :cid, :cont, :meta)
            ON CONFLICT(chunk_id) DO NOTHING;
            """
        )
        with eng.connect() as conn:
            for c in chunks:
                conn.execute(sql, {
                    "col": self.config.collection_name,
                    "doc": c.metadata.get("document_id","unknown"),
                    "cid": c.metadata["chunk_id"],
                    "cont": c.page_content,
                    "meta": json.dumps(c.metadata),
                })
            conn.commit()

    # ------------------------ QA -------------------------------------------
    def query(self, question: str, *, config: Optional[RunnableConfig] = None):
        with get_openai_callback() as cb:
            answer = self.chain.invoke({"question": question}, config=config)
            usage = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}
        sources = self.retriever.get_relevant_documents(question)
        return {
            "question": question,
            "answer": answer,
            "sources": [{"content": s.page_content[:200] + ("…" if len(s.page_content)>200 else ""), "metadata": s.metadata} for s in sources],
            "usage": usage,
        }

__all__ = ["Config", "RAGSystem"]
