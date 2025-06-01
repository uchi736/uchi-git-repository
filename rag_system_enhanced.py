"""rag_system_enhanced.py
~~~~~~~~~~~~~~~~~~~~~~~
Enhanced RAG System with Query Expansion, RAG-Fusion, and Text-to-SQL functionality

Langsmithでトレースを有効にするには、以下の環境変数を設定してください:
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="your-project-name" (オプション)
"""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from operator import itemgetter
import pandas as pd # For formatting SQL results for LLM if needed

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect as sqlalchemy_inspect

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)

try:
    from langchain_community.document_loaders import TextractLoader
except ImportError:
    TextractLoader = None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback

# Load .env early
load_dotenv()

try:
    import psycopg
    _PG_DIALECT = "psycopg"
except ModuleNotFoundError:
    try:
        import psycopg2
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
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    vector_search_k: int = int(os.getenv("VECTOR_SEARCH_K", 10))
    keyword_search_k: int = int(os.getenv("KEYWORD_SEARCH_K", 10))
    final_k: int = int(os.getenv("FINAL_K", 5))

    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))
    fts_language: str = os.getenv("FTS_LANGUAGE", "english")

    rrf_k_for_fusion: int = int(os.getenv("RRF_K_FOR_FUSION", 60))

    # Text-to-SQL用設定
    enable_text_to_sql: bool = True # Trueに設定してSQL機能を有効化
    max_sql_results: int = int(os.getenv("MAX_SQL_RESULTS", 1000)) # Max results to fetch from DB
    max_sql_preview_rows_for_llm: int = int(os.getenv("MAX_SQL_PREVIEW_ROWS_FOR_LLM", 20)) # Max results to show LLM for summarization
    # ユーザーが作成したデータテーブルのプレフィックス（get_data_tablesで使用）
    user_table_prefix: str = os.getenv("USER_TABLE_PREFIX", "data_")


###############################################################################
# Helper Function for RAG System                                              #
###############################################################################

def format_docs(docs: List[Document]) -> str:
    if not docs: return "(コンテキスト無し)" # No context
    return "\n\n".join([f"[ソース {i+1} ChunkID: {d.metadata.get('chunk_id','N/A')}]\n{d.page_content}" for i, d in enumerate(docs)]) # Source {i+1} ChunkID: ...

###############################################################################
# Hybrid Retriever                                                            #
###############################################################################

class HybridRetriever(BaseRetriever):
    vector_store: PGVector
    connection_string: str
    config_params: Config

    def _vector_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        if not self.vector_store: return []
        try: return self.vector_store.similarity_search_with_score(q, k=self.config_params.vector_search_k)
        except Exception as exc: print(f"[HybridRetriever] vector search error: {exc}"); return []

    def _keyword_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        engine = create_engine(self.connection_string); res: List[Tuple[Document, float]] = []
        sql = f"""SELECT chunk_id, content, metadata, ts_rank(to_tsvector('{self.config_params.fts_language}',content), plainto_tsquery('{self.config_params.fts_language}',:q)) AS score FROM document_chunks WHERE to_tsvector('{self.config_params.fts_language}',content) @@ plainto_tsquery('{self.config_params.fts_language}',:q) AND collection_name = :collection_name ORDER BY score DESC LIMIT :k;"""
        try:
            with engine.connect() as conn:
                db_result = conn.execute(text(sql), {"q": q, "k": self.config_params.keyword_search_k, "collection_name": self.config_params.collection_name})
                for row in db_result: md = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}"); res.append((Document(page_content=row.content, metadata=md), float(row.score)))
        except Exception as exc: print(f"[HybridRetriever] keyword search error: {exc}")
        return res

    @staticmethod
    def _rrf_hybrid(rank: int, k: int = 60) -> float: return 1.0 / (k + rank)

    def _reciprocal_rank_fusion_hybrid(self, vres: List[Tuple[Document, float]], kres: List[Tuple[Document, float]]) -> List[Document]:
        score_map: Dict[str, Dict[str, Any]] = {}; _id = lambda d: d.metadata.get("chunk_id", d.page_content[:100])
        for r, (d, _) in enumerate(vres, 1): doc_id_val = _id(d); score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf_hybrid(r)
        for r, (d, _) in enumerate(kres, 1): doc_id_val = _id(d); score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf_hybrid(r)
        ranked = sorted(score_map.values(), key=lambda x: x["s"], reverse=True)
        return [x["doc"] for x in ranked[:self.config_params.final_k]]

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any) -> List[Document]:
        config = kwargs.get("config")
        vres = self._vector_search(query, config=config)
        kres = self._keyword_search(query, config=config)
        return self._reciprocal_rank_fusion_hybrid(vres, kres)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any) -> List[Document]:
        config = kwargs.get("config")
        vres = self._vector_search(query, config=config)
        kres = self._keyword_search(query, config=config)
        return self._reciprocal_rank_fusion_hybrid(vres, kres)

###############################################################################
# RAG System                                                                  #
###############################################################################

class RAGSystem:
    def __init__(self, cfg: Config):
        if _PG_DIALECT is None: raise RuntimeError("PostgreSQL driver not installed.")
        if not cfg.openai_api_key: raise ValueError("OPENAI_API_KEY is missing.")
        self.config = cfg
        self.embeddings = OpenAIEmbeddings(openai_api_key=cfg.openai_api_key, model=cfg.embedding_model)
        self.llm = ChatOpenAI(openai_api_key=cfg.openai_api_key, model_name=cfg.llm_model, temperature=0.7)
        self.connection_string = self._conn_str(); self._init_db()
        self.vector_store = PGVector(connection_string=self.connection_string, collection_name=cfg.collection_name, embedding_function=self.embeddings, use_jsonb=True, distance_strategy=DistanceStrategy.COSINE)
        self.retriever = HybridRetriever(vector_store=self.vector_store, connection_string=self.connection_string, config_params=cfg)

        self.base_rag_prompt = ChatPromptTemplate.from_template(
            """あなたは親切で知識豊富なアシスタントです。以下のコンテキストを参考に質問に答えてください。\n\nコンテキスト:\n{context}\n\n質問: {question}\n\n回答:""" # You are a kind and knowledgeable assistant. Please answer the question based on the following context. Context: ... Question: ... Answer:
        )
        self.query_expansion_prompt = ChatPromptTemplate.from_template(
            """以下の質問に対して、より良い検索結果を得るために、関連する追加の検索クエリを3つ生成してください。\n元の質問の意図を保ちながら、異なる表現や関連する概念を含めてください。\n\n元の質問: {question}\n\n追加クエリ（改行で区切って3つ）:""" # For the following question, generate three additional related search queries to get better search results. Keep the intent of the original question, but include different expressions and related concepts. Original question: ... Additional queries (3, separated by newlines):
        )
        self._query_expansion_llm_chain = self.query_expansion_prompt | self.llm | StrOutputParser()

        # For multi-table SQL generation (used by query_unified)
        self.multi_table_text_to_sql_prompt = ChatPromptTemplate.from_template(
            """あなたはPostgreSQLエキスパートです。以下に提示される複数のテーブルスキーマの中から、ユーザーの質問に答えるために最も適切と思われるテーブルを **選択** し、必要であればそれらのテーブル間でJOINを適切に使用して、SQLクエリを生成してください。
SQLはPostgreSQL構文に準拠し、テーブル名やカラム名が日本語の場合はダブルクォーテーションで囲んでください (例: public."テーブル名", "カラム名")。
最終的な結果セットが過度に大きくならないよう、適切にLIMIT句を使用してください（例: LIMIT {max_sql_results}）。

利用可能なテーブルのスキーマ情報一覧 (各テーブルは --- で区切られています):
{schemas_info}

ユーザーの質問: {question}

SQLクエリのみを返してください:
```sql
SELECT ...
```
""" # You are a PostgreSQL expert. From the multiple table schemas presented below, **select** the most appropriate table(s) to answer the user's question, and generate an SQL query, using JOINs appropriately if necessary. The SQL should conform to PostgreSQL syntax, and Japanese table or column names should be enclosed in double quotes (e.g., public."テーブル名", "カラム名"). Use the LIMIT clause appropriately to prevent result sets from becoming too large (e.g., LIMIT {max_sql_results}). Available table schema information list (each table separated by ---): ... User's question: ... Return only the SQL query: ...
        )
        self._multi_table_sql_chain = self.multi_table_text_to_sql_prompt | self.llm | StrOutputParser()

        # For single-table SQL generation (used by execute_sql_query for Tab 6)
        self.single_table_text_to_sql_prompt = ChatPromptTemplate.from_template(
            """あなたはPostgreSQLエキスパートです。以下のテーブル情報を参考に、質問をSQLに変換してください。
SQLはPostgreSQL構文に準拠し、テーブル名やカラム名が日本語の場合はダブルクォーテーションで囲んでください (例: public."テーブル名", "カラム名")。
最終的な結果セットが過度に大きくならないよう、適切にLIMIT句を使用してください（例: LIMIT {max_sql_results}）。

テーブル情報:
{schema_info}

質問: {question}

SQLクエリのみを返してください:
```sql
SELECT ...
```
""" # You are a PostgreSQL expert. Convert the question to SQL based on the table information. SQL should conform to PostgreSQL syntax, and Japanese table or column names should be enclosed in double quotes (e.g., public."テーブル名", "カラム名"). Use the LIMIT clause appropriately to prevent result sets from becoming too large (e.g., LIMIT {max_sql_results}). Table information: ... Question: ... Return only the SQL query: ...
        )
        self._single_table_sql_chain = self.single_table_text_to_sql_prompt | self.llm | StrOutputParser()


        self.query_detection_prompt = ChatPromptTemplate.from_template(
            """この質問はSQL分析とRAG検索のどちらが適切ですか？

利用可能なデータテーブルの概要:
{tables_info}

ユーザーの質問: {question}

判断基準:
- SQLが適している場合: 具体的な数値データに基づく分析、集計、ランキング、フィルタリング、特定レコードの抽出など。質問がテーブル内のカラム名や値に直接関連している場合。
- RAGが適している場合: ドキュメントの内容に関する要約、説明、概念理解、自由形式の質問、SQLでは答えられない種類の問い合わせ。

回答は「SQL」または「RAG」のいずれか一つのみを返してください。""" # Is SQL analysis or RAG search more appropriate for this question? Overview of available data tables: ... User's question: ... Criteria: - SQL is suitable for: Analysis based on specific numerical data, aggregation, ranking, filtering, extraction of specific records, etc. When the question is directly related to column names or values in the table. - RAG is suitable for: Summarization, explanation, conceptual understanding of document content, free-form questions, inquiries that cannot be answered by SQL. Return only "SQL" or "RAG".
        )
        self._detection_chain = self.query_detection_prompt | self.llm | StrOutputParser()


        # New prompt for generating answers from SQL results
        self.sql_answer_generation_prompt = ChatPromptTemplate.from_template(
            """与えられた元の質問と、それに基づいて実行されたSQLクエリ、およびその実行結果(プレビュー)を考慮して、ユーザーにとって分かりやすい言葉で回答を生成してください。
SQLの実行結果は、全てのデータではなく一部のプレビューである可能性があります。その場合はその旨も考慮し、必要であれば「さらに詳細なデータが必要な場合はお知らせください」といった形で補足してください。
結果がない場合は、その旨を伝えてください。

元の質問: {original_question}

実行されたSQLクエリ:
```sql
{sql_query}
```

SQL実行結果のプレビュー (最大 {max_preview_rows} 件表示):
{sql_results_preview_str}
(このプレビューは全 {total_row_count} 件中の一部です)

上記の情報を踏まえた、質問に対する回答:""" # Considering the original question, the SQL query executed based on it, and its execution results (preview), generate an answer in words that are easy for the user to understand. The SQL execution results may be a partial preview, not all the data. In that case, take that into account and, if necessary, add a supplement such as "Please let me know if you need more detailed data." If there are no results, please state that. Original question: ... Executed SQL query: ... SQL execution results preview (max {max_preview_rows} items displayed): ... (This preview is a part of all {total_row_count} items) Based on the above information, answer to the question:
        )
        self._sql_answer_generation_chain = self.sql_answer_generation_prompt | self.llm | StrOutputParser()


    def _conn_str(self) -> str:
        c = self.config; return f"postgresql+{_PG_DIALECT}://{c.db_user}:{c.db_password}@{c.db_host}:{c.db_port}/{c.db_name}"

    def _init_db(self):
        engine = create_engine(self.connection_string)
        with engine.connect() as conn: # Use 'with' statement for connection
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""CREATE TABLE IF NOT EXISTS document_chunks (id SERIAL PRIMARY KEY, collection_name TEXT, document_id TEXT, chunk_id TEXT UNIQUE, content TEXT, metadata JSONB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP );"""))
            inspector = sqlalchemy_inspect(engine)
            columns = [col['name'] for col in inspector.get_columns('document_chunks')]
            if 'collection_name' not in columns:
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN collection_name TEXT;"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_collection_doc_id_chunks ON document_chunks (collection_name, document_id);"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_fts_content ON document_chunks USING GIN(to_tsvector('{self.config.fts_language}',content));"))
            conn.commit()
        # conn.close() is handled by 'with' statement

    def _generate_expanded_queries(self, original_query: str, config: Optional[RunnableConfig] = None) -> List[str]:
        try:
            expanded_str = self._query_expansion_llm_chain.invoke({"question": original_query}, config=config)
            additional_queries = [q.strip() for q in expanded_str.split('\n') if q.strip()][:3]
            return [original_query] + additional_queries
        except Exception as e:
            print(f"[Query Expansion LLM] Error: {e}"); return [original_query]

    def _retrieve_for_one_query(self, query: str, config: Optional[RunnableConfig] = None) -> List[Document]:
        """Helper to retrieve for a single query. Expects query string as input."""
        if not isinstance(query, str):
            print(f"[RetrieverOneQuery] Expected string query, got {type(query)}: {query}")
            if isinstance(query, dict) and "question" in query and isinstance(query["question"], str):
                query = query["question"]
            elif isinstance(query, dict) and any(isinstance(v, str) for v in query.values()):
                query = next(v for v in query.values() if isinstance(v,str))
            else:
                return []
        return self.retriever.invoke(query, config=config)


    def _retrieve_for_multiple_queries(self, queries: List[str], config: Optional[RunnableConfig] = None) -> List[List[Document]]:
        """Retrieves documents for multiple queries using RunnableParallel for better tracing."""
        if not queries: return []
        tasks = {f"docs_for_query_{i}": RunnableLambda(self._retrieve_for_one_query) for i in range(len(queries))}
        if not tasks: return [[] for _ in queries] # Should not happen if queries is not empty
        parallel_retriever = RunnableParallel(**tasks)
        input_dict_for_parallel = {f"docs_for_query_{i}": q_str for i, q_str in enumerate(queries)}
        results_dict = parallel_retriever.invoke(input_dict_for_parallel, config=config)
        ordered_results: List[List[Document]] = []
        for i in range(len(queries)):
            task_key = f"docs_for_query_{i}"
            ordered_results.append(results_dict.get(task_key, []))
        return ordered_results

    def _reciprocal_rank_fusion(self, list_of_document_lists: List[List[Document]]) -> List[Document]:
        fused_scores: Dict[str, float] = {}; doc_map: Dict[str, Document] = {}
        k_rrf = self.config.rrf_k_for_fusion
        for doc_list in list_of_document_lists:
            for rank, doc in enumerate(doc_list):
                chunk_id = doc.metadata.get("chunk_id")
                if not chunk_id: continue # Skip if no chunk_id
                if chunk_id not in doc_map: doc_map[chunk_id] = doc
                rrf_score = 1.0 / (k_rrf + rank + 1)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_score
        sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        return [doc_map[cid] for cid in sorted_chunk_ids][:self.config.final_k]

    def _combine_documents_simple(self, list_of_document_lists: List[List[Document]]) -> List[Document]:
        all_docs: List[Document] = []; seen_chunk_ids = set()
        for doc_list in list_of_document_lists:
            for doc in doc_list:
                chunk_id = doc.metadata.get('chunk_id', '')
                if chunk_id and chunk_id not in seen_chunk_ids: # Ensure chunk_id exists and is unique
                    seen_chunk_ids.add(chunk_id); all_docs.append(doc)
        return all_docs[:self.config.final_k]

    def _get_answer_generation_chain(self) -> RunnableSequence:
        return self.base_rag_prompt | self.llm | StrOutputParser()

    def _build_rag_pipeline(
        self,
        retrieval_chain: RunnableSequence,
        expanded_info_updater: Optional[RunnableLambda] = None
    ) -> RunnableSequence:
        context_preparation = RunnablePassthrough.assign(
            context=itemgetter("final_sources") | RunnableLambda(format_docs)
        )
        answer_logic = {
            "answer": self._get_answer_generation_chain(),
            "sources": itemgetter("final_sources"),
            "expanded_info": itemgetter("expanded_info") # Pass through expanded_info
        }
        pipeline = retrieval_chain | context_preparation
        if expanded_info_updater: # If an updater is provided, apply it
            pipeline = pipeline | expanded_info_updater
        pipeline = pipeline | answer_logic # Apply final answer logic
        return pipeline

    def query(self, question: str, *, use_query_expansion: bool = False, use_rag_fusion: bool = False, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        openai_cb_result = {}
        # Initialize expanded_info. This will be updated by the pipeline if expansion/fusion is used.
        chain_input = {
            "question": question,
            "expanded_info": {"used": False, "queries": [question], "strategy": "Standard RAG"}
        }

        if use_rag_fusion:
            retrieval_chain_for_fusion = (
                RunnablePassthrough.assign(expanded_queries=itemgetter("question") | RunnableLambda(self._generate_expanded_queries))
                | RunnablePassthrough.assign(doc_lists=itemgetter("expanded_queries") | RunnableLambda(self._retrieve_for_multiple_queries))
                | RunnablePassthrough.assign(final_sources=itemgetter("doc_lists") | RunnableLambda(self._reciprocal_rank_fusion))
            )
            # Updater to correctly set expanded_info when RAG-Fusion is used
            fusion_info_updater = RunnablePassthrough.assign(
                expanded_info=lambda x: {**x["expanded_info"], "queries": x["expanded_queries"], "strategy": "RAG-Fusion (RRF)", "used": True}
            )
            active_chain = self._build_rag_pipeline(retrieval_chain_for_fusion, fusion_info_updater)
        elif use_query_expansion:
            retrieval_chain_for_qe = (
                RunnablePassthrough.assign(expanded_queries=itemgetter("question") | RunnableLambda(self._generate_expanded_queries))
                | RunnablePassthrough.assign(doc_lists=itemgetter("expanded_queries") | RunnableLambda(self._retrieve_for_multiple_queries))
                | RunnablePassthrough.assign(final_sources=itemgetter("doc_lists") | RunnableLambda(self._combine_documents_simple))
            )
            # Updater for query expansion
            qe_info_updater = RunnablePassthrough.assign(
                expanded_info=lambda x: {**x["expanded_info"], "queries": x["expanded_queries"], "strategy": "Query Expansion (Simple Combination)", "used": True}
            )
            active_chain = self._build_rag_pipeline(retrieval_chain_for_qe, qe_info_updater)
        else: # Standard RAG (no expansion, no fusion)
            retrieval_chain_standard = RunnablePassthrough.assign(
                final_sources=itemgetter("question") | self.retriever # Direct retrieval
            )
            # No specific updater needed here as expanded_info is already initialized for standard RAG
            active_chain = self._build_rag_pipeline(retrieval_chain_standard)

        with get_openai_callback() as cb:
            result = active_chain.invoke(chain_input, config=config)
            openai_cb_result = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}

        answer_text = result.get("answer", "エラー: 回答を生成できませんでした。") # Error: Could not generate answer.
        final_sources = result.get("sources", [])
        final_expanded_info = result.get("expanded_info", chain_input["expanded_info"]) # Get the potentially updated info

        sources_data = [{"excerpt": str(s.page_content)[:200] + ("…" if len(s.page_content) > 200 else ""),
                         "full_content": str(s.page_content),
                         "metadata": s.metadata or {}} for s in final_sources]
        return {
            "question": question,
            "answer": answer_text,
            "sources": sources_data,
            "usage": openai_cb_result,
            "query_expansion": final_expanded_info # Return the correct expansion info
        }

    # ===== SQL関連のメソッド (SQL-related methods) =====

    def create_table_from_file(self, file_path: str, table_name: Optional[str] = None) -> tuple[bool, str, str]:
        """指定されたファイルからデータベースにテーブルを作成または置換します。
        テーブル名はファイル名から自動生成されるか、指定されたものを使用します。
        成功した場合、(True, メッセージ, スキーマ情報) を返します。
        失敗した場合、(False, エラーメッセージ, "") を返します。

        (Creates or replaces a table in the database from the specified file.
        The table name is automatically generated from the file name or uses the specified one.
        Returns (True, message, schema information) on success.
        Returns (False, error message, "") on failure.)
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False, f"ファイルが見つかりません: {file_path}", "" # File not found:

            if not table_name:
                # ファイル名から安全なテーブル名を生成 (Generate safe table name from file name)
                # プレフィックスを追加してユーザーテーブルであることを明示 (Add prefix to indicate user table)
                clean_stem = re.sub(r'[^a-zA-Z0-9_]', '_', path.stem.lower())
                table_name = f"{self.config.user_table_prefix}{clean_stem}"


            df: Optional[pd.DataFrame] = None
            if path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif path.suffix.lower() == '.csv':
                encodings = ['utf-8', 'shift_jis', 'cp932', 'latin1'] # Added more encodings
                for enc in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    return False, "CSVファイルのエンコーディングエラー、または読み取りに失敗しました。", "" # CSV file encoding error, or failed to read.
            else:
                return False, f"サポートされていないファイル形式です: {path.suffix}", "" # Unsupported file format:

            if df is None or df.empty: # Check if df is None as well
                return False, "ファイルが空か、データフレームの読み込みに失敗しました。", "" # File is empty or failed to load dataframe.

            # データ型の最適化を試みる (Try to optimize data types)
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # 数値に変換できるか試す (Try to convert to numeric)
                        df[col] = pd.to_numeric(df[col], errors='raise') # Use 'raise' to catch errors
                    except (ValueError, TypeError):
                        try:
                            # 日付時刻に変換できるか試す (Try to convert to datetime)
                            df[col] = pd.to_datetime(df[col], errors='raise')
                        except (ValueError, TypeError):
                            pass # 変換できない場合は元の型のまま (If not convertible, keep original type)
            
            # カラム名をSQLにとって安全な形式に変換 (Convert column names to SQL-safe format)
            # 例: スペースや特殊文字をアンダースコアに置換 (Example: Replace spaces and special characters with underscores)
            df.columns = [re.sub(r'\s+', '_', col) for col in df.columns] # Replace spaces
            df.columns = [re.sub(r'[^0-9a-zA-Z_]', '', col) for col in df.columns] # Remove non-alphanumeric (except _)
            df.columns = [f'col_{i}' if not name else name for i, name in enumerate(df.columns)] # Ensure no empty names

            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                # テーブルが存在する場合は削除 (CASCADEで関連オブジェクトも削除)
                # (If table exists, delete it (also delete related objects with CASCADE))
                conn.execute(text(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE'))
                # データフレームをSQLテーブルとして書き込み (Write dataframe as SQL table)
                df.to_sql(table_name, conn, if_exists='replace', index=False, schema='public')
                conn.commit()

            schema_info = self._get_table_schema(table_name)
            return True, f"テーブル '{table_name}' が {len(df)} 行で作成/更新されました。", schema_info # Table '{X}' created/updated with {Y} rows.
        except Exception as e:
            return False, f"テーブル作成エラー: {type(e).__name__} - {str(e)}", "" # Table creation error:

    def _get_table_schema(self, table_name: str) -> str:
        """指定されたテーブルのスキーマ情報（カラム名、データ型、サンプルデータ）を取得します。

        (Gets schema information (column names, data types, sample data) for the specified table.)
        """
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                # information_schemaからカラム情報を取得 (Get column information from information_schema)
                # テーブル名とスキーマ名をSQLインジェクションから保護するためにパラメータ化
                # (Parameterize table name and schema name to protect from SQL injection)
                stmt_cols = text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name AND table_schema = 'public'
                    ORDER BY ordinal_position
                """)
                result_cols = conn.execute(stmt_cols, {"table_name": table_name})
                columns_info = result_cols.fetchall()

                if not columns_info:
                    return (f"テーブル名: \"{table_name}\"\n"
                            f"カラム情報: (このテーブルのカラム情報は見つかりませんでした。"
                            f"テーブルが存在しないか、publicスキーマにない可能性があります。)")
                            # Table name: "{X}"\nColumn information: (Column information for this table was not found. The table may not exist or may not be in the public schema.)

                schema = f"テーブル名: \"{table_name}\" (スキーマ: public)\nカラム情報:\n" # Table name: "{X}" (Schema: public)\nColumn information:
                for col_name, col_type in columns_info:
                    schema += f"  - \"{col_name}\": {col_type}\n"

                # サンプルデータを取得 (Get sample data)
                try:
                    # テーブル名を安全にクォート (Safely quote table name)
                    # PostgreSQLでは識別子に大文字小文字の区別がある場合や特殊文字が含まれる場合にダブルクォートが必要
                    # (In PostgreSQL, double quotes are required if identifiers are case-sensitive or contain special characters)
                    stmt_sample = text(f'SELECT * FROM public."{table_name}" LIMIT 3') # 3行に増やす (Increase to 3 rows)
                    sample_result = conn.execute(stmt_sample)
                    rows = sample_result.fetchall()

                    if rows:
                        schema += f"\nサンプルデータ (上位{len(rows)}行):\n" # Sample data (top {X} rows):
                        # sample_result.keys() を使用してカラム名を取得 (Use sample_result.keys() to get column names)
                        sample_column_names = list(sample_result.keys())
                        df_sample = pd.DataFrame(rows, columns=sample_column_names)
                        schema += df_sample.to_string(index=False, max_colwidth=50)
                    else:
                        schema += "\nサンプルデータ: (テーブルは空か、プレビュー可能なデータがありません。)\n" # Sample data: (Table is empty or no previewable data.)
                
                except Exception as e_sample:
                    schema += f"\nサンプルデータの取得中にエラーが発生しました: {type(e_sample).__name__} - {str(e_sample)}\n" # An error occurred while retrieving sample data:
                
                return schema
        except Exception as e_schema:
            print(f"Error in _get_table_schema for table '{table_name}': {type(e_schema).__name__} - {e_schema}")
            return (f"テーブル \"{table_name}\" のスキーマ情報取得に失敗しました。"
                    f"エラー: {type(e_schema).__name__} - {str(e_schema)}") # Failed to get schema information for table "{X}". Error:

    # New/Refactored internal method for execution and summarization
    def _execute_and_summarize_sql(self, original_question: str, generated_sql_query: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        # This method takes an already generated SQL query
        sql_llm_usage = {"total_tokens": 0, "cost": 0.0}
        try:
            if not generated_sql_query or not generated_sql_query.strip(): # Check if empty or whitespace
                return {
                    "success": False, "error": "実行すべきSQLクエリが提供されませんでした。", # No SQL query provided for execution.
                    "natural_language_answer": "実行すべきSQLクエリが提供されませんでした。", # No SQL query provided for execution.
                    "usage": sql_llm_usage, "generated_sql": "", "columns": [], "row_count_fetched": 0,
                    "full_results_sample": [], "results_preview": []
                }

            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                db_result = conn.execute(text(generated_sql_query))
                columns = list(db_result.keys()) # Get column names from the result set
                # fetchmanyの動作を制御するため、まずは全件取得を試みるが、多すぎる場合は制限する
                # (To control the behavior of fetchmany, first try to get all records, but limit if too many)
                # ここでは self.config.max_sql_results を上限とする
                # (Here, self.config.max_sql_results is the upper limit)
                rows = db_result.fetchmany(self.config.max_sql_results) 
            
            full_results_sample = [dict(zip(columns, row)) for row in rows]
            row_count_fetched = len(full_results_sample)
            
            # LLMに渡すプレビュー行数を制限 (Limit preview rows passed to LLM)
            preview_rows_for_llm = full_results_sample[:self.config.max_sql_preview_rows_for_llm]

            if preview_rows_for_llm:
                df_preview = pd.DataFrame(preview_rows_for_llm)
                sql_results_preview_str = df_preview.to_string(index=False, max_rows=self.config.max_sql_preview_rows_for_llm)
                if row_count_fetched > len(preview_rows_for_llm):
                    sql_results_preview_str += f"\n...他 {row_count_fetched - len(preview_rows_for_llm)} 件の結果があります（全{row_count_fetched}件中、最初の{len(preview_rows_for_llm)}件を表示）。" 
                    # ...and {X} more results (displaying first {Y} of all {Z} items).
                elif row_count_fetched > 0:
                    sql_results_preview_str += f"\n（全{row_count_fetched}件の結果を表示）。"
                    # (Displaying all {X} results).
            else:
                sql_results_preview_str = "クエリは成功しましたが、結果はありませんでした。" # Query was successful, but no results found.
            
            with get_openai_callback() as cb:
                answer_generation_payload = {
                    "original_question": original_question,
                    "sql_query": generated_sql_query,
                    "sql_results_preview_str": sql_results_preview_str,
                    "max_preview_rows": self.config.max_sql_preview_rows_for_llm,
                    "total_row_count": row_count_fetched 
                }
                natural_language_answer = self._sql_answer_generation_chain.invoke(answer_generation_payload, config=config)
                sql_llm_usage = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}

            return {
                "success": True, "question": original_question, "generated_sql": generated_sql_query,
                "natural_language_answer": natural_language_answer,
                "results_preview": preview_rows_for_llm, # This is the preview shown to LLM
                "full_results_sample": full_results_sample, # This is the data up to max_sql_results for UI display
                "row_count_fetched": row_count_fetched, "columns": columns, "usage": sql_llm_usage
            }
                
        except Exception as e:
            print(f"[_execute_and_summarize_sql] Error: {type(e).__name__} - {e}")
            error_message = f"SQLクエリ「{generated_sql_query}」の実行または結果の解釈中にエラーが発生しました: {type(e).__name__} - {str(e)}" # An error occurred during execution or result interpretation of SQL query "{X}":
            return {
                "success": False, "error": str(e), "results_preview": [],
                "natural_language_answer": error_message, "usage": sql_llm_usage,
                "generated_sql": generated_sql_query, "columns": [], "row_count_fetched": 0,
                "full_results_sample": []
            }

    # This is for Tab 6 (direct SQL analysis on a selected table)
    def execute_sql_query(self, question: str, table_name: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Generates SQL for a SINGLE specified table, then executes it and generates a natural language answer.
        """
        try:
            schema_info = self._get_table_schema(table_name)
            if "スキーマ情報取得に失敗しました" in schema_info or "カラム情報は見つかりませんでした" in schema_info : # Failed to get schema information / Column information not found
                 return {
                    "success": False, "error": f"テーブル {table_name} のスキーマ取得に失敗しました。スキーマ情報: {schema_info}", # Failed to retrieve schema for table {X}. Schema info:
                    "natural_language_answer": f"テーブル「{table_name}」のスキーマ情報を取得できませんでした。テーブルが存在するか確認してください。", # Could not retrieve schema information for table "{X}". Please check if the table exists.
                    "generated_sql": "", "columns": [], "row_count_fetched": 0, 
                    "full_results_sample": [], "results_preview": [],
                    "usage": {"total_tokens": 0, "cost": 0.0}
                }
            
            # 1. Generate SQL Query using the single-table SQL chain
            #    max_sql_results をプロンプトに含める (Include max_sql_results in the prompt)
            sql_generation_payload = {
                "question": question, 
                "schema_info": schema_info,
                "max_sql_results": self.config.max_sql_results 
            }
            sql_response_str = self._single_table_sql_chain.invoke(sql_generation_payload, config=config)
            generated_sql_for_single_table = self._extract_sql(sql_response_str)

            if not generated_sql_for_single_table:
                return {
                    "success": False, "error": "単一テーブルに対するSQL生成に失敗しました", # Single-table SQL generation failed
                    "results_preview": [],
                    "natural_language_answer": "申し訳ありませんが、指定されたテーブルに対するSQLクエリを生成できませんでした。", # I'm sorry, but I couldn't generate an SQL query for the specified table.
                    "generated_sql": "", "columns": [], "row_count_fetched": 0, "full_results_sample": [],
                    "usage": {"total_tokens": 0, "cost": 0.0}
                }
            
            # 2. Execute the generated SQL and get natural language answer using the common internal method
            return self._execute_and_summarize_sql(
                original_question=question,
                generated_sql_query=generated_sql_for_single_table,
                config=config
            )
        except Exception as e:
            print(f"[execute_sql_query for single table] Error: {type(e).__name__} - {e}")
            return {
                "success": False, "error": str(e), "results_preview": [],
                "natural_language_answer": f"指定テーブル「{table_name}」に対するSQL分析処理中にエラー: {str(e)}", # Error during SQL analysis processing for specified table "{X}":
                "generated_sql": "", "columns": [], "row_count_fetched": 0, "full_results_sample": [],
                "usage": {"total_tokens": 0, "cost": 0.0}
            }

    def _extract_sql(self, llm_output: str) -> str:
        """LLMの出力からSQLクエリを抽出します。
        マークダウンのコードブロック形式 (```sql ... ```) を想定しています。

        (Extracts SQL query from LLM output.
        Assumes Markdown code block format (```sql ... ```).)
        """
        match = re.search(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # コードブロックがない場合、出力全体をSQLと見なすか、特定のキーワードで始まる行を探す
        # (If no code block, consider the entire output as SQL or look for lines starting with specific keywords)
        # ここでは、単純に出力全体を返すか、SELECTで始まる最初の行を試す
        # (Here, simply return the entire output or try the first line starting with SELECT)
        lines = llm_output.strip().split('\n')
        for line in lines:
            if line.strip().upper().startswith("SELECT"):
                # SELECTで始まる行から最後までをSQLとみなす (Consider from the line starting with SELECT to the end as SQL)
                sql_candidate = "\n".join(lines[lines.index(line):])
                return sql_candidate.strip()
        return llm_output.strip() # Fallback to the whole output if no clear SQL block


    # This is for the unified chat flow
    def query_unified(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        ユーザーの質問を解釈し、SQL分析が適切かRAG検索が適切かを判断します。
        SQL分析が適切と判断された場合、関連するテーブルのスキーマ情報を基にSQLを生成し実行、結果を要約します。
        RAG検索が適切と判断された場合、通常のRAGパイプラインを実行します。

        (Interprets the user's question and determines whether SQL analysis or RAG search is appropriate.
        If SQL analysis is deemed appropriate, it generates and executes SQL based on the schema information of relevant tables and summarizes the results.
        If RAG search is deemed appropriate, it executes the normal RAG pipeline.)
        """
        tables = self.get_data_tables() # 既存のデータテーブルを取得 (Get existing data tables)
        run_config = kwargs.get("config")
        overall_usage = {"total_tokens": 0, "cost": 0.0} # To aggregate costs

        if not self.config.enable_text_to_sql or not tables: # SQL機能が無効、またはテーブルがない場合はRAGへ (If SQL function is disabled or no tables, go to RAG)
            rag_result = self.query(question, **kwargs) # kwargsを渡してconfigを伝播 (Pass kwargs to propagate config)
            rag_result["query_type"] = "rag"
            if not self.config.enable_text_to_sql:
                 rag_result["info"] = "Text-to-SQL is disabled."
            elif not tables:
                 rag_result["info"] = "No data tables found for SQL analysis."
            return rag_result
        
        try:
            # query_detection_prompt で使用するテーブル情報のサマリーを作成
            # (Create summary of table information for use in query_detection_prompt)
            tables_info_for_detection = []
            for t in tables:
                schema_summary = t.get('schema', '')
                # カラム名のみを抽出して簡潔にする試み (Attempt to extract only column names for brevity)
                col_lines = [line.strip() for line in schema_summary.split('\n') if line.strip().startswith('- "')]
                if col_lines: 
                    extracted_cols = [cl.split(':')[0].replace('- "', '').replace('"', '').strip() for cl in col_lines]
                    schema_summary_for_detection = ", ".join(extracted_cols)
                else: # カラム抽出失敗時はスキーマの先頭部分を使用 (If column extraction fails, use the beginning of the schema)
                    schema_summary_for_detection = schema_summary[:150] + "..." if len(schema_summary) > 150 else schema_summary
                tables_info_for_detection.append(f"- テーブル名 \"{t['table_name']}\" ({t.get('row_count', 'N/A')}行, カラム例: {schema_summary_for_detection})") # Table name "{X}" ({Y} rows, column examples: )


            tables_info_str_for_detection = "\n".join(tables_info_for_detection)
            if not tables_info_str_for_detection.strip(): # サマリーが空ならRAGへ (If summary is empty, go to RAG)
                 rag_result = self.query(question, **kwargs)
                 rag_result["query_type"] = "rag"
                 rag_result["info"] = "Could not generate table summary for SQL vs RAG detection."
                 return rag_result


            detection_payload = {"question": question, "tables_info": tables_info_str_for_detection}
            # decision_chain_config の設定 (Setting decision_chain_config)
            decision_chain_config = RunnableConfig(run_name="QueryTypeDetection", tags=["sql_rag_detection"])
            if run_config and run_config.get("callbacks"): # 親のコールバックを継承 (Inherit parent's callbacks)
                decision_chain_config["callbacks"] = run_config.get("callbacks")

            decision = self._detection_chain.invoke(detection_payload, config=decision_chain_config)
            
            if "SQL" in decision.upper(): # SQLパス (SQL path)
                # 1. 全てのテーブルの完全なスキーマ情報を収集 (Collect complete schema information for all tables)
                all_full_schemas_str = "\n\n---\n\n".join(
                    [t['schema'] for t in tables if t.get('schema')]
                )
                if not all_full_schemas_str.strip():
                    return { 
                        "query_type": "sql_error", "question": question, 
                        "answer": "分析可能なデータテーブルのスキーマ情報を取得できませんでした。RAG検索を試みます。", # Could not retrieve schema information for data tables. Trying RAG search.
                        "sql_details": {"success": False, "error": "SQL生成のためのスキーマ情報が見つかりません。"}, # Schema information for SQL generation not found.
                        "sources": [], "usage": overall_usage, "query_expansion": {}
                    }

                # 2. _multi_table_sql_chain を使用してSQLを生成
                # (Generate SQL using _multi_table_sql_chain)
                sql_generation_payload = {
                    "question": question, 
                    "schemas_info": all_full_schemas_str,
                    "max_sql_results": self.config.max_sql_results # プロンプトに含める (Include in prompt)
                }
                sql_gen_config = RunnableConfig(run_name="MultiTableSQLGenerationFromQuery", tags=["text_to_sql", "multi_table"])
                if run_config and run_config.get("callbacks"):
                    sql_gen_config["callbacks"] = run_config.get("callbacks")
                
                generated_sql_from_llm = self._multi_table_sql_chain.invoke(sql_generation_payload, config=sql_gen_config)
                actual_generated_sql = self._extract_sql(generated_sql_from_llm)

                if not actual_generated_sql:
                    return {
                        "query_type": "sql_error", "question": question,
                        "answer": "SQLクエリを生成できませんでした。質問をより具体的にするか、RAG検索をお試しください。", # Could not generate SQL query. Please make your question more specific or try RAG search.
                        "sql_details": {"success": False, "error": "LLMによる複数テーブルSQL生成失敗（SQL抽出できず）。", "generated_sql": generated_sql_from_llm}, # Multi-table SQL generation failed by LLM (could not extract SQL).
                        "sources": [], "usage": overall_usage, "query_expansion": {}
                    }
                
                # 3. 生成されたSQLを実行し、結果を要約 (Execute the generated SQL and summarize the results)
                sql_execution_details = self._execute_and_summarize_sql(
                    original_question=question,
                    generated_sql_query=actual_generated_sql,
                    config=run_config # 親のconfigを渡す (Pass parent's config)
                )
                
                if sql_execution_details.get("usage"):
                    overall_usage["total_tokens"] += sql_execution_details["usage"]["total_tokens"]
                    overall_usage["cost"] += sql_execution_details["usage"]["cost"]

                return {
                    "query_type": "sql", "question": question,
                    "answer": sql_execution_details.get("natural_language_answer", "SQLベースの回答を生成できませんでした。"), # Could not generate SQL-based answer.
                    "sql_details": sql_execution_details, 
                    "sources": [], "usage": overall_usage, "query_expansion": {}
                }
            else: # RAGパス (RAG path)
                rag_result = self.query(question, **kwargs)
                rag_result["query_type"] = "rag"
                return rag_result
                
        except Exception as e:
            print(f"[Query Unified] Error during type detection or execution: {type(e).__name__} - {e}")
            # エラー発生時は安全のためRAGにフォールバック (Fallback to RAG for safety in case of error)
            rag_result = self.query(question, **kwargs) 
            rag_result["query_type"] = "rag_fallback_error" # タイプを明確化 (Clarify type)
            rag_result["error_info_fallback"] = f"統合クエリ処理中にエラーが発生しRAGにフォールバックしました。エラー: {type(e).__name__} - {str(e)}" # An error occurred during unified query processing and fell back to RAG. Error:
            return rag_result

    def get_data_tables(self) -> List[Dict[str, Any]]:
        """
        データベース内に存在するユーザー作成のデータテーブルのリストを取得します。
        各テーブルについて、テーブル名、行数、スキーマ情報を含みます。
        ユーザーテーブルは Config で定義されたプレフィックス (`user_table_prefix`) を持つと想定します。

        (Retrieves a list of user-created data tables present in the database.
        For each table, includes table name, row count, and schema information.
        Assumes user tables have a prefix defined in Config (`user_table_prefix`).)
        """
        tables_data: List[Dict[str, Any]] = []
        engine = create_engine(self.connection_string)
        try:
            with engine.connect() as conn:
                # publicスキーマ内のテーブルで、指定されたプレフィックスを持つものを取得
                # (Get tables in the public schema that have the specified prefix)
                # SQLインジェクション対策のため、プレフィックスはPython側で処理
                # (To prevent SQL injection, process the prefix on the Python side)
                stmt_tables = text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """)
                result_tables = conn.execute(stmt_tables)
                
                all_public_tables = [row[0] for row in result_tables if row and row[0]]
                
                user_tables = [
                    tbl_name for tbl_name in all_public_tables 
                    if tbl_name.startswith(self.config.user_table_prefix)
                ]

                for table_name in user_tables:
                    try:
                        # 行数を取得 (Get row count)
                        # テーブル名を安全にクォート (Safely quote table name)
                        stmt_count = text(f'SELECT COUNT(*) FROM public."{table_name}"')
                        count_result = conn.execute(stmt_count).scalar_one_or_none()
                        row_count = count_result if count_result is not None else 0

                        # スキーマ情報を取得 (Get schema information)
                        schema_info = self._get_table_schema(table_name)
                        
                        tables_data.append({
                            "table_name": table_name,
                            "row_count": row_count,
                            "schema": schema_info
                        })
                    except Exception as e_table_detail:
                        print(f"テーブル '{table_name}' の詳細情報取得中にエラー: {type(e_table_detail).__name__} - {str(e_table_detail)}") # Error while getting detailed information for table '{X}':
                        # エラーが発生したテーブルはリストに含めないか、エラー情報と共に含めるか選択
                        # (Choose whether to not include tables that caused errors in the list, or include them with error information)
                        # ここでは含めない方針 (Here, the policy is not to include them)
                        continue 
            return tables_data
        except Exception as e_main:
            print(f"データテーブルリストの取得中にエラー: {type(e_main).__name__} - {str(e_main)}") # Error while getting data table list:
            return [] # エラー時は空のリストを返す (Return empty list on error)

    def delete_data_table(self, table_name: str) -> tuple[bool, str]:
        """指定されたデータテーブルをデータベースから削除します。

        (Deletes the specified data table from the database.)
        """
        if not table_name or not table_name.startswith(self.config.user_table_prefix): # 安全のためプレフィックスも確認 (Check prefix for safety)
            return False, f"無効なテーブル名、または削除が許可されていないテーブルです: {table_name}" # Invalid table name, or table not allowed to be deleted:

        engine = create_engine(self.connection_string)
        try:
            with engine.connect() as conn:
                # テーブル名を安全にクォート (Safely quote table name)
                conn.execute(text(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE'))
                conn.commit()
            return True, f"テーブル '{table_name}' は正常に削除されました。" # Table '{X}' deleted successfully.
        except Exception as e:
            print(f"テーブル '{table_name}' の削除中にエラー: {type(e).__name__} - {str(e)}") # Error while deleting table '{X}':
            return False, f"テーブル '{table_name}' の削除中にエラーが発生しました: {str(e)}" # An error occurred while deleting table '{X}':


    def load_documents(self, paths: List[str]) -> List[Document]:
        docs: List[Document] = [];
        for p_str in paths:
            path = Path(p_str)
            if not path.exists(): print(f"File not found: {p_str}"); continue
            suf = path.suffix.lower()
            try:
                if suf == ".pdf": docs.extend(PyPDFLoader(str(path)).load())
                elif suf in {".txt", ".md"}: docs.extend(TextLoader(str(path), encoding="utf-8").load())
                elif suf == ".docx":
                    try: docs.extend(UnstructuredFileLoader(str(path), mode="single", strategy="fast").load())
                    except Exception: docs.extend(Docx2txtLoader(str(path)).load())
                elif suf == ".doc" and TextractLoader: # Check if TextractLoader is available
                    try: docs.extend(TextractLoader(str(path)).load())
                    except Exception as te: print(f"TextractLoader error for {p_str}: {te}")
            except Exception as e: print(f"Error loading {p_str}: {type(e).__name__} - {e}")
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        out: List[Document] = [];
        for i, d in enumerate(docs):
            src = d.metadata.get("source", f"doc_source_{i}") # Use a default if "source" is missing
            doc_id = Path(src).name # Get filename as document_id
            try:
                doc_splits = text_splitter.split_documents([d]) # Split one document at a time
                for j, c in enumerate(doc_splits):
                    # Ensure metadata is a dictionary and combine safely
                    new_metadata = d.metadata.copy() if d.metadata else {}
                    new_metadata.update(c.metadata if c.metadata else {})
                    new_metadata.update({
                        "chunk_id": f"{doc_id}_{i}_{j}", # Unique chunk ID
                        "document_id": doc_id,
                        "original_document_source": src,
                        "collection_name": self.config.collection_name
                    })
                    c.metadata = new_metadata
                    out.append(c)
            except Exception as e: print(f"Error splitting document {src}: {type(e).__name__} - {e}")
        return out

    def ingest_documents(self, paths: List[str]):
        docs = self.load_documents(paths);
        if not docs: print("No documents loaded for ingestion."); return
        chunks = self.chunk_documents(docs);
        if not chunks: print("No chunks created from documents."); return
        
        # チャンクの検証を強化 (Strengthen chunk validation)
        valid_chunks = [
            c for c in chunks 
            if isinstance(c, Document) and 
               isinstance(c.metadata, dict) and 
               'chunk_id' in c.metadata and 
               c.page_content and 
               c.page_content.strip()
        ]
        if not valid_chunks: print("No valid chunks to ingest after validation."); return
        
        chunk_ids_for_vectorstore = [c.metadata['chunk_id'] for c in valid_chunks]
        
        try:
            self.vector_store.add_documents(valid_chunks, ids=chunk_ids_for_vectorstore);
            self._store_chunks_for_keyword_search(valid_chunks)
            print(f"Successfully ingested {len(valid_chunks)} chunks from {len(paths)} file(s).")
        except Exception as e:
            print(f"Error during vector store addition or keyword storage: {type(e).__name__} - {e}")


    def _store_chunks_for_keyword_search(self, chunks: List[Document]):
        eng = create_engine(self.connection_string);
        # SQLクエリを修正し、全ての必須フィールドを挿入または更新するようにする
        # (Modify SQL query to insert or update all required fields)
        sql = text("""
            INSERT INTO document_chunks(collection_name, document_id, chunk_id, content, metadata, created_at) 
            VALUES(:coll_name, :doc_id, :cid, :cont, :meta, CURRENT_TIMESTAMP) 
            ON CONFLICT(chunk_id) DO UPDATE SET 
                content = EXCLUDED.content, 
                metadata = EXCLUDED.metadata, 
                document_id = EXCLUDED.document_id, 
                collection_name = EXCLUDED.collection_name,
                created_at = CURRENT_TIMESTAMP; 
        """) # Added created_at
        try:
            with eng.connect() as conn, conn.begin(): # Use transaction
                for c in chunks:
                    if not (isinstance(c.metadata, dict) and 
                            'chunk_id' in c.metadata and 
                            'document_id' in c.metadata):
                        print(f"Skipping chunk due to missing metadata: {c.page_content[:50]}...")
                        continue
                    
                    # metadataがNoneの場合やJSON変換できない場合を考慮 (Consider cases where metadata is None or cannot be JSON converted)
                    meta_json = None
                    try:
                        meta_json = json.dumps(c.metadata or {})
                    except TypeError as te:
                        print(f"Could not serialize metadata for chunk_id {c.metadata.get('chunk_id')}: {te}. Storing as empty JSON object.")
                        meta_json = json.dumps({})

                    conn.execute(sql, {
                        "coll_name": self.config.collection_name, 
                        "doc_id": c.metadata["document_id"], 
                        "cid": c.metadata["chunk_id"], 
                        "cont": c.page_content, 
                        "meta": meta_json # Use serialized metadata
                    })
        except Exception as e:
            print(f"Error storing chunks for keyword search: {type(e).__name__} - {e}")


    def delete_document_by_id(self, document_id_to_delete: str) -> tuple[bool, str]:
        if not document_id_to_delete: return False, "ドキュメントIDは空にできません。" # Document ID cannot be empty.
        engine = create_engine(self.connection_string); chunk_ids_to_delete: List[str] = []; deleted_rows_table = 0
        try:
            with engine.connect() as conn, conn.begin(): # Use transaction
                # 削除対象のチャンクIDを取得 (Get chunk IDs to be deleted)
                res_proxy = conn.execute(
                    text("SELECT chunk_id FROM document_chunks WHERE document_id = :doc_id AND collection_name = :coll"),
                    {"doc_id": document_id_to_delete, "coll": self.config.collection_name}
                )
                chunk_ids_to_delete = [row[0] for row in res_proxy if row and row[0]]

                if not chunk_ids_to_delete:
                    return True, f"ドキュメントID '{document_id_to_delete}' に該当するチャンクはコレクション '{self.config.collection_name}' に見つかりませんでした。" # Chunks corresponding to document ID '{X}' were not found in collection '{Y}'.
                
                # document_chunks テーブルから削除 (Delete from document_chunks table)
                del_res = conn.execute(
                    text("DELETE FROM document_chunks WHERE document_id = :doc_id AND collection_name = :coll"),
                    {"doc_id": document_id_to_delete, "coll": self.config.collection_name}
                )
                deleted_rows_table = del_res.rowcount
                
                # PGVector (ベクトルストア) からも削除 (Also delete from PGVector (vector store))
                if self.vector_store and hasattr(self.vector_store, 'delete') and chunk_ids_to_delete:
                    try:
                        self.vector_store.delete(ids=chunk_ids_to_delete)
                    except Exception as e_vec_del:
                        # ベクトルストアからの削除失敗はログに残すが、処理は続行 (Log failure to delete from vector store, but continue processing)
                        print(f"ベクトルストアからのチャンク削除中にエラー (ドキュメントID: {document_id_to_delete}): {e_vec_del}") # Error while deleting chunks from vector store (document ID: {X}):
                        return False, f"ドキュメントID '{document_id_to_delete}' のチャンクをデータベースから {deleted_rows_table} 個削除しましたが、ベクトルストアからの削除中にエラーが発生しました: {e_vec_del}" # Deleted {X} chunks of document ID '{Y}' from the database, but an error occurred while deleting from the vector store:

            return True, f"ドキュメントID '{document_id_to_delete}' の {deleted_rows_table} 個のチャンクをデータベースから削除し、{len(chunk_ids_to_delete)} 個のベクトルをベクトルストアから削除対象としました。" # Deleted {X} chunks of document ID '{Y}' from the database and targeted {Z} vectors for deletion from the vector store.
        except Exception as e:
            return False, f"ドキュメントID '{document_id_to_delete}' の削除中にエラーが発生しました: {type(e).__name__} - {e}" # An error occurred while deleting document ID '{X}':

__all__ = ["Config", "RAGSystem"]
