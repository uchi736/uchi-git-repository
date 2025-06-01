"""
LangChain Expression Language (LCEL), pgvector, および AWS RDS for PostgreSQL を使用した
RAG (Retrieval Augmented Generation) システムの実装。
RetrievalQA を使用しない明示的な Chain 構成。
"""

# 標準ライブラリのインポート
import os
import json
from typing import List, Dict, Any, Optional, Callable, Tuple # 型ヒント用
from dataclasses import dataclass # データクラス作成用
from operator import itemgetter  #辞書やオブジェクトから特定の要素を取得する関数

# --- LangChainおよび関連ライブラリのバージョン表示（デバッグ用）---
try:
    from importlib.metadata import version as get_version
    def print_package_version(package_name: str):
        try:
            print(f"Using {package_name} version: {get_version(package_name)}")
        except Exception:
            print(f"Could not determine version for {package_name}. Is it installed?")

    print_package_version('langchain')
    print_package_version('langchain-core')
    print_package_version('langchain-postgres') # langchain_postgresへの移行を推奨するため表示
    print_package_version('langchain-openai')
    print_package_version('langchain-community') # 現在ユーザーが使用しているPGVectorの提供元
    print_package_version('unstructured') # .docxなどのファイル読み込み用

except ImportError:
    print("importlib.metadata not available (Python < 3.8) or other import error. Version printing skipped.")
# --- バージョン表示ここまで ---

# LangChain関連のコンポーネントのインポート
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader # 各種ドキュメントローダー
from langchain.text_splitter import RecursiveCharacterTextSplitter # テキスト分割用
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # OpenAIのEmbeddingモデルとチャットモデル
from langchain_community.vectorstores import PGVector # 現在使用しているPGVector (langchain_community版)
from langchain_community.vectorstores.pgvector import DistanceStrategy # PGVectorの距離戦略
from sqlalchemy import create_engine, text # PostgreSQL操作のためのSQLAlchemy
from langchain_core.documents import Document # LangChainの基本データ構造であるDocument
from langchain_core.retrievers import BaseRetriever # カスタムRetrieverの基底クラス
from langchain_core.callbacks import CallbackManagerForRetrieverRun # Retrieverのコールバック用
from langchain_core.prompts import ChatPromptTemplate # チャットプロンプトテンプレート
from langchain_core.runnables import RunnablePassthrough, RunnableConfig # LCELのコンポーネント
from langchain_core.output_parsers import StrOutputParser # 出力パーサー（文字列として出力）
from langchain_community.callbacks.manager import get_openai_callback # OpenAIのトークン数などを取得するコールバック
from pathlib import Path # ファイルパス操作用
from dotenv import load_dotenv # .envファイルから環境変数を読み込む

# .envファイルから環境変数をロード
# (例: OPENAI_API_KEY="sk-...", DB_HOST="localhost", 等を.envファイルに記述)
load_dotenv()

# PostgreSQL接続ドライバ(psycopg)のインポート試行
try:
    import psycopg # psycopg v3 (推奨)
except ImportError:
    try:
        import psycopg2 as psycopg # psycopg v2 (フォールバック)
    except ImportError:
        psycopg = None # ドライバが見つからない場合はNoneに設定
        print("CRITICAL: psycopg or psycopg2 (PostgreSQL driver) not found.")
        print("Please install it: pip install psycopg OR pip install psycopg2-binary")

# 設定値を保持するデータクラス
@dataclass
class Config:
    """
    RAGシステム全体の設定を管理するデータクラス。
    環境変数またはデフォルト値から設定を読み込む。
    """
    db_host: str = os.getenv("DB_HOST", "localhost")  # データベースホスト
    db_port: str = os.getenv("DB_PORT", "5432")      # データベースポート
    db_name: str = os.getenv("DB_NAME", "postgres")  # データベース名
    db_user: str = os.getenv("DB_USER", "postgres")  # データベースユーザー
    db_password: str = os.getenv("DB_PASSWORD", "your-password") # データベースパスワード (要変更)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") # OpenAI APIキー
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002") # Embeddingモデル名
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o") # 大規模言語モデル(LLM)名
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000")) # ドキュメント分割時のチャンクサイズ
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200")) # チャンク間のオーバーラップ文字数
    vector_search_k: int = int(os.getenv("VECTOR_SEARCH_K", "10")) # ベクトル検索で取得する上位K件
    keyword_search_k: int = int(os.getenv("KEYWORD_SEARCH_K", "10")) # キーワード検索で取得する上位K件
    final_k: int = int(os.getenv("FINAL_K", "5")) # 最終的にLLMに渡すコンテキストの数
    collection_name: str = os.getenv("COLLECTION_NAME", "documents") # PGVectorのコレクション名
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536")) # Embeddingベクトルの次元数 (ada-002は1536)

# ハイブリッド検索を行うカスタムRetrieverクラス
class HybridRetriever(BaseRetriever):
    """
    ベクトル検索とキーワード検索の結果を組み合わせるハイブリッドRetriever。
    BaseRetrieverを継承し、Pydanticモデルとしても機能する。
    """
    vector_store: PGVector          # ベクトル検索に使用するPGVectorストア
    connection_string: str        # キーワード検索用のDB接続文字列 (SQLAlchemy形式)
    config_params: Config         # RAGシステムの設定オブジェクト

    # PydanticのBaseModelを継承する場合、フィールドを上記のように型定義しておけば、
    # これらのフィールドをキーワード引数として受け付ける __init__ が自動的に提供される。
    # 明示的な __init__ がフィールド割り当て以外に特別なロジックを持たない場合は削除可能。

    def _vector_search(self, query: str) -> List[Tuple[Document, float]]:
        """指定されたクエリでベクトル検索を実行する。"""
        if not self.vector_store:
            print("HybridRetriever: Vector store not available for vector search.")
            return []
        try:
            # PGVectorストアのsimilarity_search_with_scoreメソッドで類似度検索を実行
            return self.vector_store.similarity_search_with_score(query, k=self.config_params.vector_search_k)
        except Exception as e:
            print(f"Error during vector search: {e}")
            return []

    def _keyword_search(self, query: str) -> List[Tuple[Document, float]]:
        """指定されたクエリでキーワード検索（全文検索）を実行する。"""
        engine = create_engine(self.connection_string) # SQLAlchemyエンジンを作成
        docs_with_scores: List[Tuple[Document, float]] = []
        try:
            with engine.connect() as conn:
                # PostgreSQLの全文検索機能(ts_vector, plainto_tsquery, ts_rank)を使用
                # document_chunks テーブルに対して検索を行う
                result = conn.execute(text("""
                    SELECT chunk_id, content, metadata,
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', :query)) AS score
                    FROM document_chunks
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                    ORDER BY score DESC LIMIT :k
                """), {'query': query, 'k': self.config_params.keyword_search_k})
                
                for row in result: # type: ignore
                    # 検索結果を行ごとに処理
                    metadata_val = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or '{}') # type: ignore
                    doc = Document(page_content=str(row.content), metadata=metadata_val) # type: ignore
                    docs_with_scores.append((doc, float(row.score))) # type: ignore
        except Exception as e:
            print(f"Error during keyword search: {e}")
        return docs_with_scores

    def _reciprocal_rank_fusion(self, vector_results: List[Tuple[Document, float]], 
                                keyword_results: List[Tuple[Document, float]], k_rrf: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) を使用して、ベクトル検索とキーワード検索の結果を統合する。
        異なる検索結果のランキングを考慮してスコアを付け、最終的なランキングを作成する。
        """
        fused_scores: Dict[str, Dict[str, Any]] = {} # ドキュメントIDをキーとしたスコア格納用辞書

        # ドキュメントの一意なIDを取得するヘルパー関数 (chunk_id があればそれを優先)
        def get_doc_id(doc: Document) -> str:
            return doc.metadata.get('chunk_id', doc.page_content[:100]) # 最初の100文字をフォールバックIDとする

        # ベクトル検索結果の処理
        for rank, (doc, _) in enumerate(vector_results, 1): # rankは1から始まる
            doc_id = get_doc_id(doc)
            # RRFスコアを加算 (1 / (k + rank))
            fused_scores.setdefault(doc_id, {'document': doc, 'score': 0.0})['score'] += 1.0 / (k_rrf + rank)

        # キーワード検索結果の処理
        for rank, (doc, _) in enumerate(keyword_results, 1):
            doc_id = get_doc_id(doc)
            fused_scores.setdefault(doc_id, {'document': doc, 'score': 0.0})['score'] += 1.0 / (k_rrf + rank)
        
        # 統合スコアに基づいて結果を降順にソート
        sorted_results_with_scores = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        
        # 上位 K 件のドキュメントを返す
        return [item['document'] for item in sorted_results_with_scores[:self.config_params.final_k]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any
    ) -> List[Document]:
        """
        BaseRetrieverクラスで実装が必須のメソッド。
        指定されたクエリに関連するドキュメントのリストを返す。
        ここではハイブリッド検索（ベクトル検索 + キーワード検索 + RRF）を実行する。
        """
        vector_results = self._vector_search(query)
        keyword_results = self._keyword_search(query)
        return self._reciprocal_rank_fusion(vector_results, keyword_results)

    async def _aget_relevant_documents( # 非同期版 (現在は同期版を呼び出している)
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any
    ) -> List[Document]:
        """ BaseRetrieverクラスで実装が必須の非同期メソッド。 """
        # 本格的な非同期処理のためには、_vector_search と _keyword_search も非同期化する必要がある
        return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)


# RAGシステム全体を管理するクラス
class RAGSystem:
    """
    RAGパイプライン全体をカプセル化するクラス。
    ドキュメントの取り込み、検索、質問応答の機能を提供する。
    """
    def __init__(self, config_params: Config):
        """
        RAGSystemのコンストラクタ。
        設定オブジェクトを受け取り、必要なコンポーネントを初期化する。
        """
        self.config = config_params # 設定を格納

        # Embeddingモデルの初期化 (OpenAIのtext-embedding-ada-002など)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config_params.openai_api_key, 
            model=config_params.embedding_model
        )
        # 大規模言語モデル(LLM)の初期化 (OpenAIのgpt-4oなど)
        self.llm = ChatOpenAI(
            openai_api_key=config_params.openai_api_key, 
            model_name=config_params.llm_model, 
            temperature=0.7 # 生成テキストの多様性を調整 (0に近いほど決定的)
        )
        # テキスト分割ツールの初期化 (RecursiveCharacterTextSplitter)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config_params.chunk_size, 
            chunk_overlap=config_params.chunk_overlap
        )
        # データベース接続文字列の生成
        self.connection_string = self._get_connection_string()

        # 各コンポーネントをNoneで初期化 (後で _setup_chain で設定)
        self.vector_store: Optional[PGVector] = None
        self.retriever: Optional[HybridRetriever] = None
        self.chain = None # LCELチェーン

        # データベースの初期化（テーブル作成など）
        self._initialize_database()
        # RAGチェーンのセットアップ
        self._setup_chain()      

    def _get_connection_string(self) -> str:
        """PostgreSQLへの接続文字列を生成する (SQLAlchemy形式)。"""
        return (
            f"postgresql+psycopg://{self.config.db_user}:{self.config.db_password}"
            f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )

    def _initialize_database(self):
        """
        データベースの初期設定を行う。
        pgvector拡張機能の有効化と、キーワード検索用の `document_chunks` テーブルを作成する。
        """
        engine = create_engine(self.connection_string) # SQLAlchemyエンジンを作成
        try:
            with engine.connect() as conn:
                # pgvector拡張機能がなければ作成
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit() # トランザクションをコミット

                # キーワード検索およびドキュメントチャンク格納用のテーブルを作成
                # (embeddingカラムはPGVectorが直接管理するテーブルとは別)
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY, 
                        document_id TEXT, 
                        chunk_id TEXT UNIQUE, 
                        content TEXT,
                        metadata JSONB, 
                        embedding VECTOR({self.config.embedding_dimensions}), 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )"""))
                conn.commit()

                # contentカラムに対する全文検索インデックス(GIN)を作成
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_content_fts 
                    ON document_chunks 
                    USING GIN(to_tsvector('english', content))
                """))
                conn.commit()
                print("Database initialized (vector extension and document_chunks table ensured).")
        except Exception as e:
            print(f"Error initializing database table 'document_chunks': {e}")
            print("Please ensure PostgreSQL is running and accessible, and the user has rights.")

    def _setup_chain(self):
        """
        RAGパイプラインの主要コンポーネント（VectorStore, Retriever, LCEL Chain）をセットアップする。
        """
        pgvector_instance = None # 初期化
        try:
            # --- PGVectorの初期化 ---
            # 現在のユーザーのコードでは langchain_community.vectorstores.PGVector を使用。
            # このバージョンは __init__ で connection_string と embedding_function を取る。
            # (注意: このPGVectorは非推奨です。langchain_postgresへの移行を推奨します。)
            print(f"Attempting to initialize PGVector from 'langchain_community.vectorstores' with collection: {self.config.collection_name}")
            print("This PGVector (from langchain_community) is pending deprecation. Consider migrating to 'langchain_postgres.vectorstores.PGVector'.")
            
            pgvector_instance = PGVector( # langchain_community.vectorstores.PGVector の初期化
                connection_string=self.connection_string,
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings, # langchain_community版は embedding_function を期待
                use_jsonb=True, # メタデータをJSONB型で格納
                distance_strategy=DistanceStrategy.COSINE, # 類似度計算の戦略 (コサイン類似度)
            )
            self.vector_store = pgvector_instance # 初期化したものをRAGSystemの属性に設定
            print("PGVector (from langchain_community) initialized successfully.")

            # --- HybridRetrieverの初期化 ---
            # 修正点: HybridRetriever のインスタンス化でキーワード引数を使用
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,             # キーワード引数 vector_store に設定
                connection_string=self.connection_string,   # キーワード引数 connection_string に設定
                config_params=self.config                   # キーワード引数 config_params に設定
            )
            print("HybridRetriever initialized successfully.")

            # --- LLMに渡すプロンプトテンプレートの定義 ---
            prompt_template = ChatPromptTemplate.from_template(
                "あなたは親切で知識豊富なアシスタントです。以下のコンテキストを使用して質問に答えてください。\n"
                "回答は正確で、コンテキストに基づいている必要があります。\n"
                "コンテキストに答えが見つからない場合は、「提供された情報からは回答できません」と答えてください。\n\n"
                "コンテキスト:\n{context}\n\n質問: {question}\n\n回答:"
            )

            # --- LangChain Expression Language (LCEL) を使用したChainの構築 ---
            # 処理の流れ:
            # 1. 質問 (question) を受け取る
            # 2. Retrieverが質問に基づいて関連コンテキスト (context) を取得
            #    - context の取得: itemgetter("question")で質問を取り出し、retrieverに渡し、結果を_format_docsで整形
            # 3. 整形されたコンテキストと元の質問をプロンプトテンプレートに渡す
            # 4. プロンプトをLLMに渡し、回答を生成
            # 5. LLMの出力を文字列として解析
            self.chain = (
                {
                    "context": itemgetter("question") | self.retriever | self._format_docs, # type: ignore
                    "question": itemgetter("question") # 元の質問をそのまま渡す
                }
                | prompt_template  # プロンプトテンプレートに上記の辞書を渡す
                | self.llm         # LLMにプロンプトを渡す
                | StrOutputParser() # LLMの出力を文字列に変換
            )
            print("RAG chain setup complete.")

        except Exception as e: # _setup_chain全体でエラーが発生した場合
            error_source = "PGVector or Retriever initialization" # デフォルトのエラー源
            # Pydanticのバリデーションエラーか判定
            if 'validation error' in str(e).lower() and hasattr(e, 'errors'):
                error_source = f"HybridRetriever Pydantic validation ({len(e.errors())} errors)" # type: ignore
            
            print(f"エラー ({error_source}): {e}") # エラーメッセージを表示
            # PGVector.from_connection_stringに関するエラーだった場合の追加情報
            if "PGVector" in error_source and "from_connection_string" in str(e): # この条件は現状のコードでは通りにくい
                 print("This typically means you are trying to use an API from 'langchain_postgres' with 'langchain_community.PGVector'.")
                 print("Ensure your imports and PGVector initialization method match the library version you intend to use.")
            # _setup_chainでエラーが発生した場合、self.chainはNoneのままとなる。
            # main関数側でこの状態をチェックし、プログラムを安全に終了させる。

    def _format_docs(self, docs: List[Document]) -> str:
        """取得されたドキュメントのリストをLLMのコンテキストに適した単一の文字列に整形する。"""
        if not docs:
            return "利用可能なコンテキスト情報がありません。"
        # 各ドキュメントの内容を結合して返す
        return "\n\n".join(
            [f"[ソース {i+1} chunk_id {doc.metadata.get('chunk_id', 'N/A')}]:\n{doc.page_content}" 
             for i, doc in enumerate(docs)]
        )

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        指定されたファイルパスのリストからドキュメントを読み込む。
        PDF, TXT, MD, DOC, DOCX形式のファイルをサポート（DOC/DOCXはunstructuredが必要）。
        """
        documents = [] # 読み込んだドキュメントを格納するリスト
        for file_path_str in file_paths: # 各ファイルパスについて処理
            file_path = Path(file_path_str) # Pathオブジェクトに変換
            print(f"[load_documents] Processing: '{file_path_str}', Exists: {file_path.exists()}")

            if not file_path.exists(): # ファイルが存在しない場合はスキップ
                print(f"File not found: '{file_path_str}'. Skipping.")
                continue

            current_suffix = file_path.suffix.lower() # ファイルの拡張子を小文字で取得
            print(f"[load_documents] Suffix for '{file_path_str}': '{current_suffix}'")

            loader_used = False # このファイルに対してローダーが使用されたかのフラグ
            try:
                if current_suffix == '.pdf':
                    print(f"[load_documents] Using PyPDFLoader for '{file_path_str}'.")
                    loader = PyPDFLoader(str(file_path))
                    documents.extend(loader.load())
                    loader_used = True
                elif current_suffix in ['.txt', '.md']:
                    print(f"[load_documents] Using TextLoader for '{file_path_str}'.")
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    documents.extend(loader.load())
                    loader_used = True
                elif current_suffix in ['.doc', '.docx']:
                    print(f"[load_documents] Attempting UnstructuredFileLoader for '{file_path_str}'.")
                    try:
                        # UnstructuredFileLoaderは様々な非構造化データに対応
                        # mode="single" はファイルを単一のドキュメントとして扱う
                        # strategy="fast" は高速な処理を試みる
                        loader = UnstructuredFileLoader(str(file_path), mode="single", strategy="fast")
                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)
                        loader_used = True
                        print(f"[load_documents] Successfully loaded '{file_path_str}' with UnstructuredFileLoader ({len(loaded_docs)} docs).")
                    except ImportError: # unstructuredライブラリがない場合
                        print(f"[load_documents] ImportError for UnstructuredFileLoader. Ensure 'unstructured' and its .docx dependencies are installed.")
                        print("Try: pip install \"unstructured[docx]\"")
                    except Exception as e_unstructured: # UnstructuredFileLoaderのその他のエラー
                        print(f"[load_documents] Error using UnstructuredFileLoader for '{file_path_str}': {e_unstructured}")
                
                if not loader_used: # どのローダーにもマッチしなかった場合
                    print(f"[load_documents] No suitable loader explicitly handled suffix '{current_suffix}' for file '{file_path_str}'. Skipping.")

            except Exception as e_outer: # ファイル読み込み中の予期せぬエラー
                print(f"[load_documents] General error loading document '{file_path_str}': {e_outer}")
        
        if not documents: # 最終的に何もドキュメントが読み込めなかった場合
             print("[load_documents] No documents were successfully loaded from any of the provided paths.")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """読み込んだドキュメントを指定されたチャンクサイズとオーバーラップで分割する。"""
        all_chunks = [] # 分割されたチャンクを格納するリスト
        for doc_idx, doc in enumerate(documents): # 各ドキュメントについて処理
            try:
                # 元のファイル名などをメタデータから取得
                doc_source_path = doc.metadata.get('source', f"unknown_source_doc_idx_{doc_idx}")
                source_name = Path(doc_source_path).name # ファイル名のみ抽出

                # text_splitterでドキュメントをチャンクに分割
                doc_chunks = self.text_splitter.split_documents([doc])
                for chunk_idx, chunk in enumerate(doc_chunks):
                    if chunk.metadata is None: # チャンクのメタデータがなければ初期化
                        chunk.metadata = {}
                    
                    # 元のドキュメントのメタデータをチャンクに引き継ぎ、チャンク固有の情報を追加
                    original_doc_metadata = doc.metadata.copy() # 安全のためコピー
                    original_doc_metadata.update(chunk.metadata) # チャンクのメタデータを上書き/追加
                    chunk.metadata = original_doc_metadata
                    
                    chunk.metadata['original_document_source'] = source_name # 元のファイル名
                    chunk.metadata['document_source_path'] = doc_source_path # 元のフルパス
                    chunk.metadata['chunk_id'] = f"{source_name}_{doc_idx}_{chunk_idx}" # チャンクの一意なID
                    all_chunks.append(chunk)
            except Exception as e:
                print(f"Error chunking document (source: {doc.metadata.get('source', 'unknown')}): {e}")
        return all_chunks


    def ingest_documents(self, file_paths: List[str]):
        """
        指定されたファイルパスからドキュメントを読み込み、チャンク化し、
        ベクトルストア（PGVector）およびキーワード検索用テーブルに格納する。
        """
        print("Starting document ingestion process...")
        documents = self.load_documents(file_paths) # 1. ドキュメント読み込み
        if not documents: 
            print("No documents were loaded. Ingestion process aborted.")
            return
        print(f"Successfully loaded {len(documents)} documents from paths.")

        chunks = self.chunk_documents(documents) # 2. ドキュメントをチャンク化
        if not chunks: 
            print("No chunks were created from the loaded documents. Ingestion process aborted.")
            return
        print(f"Created {len(chunks)} chunks from the documents.")

        if not self.vector_store: # ベクトルストアが初期化されていなければ中止
            print("Vector store is not initialized (critical error during _setup_chain). Aborting ingestion.")
            return
        
        print(f"Ingesting {len(chunks)} chunks into PGVector collection '{self.config.collection_name}'...")
        try:
            # 3. チャンクをPGVectorに格納 (Embedding生成もここで行われる)
            self.vector_store.add_documents(chunks) 
            print("Chunks successfully added to PGVector.")
            
            # 4. (オプション) キーワード検索用の別テーブルにもチャンク情報を保存
            # print("Storing/updating chunk details in 'document_chunks' table for keyword search...")
            # self._store_chunks_for_keyword_search(chunks) 
            print("Document ingestion process complete!")
        except Exception as e:
            print(f"Error during document ingestion into PGVector: {e}")

    # _store_chunks_for_keyword_search: HybridRetrieverがPGVector内の情報だけで
    # キーワード検索を代替できない場合に、このメソッドで別途テーブルに情報を保存する。
    # PGVectorが管理するコレクション（langchain_pg_collectionなど）とのデータの二重管理や
    # スキーマの衝突に注意が必要。
    def _store_chunks_for_keyword_search(self, chunks: List[Document]):
        """チャンク情報をキーワード検索用の `document_chunks` テーブルに保存する。"""
        engine = create_engine(self.connection_string)
        try:
            with engine.connect() as conn:
                for chunk in chunks:
                    metadata_json = json.dumps(chunk.metadata or {}) # メタデータをJSON文字列に変換
                    # embeddingカラムに保存する場合は、ここでembeddingベクトルを取得する処理が必要
                    #例: chunk_embedding_vector = self.embeddings.embed_query(chunk.page_content)
                    conn.execute(text("""
                        INSERT INTO document_chunks (document_id, chunk_id, content, metadata) 
                        VALUES (:doc_id, :chunk_id, :content, :metadata)
                        ON CONFLICT (chunk_id) DO UPDATE  -- chunk_idが重複した場合は更新
                        SET content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata
                            -- embedding = EXCLUDED.embedding -- embeddingも更新する場合
                    """), {
                        'doc_id': chunk.metadata.get('original_document_source', 'unknown_source'),
                        'chunk_id': chunk.metadata['chunk_id'], 
                        'content': chunk.page_content,
                        'metadata': metadata_json,
                        # 'embedding': chunk_embedding_vector, # ここでベクトルを渡す
                    })
                conn.commit()
            print(f"Stored/Updated {len(chunks)} chunks in 'document_chunks' table.")
        except Exception as e:
            print(f"Error storing chunks in 'document_chunks' table: {e}")


    def query(self, question: str, query_config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        ユーザーからの質問を受け取り、RAGチェーンを実行して回答と関連ソースを返す。
        """
        print(f"\nProcessing query: {question}")
        if not self.chain: # RAGチェーンがセットアップされていなければエラー
            print("Error: RAG chain is not set up. Cannot process query.")
            return {'question': question, 'answer': "エラー: RAGシステムが準備できていません (Chain not set up)。", 'sources': []}
        
        retrieved_docs_for_display: List[Document] = [] # 表示用の取得ドキュメント
        answer_content = "エラーが発生しました。" # デフォルトの回答
        try:
            # OpenAIのコールバックを使用してトークン数とコストを計測
            with get_openai_callback() as cb:
                # LCELチェーンを実行して回答を取得
                answer_content = self.chain.invoke({"question": question}, config=query_config)
                print(f"LLM tokens used: {cb.total_tokens}, Total cost (USD): ${cb.total_cost:.6f}")
            
            # 表示用に再度ドキュメントを取得 (デバッグやUI表示のため)
            # (注意: chain内でretrieverは既に実行されているため、これは二度目の実行になる場合がある)
            if self.retriever:
                 retrieved_docs_for_display = self.retriever.get_relevant_documents(question) # type: ignore

        except Exception as e:
            print(f"Error during query processing: {e}")
            answer_content = "申し訳ありませんが、質問の処理中にエラーが発生しました。"

        # 結果を辞書形式で返す
        return {
            'question': question, 
            'answer': answer_content,
            'sources': [
                {'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content, # 内容を一部表示
                 'metadata': doc.metadata} 
                for doc in retrieved_docs_for_display
            ]
        }

# メイン実行ブロック
def main():
    """
    スクリプトのエントリーポイント。
    設定の読み込み、RAGシステムの初期化、ファイル取り込み、対話ループの実行を行う。
    """
    # PostgreSQLドライバの存在確認
    if psycopg is None:
        print("CRITICAL: PostgreSQL driver (psycopg or psycopg2) not found. Please install one.")
        print("Run: pip install psycopg OR pip install psycopg2-binary")
        return # ドライバがなければ終了

    # 設定クラスのインスタンス化 (環境変数またはデフォルト値が読み込まれる)
    app_config = Config()

    # OpenAI APIキーの存在確認
    if not app_config.openai_api_key:
        print("CRITICAL ERROR: OPENAI_API_KEY is not set in your .env file or environment variables.")
        print("The application cannot proceed without an OpenAI API key.")
        return
    
    # DBパスワードがデフォルトのままか確認 (localhost以外の場合)
    if app_config.db_password == "your-password" and app_config.db_host != "localhost":
        print("WARNING: The default database password 'your-password' is being used for a non-localhost DB.")
        print("Please set a strong, unique DB_PASSWORD in your .env file or environment variables.")
    
    print("Initializing RAGSystem...")
    rag_system_instance: Optional[RAGSystem] = None # 初期化
    try:
        # RAGSystemのインスタンスを作成 (内部で _setup_chain が呼ばれる)
        rag_system_instance = RAGSystem(app_config)
        # _setup_chain でエラーが発生した場合、rag_system_instance.chain は None になる
        if rag_system_instance.chain is None:
            print("\nCRITICAL: RAGSystem chain initialization failed. Review errors above (e.g., PGVector, Retriever).")
            print("If using 'langchain_community.vectorstores.PGVector', be aware of its deprecation status.")
            print("Strongly consider migrating to 'langchain_postgres.vectorstores.PGVector' and updating your libraries.")
            print("Exiting application.")
            return # chainがセットアップできなければ終了
        print("RAGSystem initialized successfully.")
    except Exception as e: # RAGSystemの初期化中に予期せぬエラーが発生した場合
        print(f"\nCRITICAL: An unexpected error occurred during RAGSystem initialization: {e}")
        print("Exiting application.")
        return

    # ファイル取り込みの対話的入力
    file_paths_input_str = input("取り込むファイルのパスをカンマ区切りで入力してください（スキップする場合はEnter）: ").strip()
    if file_paths_input_str: # 何か入力があった場合
        # カンマで分割し、各パスの空白を除去
        raw_paths = [p.strip() for p in file_paths_input_str.split(',') if p.strip()]
        file_paths_to_ingest = []
        for p_path_str in raw_paths:
            # パスが引用符で囲まれていたら除去 (Windowsのコピペ対策など)
            if len(p_path_str) >= 2 and p_path_str.startswith('"') and p_path_str.endswith('"'):
                file_paths_to_ingest.append(p_path_str[1:-1])
            elif len(p_path_str) >= 2 and p_path_str.startswith("'") and p_path_str.endswith("'"):
                file_paths_to_ingest.append(p_path_str[1:-1])
            else:
                file_paths_to_ingest.append(p_path_str)
        
        if file_paths_to_ingest: # 有効なパスがあれば取り込み実行
            print(f"Attempting to ingest the following files: {file_paths_to_ingest}")
            rag_system_instance.ingest_documents(file_paths_to_ingest)
        else:
            print("No valid file paths provided for ingestion.")
    else: # 何も入力されなかった場合
        print("No files specified for ingestion. Proceeding with existing data or an empty vector store.")

    # 質問応答の対話ループ
    print("\nReady to answer questions.")
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ").strip()
        if question.lower() == 'quit': # 'quit'でループ終了
            break
        if not question: # 空の入力は無視
            print("Please enter a question."); 
            continue
        
        # RAGシステムに質問を投げて結果を取得
        result = rag_system_instance.query(question)
        
        # 結果の表示
        print("\n" + "="*80)
        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        if result.get('sources'): # ソース情報があれば表示
            print("\nSources:")
            for i, source in enumerate(result['sources']):
                print(f"\n{i+1}. {source.get('content', 'N/A')}")
                print(f"   Metadata: {source.get('metadata', '{}')}")
        else:
            print("\nNo sources found for this answer.")
        print("="*80)

if __name__ == "__main__":
    # スクリプトが直接実行された場合にmain()関数を呼び出す
    main()