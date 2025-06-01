"""streamlit_rag_ui_hybrid.py – Hybrid Modern RAG System with Text-to-SQL
=======================================================
チャット画面はChatGPT風、その他は洗練されたモダンデザイン
(Chat screen is ChatGPT-style, others are sophisticated modern design)

起動: streamlit run streamlit_rag_ui_hybrid.py
(Launch: streamlit run streamlit_rag_ui_hybrid.py)

Langsmithでトレースを有効にするには、以下の環境変数を設定してください:
(To enable tracing with Langsmith, set the following environment variables:)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="your-project-name" (オプション) (optional)
"""
from __future__ import annotations

import streamlit as st

# ── Page Configuration (最優先で呼び出し) ─────────────────────────────────
# (Page Configuration (must be called first))
# Streamlitのコマンドは、これが最初に呼び出される必要がある
# (This Streamlit command must be called first)
st.set_page_config(
    page_title="RAG System • Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# その他のインポート（set_page_configの後）
# (Other imports (after set_page_config))
import os
import json
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

from langchain_core.runnables import RunnableConfig

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── Environment & Configuration ────────────────────────────────────────────
load_dotenv()
ENV_DEFAULTS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
    "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4o"),
    "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "documents"),
    "FINAL_K": int(os.getenv("FINAL_K", 5)),
}

# ── RAG System Import ─────────────────────────────────────────────────────
# この部分の st.error() や st.stop() は st.set_page_config() の後になる
# (st.error() or st.stop() in this part must be after st.set_page_config())
try:
    from rag_system_enhanced import Config, RAGSystem
except ModuleNotFoundError:
    try:
        from rag_system import Config, RAGSystem # Fallback to original if enhanced not found
        st.warning("⚠️ rag_system_enhanced.py が見つからなかったため、rag_system.py を使用します。") # ⚠️ rag_system_enhanced.py not found, using rag_system.py.
    except ModuleNotFoundError:
        st.error("❌ rag_system_enhanced.py または rag_system.py が見つかりません。アプリケーションを起動できません。") # ❌ rag_system_enhanced.py or rag_system.py not found. Cannot launch application.
        st.stop()
    except ImportError as e:
        st.error(f"❌ RAGシステムのインポート中にエラーが発生しました: {e}") # ❌ Error during RAG system import:
        st.stop()


# ── Hybrid CSS Design ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        /* ダークテーマカラー (Dark theme colors) */
        --bg-primary: #0a0a0a; --bg-secondary: #141414; --bg-tertiary: #1a1a1a;
        --surface: #242424; --surface-hover: #2a2a2a; --border: #333333;
        /* ChatGPT風カラー（チャット部分用） (ChatGPT-style colors (for chat part)) */
        --chat-bg: #343541; --sidebar-bg: #202123; --user-msg-bg: #343541;
        --ai-msg-bg: #444654; --chat-border: #4e4f60;
        /* テキストカラー (Text colors) */
        --text-primary: #ffffff; --text-secondary: #b3b3b3; --text-tertiary: #808080;
        /* アクセントカラー (Accent colors) */
        --accent: #7c3aed; --accent-hover: #8b5cf6; --accent-light: rgba(124, 58, 237, 0.15);
        --accent-green: #10a37f;
        /* ステータスカラー (Status colors) */
        --success: #10b981; --error: #ef4444; --warning: #f59e0b; --info: #3b82f6;
    }
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: var(--bg-primary); color: var(--text-primary); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #666; }

    /* ヘッダーの修正 (Header correction) */
    .main-header {
        background: linear-gradient(135deg, var(--accent) 0%, #a855f7 100%);
        padding: 0.1rem 1rem; /* 上下パディングを 0.8rem に変更 (Vertical padding changed to 0.8rem) */
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3);
        max-width: 100%;      /* 最大幅を100%に設定 (Max width set to 100%) */
        margin-left: auto;    /* 中央揃え (Center align) */
        margin-right: auto;   /* 中央揃え (Center align) */
    }
    .header-title { font-size: 2.5rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
    .header-subtitle { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .chat-welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; text-align: center; margin-top: -50px; }
    .chat-welcome h2 { color: var(--text-primary); font-size: 2rem; margin-bottom: 1rem; }
    .initial-input-container { margin-top: -100px; width: 100%; max-width: 700px; margin-left: auto; margin-right: auto; }
    .messages-area { padding: 20px 0; min-height: 400px; max-height: calc(100vh - 400px); overflow-y: auto; }
    .message-row { display: flex; padding: 16px 20px; gap: 16px; margin-bottom: 8px; } .user-message-row { background-color: var(--user-msg-bg); } .ai-message-row { background-color: var(--ai-msg-bg); }
    .avatar { width: 36px; height: 36px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 600; flex-shrink: 0; }
    .user-avatar { background-color: #5436DA; color: white; } .ai-avatar { background-color: var(--accent-green); color: white; }
    .message-content { color: var(--text-primary); line-height: 1.6; flex: 1; } .message-content p { margin: 0; }
    .chat-input-area { border-top: 1px solid var(--chat-border); padding: 20px; background-color: var(--chat-bg); border-radius: 0 0 12px 12px; }
    .source-container { background: var(--bg-secondary); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border); margin-top: 1rem; }
    .source-item { background: var(--surface); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; border-left: 3px solid var(--accent); cursor: pointer; transition: all 0.2s ease; }
    .source-item:hover { transform: translateX(4px); background: var(--surface-hover); } .source-title { font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem; }
    .source-excerpt { font-size: 0.875rem; color: var(--text-secondary); line-height: 1.5; }
    .full-text-container { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; max-height: 300px; overflow-y: auto; margin-top: 0.5rem; font-size: 0.875rem; line-height: 1.6; color: var(--text-primary); }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; text-align: center; transition: transform 0.2s ease; } .stat-card:hover { transform: translateY(-2px); }
    .stat-number { font-size: 2rem; font-weight: 700; color: var(--accent); } .stat-label { color: var(--text-secondary); font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.5rem; }
    .stButton > button { background: var(--accent); color: white; border: none; border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 500; transition: all 0.2s ease; } .stButton > button:hover { background: var(--accent-hover); transform: translateY(-1px); }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { background: var(--surface); border: 1px solid var(--border); color: var(--text-primary); border-radius: 8px; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-light); }
    .stFormLabel { color: var(--text-primary) !important; font-weight: 500; }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-secondary) !important; }
    .stTextInput input, .stTextArea textarea, .stSelectbox > div > div > div[data-baseweb="select"] > div { color: var(--text-primary) !important; font-size: 1rem !important; }
    .stSelectbox > div > div > div { background: var(--surface); border: 1px solid var(--border); color: var(--text-primary); }
    .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { background: var(--surface); color: var(--text-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: var(--accent); color: white; border-color: var(--accent); }
    .stFileUploader > div { background: var(--surface); border: 2px dashed var(--border); border-radius: 12px; } .stFileUploader > div:hover { border-color: var(--accent); background: var(--surface-hover); }
    .stProgress > div > div > div > div { background: var(--accent); } div[data-testid="metric-container"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
    .css-1d391kg { background: var(--bg-secondary); } .stAlert { background: var(--surface); color: var(--text-primary); border: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────
def _persist_uploaded_file(uploaded_file) -> Path:
    if uploaded_file is None:
        raise ValueError("Uploaded file cannot be None")
    tmp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / uploaded_file.name
    with open(tmp_path, "wb") as f: # Ensure binary mode for write_bytes equivalent
        f.write(uploaded_file.getbuffer())
    return tmp_path

@st.cache_resource(show_spinner=False)
def initialize_rag_system(config_obj: Config) -> RAGSystem: # Renamed config to config_obj to avoid conflict
    return RAGSystem(config_obj) # Use renamed config_obj

def get_collection_statistics(rag: RAGSystem) -> Dict[str, Any]:
    if not rag:
        return {"documents": 0, "chunks": 0, "collection_name": "N/A"}
    try:
        engine = create_engine(rag.connection_string)
        with engine.connect() as conn: # Use 'with' statement for connection
            query = text("SELECT COUNT(DISTINCT document_id) AS num_documents, COUNT(*) AS num_chunks FROM document_chunks WHERE collection_name = :collection")
            result = conn.execute(query, {"collection": rag.config.collection_name}).first()
        # conn.close() is handled by 'with' statement
        return {
            "documents": result.num_documents if result else 0,
            "chunks": result.num_chunks if result else 0,
            "collection_name": rag.config.collection_name
        }
    except Exception as e:
        st.error(f"統計情報の取得に失敗: {e}") # Failed to get statistics:
        return {"documents": 0, "chunks": 0, "collection_name": rag.config.collection_name if rag else "N/A"}


def get_documents_dataframe(rag: RAGSystem) -> pd.DataFrame:
    if not rag:
        return pd.DataFrame()
    try:
        engine = create_engine(rag.connection_string)
        with engine.connect() as conn: # Use 'with' statement
            query = text("SELECT document_id, COUNT(*) as chunk_count, MAX(created_at) as last_updated FROM document_chunks WHERE collection_name = :collection GROUP BY document_id ORDER BY last_updated DESC")
            result = conn.execute(query, {"collection": rag.config.collection_name})
            df = pd.DataFrame(result.fetchall(), columns=["Document ID", "Chunks", "Last Updated"])
        if not df.empty and "Last Updated" in df.columns:
            df["Last Updated"] = pd.to_datetime(df["Last Updated"]).dt.strftime("%Y-%m-%d %H:%M")
        return df
    except Exception as e: # Consider logging the exception
        st.error(f"登録済みドキュメントリストの取得に失敗: {e}") # Failed to get registered document list:
        return pd.DataFrame()


def get_query_history_data(days: int = 30) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    # Simulating some query data variability
    queries = [20 + int(10 * abs(np.sin(i / 5.0))) + np.random.randint(-3, 4) for i in range(days)]
    queries = [max(0, q) for q in queries] # Ensure non-negative
    return pd.DataFrame({'Date': dates, 'Queries': queries})

def render_simple_chart(df: pd.DataFrame):
    """簡単なチャート描画 (Simple chart rendering)"""
    try:
        if df.empty:
            st.info("チャートを描画するデータがありません。") # No data to draw chart.
            return

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols: # Check if numeric_cols is empty
            st.info("数値型の列がないため、チャートを描画できません。") # Cannot draw chart because there are no numeric columns.
            return

        # Ensure at least one non-numeric column for x-axis if possible
        categorical_cols = df.select_dtypes(include=['object', 'category', 'datetime64']).columns.tolist()


        chart_type_options = ["なし"] # None
        if len(df.columns) >= 2 and categorical_cols and numeric_cols:
            chart_type_options.append("棒グラフ") # Bar chart
        if numeric_cols: # Line chart can be plotted against index if no clear x-axis
             chart_type_options.append("折れ線グラフ") # Line chart
        if len(numeric_cols) >= 2 : # Scatter plot needs at least two numeric cols
            chart_type_options.append("散布図") # Scatter plot


        if len(chart_type_options) == 1: # Only "なし" (None) is available
            st.info("適切なデータ形式ではないため、チャートタイプを選択できません。") # Cannot select chart type due to inappropriate data format.
            return

        chart_type = st.selectbox("可視化タイプを選択:", chart_type_options, key=f"sql_chart_type_selector_{df.shape[0]}_{df.shape[1]}") # Select visualization type: (More unique key)

        if chart_type == "棒グラフ": # Bar chart
            if categorical_cols and numeric_cols:
                x_col_bar = st.selectbox("X軸 (カテゴリ/日付)", categorical_cols, key=f"bar_x_sql_{df.shape[0]}") # X-axis (category/date)
                y_col_bar = st.selectbox("Y軸 (数値)", numeric_cols, key=f"bar_y_sql_{df.shape[0]}") # Y-axis (numeric)
                if x_col_bar and y_col_bar:
                    fig = px.bar(df.head(25), x=x_col_bar, y=y_col_bar, title=f"{y_col_bar} by {x_col_bar}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("棒グラフにはカテゴリ列と数値列が必要です。") # Bar chart requires a category column and a numeric column.


        elif chart_type == "折れ線グラフ": # Line chart
            y_cols_line = st.multiselect("Y軸 (数値 - 複数選択可)", numeric_cols, default=numeric_cols[0] if numeric_cols else None, key=f"line_y_sql_{df.shape[0]}") # Y-axis (numeric - multiple selection allowed)
            # Try to find a date/time column for x-axis, otherwise use index or first categorical
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            x_col_line_options = ["(インデックス)"] + categorical_cols # Allow choosing index ((Index))
            
            chosen_x_col = None # Initialize chosen_x_col
            if date_cols:
                x_col_line_options = ["(インデックス)"] + date_cols + [c for c in categorical_cols if c not in date_cols] # (Index)
                chosen_x_col = date_cols[0] # Default to first date column
            elif categorical_cols:
                 chosen_x_col = categorical_cols[0] # Default to first categorical if no date

            x_col_line = st.selectbox("X軸", x_col_line_options, index=x_col_line_options.index(chosen_x_col) if chosen_x_col and chosen_x_col in x_col_line_options else 0, key=f"line_x_sql_{df.shape[0]}") # X-axis


            if y_cols_line:
                title_ys = ", ".join(y_cols_line)
                if x_col_line and x_col_line != "(インデックス)": # (Index)
                    fig = px.line(df.head(100), x=x_col_line, y=y_cols_line, title=f"{title_ys} over {x_col_line}", markers=True)
                else: # If no suitable x-axis or index is chosen, plot against index
                    fig = px.line(df.head(100), y=y_cols_line, title=f"{title_ys} Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "散布図": # Scatter plot
            if len(numeric_cols) >= 2:
                x_col_scatter = st.selectbox("X軸 (数値)", numeric_cols, key=f"scatter_x_sql_{df.shape[0]}") # X-axis (numeric)
                y_col_scatter = st.selectbox("Y軸 (数値)", [nc for nc in numeric_cols if nc != x_col_scatter], key=f"scatter_y_sql_{df.shape[0]}") # Y-axis (numeric)
                color_col_scatter_options = ["なし"] + categorical_cols + [nc for nc in numeric_cols if nc != x_col_scatter and nc != y_col_scatter] # None
                color_col_scatter = st.selectbox("色分け (任意)", color_col_scatter_options, key=f"scatter_color_sql_{df.shape[0]}") # Color coding (optional)

                if x_col_scatter and y_col_scatter:
                    fig = px.scatter(
                        df.head(500), # Allow more points for scatter
                        x=x_col_scatter, 
                        y=y_col_scatter, 
                        color=color_col_scatter if color_col_scatter != "なし" else None, # None
                        title=f"{y_col_scatter} vs {x_col_scatter}" + (f" by {color_col_scatter}" if color_col_scatter != "なし" else "") # None
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("散布図には少なくとも2つの数値列が必要です。") # Scatter plot requires at least two numeric columns.


    except Exception as e:
        st.error(f"チャート描画エラー: {type(e).__name__} - {e}") # Chart rendering error:
        # import traceback
        # st.text(traceback.format_exc())


def render_sql_result_in_chat(sql_details_dict: Dict[str, Any]): # Renamed for clarity
    """チャット内でのSQL関連情報表示 (SQL Details Dict を受け取る)
    (Display SQL-related information in chat (receives SQL Details Dict))"""
    if not sql_details_dict or not isinstance(sql_details_dict, dict): # Check if sql_details_dict is valid
        st.warning("チャット表示用のSQL詳細情報がありません。") # No SQL details for chat display.
        return

    # Natural language answer is already part of the main message.content.
    # This function now focuses on displaying the SQL query and a data preview.

    with st.expander("🔍 実行されたSQL (チャット内)", expanded=False): # 🔍 Executed SQL (in chat)
        st.code(sql_details_dict.get("generated_sql", "SQLが生成されませんでした。"), language="sql") # SQL was not generated.

    # Display a preview of the data returned by the SQL query
    results_data_preview = sql_details_dict.get("results_preview") # This is already limited by max_sql_preview_rows_for_llm
    if results_data_preview and isinstance(results_data_preview, list) and len(results_data_preview) > 0:
        with st.expander("📊 SQL実行結果プレビュー (チャット内)", expanded=False): # 📊 SQL execution results preview (in chat)
            try:
                df_chat_preview = pd.DataFrame(results_data_preview)
                st.dataframe(df_chat_preview, use_container_width=True, height = min(300, (len(df_chat_preview) + 1) * 35 + 3)) # Dynamic height
                
                total_fetched = sql_details_dict.get("row_count_fetched", 0)
                preview_count = len(results_data_preview)
                if total_fetched > preview_count:
                    st.caption(f"結果の最初の{preview_count}件を表示（全{total_fetched}件取得）。") # Displaying first {X} of {Y} results retrieved.
                elif total_fetched > 0:
                     st.caption(f"全{total_fetched}件の結果を表示。") # Displaying all {X} results.
            except Exception as e:
                st.error(f"チャット内でのSQL結果プレビュー表示エラー: {e}") # Error displaying SQL results preview in chat:
    elif sql_details_dict.get("success"): # Query was successful but no rows returned
        with st.expander("📊 SQL実行結果プレビュー (チャット内)", expanded=False): # 📊 SQL execution results preview (in chat)
            st.info("SQLクエリは成功しましたが、該当するデータはありませんでした。") # SQL query was successful, but no corresponding data was found.


# ── Initialize Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "last_query_expansion" not in st.session_state:
    st.session_state.last_query_expansion = {}
if "use_query_expansion" not in st.session_state:
    st.session_state.use_query_expansion = False
if "use_rag_fusion" not in st.session_state:
    st.session_state.use_rag_fusion = False
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
# sql_analysis_history はSQLタブがなくなったため、初期化は不要になる可能性がありますが、
# チャット内でのSQL実行履歴を別途持ちたい場合は残します。
# 今回はSQLタブ自体をなくすため、この履歴も不要と判断できます。
# if "sql_analysis_history" not in st.session_state:
#    st.session_state.sql_analysis_history = []


if "rag_system" not in st.session_state:
    if ENV_DEFAULTS["OPENAI_API_KEY"]:
        try:
            # Assuming Config is imported from rag_system_enhanced or rag_system
            app_config = Config(
                openai_api_key=ENV_DEFAULTS["OPENAI_API_KEY"], # Pass explicitly
                db_password=os.getenv("DB_PASSWORD", "postgres") # Other sensitive or non-UI defaults
            )
            st.session_state.rag_system = initialize_rag_system(app_config)
            st.toast("✅ RAGシステムが正常に初期化されました", icon="🎉") # ✅ RAG system initialized successfully
        except Exception as e:
            st.error(f"RAGシステムの初期化中にエラーが発生しました: {type(e).__name__} - {e}") # An error occurred during RAG system initialization:
            st.warning("""
### 🔧 データベース接続エラーの解決方法 (一般的な例)
(Troubleshooting database connection errors (general examples))

1.  **.envファイルを確認してください**: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` が正しく設定されているか確認してください。
    (Check your .env file: Ensure `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` are set correctly.)
    例 (Example):
    ```
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=mydatabase
    DB_USER=myuser
    DB_PASSWORD=mypassword
    ```
2.  **PostgreSQLサービスが実行中か確認**: データベースサーバーが起動していることを確認してください。
    (Check if PostgreSQL service is running: Ensure the database server is running.)
3.  **ファイアウォール設定**: ファイアウォールがPostgreSQLのポート (デフォルト5432) をブロックしていないか確認してください。
    (Firewall settings: Check if the firewall is blocking the PostgreSQL port (default 5432).)
4.  **認証情報**: ユーザー名とパスワードが正しいか、そのユーザーにデータベースへのアクセス権があるか確認してください。
    (Authentication information: Check if the username and password are correct and if the user has access rights to the database.)
            """)
            st.session_state.rag_system = None # Ensure it's None on failure
    else:
        st.warning("OPENAI_API_KEYが設定されていません。チャット機能を利用できません。サイドバーから設定してください。") # OPENAI_API_KEY is not set. Chat function cannot be used. Please set it from the sidebar.
        st.session_state.rag_system = None


# ── Main Header & Langsmith Info ───────────────────────────────────────────
st.markdown("""<div class="main-header"><h1 class="header-title">iRAG</h1><p class="header-subtitle">IHI's Smart Knowledge Base with SQL Analytics</p></div>""", unsafe_allow_html=True)

# LangSmith Tracing Info (Optional)
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_project = os.getenv("LANGCHAIN_PROJECT")
if langsmith_api_key:
    st.sidebar.success(f"ιχ LangSmith Tracing: ENABLED{' (Project: ' + langsmith_project + ')' if langsmith_project else ''}")
else:
    st.sidebar.info("ιχ LangSmith Tracing: DISABLED (環境変数を設定してください)") # (Set environment variables)


rag: RAGSystem | None = st.session_state.get("rag_system")

# ── Sidebar Configuration ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color: var(--text-primary);'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    if rag:
        st.success(f"✅ System Online - Collection: **{rag.config.collection_name}**")
    else:
        st.warning("⚠️ System Offline - APIキーまたはDB設定を確認してください。") # ⚠️ System Offline - Check API key or DB settings.


    with st.form("config_form"):
        st.markdown("### 🔑 OpenAI APIキー") # OpenAI API Key
        openai_api_key_input = st.text_input(
            "OpenAI API Key",
            value=ENV_DEFAULTS["OPENAI_API_KEY"] or "",
            type="password",
            help="OpenAI APIキーを入力してください。変更を適用するとシステムが再初期化されます。" # Enter your OpenAI API key. Applying changes will reinitialize the system.
        )

        st.markdown("### 🤖 Model Settings")
        embedding_model_options = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        llm_model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

        # Default values if rag or rag.config is None
        default_emb_model = ENV_DEFAULTS["EMBEDDING_MODEL"]
        default_llm_model = ENV_DEFAULTS["LLM_MODEL"]
        default_collection_name = ENV_DEFAULTS["COLLECTION_NAME"]
        default_final_k = ENV_DEFAULTS["FINAL_K"]
        # RRF_K_FOR_FUSION is part of Config, so it should be handled like other config values
        default_rrf_k_fusion = int(os.getenv("RRF_K_FOR_FUSION", 60))


        current_emb_model = rag.config.embedding_model if rag and hasattr(rag, 'config') else default_emb_model
        current_llm_model = rag.config.llm_model if rag and hasattr(rag, 'config') else default_llm_model
        current_collection_name = rag.config.collection_name if rag and hasattr(rag, 'config') else default_collection_name
        current_final_k = rag.config.final_k if rag and hasattr(rag, 'config') else default_final_k
        current_rrf_k = rag.config.rrf_k_for_fusion if rag and hasattr(rag, 'config') else default_rrf_k_fusion


        embedding_model_idx = embedding_model_options.index(current_emb_model) if current_emb_model in embedding_model_options else 0
        llm_model_idx = llm_model_options.index(current_llm_model) if current_llm_model in llm_model_options else 0

        embedding_model_sb = st.selectbox("Embedding Model", embedding_model_options, index=embedding_model_idx)
        llm_model_sb = st.selectbox("Language Model", llm_model_options, index=llm_model_idx)

        st.markdown("### 🔍 Search Settings")
        collection_name_ti = st.text_input("Collection Name", value=current_collection_name)
        final_k_sl = st.slider("検索結果数 (Final K)", 1, 20, current_final_k, help="LLMに渡す最終的なチャンク数") # Number of search results (Final K) (Final number of chunks to pass to LLM)

        apply_button = st.form_submit_button("Apply Settings", use_container_width=True)

if apply_button:
    # Use current or default for settings not exposed in this simplified sidebar form
    # These would come from the more detailed settings tab or .env defaults.
    # This sidebar form is for quick changes to common settings.

    # Create a new Config object with updated values from the sidebar form
    # For unexposed settings, use current values if rag system exists, otherwise use .env defaults
    # This requires fetching all necessary defaults or current config values.
    
    cfg_for_update = Config() # Start with all .env defaults or dataclass defaults

    if rag and hasattr(rag, 'config'): # If system is initialized, use its current config as base
        existing_cfg_dict = rag.config.__dict__
        for key in cfg_for_update.__dict__:
            if key in existing_cfg_dict:
                setattr(cfg_for_update, key, existing_cfg_dict[key])
    
    # Override with values from the sidebar form
    cfg_for_update.openai_api_key = openai_api_key_input or ENV_DEFAULTS["OPENAI_API_KEY"] # Update API key
    cfg_for_update.embedding_model = embedding_model_sb
    cfg_for_update.llm_model = llm_model_sb
    cfg_for_update.collection_name = collection_name_ti
    cfg_for_update.final_k = int(final_k_sl)
    # rrf_k_for_fusion is not in this form, so it will retain its current or default value from cfg_for_update initialization

    try:
        with st.spinner("設定を適用し、システムを再初期化しています..."): # Applying settings and reinitializing system...
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"] # Clear previous instance to force reinitialization
                st.cache_resource.clear() # Clear cached resources, including initialize_rag_system
            
            st.session_state.rag_system = initialize_rag_system(cfg_for_update)
            rag = st.session_state.rag_system # Update global rag variable
        st.success("✅ 設定が正常に適用され、システムが再初期化されました。") # ✅ Settings applied successfully and system reinitialized.
        time.sleep(1) # Give time for toast to be seen
        st.rerun()
    except Exception as e:
        st.error(f"設定適用エラー: {type(e).__name__} - {e}") # Settings application error:


# ── Main Tabs ─────────────────────────────────────────────────────────────
# タブのタイトルと順序を変更 (Change tab titles and order)
tab_titles = ["💬 Chat", "📊 Analytics", "🗃️ Data", "📁 Documents", "⚙️ Settings"]
# タブの数を5つに変更 (Change number of tabs to 5)
tabs = st.tabs(tab_titles)
tab_chat, tab_analytics, tab_data, tab_documents, tab_settings = tabs


# ── Tab 1: Chat Interface (ChatGPT Style) ────────────────────────────────
with tab_chat: # tab1 から tab_chat に変更 (Change from tab1 to tab_chat)
    if not rag:
        st.info("🔧 RAGシステムが初期化されていません。サイドバーでOpenAI APIキーを設定し、「Apply Settings」をクリックするか、データベース設定を確認してください。") # 🔧 RAG system is not initialized. Set OpenAI API key in the sidebar and click "Apply Settings", or check database settings.
    else:
        if hasattr(rag, 'get_data_tables'): # Check if the method exists
            data_tables = rag.get_data_tables()
            if data_tables:
                table_names = [t['table_name'] for t in data_tables]
 
        has_messages = len(st.session_state.messages) > 0
        if not has_messages:
            st.markdown("""
            <div class="chat-welcome">
                <h2>Chat with your data</h2>
                <p style="color: var(--text-secondary);">
                    アップロードされたドキュメントから関連情報を検索し、AIが回答します<br>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="initial-input-container">', unsafe_allow_html=True)

            st.markdown("<h6>高度なRAG設定:</h6>", unsafe_allow_html=True) # Advanced RAG settings:
            opt_cols_initial = st.columns(2)
            with opt_cols_initial[0]:
                use_qe_initial = st.checkbox("クエリ拡張", value=st.session_state.use_query_expansion, key="use_qe_initial_v7_tab_chat", help="質問を自動的に拡張して検索 (RRFなし)") # Query expansion (Search by automatically expanding questions (without RRF))
            with opt_cols_initial[1]:
                use_rf_initial = st.checkbox("RAG-Fusion", value=st.session_state.use_rag_fusion, key="use_rf_initial_v7_tab_chat", help="クエリ拡張とRRFで結果を統合") # RAG-Fusion (Integrate results with query expansion and RRF)

            user_input_initial = st.text_area("質問を入力:", placeholder="例：このドキュメントの要約を教えてください / 売上上位10件を表示して", height=100, key="initial_input_textarea_v7_tab_chat", label_visibility="collapsed") # Enter question: (Example: Summarize this document / Show top 10 sales items)

            if st.button("送信", type="primary", use_container_width=True, key="initial_send_button_v7_tab_chat"): # Send
                if user_input_initial:
                    st.session_state.messages.append({"role": "user", "content": user_input_initial})
                    st.session_state.use_query_expansion = use_qe_initial
                    st.session_state.use_rag_fusion = use_rf_initial

                    with st.spinner("考え中..."): # Thinking...
                        try:
                            trace_config = RunnableConfig(
                                run_name="RAG Initial Query Unified", # Updated name
                                tags=["streamlit", "rag", "initial_query", st.session_state.session_id], # Added session_id
                                metadata={
                                    "session_id": st.session_state.session_id,
                                    "user_query": user_input_initial,
                                    "use_query_expansion": st.session_state.use_query_expansion,
                                    "use_rag_fusion": st.session_state.use_rag_fusion,
                                    "query_source": "initial_input"
                                }
                            )
                            # Always use query_unified if available
                            if hasattr(rag, 'query_unified'):
                                response = rag.query_unified(
                                    user_input_initial,
                                    use_query_expansion=st.session_state.use_query_expansion,
                                    use_rag_fusion=st.session_state.use_rag_fusion,
                                    config=trace_config
                                )
                            else: # Fallback for older RAGSystem versions without query_unified
                                st.warning("警告: `query_unified` メソッドが見つかりません。標準の `query` メソッドを使用します。SQL自動判別は機能しません。") # Warning: `query_unified` method not found. Using standard `query` method. SQL auto-detection will not work.
                                response = rag.query(
                                    user_input_initial,
                                    use_query_expansion=st.session_state.use_query_expansion,
                                    use_rag_fusion=st.session_state.use_rag_fusion, # Assuming rag.query handles this
                                    config=trace_config
                                )


                            answer = response.get("answer", "申し訳ございません。回答を生成できませんでした。") # I'm sorry. I couldn't generate an answer.
                            message_data: Dict[str, Any] = {"role": "assistant", "content": answer}

                            # If query_unified returned SQL details, attach them to the message
                            if response.get("query_type") == "sql" and response.get("sql_details"):
                                message_data["sql_details"] = response["sql_details"]
                            elif response.get("sql_details"): # Also handle if sql_details is present without query_type explicitly set
                                message_data["sql_details"] = response["sql_details"]


                            st.session_state.messages.append(message_data)
                            st.session_state.current_sources = response.get("sources", [])
                            st.session_state.last_query_expansion = response.get("query_expansion", {})
                        except Exception as e:
                            st.error(f"チャット処理中にエラーが発生しました: {type(e).__name__} - {e}") # An error occurred during chat processing:
                            # import traceback
                            # st.text(traceback.format_exc()) # For debugging
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else: # Messages exist, render chat history and continued input
            chat_col, source_col = st.columns([2, 1]) # Adjust column ratio if needed
            with chat_col:
                # Use a container with a specific height for messages to make it scrollable
                message_container_height = 600 # Adjust height as needed
                with st.container(height=message_container_height):
                    for idx, message in enumerate(st.session_state.messages):
                        avatar_char = "👤" if message['role'] == 'user' else "🤖"
                        avatar_class = 'user-avatar' if message['role'] == 'user' else 'ai-avatar'
                        avatar_html = f"<div class='avatar {avatar_class}'>{avatar_char}</div>"

                        # Display the main content (text answer)
                        # Ensure message content is properly escaped or handled if it can contain user-generated HTML/Markdown
                        st.markdown(f"<div class='message-row {'user-message-row' if message['role'] == 'user' else 'ai-message-row'}'>{avatar_html}<div class='message-content'>{message['content']}</div></div>", unsafe_allow_html=True)

                        # If the assistant's message has SQL details, render them
                        if message['role'] == 'assistant' and message.get("sql_details"):
                            render_sql_result_in_chat(message["sql_details"])


                st.markdown("---") # Separator before input area

                opt_cols_chat = st.columns(2)
                with opt_cols_chat[0]:
                    use_qe_chat = st.checkbox("クエリ拡張", value=st.session_state.use_query_expansion, key="use_qe_chat_continued_v7_tab_chat", help="クエリ拡張 (RRFなし)") # Query expansion (without RRF)
                with opt_cols_chat[1]:
                    use_rf_chat = st.checkbox("RAG-Fusion", value=st.session_state.use_rag_fusion, key="use_rf_chat_continued_v7_tab_chat", help="RAG-Fusion (拡張+RRF)") # RAG-Fusion (expansion+RRF)

                # Chat input area at the bottom
                # For Streamlit's chat_input, it's usually placed outside the main message loop
                # Using text_input + button for more control over placement and behavior here.
                # Consider using st.chat_input if a simpler, bottom-docked input is desired.
                user_input_continued = st.text_area( # Changed to text_area for consistency with initial input
                    "メッセージを入力:", # Enter message:
                    placeholder="続けて質問してください...", # Continue asking questions...
                    label_visibility="collapsed",
                    key=f"chat_input_continued_text_v7_tab_chat_{len(st.session_state.messages)}" # More unique key
                )

                if st.button("送信", type="primary", key=f"chat_send_button_continued_v7_tab_chat_{len(st.session_state.messages)}", use_container_width=True): # Send
                    if user_input_continued: # Process if there is input
                        st.session_state.messages.append({"role": "user", "content": user_input_continued})
                        st.session_state.use_query_expansion = use_qe_chat
                        st.session_state.use_rag_fusion = use_rf_chat
                        with st.spinner("考え中..."): # Thinking...
                            try:
                                trace_config_cont = RunnableConfig(
                                    run_name="RAG Chat Query Unified", # Updated name
                                    tags=["streamlit", "rag", "chat_query", st.session_state.session_id], # Added session_id
                                    metadata={
                                        "session_id": st.session_state.session_id,
                                        "user_query": user_input_continued,
                                        "use_query_expansion": st.session_state.use_query_expansion,
                                        "use_rag_fusion": st.session_state.use_rag_fusion,
                                        "query_source": "continued_chat"
                                    }
                                )
                                if hasattr(rag, 'query_unified'):
                                    response = rag.query_unified(
                                        user_input_continued,
                                        use_query_expansion=st.session_state.use_query_expansion,
                                        use_rag_fusion=st.session_state.use_rag_fusion,
                                        config=trace_config_cont
                                    )
                                else:
                                    st.warning("警告: `query_unified` メソッドが見つかりません。標準の `query` メソッドを使用します。SQL自動判別は機能しません。") # Warning: `query_unified` method not found. Using standard `query` method. SQL auto-detection will not work.
                                    response = rag.query(
                                        user_input_continued,
                                        use_query_expansion=st.session_state.use_query_expansion,
                                        use_rag_fusion=st.session_state.use_rag_fusion,
                                        config=trace_config_cont
                                    )

                                answer = response.get("answer", "申し訳ございません。回答を生成できませんでした。") # I'm sorry. I couldn't generate an answer.
                                message_data_cont: Dict[str, Any] = {"role": "assistant", "content": answer}

                                if response.get("query_type") == "sql" and response.get("sql_details"):
                                    message_data_cont["sql_details"] = response["sql_details"]
                                elif response.get("sql_details"):
                                     message_data_cont["sql_details"] = response["sql_details"]


                                st.session_state.messages.append(message_data_cont)
                                st.session_state.current_sources = response.get("sources", [])
                                st.session_state.last_query_expansion = response.get("query_expansion", {})
                                # Clear the input field after sending by rerunning
                            except Exception as e:
                                st.error(f"チャット処理中にエラーが発生しました: {type(e).__name__} - {e}") # An error occurred during chat processing:
                                # import traceback
                                # st.text(traceback.format_exc()) # For debugging
                        st.rerun()


                button_col, info_col = st.columns([1, 3])
                with button_col:
                    if st.button("🗑️ 会話をクリア", use_container_width=True, key="clear_chat_button_v7_tab_chat"): # 🗑️ Clear conversation
                        st.session_state.messages = []
                        st.session_state.current_sources = []
                        st.session_state.last_query_expansion = {}
                        st.rerun()
                with info_col:
                    last_expansion = st.session_state.get("last_query_expansion", {})
                    if last_expansion and last_expansion.get("used", False): # Check if last_expansion is not empty
                        with st.expander(f"📋 拡張クエリ詳細 ({last_expansion.get('strategy', 'N/A')})", expanded=False): # 📋 Expanded query details
                            queries = last_expansion.get("queries", [])
                            st.caption("以下のクエリで検索しました（該当する場合）：") # Searched with the following queries (if applicable):
                            for i, q_text in enumerate(queries):
                                st.write(f"• {'**' if i == 0 else ''}{q_text}{'** (元の質問)' if i == 0 else ''}") # (Original question)
                    elif any(msg.get("sql_details") for msg in st.session_state.messages if msg["role"] == "assistant"):
                         st.caption("SQL分析が実行されました。詳細はメッセージ内の実行結果をご確認ください。") # SQL analysis was executed. Check the execution results in the message for details.


            with source_col: # Source display column
                st.markdown("""<div style="position: sticky; top: 1rem;"><h4 style="color: var(--text-primary); margin-bottom: 1rem;">📚 参照ソース (RAG)</h4></div>""", unsafe_allow_html=True) # 📚 Referenced sources (RAG)
                if st.session_state.current_sources:
                    for i, source in enumerate(st.session_state.current_sources):
                        # Ensure source is a dictionary and has metadata
                        doc_id = source.get('metadata', {}).get('document_id', 'Unknown Document')
                        chunk_id_val = source.get('metadata', {}).get('chunk_id', f'N/A_{i}') # Renamed and added unique fallback
                        excerpt = source.get('excerpt', '抜粋なし') # No excerpt
                        
                        # Unique key for expander using chunk_id_val for better stability
                        expander_key = f"source_expander_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"
                        
                        with st.expander(f"ソース {i+1}: {doc_id} (Chunk: {chunk_id_val})", expanded=False): # Source {i+1}:
                            st.markdown(f"""<div class="source-excerpt" style="margin-bottom: 1rem;">{excerpt}</div>""", unsafe_allow_html=True)
                            
                            # Unique keys for button and session state variable for showing full text
                            button_key = f"full_text_btn_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"
                            show_full_text_key = f"show_full_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"

                            if st.button(f"全文を表示##{chunk_id_val}", key=button_key): # Added chunk_id_val to make button text unique if needed (Show full text)
                                st.session_state[show_full_text_key] = not st.session_state.get(show_full_text_key, False)
                            
                            if st.session_state.get(show_full_text_key, False):
                                full_text = source.get('full_content', 'コンテンツなし') # No content
                                st.markdown(f"""<div class="full-text-container">{full_text}</div>""", unsafe_allow_html=True)
                else:
                    st.info("RAG検索が実行されると、参照したソースがここに表示されます。") # When RAG search is executed, referenced sources will be displayed here.

# ── Tab 2: Analytics Dashboard ────────────────────────────────────────────
with tab_analytics: # tab2 から tab_analytics に変更 (Change from tab2 to tab_analytics)
    if rag:
        stats = get_collection_statistics(rag)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{stats.get('documents', 0):,}</div><div class="stat-label">Documents</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{stats.get('chunks', 0):,}</div><div class="stat-label">Total Chunks</div></div>""", unsafe_allow_html=True)
        with col3:
            avg_chunks = stats.get('chunks', 0) // max(stats.get('documents', 1), 1) if stats.get('documents', 0) > 0 else 0
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{avg_chunks}</div><div class="stat-label">Avg Chunks/Doc</div></div>""", unsafe_allow_html=True)
        with col4:
            query_count_user = len([m for m in st.session_state.messages if m['role'] == 'user']) # Count only user messages
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{query_count_user}</div><div class="stat-label">Total Queries (User)</div></div>""", unsafe_allow_html=True)


        st.markdown("### 📈 質問数の推移（日別）") # 📈 Trend of number of questions (daily)
        st.caption("過去30日間の質問数の推移を表示しています（シミュレーションデータ）") # Displaying trend of number of questions for the past 30 days (simulation data)

        if PLOTLY_AVAILABLE:
            query_data = get_query_history_data(30)
            if not query_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=query_data['Date'],
                    y=query_data['Queries'],
                    mode='lines+markers',
                    name='質問数', # Number of questions
                    line=dict(color='var(--accent)', width=3), # Use CSS variable
                    marker=dict(size=8, color='var(--accent-hover)'), # Use CSS variable
                    fill='tozeroy',
                    fillcolor='var(--accent-light)' # Use CSS variable
                ))
                fig.update_layout(
                    template="plotly_dark", # Using Plotly's dark theme
                    height=400,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20), # Added some margin
                    plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
                    paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
                    xaxis=dict(showgrid=False, title="日付"), # Date
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="質問数"), # Light gridlines (Number of questions)
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("質問履歴データがありません。") # No question history data.

        else:
            st.warning("Plotlyが利用できません。グラフは表示されません。`pip install plotly` を実行してください。") # Plotly is not available. Graph will not be displayed. Run `pip install plotly`.


        st.markdown("### 🕐 最近の質問 (ユーザー)") # 🕐 Recent questions (user)
        user_questions_list = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        if user_questions_list:
            recent_questions_data = []
            for i, msg in enumerate(reversed(user_questions_list[-10:])): # Show latest 10
                content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                # For simplicity, not adding exact time here, can be added if needed by storing timestamps in messages
                recent_questions_data.append({"質問": content_preview}) # Question
            if recent_questions_data:
                st.dataframe(pd.DataFrame(recent_questions_data), use_container_width=True, hide_index=True)
            else: # Should not happen if user_questions_list is not empty
                st.info("まだユーザーからの質問がありません。") # No questions from user yet.
        else:
            st.info("まだユーザーからの質問がありません。") # No questions from user yet.
    else:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。") # RAG system is not initialized. Check settings in the sidebar.

# ── Tab 3: Data Management (SQL用テーブル) ───────────────────────────────────
# (Tab 3: Data Management (for SQL tables)) - 元のTab5の内容をここに移動 (Move content of original Tab5 here)
with tab_data: # tab5 から tab_data に変更 (Change from tab5 to tab_data)
    if not rag:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。") # RAG system is not initialized. Check settings in the sidebar.
    elif not all(hasattr(rag, attr) for attr in ['create_table_from_file', 'get_data_tables', 'delete_data_table']):
        st.warning("RAGシステムがデータテーブル管理機能（create_table_from_file, get_data_tables, delete_data_tableのいずれか）をサポートしていません。rag_system_enhanced.py を確認してください。") # RAG system does not support data table management functions (any of create_table_from_file, get_data_tables, delete_data_table). Check rag_system_enhanced.py.
    else:
        st.markdown("### 📊 データファイル管理 (SQL分析用)") # 📊 Data File Management (for SQL analysis)
        st.caption("Excel/CSVファイルをアップロードして、SQLで分析可能なテーブルを作成・管理します。") # Upload Excel/CSV files to create and manage tables analyzable with SQL.

        uploaded_sql_data_files_list = st.file_uploader( # Renamed variable
            "Excel/CSVファイルを選択 (.xlsx, .xls, .csv)", # Select Excel/CSV files
            accept_multiple_files=True,
            type=["xlsx", "xls", "csv"],
            key="sql_data_file_uploader_v7_tab_data" # キーをタブ名に合わせて変更 (Change key according to tab name)
        )

        if uploaded_sql_data_files_list:
            if st.button("🚀 選択したファイルからテーブルを作成/更新", type="primary", key="create_table_button_v7_tab_data"): # 🚀 Create/Update table from selected files
                progress_bar_sql_data_create = st.progress(0, text="処理開始...") # Renamed variable (Processing started...)
                status_text_sql_data_create = st.empty() # Renamed variable

                for i, file_item_sql in enumerate(uploaded_sql_data_files_list): # Renamed loop variable
                    status_text_sql_data_create.info(f"処理中: {file_item_sql.name}") # Processing:
                    try:
                        # Persist file temporarily for RAGSystem to access by path
                        temp_dir_for_sql_data_path = Path(tempfile.gettempdir()) / "rag_sql_data_uploads" # Renamed variable
                        temp_dir_for_sql_data_path.mkdir(parents=True, exist_ok=True)
                        temp_file_path_sql = temp_dir_for_sql_data_path / file_item_sql.name # Renamed variable
                        with open(temp_file_path_sql, "wb") as f:
                            f.write(file_item_sql.getbuffer())

                        # テーブル作成/更新 (Table creation/update)
                        success_create, message_create, schema_info_create = rag.create_table_from_file(str(temp_file_path_sql)) # Renamed variables
                        if success_create:
                            st.success(f"✅ {file_item_sql.name}: {message_create}")
                            if schema_info_create:
                                st.text("作成/更新されたテーブルスキーマ:") # Created/Updated table schema:
                                st.code(schema_info_create, language='text') # Changed language to text for broader schema format
                        else:
                            st.error(f"❌ {file_item_sql.name}: {message_create}")
                    except Exception as e_upload_sql: # Renamed exception variable
                        st.error(f"❌ {file_item_sql.name} の処理中にエラー: {type(e_upload_sql).__name__} - {e_upload_sql}") # ❌ Error during processing of {X}:
                    finally:
                        progress_bar_sql_data_create.progress((i + 1) / len(uploaded_sql_data_files_list), text=f"完了: {file_item_sql.name}") # Complete:

                if 'progress_bar_sql_data_create' in locals(): progress_bar_sql_data_create.empty()
                if 'status_text_sql_data_create' in locals(): status_text_sql_data_create.empty()
                st.rerun() # Refresh table list

        st.markdown("---")
        st.markdown("### 📋 登録済みデータテーブル") # 📋 Registered data tables
        tables_list_display = rag.get_data_tables() # Fetch fresh list, Renamed variable
        if tables_list_display:
            for table_info_item in tables_list_display: # Renamed loop variable
                table_name_display = table_info_item.get('table_name', '不明なテーブル') # Renamed variable (Unknown table)
                row_count_display = table_info_item.get('row_count', 'N/A') # Renamed variable
                schema_display_text = table_info_item.get('schema', 'スキーマ情報なし') # Renamed variable (No schema information)
                with st.expander(f"📊 {table_name_display} ({row_count_display:,}行)"): # {X} rows
                    st.code(schema_display_text, language='text') # Changed language to text
                    st.warning(f"**注意:** テーブル '{table_name_display}' を削除すると元に戻せません。") # **Caution:** Deleting table '{X}' cannot be undone.
                    if st.button(f"🗑️ テーブル '{table_name_display}' を削除", key=f"delete_table_{table_name_display}_v7_tab_data", type="secondary"): # 🗑️ Delete table '{X}'
                        with st.spinner(f"テーブル '{table_name_display}' を削除中..."): # Deleting table '{X}'...
                            del_success_flag, del_msg_text = rag.delete_data_table(table_name_display) # Renamed variables
                        if del_success_flag:
                            st.success(del_msg_text)
                            st.rerun() # Refresh list
                        else:
                            st.error(del_msg_text)
        else:
            st.info("分析可能なデータテーブルはまだありません。上記からファイルをアップロードしてください。") # No data tables available for analysis yet. Upload files from above.

# ── Tab 4: Document Management ────────────────────────────────────────────
with tab_documents: # tab3 から tab_documents に変更 (Change from tab3 to tab_documents)
    if rag:
        st.markdown("### 📤 ドキュメントアップロード") # 📤 Document upload
        uploaded_docs_list = st.file_uploader( # Renamed variable
            "ファイルを選択またはドラッグ&ドロップ (.pdf, .txt, .md, .docx, .doc)", # More specific help (Select or drag & drop files)
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx", "doc"],
            label_visibility="collapsed",
            key=f"doc_uploader_v7_tab_documents_{rag.config.collection_name if rag else 'default'}" # Make key unique per collection
        )

        if uploaded_docs_list:
            st.markdown(f"#### 選択されたファイル ({len(uploaded_docs_list)})") # Selected files ({X})
            file_info_display_list = [] # Renamed variable
            for file_item in uploaded_docs_list:
                file_info_display_list.append({
                    "ファイル名": file_item.name, # File name
                    "サイズ": f"{file_item.size / 1024:.1f} KB", # Size
                    "タイプ": file_item.type or "不明" # Type (Unknown)
                })
            st.dataframe(pd.DataFrame(file_info_display_list), use_container_width=True, hide_index=True)

            if st.button("🚀 ドキュメントを処理 (インジェスト)", type="primary", use_container_width=True, key="process_docs_button_v7_tab_documents"): # 🚀 Process documents (ingest)
                progress_bar_docs_ingest = st.progress(0, text="処理開始...") # Renamed variable, added initial text (Processing started...)
                status_text_docs_ingest = st.empty()    # Renamed variable
                try:
                    paths_to_ingest_list = [] # Renamed variable
                    for i, file_item_to_ingest in enumerate(uploaded_docs_list): # Renamed loop variable
                        status_text_docs_ingest.info(f"一時保存中: {file_item_to_ingest.name}") # Use info for better visibility (Temporarily saving:)
                        paths_to_ingest_list.append(str(_persist_uploaded_file(file_item_to_ingest)))
                        # Update progress after each file is persisted
                        progress_bar_docs_ingest.progress((i + 1) / (len(uploaded_docs_list) * 2), text=f"一時保存完了: {file_item_to_ingest.name}") # Temporary save complete:


                    status_text_docs_ingest.info(f"インデックスを構築中... ({len(paths_to_ingest_list)}件のファイル)") # Building index... ({X} files)
                    # Assuming rag.ingest_documents handles multiple paths
                    # For now, just update progress after the call, or consider adding a callback to ingest_documents
                    rag.ingest_documents(paths_to_ingest_list)
                    progress_bar_docs_ingest.progress(1.0, text="インジェスト完了！") # Full progress after ingestion (Ingest complete!)
                    st.success(f"✅ {len(uploaded_docs_list)}個のファイルが正常に処理されました！") # ✅ {X} files processed successfully!
                    time.sleep(1) # Keep success message visible
                    st.balloons()
                    # Clear uploader and refresh document list after successful upload
                    st.rerun()
                except Exception as e_ingest: # Renamed exception variable
                    st.error(f"ドキュメント処理中にエラーが発生しました: {type(e_ingest).__name__} - {e_ingest}") # An error occurred during document processing:
                finally:
                    # Clear progress bar and status text
                    if 'progress_bar_docs_ingest' in locals(): progress_bar_docs_ingest.empty()
                    if 'status_text_docs_ingest' in locals(): status_text_docs_ingest.empty()


        st.markdown("### 📚 登録済みドキュメント") # 📚 Registered documents
        docs_df_display = get_documents_dataframe(rag) # This function should be robust
        if not docs_df_display.empty:
            st.dataframe(docs_df_display, use_container_width=True, hide_index=True)

            st.markdown("### 🗑️ ドキュメント削除") # 🗑️ Delete document
            doc_ids_for_deletion_options = ["選択してください..."] + docs_df_display["Document ID"].tolist() # Select...
            doc_to_delete_selected = st.selectbox( # Renamed variable
                "削除するドキュメントIDを選択:", # Select document ID to delete:
                doc_ids_for_deletion_options,
                label_visibility="collapsed", # Already set, but good to note
                key=f"doc_delete_selectbox_v7_tab_documents_{rag.config.collection_name if rag else 'default'}"
            )
            if doc_to_delete_selected != "選択してください...": # Select...
                st.warning(f"**警告:** ドキュメント '{doc_to_delete_selected}' を削除すると、関連する全てのチャンクがデータベースとベクトルストアから削除されます。この操作は元に戻せません。") # **Warning:** Deleting document '{X}' will remove all related chunks from the database and vector store. This operation cannot be undone.
                if st.button(f"'{doc_to_delete_selected}' を削除実行", type="secondary", key="doc_delete_button_v7_tab_documents"): # More specific button text (Execute deletion of '{X}')
                    try:
                        with st.spinner(f"削除中: {doc_to_delete_selected}"): # Deleting:
                            success, message = rag.delete_document_by_id(doc_to_delete_selected)
                        if success:
                            st.success(message)
                            time.sleep(1)
                            st.rerun() # Refresh list
                        else:
                            st.error(message)
                    except Exception as e_delete: # Renamed exception variable
                        st.error(f"ドキュメント削除中にエラーが発生しました: {type(e_delete).__name__} - {e_delete}") # An error occurred during document deletion:
        else:
            st.info("まだドキュメントが登録されていません。上のセクションからアップロードしてください。") # No documents registered yet. Upload from the section above.
    else:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。") # RAG system is not initialized. Check settings in the sidebar.


# ── Tab 5: Settings ───────────────────────────────────────────────────────
with tab_settings: # tab4 から tab_settings に変更 (Change from tab4 to tab_settings)
    st.markdown("### ⚙️ システム詳細設定") # ⚙️ System detailed settings
    st.caption("RAGシステムの詳細な設定を行います。変更後は「設定を適用」ボタンをクリックしてください。システムの再初期化が必要な場合があります。") # Perform detailed settings for the RAG system. After changes, click the "Apply Settings" button. System reinitialization may be required.

    # Default values for form elements if rag or rag.config is not available
    # Use a temporary Config instance with .env defaults to populate the form initially
    # This ensures that if rag is None, the form still shows meaningful defaults.
    temp_default_cfg = Config() # Initializes with .env or dataclass defaults

    current_values_dict: Dict[str, Any] = {}
    if rag and hasattr(rag, 'config'):
        current_values_dict = rag.config.__dict__ # Use actual config if available
    else:
        current_values_dict = temp_default_cfg.__dict__ # Fallback to defaults

    # Ensure all expected keys exist in current_values_dict, falling back to temp_default_cfg if a key is missing
    # This is mostly for robustness if Config structure changes or rag.config is somehow incomplete
    for key, default_val in temp_default_cfg.__dict__.items():
        if key not in current_values_dict:
            current_values_dict[key] = default_val


    with st.form("detailed_settings_form_v7_tab_settings"):
        col1_settings, col2_settings = st.columns(2) # Renamed columns
        with col1_settings:
            st.markdown("#### 🔑 APIキー設定") # 🔑 API Key Settings
            form_openai_api_key = st.text_input(
                "OpenAI APIキー", # OpenAI API Key
                value=current_values_dict.get("openai_api_key", "") or "", # Handle None from getenv
                type="password",
                key="setting_openai_key_v7_tab_settings",
                help="OpenAI APIキー。変更するとシステムが再初期化されます。" # OpenAI API Key. Changing it will reinitialize the system.
            )

            st.markdown("#### 🤖 AIモデル設定") # 🤖 AI Model Settings
            emb_opts_form = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
            current_emb_model_form = current_values_dict.get("embedding_model", temp_default_cfg.embedding_model)
            emb_idx_form = emb_opts_form.index(current_emb_model_form) if current_emb_model_form in emb_opts_form else 0
            embedding_model_form_val = st.selectbox("埋め込みモデル", emb_opts_form, index=emb_idx_form, help="ドキュメントのベクトル化に使用するモデル", key="setting_emb_model_v7_tab_settings") # Embedding model (Model used for document vectorization)

            llm_opts_form = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
            current_llm_model_form = current_values_dict.get("llm_model", temp_default_cfg.llm_model)
            llm_idx_form = llm_opts_form.index(current_llm_model_form) if current_llm_model_form in llm_opts_form else 0
            llm_model_form_val = st.selectbox("言語モデル", llm_opts_form, index=llm_idx_form, help="回答生成に使用するGPTモデル", key="setting_llm_model_v7_tab_settings") # Language model (GPT model used for answer generation)

            st.markdown("#### 📄 チャンク設定") # 📄 Chunk Settings
            chunk_size_form_val = st.number_input("チャンクサイズ", 100, 5000, current_values_dict.get("chunk_size", temp_default_cfg.chunk_size), 100, help="1つのチャンクの最大文字数", key="setting_chunk_size_v7_tab_settings") # Chunk size (Max characters per chunk)
            chunk_overlap_form_val = st.number_input("チャンクオーバーラップ", 0, 1000, current_values_dict.get("chunk_overlap", temp_default_cfg.chunk_overlap), 50, help="隣接するチャンク間で重複する文字数", key="setting_chunk_overlap_v7_tab_settings") # Chunk overlap (Number of overlapping characters between adjacent chunks)

        with col2_settings:
            st.markdown("#### 🔍 検索・RAG設定") # 🔍 Search & RAG Settings
            collection_name_form_val = st.text_input("コレクション名", current_values_dict.get("collection_name", temp_default_cfg.collection_name), help="ドキュメントを格納するコレクションの名前", key="setting_collection_name_v7_tab_settings") # Collection name (Name of the collection to store documents)
            final_k_form_val = st.slider("最終検索結果数 (Final K)", 1, 20, current_values_dict.get("final_k", temp_default_cfg.final_k), help="LLMに渡す最終的なチャンク数", key="setting_final_k_v7_tab_settings") # Final number of search results (Final K) (Final number of chunks to pass to LLM)
            vector_search_k_form_val = st.number_input("ベクトル検索数 (Vector K)", 1, 50, current_values_dict.get("vector_search_k", temp_default_cfg.vector_search_k), help="ベクトル検索で取得する候補数", key="setting_vector_k_v7_tab_settings") # Number of vector search results (Vector K) (Number of candidates to retrieve in vector search)
            keyword_search_k_form_val = st.number_input("キーワード検索数 (Keyword K)", 1, 50, current_values_dict.get("keyword_search_k", temp_default_cfg.keyword_search_k), help="キーワード検索で取得する候補数", key="setting_keyword_k_v7_tab_settings") # Number of keyword search results (Keyword K) (Number of candidates to retrieve in keyword search)
            rrf_k_for_fusion_form_val = st.number_input("RAG-Fusion用RRF係数 (k)", 1, 100, current_values_dict.get("rrf_k_for_fusion", temp_default_cfg.rrf_k_for_fusion), help="RAG-Fusion時のRRFで使用するk値 (通常60程度)", key="setting_rrf_k_v7_tab_settings") # RRF coefficient for RAG-Fusion (k) (k value used in RRF for RAG-Fusion (usually around 60))
            embedding_dimensions_form_val = st.number_input("埋め込み次元数", value=current_values_dict.get("embedding_dimensions", temp_default_cfg.embedding_dimensions), min_value=128, max_value=8192, step=128, help="埋め込みモデルの次元数。モデルに合わせて変更してください。", key="setting_emb_dim_v7_tab_settings") # Embedding dimensions (Dimensionality of the embedding model. Change according to the model.)


        st.markdown("---")
        st.markdown("#### 🗄️ データベース設定 (変更には注意が必要です)") # 🗄️ Database Settings (Be careful when changing)
        db_col1_settings, db_col2_settings = st.columns(2) # Renamed columns
        with db_col1_settings:
            db_host_form_val = st.text_input("DBホスト", current_values_dict.get("db_host", temp_default_cfg.db_host), key="setting_db_host_v7_tab_settings") # DB Host
            db_name_form_val = st.text_input("DB名", current_values_dict.get("db_name", temp_default_cfg.db_name), key="setting_db_name_v7_tab_settings") # DB Name
            db_user_form_val = st.text_input("DBユーザー", current_values_dict.get("db_user", temp_default_cfg.db_user), key="setting_db_user_v7_tab_settings") # DB User
        with db_col2_settings:
            db_port_form_val = st.text_input("DBポート", current_values_dict.get("db_port", temp_default_cfg.db_port), key="setting_db_port_v7_tab_settings") # DB Port
            db_password_form_val = st.text_input("DBパスワード", current_values_dict.get("db_password", temp_default_cfg.db_password), type="password", key="setting_db_pass_v7_tab_settings") # DB Password
            # FTS language example (assuming it's a simple select for now)
            fts_language_options = ["english", "japanese", "simple", "german", "french"] # Add more as needed
            current_fts_lang = current_values_dict.get("fts_language", temp_default_cfg.fts_language)
            fts_lang_idx = fts_language_options.index(current_fts_lang) if current_fts_lang in fts_language_options else 0
            fts_language_form_val = st.selectbox("FTS言語", fts_language_options, index=fts_lang_idx, key="setting_fts_lang_v7_tab_settings", help="全文検索インデックスで使用する言語") # FTS Language (Language used for full-text search index)

        st.markdown("#### 📈 SQL分析設定") # 📈 SQL Analysis Settings
        max_sql_results_form_val = st.number_input(
            "SQL最大取得行数", 10, 10000, # SQL Max Rows to Retrieve
            current_values_dict.get("max_sql_results", temp_default_cfg.max_sql_results), 10,
            help="SQLクエリでデータベースから取得する最大行数。", # Max number of rows to retrieve from the database with an SQL query.
            key="setting_max_sql_results_v7_tab_settings"
        )
        max_sql_preview_llm_form_val = st.number_input(
            "SQL結果LLMプレビュー行数", 1, 100, # SQL Result LLM Preview Rows
            current_values_dict.get("max_sql_preview_rows_for_llm", temp_default_cfg.max_sql_preview_rows_for_llm), 1,
            help="SQL実行結果をLLMに渡して要約させる際の最大プレビュー行数。", # Max number of preview rows when passing SQL execution results to LLM for summarization.
            key="setting_max_sql_preview_llm_v7_tab_settings"
        )


        s_col_form, r_col_form = st.columns([3,1]) # Renamed columns
        apply_settings_button_form = s_col_form.form_submit_button("🔄 設定を適用", type="primary", use_container_width=True) # Renamed variable (🔄 Apply Settings)
        reset_settings_button_form = r_col_form.form_submit_button("↩️ デフォルトにリセット", use_container_width=True) # Renamed variable (↩️ Reset to Default)

    if apply_settings_button_form:
        try:
            # Create a new Config object using all values from the form
            new_app_config_obj = Config( # Renamed variable
                openai_api_key=form_openai_api_key, # Get API key from form
                embedding_model=embedding_model_form_val,
                llm_model=llm_model_form_val,
                collection_name=collection_name_form_val,
                final_k=int(final_k_form_val),
                chunk_size=chunk_size_form_val,
                chunk_overlap=chunk_overlap_form_val,
                vector_search_k=vector_search_k_form_val,
                keyword_search_k=keyword_search_k_form_val,
                db_host=db_host_form_val,
                db_port=db_port_form_val,
                db_name=db_name_form_val,
                db_user=db_user_form_val,
                db_password=db_password_form_val, # Password from form
                embedding_dimensions=embedding_dimensions_form_val,
                fts_language=fts_language_form_val,
                rrf_k_for_fusion=rrf_k_for_fusion_form_val,
                max_sql_results=max_sql_results_form_val,
                max_sql_preview_rows_for_llm=max_sql_preview_llm_form_val
                # enable_text_to_sql is not directly set here, defaults in Config dataclass
            )

            cfg_changed_flag = False # Renamed variable
            if rag and hasattr(rag, 'config'):
                # Compare new_app_config_obj with rag.config
                # A simple way is to compare their dict representations for relevant keys
                # More robust: compare field by field
                if new_app_config_obj != rag.config: # Dataclass comparison works if __eq__ is default
                    cfg_changed_flag = True
            else: # If rag system wasn't initialized, any setting is a "change" to initialize
                cfg_changed_flag = True


            if cfg_changed_flag:
                st.info("設定が変更されました。システムを再初期化します...") # Settings have been changed. Reinitializing system...

            with st.spinner("設定を適用し、システムを初期化しています..."): # Applying settings and initializing system...
                if "rag_system" in st.session_state:
                    del st.session_state["rag_system"] # Clear previous instance
                    st.cache_resource.clear() # Clear all cached resources, including initialize_rag_system
                
                st.session_state.rag_system = initialize_rag_system(new_app_config_obj)
                rag = st.session_state.rag_system # Update global rag variable
            st.success("✅ 設定が正常に適用され、システムが初期化されました！") # ✅ Settings applied successfully and system initialized!
            time.sleep(1)
            st.rerun()
        except Exception as e_apply_settings: # Renamed exception variable
            st.error(f"❌ 設定の適用中にエラーが発生しました: {type(e_apply_settings).__name__} - {e_apply_settings}") # ❌ An error occurred while applying settings:

    if reset_settings_button_form:
        st.info("設定をデフォルト値にリセットし、システムを再初期化します...") # Resetting settings to default values and reinitializing system...
        # Use a new Config() instance to get all .env or dataclass defaults
        default_config_for_reset_obj = Config(openai_api_key=ENV_DEFAULTS["OPENAI_API_KEY"]) # Ensure API key default is also considered

        with st.spinner("デフォルト設定でシステムを初期化しています..."): # Initializing system with default settings...
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
                st.cache_resource.clear()
            st.session_state.rag_system = initialize_rag_system(default_config_for_reset_obj)
            rag = st.session_state.rag_system
        st.success("✅ 設定がデフォルトにリセットされ、システムが初期化されました！") # ✅ Settings reset to default and system initialized!
        time.sleep(1)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 現在の有効な設定") # 📋 Current effective settings
    if rag and hasattr(rag, 'config'):
        # Display current effective configuration from the active RAGSystem instance
        # Convert dataclass to dict for easier iteration, but exclude sensitive fields like password
        config_display_dict = rag.config.__dict__.copy()
        if "db_password" in config_display_dict:
            config_display_dict["db_password"] = "********" # Mask password
        if "openai_api_key" in config_display_dict and config_display_dict["openai_api_key"]:
            config_display_dict["openai_api_key"] = f"sk-...{config_display_dict['openai_api_key'][-4:]}" if len(config_display_dict['openai_api_key']) > 8 else "********"


        # Create two columns for display
        col1_disp_settings, col2_disp_settings = st.columns(2) # Renamed columns
        
        # Split items for two-column display
        items_to_display_list = list(config_display_dict.items()) # Renamed variable
        mid_point_display = (len(items_to_display_list) + 1) // 2 # Renamed variable

        with col1_disp_settings:
            for key_disp, value_disp in items_to_display_list[:mid_point_display]:
                st.markdown(f"**{key_disp.replace('_', ' ').capitalize()}:** `{str(value_disp)}`")
        with col2_disp_settings:
            for key_disp, value_disp in items_to_display_list[mid_point_display:]:
                st.markdown(f"**{key_disp.replace('_', ' ').capitalize()}:** `{str(value_disp)}`")
    else:
        st.info("システムが初期化されていません。上記フォームから設定を適用してください。") # System is not initialized. Apply settings from the form above.

# SQL分析タブ (Tab 6) のコンテンツは削除されました。
# (Content of SQL Analysis Tab (Tab 6) has been removed.)
