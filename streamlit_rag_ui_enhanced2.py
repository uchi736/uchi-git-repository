"""streamlit_rag_ui_refined.py – Refined Modern RAG System
========================================================
洗練されたモダンなRAGシステムUI - 実用性とスタイルの融合

起動: streamlit run streamlit_rag_ui_refined.py
"""
from __future__ import annotations

import os
import json
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

# Plotlyのインポート（オプション）
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
try:
    from rag_system import Config, RAGSystem 
except ModuleNotFoundError:
    st.error("❌ rag_system.py が見つかりません。")
    st.stop()

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System • Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Refined CSS Design ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        /* ダークテーマカラー（視認性改善） */
        --bg-primary: #0a0a0a;
        --bg-secondary: #141414;
        --bg-tertiary: #1a1a1a;
        --surface: #242424;
        --surface-hover: #2a2a2a;
        --border: #333333;
        
        /* テキストカラー（コントラスト改善） */
        --text-primary: #ffffff;
        --text-secondary: #b3b3b3;
        --text-tertiary: #808080;
        
        /* アクセントカラー */
        --accent: #7c3aed;
        --accent-hover: #8b5cf6;
        --accent-light: rgba(124, 58, 237, 0.15);
        
        /* ステータスカラー */
        --success: #10b981;
        --error: #ef4444;
        --warning: #f59e0b;
        --info: #3b82f6;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Streamlitのデフォルト要素を非表示 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* スクロールバー */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #666;
    }

    /* ヘッダー */
    .main-header {
        background: linear-gradient(135deg, var(--accent) 0%, #a855f7 100%);
        padding: 2rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3);
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }

    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* カード */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* チャットコンテナ - 高さを削除してコンテンツに応じて伸縮 */
    .chat-container {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 1.5rem;
        overflow-y: auto;
        border: 1px solid var(--border);
        max-height: 600px;
    }

    /* メッセージ */
    .message {
        margin-bottom: 1rem;
        display: flex;
    }

    .user-message {
        margin-left: auto;
        max-width: 70%;
    }

    .ai-message {
        margin-right: auto;
        max-width: 70%;
    }

    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        word-wrap: break-word;
    }

    .user-bubble {
        background: var(--accent);
        color: white;
        border-bottom-right-radius: 4px;
    }

    .ai-bubble {
        background: var(--surface);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-bottom-left-radius: 4px;
    }

    /* ソースカード */
    .source-container {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border);
        height: 500px;
        overflow-y: auto;
    }

    .source-item {
        background: var(--surface);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid var(--accent);
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .source-item:hover {
        transform: translateX(4px);
        background: var(--surface-hover);
    }

    .source-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }

    .source-excerpt {
        font-size: 0.875rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* 全文表示エリア */
    .full-text-container {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        max-height: 300px;
        overflow-y: auto;
        margin-top: 0.5rem;
        font-size: 0.875rem;
        line-height: 1.6;
        color: var(--text-primary);
    }

    /* 統計カード */
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }

    /* ボタン */
    .stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: var(--accent-hover);
        transform: translateY(-1px);
    }

    /* 入力フィールド */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px var(--accent-light);
    }

    /* セレクトボックス */
    .stSelectbox > div > div > div {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text-primary);
    }

    /* タブ */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--surface);
        color: var(--text-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
    }

    /* ファイルアップローダー */
    .stFileUploader > div {
        background: var(--surface);
        border: 2px dashed var(--border);
        border-radius: 12px;
    }

    .stFileUploader > div:hover {
        border-color: var(--accent);
        background: var(--surface-hover);
    }

    /* プログレスバー */
    .stProgress > div > div > div > div {
        background: var(--accent);
    }

    /* メトリクス */
    div[data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }

    /* サイドバー */
    .css-1d391kg {
        background: var(--bg-secondary);
    }

    /* フォームラベル */
    .stFormLabel {
        color: var(--text-primary) !important;
        font-weight: 500;
    }

    /* スライダー */
    .stSlider > div > div > div > div {
        background: var(--accent);
    }

    /* チェックボックス・ラジオボタン */
    .stCheckbox > label > span,
    .stRadio > div > label > span {
        color: var(--text-primary);
    }

    /* 情報ボックス */
    .stAlert {
        background: var(--surface);
        color: var(--text-primary);
        border: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────

def _persist_uploaded_file(uploaded_file) -> Path:
    """ファイルの一時保存"""
    if uploaded_file is None:
        raise ValueError("Uploaded file cannot be None")
    tmp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / uploaded_file.name
    tmp_path.write_bytes(uploaded_file.getbuffer())
    return tmp_path

@st.cache_resource(show_spinner=False)
def initialize_rag_system(config: Config) -> RAGSystem:
    """RAGシステムの初期化"""
    return RAGSystem(config)

def get_collection_statistics(rag: RAGSystem) -> Dict[str, Any]:
    """コレクション統計の取得"""
    if not rag:
        return {"documents": 0, "chunks": 0, "collection_name": "N/A"}
    
    try:
        engine = create_engine(rag.connection_string)
        with engine.connect() as conn:
            query = text("""
                SELECT COUNT(DISTINCT document_id) AS num_documents,
                       COUNT(*) AS num_chunks
                FROM document_chunks
                WHERE collection_name = :collection
            """)
            result = conn.execute(query, {"collection": rag.config.collection_name}).first()
            
            return {
                "documents": result.num_documents if result else 0,
                "chunks": result.num_chunks if result else 0,
                "collection_name": rag.config.collection_name
            }
    except Exception as e:
        st.error(f"統計情報の取得に失敗: {e}")
        return {"documents": 0, "chunks": 0, "collection_name": rag.config.collection_name}

def get_documents_dataframe(rag: RAGSystem) -> pd.DataFrame:
    """ドキュメント一覧の取得"""
    if not rag:
        return pd.DataFrame()
    
    try:
        engine = create_engine(rag.connection_string)
        with engine.connect() as conn:
            query = text("""
                SELECT document_id, COUNT(*) as chunk_count, MAX(created_at) as last_updated
                FROM document_chunks
                WHERE collection_name = :collection
                GROUP BY document_id
                ORDER BY last_updated DESC
            """)
            result = conn.execute(query, {"collection": rag.config.collection_name})
            df = pd.DataFrame(result.fetchall(), columns=["Document ID", "Chunks", "Last Updated"])
            if not df.empty and "Last Updated" in df.columns:
                df["Last Updated"] = pd.to_datetime(df["Last Updated"]).dt.strftime("%Y-%m-%d %H:%M")
            return df
    except Exception:
        return pd.DataFrame()

def get_query_history_data(days: int = 30) -> pd.DataFrame:
    """クエリ履歴データの取得（デモ用）"""
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    queries = [20 + int(10 * abs(np.sin(i/5))) for i in range(days)]
    return pd.DataFrame({'Date': dates, 'Queries': queries})

# ── Initialize Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "rag_system" not in st.session_state:
    if ENV_DEFAULTS["OPENAI_API_KEY"]:
        try:
            config = Config()
            st.session_state.rag_system = initialize_rag_system(config)
            st.toast("✅ システムが正常に初期化されました", icon="✅")
        except Exception as e:
            st.error(f"初期化エラー: {e}")

# ── Main Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 class="header-title">RAG System</h1>
    <p class="header-subtitle">Intelligent Document Analysis & Question Answering</p>
</div>
""", unsafe_allow_html=True)

# ── Get RAG System ────────────────────────────────────────────────────────
rag = st.session_state.get("rag_system")

# ── Sidebar Configuration ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color: var(--text-primary);'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    if rag:
        # System Status
        st.success(f"✅ System Online - Collection: **{rag.config.collection_name}**")
    
    # Configuration Form
    with st.form("config_form"):
        st.markdown("### 🤖 Model Settings")
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
            index=0 if not rag else (
                0 if rag.config.embedding_model == "text-embedding-ada-002" else
                1 if rag.config.embedding_model == "text-embedding-3-small" else 2
            )
        )
        
        llm_model = st.selectbox(
            "Language Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            index=0 if not rag else (
                0 if rag.config.llm_model == "gpt-4o" else
                1 if rag.config.llm_model == "gpt-4-turbo" else
                2 if rag.config.llm_model == "gpt-4" else 3
            )
        )
        
        st.markdown("### 🔍 Search Settings")
        
        collection_name = st.text_input(
            "Collection Name",
            value=rag.config.collection_name if rag else ENV_DEFAULTS["COLLECTION_NAME"]
        )
        
        final_k = st.slider(
            "検索結果数 (Final K)",
            min_value=1,
            max_value=20,
            value=rag.config.final_k if rag else ENV_DEFAULTS["FINAL_K"],
            help="LLMに渡す最終的なチャンク数"
        )
        
        st.markdown("### 📊 Chunking Settings")
        
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=5000,
            value=int(os.getenv("CHUNK_SIZE", 1000)),
            step=100
        )
        
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            value=int(os.getenv("CHUNK_OVERLAP", 200)),
            step=50
        )
        
        apply_button = st.form_submit_button("Apply Settings", use_container_width=True)

# Handle configuration update
if apply_button:
    updated_config = Config(
        openai_api_key=ENV_DEFAULTS["OPENAI_API_KEY"],
        embedding_model=embedding_model,
        llm_model=llm_model,
        collection_name=collection_name,
        final_k=int(final_k),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
        db_name=os.getenv("DB_NAME", "postgres"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "your-password"),
    )
    
    try:
        with st.spinner("設定を適用しています..."):
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
            st.session_state["rag_system"] = initialize_rag_system(updated_config)
            rag = st.session_state["rag_system"]
        st.success("✅ 設定が正常に適用されました")
        st.rerun()
    except Exception as e:
        st.error(f"エラー: {e}")

# ── Main Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💬 **Chat**", "📊 **Analytics**", "📁 **Documents**", "⚙️ **Settings**"])

# ── Tab 1: Chat Interface ─────────────────────────────────────────────────
with tab1:
    if not rag:
        st.info("🔧 システムを初期化してください。サイドバーで設定を確認してください。")
    else:
        # レイアウト: 入力エリアを上部に配置
        # Input Area
        st.markdown("### 💬 質問を入力")
        user_input = st.text_area(
            "質問を入力",
            placeholder="ドキュメントについて質問してください...",
            height=80,
            key="user_input",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            send_button = st.button("送信", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("クリア", use_container_width=True)
        
        # Handle send
        if send_button and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("回答を生成中..."):
                try:
                    response = rag.query(user_input)
                    answer = response.get("answer", "申し訳ございません。回答を生成できませんでした。")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.current_sources = response.get("sources", [])
                    
                except Exception as e:
                    st.error(f"エラー: {e}")
            
            st.rerun()
        
        # Handle clear
        if clear_button:
            st.session_state.messages = []
            st.session_state.current_sources = []
            st.rerun()
        
        # チャット履歴とソースを横並びで表示
        st.markdown("---")
        chat_col, source_col = st.columns([2, 1])
        
        with chat_col:
            st.markdown("#### 会話履歴")
            # Chat History
            chat_container = st.container()
            with chat_container:
                if st.session_state.messages:
                    for message in st.session_state.messages:
                        if message["role"] == "user":
                            st.markdown(f"""
                            <div class="message user-message">
                                <div class="message-bubble user-bubble">
                                    {message["content"]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="message ai-message">
                                <div class="message-bubble ai-bubble">
                                    {message["content"]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("まだ会話履歴がありません。上の入力欄から質問を始めてください。")
        
        with source_col:
            st.markdown("#### 📚 参照ソース")
            source_container = st.container()
            with source_container:
                if st.session_state.current_sources:
                    for i, source in enumerate(st.session_state.current_sources):
                        doc_id = source.get('metadata', {}).get('document_id', 'Unknown')
                        chunk_id = source.get('metadata', {}).get('chunk_id', 'N/A')
                        excerpt = source.get('excerpt', '')[:150] + '...' if len(source.get('excerpt', '')) > 150 else source.get('excerpt', '')
                        
                        # ソースカード
                        st.markdown(f"""
                        <div class="source-item">
                            <div class="source-title">ソース {i+1}: {doc_id}</div>
                            <div class="source-excerpt">{excerpt}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 全文表示ボタン
                        if st.button(f"全文を表示", key=f"show_full_{i}"):
                            st.session_state[f"show_full_text_{i}"] = not st.session_state.get(f"show_full_text_{i}", False)
                        
                        # 全文表示
                        if st.session_state.get(f"show_full_text_{i}", False):
                            full_text = source.get('full_content', 'コンテンツなし')
                            st.markdown(f"""
                            <div class="full-text-container">
                                <strong>Chunk ID:</strong> {chunk_id}<br><br>
                                {full_text}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("質問を送信すると、参照したソースがここに表示されます")

# ── Tab 2: Analytics Dashboard ────────────────────────────────────────────
with tab2:
    if rag:
        stats = get_collection_statistics(rag)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['documents']:,}</div>
                <div class="stat-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['chunks']:,}</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_chunks = stats['chunks'] // max(stats['documents'], 1)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{avg_chunks}</div>
                <div class="stat-label">Avg Chunks/Doc</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            query_count = len(st.session_state.messages) // 2
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{query_count}</div>
                <div class="stat-label">Total Queries</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Query History Chart
        st.markdown("### 📈 質問数の推移（日別）")
        st.caption("過去30日間の質問数の推移を表示しています")
        
        if PLOTLY_AVAILABLE:
            # Plotlyを使用
            query_data = get_query_history_data(30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=query_data['Date'],
                y=query_data['Queries'],
                mode='lines+markers',
                name='質問数',
                line=dict(color='#7c3aed', width=3),
                marker=dict(size=8, color='#8b5cf6'),
                fill='tozeroy',
                fillcolor='rgba(124, 58, 237, 0.1)'
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    title="日付"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title="質問数"
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Streamlit標準のチャート
            query_data = get_query_history_data(30)
            st.line_chart(query_data.set_index('Date'))
        
        # Recent Activity
        st.markdown("### 🕐 最近の質問")
        
        if st.session_state.messages:
            recent_questions = []
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    recent_questions.append({
                        "質問": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"],
                        "時刻": (datetime.now() - timedelta(minutes=len(st.session_state.messages)-i)).strftime("%H:%M")
                    })
            
            if recent_questions:
                df_questions = pd.DataFrame(recent_questions[-5:])  # 最新5件
                st.dataframe(df_questions, use_container_width=True, hide_index=True)
            else:
                st.info("まだ質問がありません")
        else:
            st.info("まだ質問がありません")
    else:
        st.info("システムを初期化してください")

# ── Tab 3: Document Management ────────────────────────────────────────────
with tab3:
    if rag:
        st.markdown("### 📤 ドキュメントアップロード")
        
        uploaded_files = st.file_uploader(
            "ファイルを選択またはドラッグ&ドロップ",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx", "doc"],
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.markdown(f"#### 選択されたファイル ({len(uploaded_files)})")
            
            # File preview
            file_data = []
            for file in uploaded_files:
                file_data.append({
                    "ファイル名": file.name,
                    "サイズ": f"{file.size / 1024:.1f} KB",
                    "タイプ": file.type or "不明"
                })
            
            st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)
            
            # Process button
            if st.button("🚀 ドキュメントを処理", type="primary", use_container_width=True):
                progress = st.progress(0)
                status_text = st.empty()
                
                try:
                    paths = []
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"処理中: {file.name}")
                        progress.progress((i + 1) / len(uploaded_files))
                        
                        path = _persist_uploaded_file(file)
                        paths.append(str(path))
                    
                    status_text.text("インデックスを構築中...")
                    rag.ingest_documents(paths)
                    
                    progress.progress(1.0)
                    st.success(f"✅ {len(uploaded_files)}個のファイルが正常に処理されました！")
                    time.sleep(1)
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"エラー: {e}")
                finally:
                    progress.empty()
                    status_text.empty()
        
        # Document List
        st.markdown("### 📚 登録済みドキュメント")
        
        docs_df = get_documents_dataframe(rag)
        if not docs_df.empty:
            st.dataframe(docs_df, use_container_width=True, hide_index=True)
            
            # Delete functionality
            st.markdown("### 🗑️ ドキュメント削除")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                doc_to_delete = st.selectbox(
                    "削除対象",
                    ["選択してください..."] + docs_df["Document ID"].tolist(),
                    label_visibility="collapsed"
                )
            
            with col2:
                if doc_to_delete != "選択してください...":
                    if st.button("削除実行", type="secondary"):
                        try:
                            with st.spinner(f"削除中: {doc_to_delete}"):
                                success, message = rag.delete_document_by_id(doc_to_delete)
                                if success:
                                    st.success(message)
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(message)
                        except Exception as e:
                            st.error(f"エラー: {e}")
        else:
            st.info("まだドキュメントが登録されていません")
    else:
        st.info("システムを初期化してください")

# ── Tab 4: Settings ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### ⚙️ システム設定")
    st.caption("RAGシステムの詳細な設定を行います。変更後は「設定を適用」ボタンをクリックしてください。")
    
    # 現在の設定値を取得
    current_config = {
        "embedding_model": rag.config.embedding_model if rag else ENV_DEFAULTS["EMBEDDING_MODEL"],
        "llm_model": rag.config.llm_model if rag else ENV_DEFAULTS["LLM_MODEL"],
        "collection_name": rag.config.collection_name if rag else ENV_DEFAULTS["COLLECTION_NAME"],
        "final_k": rag.config.final_k if rag else ENV_DEFAULTS["FINAL_K"],
        "chunk_size": rag.config.chunk_size if rag else int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap": rag.config.chunk_overlap if rag else int(os.getenv("CHUNK_OVERLAP", 200)),
        "vector_search_k": rag.config.vector_search_k if rag else int(os.getenv("VECTOR_SEARCH_K", 10)),
        "keyword_search_k": rag.config.keyword_search_k if rag else int(os.getenv("KEYWORD_SEARCH_K", 10)),
    }
    
    with st.form("detailed_settings_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 AIモデル設定")
            
            # Embedding Model
            embedding_model = st.selectbox(
                "埋め込みモデル",
                ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                index=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"].index(current_config["embedding_model"]) if current_config["embedding_model"] in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"] else 0,
                help="ドキュメントのベクトル化に使用するモデル"
            )
            
            # LLM Model
            llm_model = st.selectbox(
                "言語モデル",
                ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
                index=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"].index(current_config["llm_model"]) if current_config["llm_model"] in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"] else 0,
                help="回答生成に使用するGPTモデル"
            )
            
            st.markdown("#### 📊 チャンク設定")
            
            # Chunk Size
            chunk_size = st.number_input(
                "チャンクサイズ",
                min_value=100,
                max_value=5000,
                value=current_config["chunk_size"],
                step=100,
                help="1つのチャンクの最大文字数"
            )
            
            # Chunk Overlap
            chunk_overlap = st.number_input(
                "チャンクオーバーラップ",
                min_value=0,
                max_value=1000,
                value=current_config["chunk_overlap"],
                step=50,
                help="隣接するチャンク間で重複する文字数"
            )
        
        with col2:
            st.markdown("#### 🔍 検索設定")
            
            # Collection Name
            collection_name = st.text_input(
                "コレクション名",
                value=current_config["collection_name"],
                help="ドキュメントを格納するコレクションの名前"
            )
            
            # Final K
            final_k = st.slider(
                "最終検索結果数 (Final K)",
                min_value=1,
                max_value=20,
                value=current_config["final_k"],
                help="LLMに渡す最終的なチャンク数"
            )
            
            # Vector Search K
            vector_search_k = st.number_input(
                "ベクトル検索数 (Vector Search K)",
                min_value=1,
                max_value=50,
                value=current_config["vector_search_k"],
                help="ベクトル検索で取得する候補数"
            )
            
            # Keyword Search K
            keyword_search_k = st.number_input(
                "キーワード検索数 (Keyword Search K)",
                min_value=1,
                max_value=50,
                value=current_config["keyword_search_k"],
                help="キーワード検索で取得する候補数"
            )
        
        st.markdown("---")
        
        # データベース設定（展開可能）
        with st.expander("🗄️ データベース設定（高度な設定）"):
            db_col1, db_col2 = st.columns(2)
            
            with db_col1:
                db_host = st.text_input(
                    "データベースホスト",
                    value=os.getenv("DB_HOST", "localhost"),
                    help="PostgreSQLサーバーのホスト名"
                )
                
                db_name = st.text_input(
                    "データベース名",
                    value=os.getenv("DB_NAME", "postgres"),
                    help="使用するデータベース名"
                )
                
                db_user = st.text_input(
                    "ユーザー名",
                    value=os.getenv("DB_USER", "postgres"),
                    help="データベースユーザー名"
                )
            
            with db_col2:
                db_port = st.text_input(
                    "ポート",
                    value=os.getenv("DB_PORT", "5432"),
                    help="PostgreSQLサーバーのポート番号"
                )
                
                db_password = st.text_input(
                    "パスワード",
                    value=os.getenv("DB_PASSWORD", "your-password"),
                    type="password",
                    help="データベースパスワード"
                )
                
                embedding_dimensions = st.number_input(
                    "埋め込み次元数",
                    min_value=384,
                    max_value=3072,
                    value=int(os.getenv("EMBEDDING_DIMENSIONS", 1536)),
                    help="使用する埋め込みモデルの次元数"
                )
        
        # Submit button
        col_submit, col_reset = st.columns([3, 1])
        with col_submit:
            apply_settings = st.form_submit_button("🔄 設定を適用", type="primary", use_container_width=True)
        with col_reset:
            reset_settings = st.form_submit_button("↩️ リセット", use_container_width=True)
    
    # Handle settings update
    if apply_settings:
        try:
            # 新しい設定で Config を作成
            updated_config = Config(
                openai_api_key=ENV_DEFAULTS["OPENAI_API_KEY"],
                embedding_model=embedding_model,
                llm_model=llm_model,
                collection_name=collection_name,
                final_k=int(final_k),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                vector_search_k=vector_search_k,
                keyword_search_k=keyword_search_k,
                db_host=db_host,
                db_port=db_port,
                db_name=db_name,
                db_user=db_user,
                db_password=db_password,
                embedding_dimensions=embedding_dimensions,
                fts_language=os.getenv("FTS_LANGUAGE", "english"),
            )
            
            # 設定が変更されたかチェック
            config_changed = False
            if rag:
                if (updated_config.embedding_model != rag.config.embedding_model or
                    updated_config.llm_model != rag.config.llm_model or
                    updated_config.collection_name != rag.config.collection_name or
                    updated_config.chunk_size != rag.config.chunk_size or
                    updated_config.chunk_overlap != rag.config.chunk_overlap):
                    config_changed = True
            
            if config_changed:
                st.warning("⚠️ 重要な設定が変更されました。システムを再初期化します...")
            
            with st.spinner("設定を適用しています..."):
                if "rag_system" in st.session_state:
                    del st.session_state["rag_system"]
                st.session_state["rag_system"] = initialize_rag_system(updated_config)
                rag = st.session_state["rag_system"]
            
            st.success("✅ 設定が正常に適用されました！")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ 設定の適用中にエラーが発生しました: {e}")
    
    # Handle reset
    if reset_settings:
        st.info("設定をデフォルト値にリセットします...")
        time.sleep(1)
        st.rerun()
    
    # 現在の設定を表示
    st.markdown("---")
    st.markdown("### 📋 現在の設定")
    
    if rag:
        config_display = {
            "埋め込みモデル": rag.config.embedding_model,
            "言語モデル": rag.config.llm_model,
            "コレクション名": rag.config.collection_name,
            "チャンクサイズ": f"{rag.config.chunk_size} 文字",
            "オーバーラップ": f"{rag.config.chunk_overlap} 文字",
            "最終検索結果数": rag.config.final_k,
            "ベクトル検索数": rag.config.vector_search_k,
            "キーワード検索数": rag.config.keyword_search_k,
        }
        
        # 2列で表示
        col1, col2 = st.columns(2)
        items = list(config_display.items())
        half = len(items) // 2
        
        with col1:
            for key, value in items[:half]:
                st.markdown(f"**{key}:** {value}")
        
        with col2:
            for key, value in items[half:]:
                st.markdown(f"**{key}:** {value}")
    else:
        st.info("システムが初期化されていません")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top: 3rem; padding: 1.5rem 0; text-align: center; 
            color: var(--text-tertiary); border-top: 1px solid var(--border);">
    <p style="margin: 0;">RAG System • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)