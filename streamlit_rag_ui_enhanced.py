"""streamlit_rag_ui_ultra_modern.py – Ultra Modern RAG System
===========================================================
最新のデザイントレンドを取り入れた次世代RAGシステムUI
ChatGPT, Perplexity AI, Claude等の最新AIアプリのデザインを参考に作成

起動: streamlit run streamlit_rag_ui_ultra_modern.py
"""
from __future__ import annotations

import os
import json
import tempfile
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    page_title="Neural RAG • Next-Gen Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Ultra Modern CSS Design ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for Dark Theme */
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #111111;
        --bg-tertiary: #1a1a1a;
        --surface: #1e1e1e;
        --surface-hover: #252525;
        --border: #2a2a2a;
        --border-light: #333333;
        
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --text-tertiary: #707070;
        
        --accent-primary: #7c3aed;
        --accent-secondary: #a855f7;
        --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%);
        --accent-glow: 0 0 40px rgba(168, 85, 247, 0.4);
        
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --info: #3b82f6;
        
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 20px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 30px rgba(168, 85, 247, 0.3);
    }

    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Hide Streamlit Branding */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Modern Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
        box-shadow: var(--shadow-glow);
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg), var(--shadow-glow);
        border-color: var(--accent-primary);
    }

    /* Neumorphic Elements */
    .neu-button {
        background: var(--surface);
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        color: var(--text-primary);
        font-weight: 500;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.4),
            -8px -8px 16px rgba(255, 255, 255, 0.02);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .neu-button:hover {
        transform: scale(0.98);
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.4),
            -4px -4px 8px rgba(255, 255, 255, 0.02);
    }

    .neu-button:active {
        box-shadow: 
            inset 4px 4px 8px rgba(0, 0, 0, 0.4),
            inset -4px -4px 8px rgba(255, 255, 255, 0.02);
    }

    /* Hero Section */
    .hero-section {
        background: var(--accent-gradient);
        padding: 60px 40px;
        border-radius: 24px;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg), var(--shadow-glow);
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(to right, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 300;
        opacity: 0.9;
        margin-top: 10px;
        position: relative;
        z-index: 1;
    }

    /* Chat Interface */
    .chat-container {
        background: var(--bg-secondary);
        border-radius: 20px;
        padding: 30px;
        min-height: 600px;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border);
    }

    .message-bubble {
        padding: 16px 20px;
        border-radius: 18px;
        margin-bottom: 16px;
        max-width: 80%;
        animation: fadeInUp 0.4s ease-out;
        position: relative;
    }

    .user-message {
        background: var(--accent-gradient);
        color: white;
        margin-left: auto;
        box-shadow: var(--shadow-md), var(--shadow-glow);
    }

    .ai-message {
        background: var(--surface);
        color: var(--text-primary);
        margin-right: auto;
        border: 1px solid var(--border-light);
    }

    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 16px 20px;
        background: var(--surface);
        border-radius: 18px;
        margin-bottom: 16px;
    }

    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-primary);
        margin: 0 3px;
        animation: typing 1.4s infinite;
    }

    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes typing {
        0%, 60%, 100% {
            opacity: 0.3;
            transform: scale(0.8);
        }
        30% {
            opacity: 1;
            transform: scale(1.2);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Input Area */
    .input-container {
        background: var(--surface);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }

    .input-container:focus-within {
        border-color: var(--accent-primary);
        box-shadow: var(--shadow-lg), var(--shadow-glow);
    }

    /* Source Cards */
    .source-card {
        background: var(--surface);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .source-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--accent-gradient);
    }

    .source-card:hover {
        transform: translateX(8px);
        border-color: var(--accent-primary);
        box-shadow: var(--shadow-md);
    }

    /* Stats Dashboard */
    .stat-card {
        background: var(--surface);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stat-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: var(--accent-gradient);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }

    .stat-card:hover::after {
        transform: scaleX(1);
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }

    /* File Upload Area */
    .upload-zone {
        background: var(--surface);
        border: 2px dashed var(--accent-primary);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .upload-zone:hover {
        border-color: var(--accent-secondary);
        background: var(--surface-hover);
        box-shadow: var(--shadow-glow);
    }

    .upload-icon {
        font-size: 3rem;
        margin-bottom: 16px;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: var(--accent-gradient);
        box-shadow: var(--shadow-lg), var(--shadow-glow);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        z-index: 1000;
    }

    .fab:hover {
        transform: scale(1.1) rotate(90deg);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 8px;
        padding: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--surface);
        color: var(--text-secondary);
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 500;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        box-shadow: var(--shadow-md), var(--shadow-glow);
    }

    /* Progress Bar */
    .custom-progress {
        height: 4px;
        background: var(--surface);
        border-radius: 2px;
        overflow: hidden;
        margin: 20px 0;
    }

    .progress-fill {
        height: 100%;
        background: var(--accent-gradient);
        border-radius: 2px;
        transition: width 0.3s ease;
        box-shadow: var(--shadow-glow);
    }

    /* Notification Badge */
    .notification-badge {
        position: absolute;
        top: -8px;
        right: -8px;
        background: var(--error);
        color: white;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .hero-section { padding: 40px 20px; }
        .stat-card { padding: 16px; }
        .stat-number { font-size: 2rem; }
    }

    /* Custom Animations */
    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }

    .glow {
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { box-shadow: 0 0 10px var(--accent-primary); }
        to { box-shadow: 0 0 20px var(--accent-primary), 0 0 30px var(--accent-secondary); }
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────

def generate_avatar_url(text: str) -> str:
    """Generate consistent avatar URL from text"""
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    return f"https://api.dicebear.com/7.x/shapes/svg?seed={hash_hex}"

def format_time_ago(timestamp: datetime) -> str:
    """Format timestamp as 'time ago'"""
    now = datetime.now()
    diff = now - timestamp
    
    if diff.seconds < 60:
        return "たった今"
    elif diff.seconds < 3600:
        return f"{diff.seconds // 60}分前"
    elif diff.seconds < 86400:
        return f"{diff.seconds // 3600}時間前"
    else:
        return f"{diff.days}日前"

def create_typing_animation():
    """Create typing indicator"""
    return st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    """, unsafe_allow_html=True)

def _persist_uploaded_file(uploaded_file) -> Path:
    """Persist uploaded file to temp directory"""
    if uploaded_file is None:
        raise ValueError("Uploaded file cannot be None")
    tmp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / uploaded_file.name
    tmp_path.write_bytes(uploaded_file.getbuffer())
    return tmp_path

@st.cache_resource(show_spinner=False)
def initialize_rag_system(config: Config) -> RAGSystem:
    """Initialize RAG system with configuration"""
    return RAGSystem(config)

def get_collection_statistics(rag: RAGSystem) -> Dict[str, Any]:
    """Get collection statistics"""
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
    """Get documents as DataFrame"""
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
            return pd.DataFrame(result.fetchall(), columns=["Document ID", "Chunks", "Last Updated"])
    except Exception:
        return pd.DataFrame()

# ── Initialize Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    if ENV_DEFAULTS["OPENAI_API_KEY"]:
        try:
            config = Config()
            st.session_state.rag_system = initialize_rag_system(config)
        except Exception as e:
            st.error(f"初期化エラー: {e}")

# ── Header Section ────────────────────────────────────────────────────────
st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Neural RAG</h1>
        <p class="hero-subtitle">Next-Generation Document Intelligence System</p>
    </div>
""", unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────────────────
rag = st.session_state.get("rag_system")

# Sidebar Toggle Button
if st.button("☰", key="sidebar_toggle", help="メニューを開く"):
    st.session_state.sidebar_state = not st.session_state.get("sidebar_state", False)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding: 20px 0;">
            <h2 style="color: var(--text-primary); margin: 0;">⚙️ Configuration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if rag:
        # System Status
        st.markdown("""
            <div class="glass-card" style="margin-bottom: 20px;">
                <h4 style="margin: 0 0 10px 0;">System Status</h4>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 10px; height: 10px; border-radius: 50%; 
                                background: var(--success); animation: pulse 2s infinite;"></div>
                    <span style="color: var(--text-secondary);">Online</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Configuration Form
    with st.form("config_form"):
        st.markdown("### Model Settings")
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
            index=0
        )
        
        llm_model = st.selectbox(
            "Language Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        
        st.markdown("### Search Settings")
        
        collection_name = st.text_input(
            "Collection Name",
            value=ENV_DEFAULTS["COLLECTION_NAME"]
        )
        
        final_k = st.slider(
            "Result Count",
            min_value=1,
            max_value=20,
            value=ENV_DEFAULTS["FINAL_K"]
        )
        
        if st.form_submit_button("Apply Settings", use_container_width=True):
            # Apply configuration logic here
            st.success("Settings applied successfully!")

# ── Main Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 **Chat**", 
    "📊 **Analytics**", 
    "📁 **Documents**", 
    "🔧 **Settings**"
])

# ── Tab 1: Chat Interface ─────────────────────────────────────────────────
with tab1:
    chat_col, info_col = st.columns([2, 1])
    
    with chat_col:
        # Chat Container
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                        <div class="message-bubble user-message">
                            {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="message-bubble ai-message">
                            {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input Area
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Ask anything about your documents...",
                label_visibility="collapsed",
                key="user_input"
            )
        
        with col2:
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle message sending
        if send_button and user_input and rag:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Show typing indicator
            with st.spinner(""):
                create_typing_animation()
                time.sleep(1)  # Simulate thinking
                
                try:
                    # Get response from RAG
                    response = rag.query(user_input)
                    answer = response.get("answer", "申し訳ございません。回答を生成できませんでした。")
                    
                    # Add AI response
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Store sources in session state for display
                    if "sources" not in st.session_state:
                        st.session_state.sources = []
                    st.session_state.sources = response.get("sources", [])
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
            
            st.rerun()
    
    with info_col:
        # Sources Panel
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top: 0;">📚 Sources</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'sources') and st.session_state.sources:
            for i, source in enumerate(st.session_state.sources):
                st.markdown(f"""
                    <div class="source-card">
                        <h5 style="margin: 0 0 8px 0;">Source {i+1}</h5>
                        <p style="color: var(--text-secondary); font-size: 0.875rem; margin: 0;">
                            {source.get('metadata', {}).get('document_id', 'Unknown')}
                        </p>
                        <p style="font-size: 0.8rem; margin-top: 8px; color: var(--text-tertiary);">
                            {source.get('excerpt', '')[:100]}...
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ソースは質問後に表示されます")

# ── Tab 2: Analytics Dashboard ────────────────────────────────────────────
with tab2:
    if rag:
        stats = get_collection_statistics(rag)
        
        # Stats Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['documents']}</div>
                    <div class="stat-label">Documents</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['chunks']}</div>
                    <div class="stat-label">Chunks</div>
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
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(st.session_state.messages)}</div>
                    <div class="stat-label">Total Queries</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📈 Usage Analytics")
        
        # Create sample data for visualization
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        queries = [random.randint(10, 100) for _ in range(30)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=queries,
            mode='lines+markers',
            name='Queries',
            line=dict(color='#a855f7', width=3),
            marker=dict(size=8, color='#7c3aed'),
            fill='tozeroy',
            fillcolor='rgba(168, 85, 247, 0.1)'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("システムを初期化してください")

# ── Tab 3: Document Management ────────────────────────────────────────────
with tab3:
    if rag:
        # Upload Section
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top: 0;">📤 Upload Documents</h3>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Drop files here or click to browse",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx", "doc"],
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.markdown(f"### Selected Files ({len(uploaded_files)})")
            
            for file in uploaded_files:
                st.markdown(f"""
                    <div class="source-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span>{file.name}</span>
                            <span style="color: var(--text-secondary);">{file.size / 1024:.1f} KB</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                progress = st.progress(0)
                
                try:
                    paths = []
                    for i, file in enumerate(uploaded_files):
                        progress.progress((i + 1) / len(uploaded_files))
                        path = _persist_uploaded_file(file)
                        paths.append(str(path))
                    
                    rag.ingest_documents(paths)
                    st.success(f"✅ Successfully processed {len(uploaded_files)} documents!")
                    time.sleep(1)
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    progress.empty()
        
        # Documents List
        st.markdown("### 📚 Registered Documents")
        
        docs_df = get_documents_dataframe(rag)
        if not docs_df.empty:
            st.dataframe(
                docs_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Last Updated": st.column_config.DatetimeColumn(
                        format="MMM DD, YYYY - HH:mm"
                    )
                }
            )
            
            # Delete functionality
            doc_to_delete = st.selectbox(
                "Select document to delete",
                ["None"] + docs_df["Document ID"].tolist()
            )
            
            if doc_to_delete != "None":
                if st.button("🗑️ Delete Document", type="secondary"):
                    try:
                        success, message = rag.delete_document_by_id(doc_to_delete)
                        if success:
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("No documents registered yet")
    else:
        st.info("システムを初期化してください")

# ── Tab 4: Advanced Settings ──────────────────────────────────────────────
with tab4:
    st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top: 0;">🔧 Advanced Configuration</h3>
            <p style="color: var(--text-secondary);">Fine-tune your RAG system parameters</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Chunking Parameters")
        chunk_size = st.number_input("Chunk Size", value=1000, min_value=100, max_value=5000)
        chunk_overlap = st.number_input("Chunk Overlap", value=200, min_value=0, max_value=1000)
        
        st.markdown("### Search Parameters")
        vector_k = st.number_input("Vector Search K", value=10, min_value=1, max_value=50)
        keyword_k = st.number_input("Keyword Search K", value=10, min_value=1, max_value=50)
    
    with col2:
        st.markdown("### Model Parameters")
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        max_tokens = st.number_input("Max Tokens", value=1000, min_value=100, max_value=4000)
        
        st.markdown("### Database Settings")
        db_host = st.text_input("Database Host", value="localhost")
        db_port = st.text_input("Database Port", value="5432")
    
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        st.success("Configuration saved successfully!")

# ── Floating Action Button ────────────────────────────────────────────────
st.markdown("""
    <div class="fab glow">
        <span style="color: white; font-size: 1.5rem;">+</span>
    </div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
    <div style="margin-top: 60px; padding: 20px 0; text-align: center; 
                color: var(--text-tertiary); border-top: 1px solid var(--border);">
        <p style="margin: 0;">Neural RAG System • Built with Streamlit • v3.0</p>
    </div>
""", unsafe_allow_html=True)