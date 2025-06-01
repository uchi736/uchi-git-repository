"""streamlit_rag_ui.py – GUI for RAG (no DB UI)
================================================
* .env に DB_* / OPENAI_API_KEY を設定しておけば自動初期化。
* サイドバーでは **モデル／検索設定** のみ編集可能
  - コレクション名 (`COLLECTION_NAME`)
  - 取得チャンク数 (`FINAL_K`)
  - Embedding / LLM モデル名
* QA タブに「🗑️ クリア」ボタンを追加（widget 再生成でエラーにならないよう修正）
* モニタリングタブで現在のコレクション統計（ドキュメント数 / チャンク数）を表示

起動::  streamlit run streamlit_rag_ui.py
"""
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ── .env ───────────────────────────────────────────────────────────────────────
load_dotenv()
ENV_DEFAULTS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
    "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4o"),
    "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "documents"),
    "FINAL_K": int(os.getenv("FINAL_K", 5)),
}

# ── RAG import ────────────────────────────────────────────────────────────────
try:
    from rag_system import Config, RAGSystem  # type: ignore
except ModuleNotFoundError:
    st.error("rag_system.py が見つかりません。プロジェクト直下に配置してください。")
    st.stop()

st.set_page_config(page_title="RAG System UI", layout="wide")

# ── helper ────────────────────────────────────────────────────────────────────

def _persist(uploaded) -> Path:  # type: ignore
    tmp = Path(tempfile.gettempdir()) / "rag_uploads"
    tmp.mkdir(exist_ok=True)
    p = tmp / uploaded.name
    with open(p, "wb") as f:
        f.write(uploaded.getbuffer())
    return p


@st.cache_resource(show_spinner=False)
def _init_rag(cfg: Config) -> RAGSystem:
    return RAGSystem(cfg)


def _collection_stats(rag: RAGSystem) -> dict[str, int]:
    """Return simple stats for monitoring."""
    try:
        eng = create_engine(rag.connection_string)
        with eng.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT count(*) AS chunks,
                           count(DISTINCT document_id) AS documents
                    FROM document_chunks;
                    """
                )
            ).first()
            return {"documents": int(row.documents), "chunks": int(row.chunks)}  # type: ignore[attr-defined]
    except Exception:
        return {"documents": 0, "chunks": 0}


# ── auto init from .env ───────────────────────────────────────────────────────
if "rag" not in st.session_state and ENV_DEFAULTS["OPENAI_API_KEY"]:
    try:
        st.session_state["rag"] = _init_rag(Config())
        st.toast("✅ .env から自動初期化", icon="🎉")
    except Exception as e:
        st.warning(f"自動初期化失敗: {e}")

rag: Optional[RAGSystem] = st.session_state.get("rag")

# ╭──────────────────────── Sidebar ─────────────────────────╮
st.sidebar.title("⚙️ 設定")
with st.sidebar.form("settings_form"):
    st.subheader("モデル")
    embed_model = st.text_input(
        "Embedding Model",
        value=st.session_state.get("EMBEDDING_MODEL", ENV_DEFAULTS["EMBEDDING_MODEL"]),
    )
    llm_model = st.text_input(
        "LLM Model",
        value=st.session_state.get("LLM_MODEL", ENV_DEFAULTS["LLM_MODEL"]),
    )

    st.subheader("検索設定")
    collection_name = st.text_input(
        "コレクション名",
        value=st.session_state.get("COLLECTION_NAME", ENV_DEFAULTS["COLLECTION_NAME"]),
    )
    final_k = st.number_input(
        "LLM に渡すチャンク数",
        min_value=1,
        max_value=50,
        value=st.session_state.get("FINAL_K", ENV_DEFAULTS["FINAL_K"]),
    )

    submitted = st.form_submit_button("🔄 再初期化")

if submitted:
    st.session_state.update(
        EMBEDDING_MODEL=embed_model,
        LLM_MODEL=llm_model,
        COLLECTION_NAME=collection_name,
        FINAL_K=final_k,
    )
    cfg = Config(
        embedding_model=embed_model,
        llm_model=llm_model,
        collection_name=collection_name,
        final_k=int(final_k),
    )
    try:
        with st.spinner("RAG を再初期化中…"):
            st.session_state["rag"] = _init_rag(cfg)
        rag = st.session_state["rag"]
        st.success("再初期化完了")
    except Exception as e:
        st.error(f"再初期化失敗: {e}")
        st.exception(e)

# ╭──────────────────── Tabs ─────────────────────╮
t_ingest, t_qa, t_monitor = st.tabs(["📂 取り込み", "💬 QA", "📈 モニタ"])

# ── 取り込み ───────────────────────────────────────────
with t_ingest:
    st.header("📂 ドキュメント取り込み")
    if rag is None:
        st.info(".env が未設定、または再初期化が必要です。サイドバーから設定してください。")
    else:
        files = st.file_uploader(
            "ファイルをアップロード",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "doc", "docx"],
        )
        if st.button("🚀 取り込む", disabled=not files):
            with st.spinner("Embedding & インデックス登録中…"):
                paths = [_persist(f) for f in files]  # type: ignore[arg-type]
                rag.ingest_documents([str(p) for p in paths])
            st.success("取り込み完了")

# ── QA ────────────────────────────────────────────────
with t_qa:
    st.header("💬 質問応答 (RAG)")
    if rag is None:
        st.info("RAG が未初期化です。サイドバーで設定してください。")
    else:
        # --- セッション state 初期化 / クリアフラグ処理 -----------------------
        if st.session_state.get("clear_request"):
            st.session_state.pop("qa_input", None)
            st.session_state.pop("qa_answer", None)
            st.session_state["clear_request"] = False

        question_val = st.session_state.get("qa_input", "")

        # -------------------- UI ---------------------------------------------
        question = st.text_area("質問", value=question_val, height=100, key="qa_input")
        col_run, col_clear = st.columns([1, 1])
        with col_run:
            run_clicked = st.button("🧠 回答を生成", disabled=not question.strip())
        with col_clear:
            clear_clicked = st.button("🗑️ クリア")

        # -------------------- ボタン処理 --------------------------------------
        if run_clicked:
            with st.spinner("推論中…"):
                res = rag.query(question)
            st.session_state["qa_answer"] = res

        if clear_clicked:
            # フラグを立てて次回レンダリング時に State をリセット
            st.session_state["clear_request"] = True
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

        # -------------------- 回答表示 ---------------------------------------
        ans = st.session_state.get("qa_answer")
        if isinstance(ans, dict) and ans:
            st.markdown("#### 📝 回答")
            st.write(ans["answer"])

            st.markdown("---")
            st.markdown("#### 🔍 参照ソース")
            if ans["sources"]:
                for i, s in enumerate(ans["sources"], 1):
                    cid = s["metadata"].get("chunk_id", "N/A")
                    st.markdown(f"**{i}.** `{cid}`")
                    st.write(s["content"])
                    with st.expander("メタデータ"):
                        st.code(json.dumps(s["metadata"], ensure_ascii=False, indent=2), language="json")
            else:
                st.info("ソースなし")

# ── モニタリング ───────────────────────────────────────────
with t_monitor:
    st.header("📈 モニタリング / デバッグ")
    if rag:
        # --- 集計メトリクス --------------------------------------------------
        stats = _collection_stats(rag)
        st.metric("登録ドキュメント数", stats["documents"])
        st.metric("登録チャンク数", stats["chunks"])

         # --- ドキュメント一覧 ----------------------------------------------
        try:
            eng = create_engine(rag.connection_string)
            sql = text(
                """
                SELECT collection_name                 AS "コレクション",
                       document_id                     AS "ドキュメントID",
                       COUNT(*)                        AS "チャンク数",
                       MAX(created_at)                 AS "最終登録"
                FROM document_chunks
                GROUP BY collection_name, document_id
                ORDER BY "最終登録" DESC;
                """
            )
            with eng.connect() as conn:
                rows = [dict(row._mapping) for row in conn.execute(sql)]  # ← ここがポイント
            if rows:
                st.markdown("### 📄 登録ドキュメント一覧")
                st.table(rows)  # list[dict] は直接表示可能
            else:
                st.info("ドキュメントがまだ登録されていません。")
        except Exception as e:
            st.warning(f"統計取得に失敗: {e}")