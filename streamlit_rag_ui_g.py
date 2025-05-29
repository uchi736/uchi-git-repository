"""streamlit_rag_ui.py – GUI for RAG
================================================
* .env に DB_* / OPENAI_API_KEY を設定しておけば自動初期化。
* サイドバーではモデル／検索設定の編集、RAGシステムの再初期化が可能。
* QA タブで質問応答を実行。参照ソースの全文表示機能を追加。
* モニタリングタブで統計情報、登録ドキュメント一覧の表示、およびドキュメント削除機能を提供。

起動:: streamlit run streamlit_rag_ui.py
"""
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

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
    from rag_system import Config, RAGSystem 
except ModuleNotFoundError:
    st.error("rag_system.py が見つかりません。プロジェクトのルートディレクトリに配置してください。")
    st.stop()

st.set_page_config(page_title="RAG System UI", layout="wide", initial_sidebar_state="expanded")

# ── Helper Functions ──────────────────────────────────────────────────────────

def _persist_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    if uploaded_file is None:
        raise ValueError("Uploaded file cannot be None for persistence.")
    tmp_upload_dir = Path(tempfile.gettempdir()) / "rag_streamlit_uploads"
    tmp_upload_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = tmp_upload_dir / uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_file_path

@st.cache_resource(show_spinner="RAGシステムを初期化中...")
def initialize_rag_system(config_obj: Config) -> RAGSystem:
    return RAGSystem(config_obj)

def get_collection_statistics(rag_sys: RAGSystem) -> Dict[str, Any]:
    if not rag_sys:
        return {"documents": 0, "chunks": 0, "collection_name": "N/A"}
    try:
        engine = create_engine(rag_sys.connection_string)
        with engine.connect() as conn:
            query = text(
                """
                SELECT COUNT(DISTINCT document_id) AS num_documents,
                       COUNT(*) AS num_chunks
                FROM document_chunks
                WHERE collection_name = :current_collection;
                """
            )
            result = conn.execute(query, {"current_collection": rag_sys.config.collection_name}).first()
            return {
                "documents": int(result.num_documents) if result else 0,
                "chunks": int(result.num_chunks) if result else 0,
                "collection_name": rag_sys.config.collection_name
            }
    except Exception as e:
        st.warning(f"コレクション統計の取得に失敗しました (Collection: {rag_sys.config.collection_name}): {e}")
        return {"documents": 0, "chunks": 0, "collection_name": rag_sys.config.collection_name}

def get_registered_documents_df(rag_sys: RAGSystem) -> pd.DataFrame:
    if not rag_sys:
        return pd.DataFrame()
    try:
        engine = create_engine(rag_sys.connection_string)
        with engine.connect() as conn:
            query = text(
                """
                SELECT document_id AS "ドキュメントID",
                       COUNT(*) AS "チャンク数",
                       MAX(created_at) AS "最終登録日時"
                FROM document_chunks
                WHERE collection_name = :current_collection
                GROUP BY document_id
                ORDER BY "最終登録日時" DESC, "ドキュメントID" ASC;
                """
            )
            rows = conn.execute(query, {"current_collection": rag_sys.config.collection_name}).fetchall()
        if rows:
            return pd.DataFrame(rows, columns=["ドキュメントID", "チャンク数", "最終登録日時"])
        return pd.DataFrame() 
    except Exception as e:
        st.warning(f"登録ドキュメント一覧の取得に失敗しました (Collection: {rag_sys.config.collection_name}): {e}")
        return pd.DataFrame()

# ── Session State Initialization & RAG System Setup ───────────────────────────
if "rag_system_instance" not in st.session_state and ENV_DEFAULTS["OPENAI_API_KEY"]:
    try:
        initial_config = Config() 
        st.session_state["rag_system_instance"] = initialize_rag_system(initial_config)
        st.toast("✅ RAGシステムが .env 設定に基づいて初期化されました。", icon="✅") 
    except Exception as e:
        st.error(f"RAGシステムの自動初期化中にエラーが発生しました: {e}")
        st.exception(e)

rag: Optional[RAGSystem] = st.session_state.get("rag_system_instance")

# ╭──────────────────────── Sidebar: Settings & Re-initialization ─────────────────────────╮
with st.sidebar:
    st.title("⚙️ RAGシステム設定")
    st.markdown("---") # Divider
    with st.form("rag_settings_form"):
        st.subheader("🤖 OpenAIモデル設定")
        current_embed_model = rag.config.embedding_model if rag else ENV_DEFAULTS["EMBEDDING_MODEL"]
        current_llm_model = rag.config.llm_model if rag else ENV_DEFAULTS["LLM_MODEL"]
        
        embed_model_input = st.text_input(
            "Embeddingモデル名", value=current_embed_model,
            help="例: text-embedding-ada-002, text-embedding-3-small"
        )
        llm_model_input = st.text_input(
            "LLMモデル名", value=current_llm_model,
            help="例: gpt-4o, gpt-3.5-turbo"
        )
        st.divider() # Visual separation
        st.subheader("📚 検索とコレクション設定")
        current_collection_name = rag.config.collection_name if rag else ENV_DEFAULTS["COLLECTION_NAME"]
        current_final_k = rag.config.final_k if rag else ENV_DEFAULTS["FINAL_K"]

        collection_name_input = st.text_input(
            "コレクション名", value=current_collection_name,
            help="ベクトルストア内でドキュメントをグループ化するための名前です。"
        )
        final_k_input = st.number_input(
            "LLMに渡す最終チャンク数 (Final K)", min_value=1, max_value=50, step=1,
            value=current_final_k,
            help="検索結果から絞り込み、最終的にLLMのコンテキストとして使用されるチャンクの数です。"
        )
        st.caption(f"注意: OpenAI APIキーは `.env` ファイルから読み込まれます。")
        apply_settings_button = st.form_submit_button("🔄 設定を適用し再初期化する", type="primary", use_container_width=True)

if apply_settings_button: # This logic remains outside the `with st.sidebar:` block
    updated_config = Config(
        openai_api_key=ENV_DEFAULTS["OPENAI_API_KEY"], 
        embedding_model=embed_model_input, llm_model=llm_model_input,
        collection_name=collection_name_input, final_k=int(final_k_input),
        db_host=os.getenv("DB_HOST", "localhost"), db_port=os.getenv("DB_PORT", "5432"),
        db_name=os.getenv("DB_NAME", "postgres"), db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "your-password"),
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)), chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        vector_search_k=int(os.getenv("VECTOR_SEARCH_K", 10)), keyword_search_k=int(os.getenv("KEYWORD_SEARCH_K", 10)),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", 1536)), fts_language=os.getenv("FTS_LANGUAGE", "english"),
    )
    try:
        with st.spinner("RAGシステムを新しい設定で再初期化しています..."):
            if "rag_system_instance" in st.session_state:
                del st.session_state["rag_system_instance"] 
            st.session_state["rag_system_instance"] = initialize_rag_system(updated_config)
            rag = st.session_state["rag_system_instance"] 
        st.success(f"RAGシステムが正常に再初期化されました (コレクション: '{rag.config.collection_name}')。")
        st.rerun() 
    except Exception as e:
        st.error(f"RAGシステムの再初期化中にエラーが発生しました: {e}")
        st.exception(e)

# ╭──────────────────── Tabs: Ingest, QA, Monitor ─────────────────────╮
tab_ingest, tab_qa, tab_monitor = st.tabs(["📁 **ドキュメント取り込み**", "💬 **QA (質問応答)**", "📈 **モニタリング**"])

# ── Tab: Document Ingestion ───────────────────────────────────────────
with tab_ingest:
    st.header("📁 ドキュメントの取り込み")
    if rag:
        st.caption(f"現在の作業コレクション: `{rag.config.collection_name}`")
    else:
        st.warning("RAGシステムが初期化されていません。")
    st.divider()

    if rag:
        with st.container(border=True):
            st.subheader("アップロードと取り込み処理")
            uploaded_files_list: List[st.runtime.uploaded_file_manager.UploadedFile] | None = st.file_uploader(
                "処理するファイルを選択 (PDF, TXT, MD, DOCX, DOC):",
                accept_multiple_files=True, type=["pdf", "txt", "md", "docx", "doc"], 
                key="file_uploader_ingest" 
            )
            
            if st.button("🚀 選択したファイルを取り込む", disabled=not uploaded_files_list, key="ingest_button_main", type="primary", use_container_width=True):
                if uploaded_files_list:
                    with st.spinner("ファイルを処理し、インデックスを構築しています... (時間がかかる場合があります)"):
                        persisted_file_paths_str: List[str] = []
                        for up_file_item in uploaded_files_list:
                            if up_file_item is not None:
                                try:
                                    persisted_path = _persist_uploaded_file(up_file_item)
                                    persisted_file_paths_str.append(str(persisted_path))
                                except Exception as e_persist:
                                    st.error(f"ファイル '{up_file_item.name}' の一時保存中にエラー: {e_persist}")
                        
                        if persisted_file_paths_str:
                            try:
                                rag.ingest_documents(persisted_file_paths_str)
                                st.success(f"{len(persisted_file_paths_str)} 件のファイルの取り込み処理が完了しました。コレクション '{rag.config.collection_name}' が更新されました。")
                            except Exception as e_ingest:
                                st.error(f"ドキュメント取り込み処理中にエラーが発生しました: {e_ingest}")
                                st.exception(e_ingest)
                        elif uploaded_files_list: 
                             st.warning("ファイルは選択されましたが、処理できるファイルがありませんでした。ファイル形式や内容を確認してください。")
    else:
        st.info("RAGシステムを初期化するために、サイドバーで設定を確認し、OpenAI APIキーを `.env` ファイルに設定してください。")


# ── Tab: Question Answering (QA) ──────────────────────────────────────
with tab_qa:
    st.header("💬 QA (質問応答)")
    if rag:
        st.caption(f"現在の作業コレクション: `{rag.config.collection_name}`")
    else:
        st.warning("RAGシステムが初期化されていません。")
    st.divider()

    if rag:
        # Initialize session state for QA tab and full text dialog
        if "qa_question_text" not in st.session_state: st.session_state.qa_question_text = ""
        if "qa_api_response" not in st.session_state: st.session_state.qa_api_response = None
        if "qa_clear_fields_requested" not in st.session_state: st.session_state.qa_clear_fields_requested = False
        if "show_full_text_dialog_flag" not in st.session_state: st.session_state.show_full_text_dialog_flag = False
        if "dialog_content_to_show" not in st.session_state: st.session_state.dialog_content_to_show = ""
        if "dialog_title_to_show" not in st.session_state: st.session_state.dialog_title_to_show = ""

        if st.session_state.qa_clear_fields_requested:
            st.session_state.qa_question_text = ""
            st.session_state.qa_api_response = None
            st.session_state.qa_clear_fields_requested = False
            st.session_state.show_full_text_dialog_flag = False 
            st.session_state.dialog_content_to_show = ""
            st.session_state.dialog_title_to_show = ""

        with st.container(border=True):
            st.subheader("質問入力")
            current_question = st.session_state.qa_question_text
            question_input = st.text_area(
                "質問を入力してください:", value=current_question, height=100, key="qa_question_input_area" 
            )
            if question_input != current_question:
                st.session_state.qa_question_text = question_input

            col_qa_run, col_qa_clear = st.columns([0.7, 0.3]) # Adjusted ratio
            with col_qa_run:
                run_qa_button_clicked = st.button(
                    "� 回答を生成する", disabled=not st.session_state.qa_question_text.strip(), 
                    key="qa_run_button", type="primary", use_container_width=True
                )
            with col_qa_clear:
                clear_qa_button_clicked = st.button("🗑️ クリア", key="qa_clear_button", use_container_width=True)

        if run_qa_button_clicked and st.session_state.qa_question_text.strip():
            with st.spinner("関連情報を検索し、AIが回答を生成しています..."):
                try:
                    api_response = rag.query(st.session_state.qa_question_text)
                    st.session_state.qa_api_response = api_response
                except Exception as e_query_ui:
                    st.error(f"質問応答処理中にエラーが発生しました: {e_query_ui}")
                    st.exception(e_query_ui)
                    st.session_state.qa_api_response = None 

        if clear_qa_button_clicked:
            st.session_state.qa_clear_fields_requested = True
            st.rerun() 

        if st.session_state.qa_api_response:
            response_content = st.session_state.qa_api_response
            st.divider()
            with st.container(border=True):
                st.subheader("📝 生成された回答")
                st.info(response_content.get("answer", "回答の取得に失敗しました。"))

            if response_content.get("sources"):
                st.subheader("🔍 主な参照ソース")
                for i, src_item in enumerate(response_content["sources"], 1):
                    with st.container(border=True): # Card-like display for each source
                        src_doc_id = src_item.get("metadata", {}).get("document_id", "N/A")
                        src_chunk_id = src_item.get("metadata", {}).get("chunk_id", "N/A")
                        
                        st.markdown(f"**ソース {i}** | ドキュメント: `{src_doc_id}` | チャンク: `{src_chunk_id}`")
                        st.caption("抜粋:")
                        st.markdown(f"<div style='max-height: 100px; overflow-y: auto; padding: 5px; border: 1px solid #eee; border-radius: 5px;'>{src_item.get('excerpt', '抜粋なし')}</div>", unsafe_allow_html=True)
                        
                        if src_item.get("full_content"):
                            button_key = f"show_full_text_btn_{src_chunk_id}_{i}" # Unique key
                            if st.button("📄 全文表示", key=button_key, help="このチャンクの全文をポップアップで表示します。", type="secondary", use_container_width=True):
                                st.session_state.dialog_title_to_show = f"チャンク全文: {src_chunk_id} (ドキュメント: {src_doc_id})"
                                st.session_state.dialog_content_to_show = src_item.get("full_content")
                                st.session_state.show_full_text_dialog_flag = True
                                st.rerun() 
                        else:
                            st.caption("全文データなし")
                        st.write("") # Adds a bit of vertical space inside the container
                st.markdown("---") 
            else: 
                st.markdown("参照されたソース情報はありませんでした。")
            
            if response_content.get("usage"):
                with st.expander("利用トークン数 (OpenAI API)"):
                    st.json(response_content["usage"])
        
        # Handle dialog display (must be outside the loop)
        if st.session_state.get("show_full_text_dialog_flag", False):
            dialog_title = st.session_state.get("dialog_title_to_show", "チャンク全文")
            dialog_content_md = st.session_state.get("dialog_content_to_show", "")
            dialog_rendered_successfully = False

            try:
                with st.dialog(title=dialog_title, width="large"):
                    st.markdown(f"<div style='max-height: 60vh; overflow-y: auto;'>{dialog_content_md}</div>", unsafe_allow_html=True)
                    if st.button("閉じる", key="close_stable_dialog_btn_qa", use_container_width=True):
                        st.session_state.show_full_text_dialog_flag = False
                        st.rerun()
                dialog_rendered_successfully = True
            except TypeError as e_stable:
                if "'function' object does not support the context manager protocol" in str(e_stable).lower():
                    st.warning("`st.dialog` がコンテキストマネージャとして機能しませんでした。`st.experimental_dialog` を試みます。Streamlitのバージョンアップをご検討ください。")
                    if hasattr(st, "experimental_dialog"):
                        try:
                            with st.experimental_dialog(title=dialog_title, width="large"): 
                                st.markdown(f"<div style='max-height: 60vh; overflow-y: auto;'>{dialog_content_md}</div>", unsafe_allow_html=True)
                                if st.button("閉じる", key="close_experimental_dialog_btn_qa", use_container_width=True):
                                    st.session_state.show_full_text_dialog_flag = False
                                    st.rerun()
                            dialog_rendered_successfully = True
                        except TypeError as e_experimental:
                            if "'function' object does not support the context manager protocol" in str(e_experimental).lower():
                                st.error("`st.experimental_dialog` もコンテキストマネージャとして機能しませんでした。")
                            else:
                                st.error(f"`st.experimental_dialog` で予期せぬTypeErrorが発生しました: {e_experimental}")
                        except Exception as e_exp_other:
                             st.error(f"`st.experimental_dialog` で予期せぬエラーが発生しました: {e_exp_other}")
                    else:
                        st.error("`st.experimental_dialog` が見つかりませんでした。")
                else:
                    st.error(f"`st.dialog` で予期せぬTypeErrorが発生しました: {e_stable}")
            except Exception as general_error:
                st.error(f"ダイアログ表示中に予期せぬ一般エラーが発生しました: {general_error}")

            if not dialog_rendered_successfully and st.session_state.get("show_full_text_dialog_flag", False):
                st.error("ポップアップダイアログの表示に失敗したため、コンテンツをページに直接表示します。Streamlitのバージョンが古いか、サポートされていない環境の可能性があります。")
                with st.container(border=True):
                    st.subheader(dialog_title)
                    st.markdown(f"<div style='max-height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ccc;'>{dialog_content_md}</div>", unsafe_allow_html=True)
                    if st.button("コンテンツを隠す", key="hide_fallback_content_btn_qa_final", use_container_width=True):
                        st.session_state.show_full_text_dialog_flag = False
                        st.rerun()
    else:
        st.info("RAGシステムを初期化するために、サイドバーで設定を確認し、OpenAI APIキーを `.env` ファイルに設定してください。")


# ── Tab: Monitoring & Document Management ───────────────────────────────────
with tab_monitor:
    st.header("📈 モニタリングとドキュメント管理")
    if rag:
        st.caption(f"現在の作業コレクション: `{rag.config.collection_name}`")
    else:
        st.warning("RAGシステムが初期化されていません。")
    st.divider()
    
    if rag:
        with st.container(border=True):
            st.subheader("📊 コレクション統計")
            collection_stats = get_collection_statistics(rag)
            col_monitor_stats_docs, col_monitor_stats_chunks = st.columns(2)
            with col_monitor_stats_docs:
                st.metric("登録ドキュメント総数", collection_stats["documents"])
            with col_monitor_stats_chunks:
                st.metric("登録チャンク総数", collection_stats["chunks"])
        
        st.divider()
        with st.container(border=True):
            st.subheader(f"📄 「{rag.config.collection_name}」コレクション内の登録ドキュメント一覧")
            registered_docs_df = get_registered_documents_df(rag)
            if not registered_docs_df.empty:
                st.dataframe(registered_docs_df, hide_index=True, use_container_width=True)
            else: 
                st.info(f"コレクション「{rag.config.collection_name}」には、現在登録されているドキュメントはありません。")
        
        if not registered_docs_df.empty: # Only show delete section if there are documents
            st.divider()
            with st.container(border=True):
                st.subheader("🗑️ 登録ドキュメントの削除")
                st.caption(f"注意: ここでの削除操作はコレクション「{rag.config.collection_name}」内のドキュメントにのみ影響します。")

                doc_ids_options: List[str] = ["削除するドキュメントを選択..."] + \
                                          sorted([str(doc_id) for doc_id in registered_docs_df["ドキュメントID"].unique() if doc_id is not None])
                
                CONFIRM_DELETE_KEY_MONITOR = f"confirm_delete_doc_id_monitor_{rag.config.collection_name}"
                SELECTED_DOC_KEY_MONITOR = f"selectbox_delete_doc_id_monitor_{rag.config.collection_name}"

                if CONFIRM_DELETE_KEY_MONITOR not in st.session_state: st.session_state[CONFIRM_DELETE_KEY_MONITOR] = None
                if SELECTED_DOC_KEY_MONITOR not in st.session_state: st.session_state[SELECTED_DOC_KEY_MONITOR] = doc_ids_options[0]

                selected_doc_to_delete = st.selectbox(
                    "削除対象のドキュメントID:", options=doc_ids_options,
                    index=doc_ids_options.index(st.session_state[SELECTED_DOC_KEY_MONITOR]) if st.session_state[SELECTED_DOC_KEY_MONITOR] in doc_ids_options else 0,
                    key=f"actual_selectbox_for_{SELECTED_DOC_KEY_MONITOR}" 
                )
                if selected_doc_to_delete != st.session_state[SELECTED_DOC_KEY_MONITOR]:
                    st.session_state[SELECTED_DOC_KEY_MONITOR] = selected_doc_to_delete
                    st.session_state[CONFIRM_DELETE_KEY_MONITOR] = None 
                    st.rerun()

                if selected_doc_to_delete != "削除するドキュメントを選択...":
                    st.warning(
                        f"**警告:** ドキュメント「{selected_doc_to_delete}」に関連する全てのチャンクデータが、"
                        f"コレクション「{rag.config.collection_name}」から完全に削除されます。"
                        "この操作は元に戻すことができません。"
                    )
                    delete_button_unique_key = f"delete_button_for_{selected_doc_to_delete.replace('.', '_')}_{rag.config.collection_name}"
                    if st.button(f"ドキュメント「{selected_doc_to_delete}」を完全に削除する", type="primary", key=delete_button_unique_key, use_container_width=True): 
                        if st.session_state[CONFIRM_DELETE_KEY_MONITOR] == selected_doc_to_delete:
                            try:
                                with st.spinner(f"ドキュメント「{selected_doc_to_delete}」をコレクション「{rag.config.collection_name}」から削除しています..."):
                                    success, message = rag.delete_document_by_id(selected_doc_to_delete)
                                if success:
                                    st.success(f"ドキュメント「{selected_doc_to_delete}」の削除処理が完了しました。メッセージ: {message}")
                                else:
                                    st.error(f"ドキュメント「{selected_doc_to_delete}」の削除中にエラーが発生しました。メッセージ: {message}")
                                st.session_state[CONFIRM_DELETE_KEY_MONITOR] = None 
                                st.session_state[SELECTED_DOC_KEY_MONITOR] = "削除するドキュメントを選択..."
                                st.rerun() 
                            except Exception as e_del_final:
                                st.error(f"ドキュメント「{selected_doc_to_delete}」の削除処理中に予期せぬUIエラーが発生しました: {e_del_final}")
                                st.exception(e_del_final)
                                st.session_state[CONFIRM_DELETE_KEY_MONITOR] = None 
                        else:
                            st.session_state[CONFIRM_DELETE_KEY_MONITOR] = selected_doc_to_delete
                            st.error(
                                f"**削除確認:** 本当にドキュメント「{selected_doc_to_delete}」をコレクション「{rag.config.collection_name}」から削除しますか？ "
                                "よろしければ、上記の「完全に削除する」ボタンをもう一度押してください。"
                            )
                            st.rerun() 
    else: 
        st.info("RAGシステムを初期化するために、サイドバーで設定を確認し、OpenAI APIキーを `.env` ファイルに設定してください。")