"""DocChat-AI Streamlit Application.

Interactive RAG document question-answering interface featuring multi-file uploads,
session-isolated vector storage, streamed answers with Gemini, daily quota handling,
in-memory caching, and cited source filtering.
"""

from collections import defaultdict
import html
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from pypdf import PdfReader
import streamlit as st

# Set up module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Page Configuration
st.set_page_config(
    page_title="DocChat-AI",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 2. Environment & Secrets Management
load_dotenv()
try:
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
        if secret_key:
            os.environ["GEMINI_API_KEY"] = secret_key
except Exception as secret_exc:
    logger.exception("Error loading GEMINI_API_KEY from st.secrets: %s", secret_exc)

# Import core modules and configs
from src.chunker import chunk_pages
from src.config import (
    MAX_CHUNKS_PER_SESSION,
    MAX_PAGES,
    MAX_QUESTIONS_PER_SESSION,
    MAX_UPLOAD_MB,
    check_and_increment_daily_llm_calls,
)
from src.loader import load_document
from src.rag import QuotaExceededError, clear_answer_cache, stream_answer
from src.vector_store import (
    add_chunks,
    cleanup_old_collections,
    delete_file,
    get_collection,
    list_files,
)

# Custom CSS for polished aesthetics
st.markdown(
    """
    <style>
    /* Styling for source cards */
    .source-card {
        background-color: rgba(125, 125, 125, 0.08);
        border-left: 3px solid #6366f1;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .source-meta {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6366f1;
        margin-bottom: 4px;
    }
    .score-badge {
        background-color: rgba(99, 102, 241, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.78rem;
    }
    .source-text {
        font-size: 0.88rem;
        line-height: 1.45;
    }
    /* Footer styling */
    .app-footer {
        text-align: center;
        margin-top: 45px;
        padding-top: 15px;
        border-top: 1px solid rgba(125, 125, 125, 0.15);
        font-size: 0.8rem;
        opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state() -> str:
    """Initialize session state variables including session isolation collection.

    Returns:
        The session collection name (e.g. 's_a1b2c3d4e5f6').
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

    session_collection = f"s_{st.session_state.session_id[:12]}"
    st.session_state.session_collection = session_collection

    if "cleaned_old_collections" not in st.session_state:
        try:
            cleaned = cleanup_old_collections(max_session_collections=20)
            if cleaned > 0:
                logger.info("Cleaned up %d old session collection(s).", cleaned)
        except Exception as clean_err:
            logger.exception("Error cleaning up old collections: %s", clean_err)
        st.session_state.cleaned_old_collections = True

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    return session_collection


def render_sources(sources: list[dict[str, Any]], uncited: bool = False) -> None:
    """Render citations grouped by document filename and page number.

    Uses uncited flag to choose between 'Retrieved passages' and 'Sources'.
    Passage text and filenames are HTML-escaped for safety.

    Args:
        sources: List of source dictionaries containing 'file', 'page', 'score', and 'text'.
        uncited: Boolean indicating whether citations were absent or invalid.
    """
    if not sources:
        return

    expander_label = "Retrieved passages" if uncited else "Sources"

    with st.expander(expander_label, expanded=False):
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for src in sources:
            file_name = src.get("file", "Unknown document")
            page_num = src.get("page", 1)
            grouped[(file_name, page_num)].append(src)

        for (file_name, page_num), chunks in grouped.items():
            escaped_file = html.escape(str(file_name))
            st.markdown(f"**📄 `{escaped_file}` — Page {page_num}**")
            for idx, chunk in enumerate(chunks, start=1):
                score = chunk.get("score", 0.0)
                passage = html.escape(str(chunk.get("text", "")).strip())
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-meta">
                            Passage {idx} &nbsp;•&nbsp;
                            <span class="score-badge">Similarity: {score:.4f}</span>
                        </div>
                        <div class="source-text">{passage}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_sidebar(session_collection: str) -> None:
    """Render sidebar controls including uploads, limits, demo loader, and file management."""
    with st.sidebar:
        st.title("📄 DocChat-AI")
        st.caption("Ask questions and get grounded answers cited directly from your documents.")

        # Check API key status
        if not os.getenv("GEMINI_API_KEY"):
            st.error("⚠️ `GEMINI_API_KEY` is not found. Please set it in `.env` or Streamlit secrets.")

        st.divider()

        # Demo Document & Examples
        st.subheader("💡 Demo & Quick Start")
        demo_path = Path("data/sample.pdf").resolve()
        if st.button("📥 Load demo document", use_container_width=True, help="Load data/sample.pdf into this session"):
            if not demo_path.exists():
                st.error("Demo document 'data/sample.pdf' was not found.")
            else:
                with st.spinner("Indexing demo document..."):
                    try:
                        pages = load_document(demo_path)
                        chunks = chunk_pages(pages)
                        cur_count = get_collection(session_collection).count()
                        if cur_count + len(chunks) > MAX_CHUNKS_PER_SESSION:
                            st.warning(
                                f"Cannot load demo document: adding {len(chunks)} chunks would exceed "
                                f"the limit of {MAX_CHUNKS_PER_SESSION} chunks."
                            )
                        else:
                            delete_file(demo_path.name, collection_name=session_collection)
                            add_chunks(chunks, collection_name=session_collection)
                            clear_answer_cache(collection_name=session_collection)
                            st.session_state.flash_message = (
                                f"Loaded demo document ({demo_path.name}) with {len(chunks)} chunks!"
                            )
                            st.rerun()
                    except Exception as demo_err:
                        logger.exception("Error loading demo document: %s", demo_err)
                        st.error(f"Failed to load demo document: {demo_err}")

        # Example question buttons
        indexed_files = list_files(collection_name=session_collection)
        has_documents = bool(indexed_files)

        st.caption("Example Questions:")
        example_questions = [
            "What is the primary purpose of DocChat-AI?",
            "What vector database does DocChat-AI use?",
            "What API is used for embeddings and answer generation?",
        ]
        for idx, ex_q in enumerate(example_questions, start=1):
            if st.button(f"❓ {ex_q}", key=f"ex_btn_{idx}", use_container_width=True, disabled=not has_documents):
                st.session_state.pending_question = ex_q
                st.rerun()

        st.divider()

        # Document uploader
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            f"Select PDF or TXT files (Max {MAX_UPLOAD_MB}MB per file)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help=f"Upload documents up to {MAX_UPLOAD_MB}MB and {MAX_PAGES} pages.",
        )

        # Process documents button
        if st.button("⚡ Process documents", use_container_width=True, disabled=not uploaded_files):
            with st.spinner("Parsing, chunking, and indexing documents..."):
                try:
                    total_chunks_added = 0
                    processed_files_count = 0

                    for file in uploaded_files:
                        filename = file.name

                        # 1. Check file size limit
                        file_size_mb = getattr(file, "size", 0) / (1024 * 1024)
                        if file_size_mb > MAX_UPLOAD_MB:
                            st.warning(
                                f"File '{filename}' ({file_size_mb:.1f} MB) exceeds the maximum "
                                f"allowed size of {MAX_UPLOAD_MB} MB."
                            )
                            continue

                        # 2. Check PDF page limit and scan for empty text layers
                        num_pdf_pages = None
                        if filename.lower().endswith(".pdf"):
                            try:
                                if hasattr(file, "seek"):
                                    file.seek(0)
                                reader = PdfReader(file)
                                num_pdf_pages = len(reader.pages)
                                if hasattr(file, "seek"):
                                    file.seek(0)
                                if num_pdf_pages > MAX_PAGES:
                                    st.warning(
                                        f"File '{filename}' exceeds the limit of {MAX_PAGES} pages "
                                        f"({num_pdf_pages} pages)."
                                    )
                                    continue
                            except Exception as pdf_err:
                                logger.exception("Failed reading PDF metadata for %s: %s", filename, pdf_err)

                        # 3. Load pages
                        pages = load_document(file)
                        if not pages:
                            st.warning(f"No extractable text found in '{filename}'.")
                            continue

                        # 4. Warn if pages have no text layer
                        if num_pdf_pages is not None:
                            extracted_count = len(pages)
                            empty_pages = num_pdf_pages - extracted_count
                            if empty_pages > 0:
                                st.warning(
                                    f"File '{filename}': {empty_pages} page(s) have no text layer "
                                    "(scanned images are not supported)."
                                )

                        # 5. Chunk pages
                        chunks = chunk_pages(pages)
                        if not chunks:
                            continue

                        # 6. Check session chunk limit
                        cur_count = get_collection(session_collection).count()
                        if cur_count + len(chunks) > MAX_CHUNKS_PER_SESSION:
                            st.warning(
                                f"Cannot add '{filename}': adding {len(chunks)} chunks would exceed "
                                f"the session limit of {MAX_CHUNKS_PER_SESSION} chunks (currently {cur_count})."
                            )
                            continue

                        # Delete any old chunks of the same filename in this session first
                        delete_file(filename, collection_name=session_collection)
                        add_chunks(chunks, collection_name=session_collection)
                        total_chunks_added += len(chunks)
                        processed_files_count += 1

                    if processed_files_count > 0:
                        clear_answer_cache(collection_name=session_collection)
                        st.session_state.flash_message = (
                            f"Processed {processed_files_count} file(s) successfully! "
                            f"Added {total_chunks_added} chunks to vector store."
                        )
                        st.rerun()
                    else:
                        st.warning("No readable text could be indexed from the selected files.")
                except Exception as exc:
                    logger.exception("Failed to process documents: %s", exc)
                    st.error("Failed to process documents. Please check file formatting and try again.")

        st.divider()

        # Indexed documents list with delete buttons
        st.subheader("📁 Indexed Documents")
        if indexed_files:
            for fname in indexed_files:
                col_name, col_del = st.columns([0.8, 0.2])
                with col_name:
                    st.text(f"• {fname}")
                with col_del:
                    if st.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                        try:
                            delete_file(fname, collection_name=session_collection)
                            clear_answer_cache(collection_name=session_collection)
                            st.session_state.flash_message = f"Deleted {fname}"
                            st.rerun()
                        except Exception as del_err:
                            logger.exception("Error deleting file %s: %s", fname, del_err)
                            st.error(f"Error deleting file: {del_err}")
        else:
            st.caption("No documents are currently indexed.")

        st.divider()

        # Clear chat button
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.question_count = 0
            clear_answer_cache(collection_name=session_collection)
            st.rerun()


def main() -> None:
    """Main application loop."""
    session_collection = initialize_session_state()

    # Display flash messages after rerun
    if st.session_state.get("flash_message"):
        flash_text = st.session_state.pop("flash_message")
        st.toast(flash_text, icon="✅")
        st.success(flash_text)

    render_sidebar(session_collection)

    st.subheader("💬 Document Q&A")

    # Check indexed files status
    indexed_files = list_files(collection_name=session_collection)
    has_documents = bool(indexed_files)

    if not has_documents:
        st.info(
            "👋 Welcome to **DocChat-AI**! No documents are currently indexed in this session.\n\n"
            "Please upload documents or click **'Load demo document'** in the sidebar to start asking questions."
        )

    # Render conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"], uncited=message.get("uncited", False))

    # Check question limit for session
    is_at_q_limit = st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION
    if is_at_q_limit:
        st.warning(
            f"You have reached the limit of {MAX_QUESTIONS_PER_SESSION} questions for this session. "
            "Please clear chat or start a new session to continue."
        )

    # Chat input
    user_input = st.chat_input(
        placeholder="Ask a question about your uploaded documents...",
        disabled=not has_documents or is_at_q_limit,
    )

    # Check for pending example question from sidebar button
    if not user_input and st.session_state.get("pending_question"):
        user_input = st.session_state.pop("pending_question")

    if user_input:
        user_question = user_input.strip()

        # Enforce question limit
        if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
            st.warning(f"Session question limit reached ({MAX_QUESTIONS_PER_SESSION} max).")
            return

        # 1. Display user query
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # 2. Enforce daily API limit before calling model
        if not check_and_increment_daily_llm_calls():
            quota_msg = "The AI model's free daily limit has been reached. Please try again later."
            st.warning(quota_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": quota_msg,
                    "sources": [],
                    "uncited": False,
                }
            )
            return

        # 3. Generate assistant response
        with st.chat_message("assistant"):
            # Extract last 4 turns of history before this message
            recent_history = st.session_state.messages[:-1][-4:]

            try:
                # Stream answer using rag.stream_answer
                stream = stream_answer(
                    user_question,
                    history=recent_history,
                    collection_name=session_collection,
                )
                full_answer = st.write_stream(stream)
                sources = getattr(stream, "sources", [])
                uncited = getattr(stream, "uncited", False)

                # Render sources expander if sources exist
                if sources:
                    render_sources(sources, uncited=uncited)

                # Increment session question count on success
                st.session_state.question_count += 1

                # Save assistant message and sources to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "sources": sources,
                        "uncited": uncited,
                    }
                )
            except QuotaExceededError:
                quota_warning = "The AI model's free daily limit has been reached. Please try again later."
                st.warning(quota_warning)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": quota_warning,
                        "sources": [],
                        "uncited": False,
                    }
                )
            except Exception as exc:
                logger.exception("Error generating answer: %s", exc)
                clean_error = "An error occurred while generating the answer. Please try again in a moment."
                st.error(clean_error)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": clean_error,
                        "sources": [],
                        "uncited": False,
                    }
                )

    # Small footer
    st.markdown(
        """
        <div class="app-footer">
            Answers are generated from your uploaded documents only.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
