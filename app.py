"""DocChat-AI Streamlit Application.

Interactive RAG document question-answering interface featuring multi-file uploads,
session-isolated vector storage, streamed answers with Gemini, daily quota handling,
in-memory caching, and cited source filtering.
"""

from collections import defaultdict
import datetime
import html
import hmac
import logging
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from pypdf import PdfReader
import streamlit as st

# Set up module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Page Configuration
st.set_page_config(
    page_title="DocChat",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 2. Environment & Secrets Management
load_dotenv()
import importlib
import src.config
if not hasattr(src.config, "DEMO_MODE"):
    importlib.reload(src.config)
from src.config import (
    DEMO_MODE,
    MAX_CHUNKS_PER_SESSION,
    MAX_HISTORY_TURNS,
    MAX_LLM_CALLS_PER_DAY,
    MAX_PAGES,
    MAX_QUESTIONS_PER_SESSION,
    MAX_SESSIONS_PER_DAY,
    MAX_UPLOAD_MB,
    OWNER_KEY,
    get_secret,
)

gemini_key = get_secret("GEMINI_API_KEY")
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

# Import core modules and configs
from src.chunker import chunk_pages
from src.limits import (
    check_and_increment_daily_llm_calls,
    check_and_register_session,
    get_daily_llm_calls_remaining,
    is_session_registered,
)
from src.loader import format_unreadable_pages, get_unreadable_pages, load_document
from src.rag import (
    QuotaExceededError,
    clear_answer_cache,
    stream_answer,
    summarize_document,
)
from src.vector_store import (
    add_chunks,
    cleanup_old_collections,
    delete_file,
    get_collection,
    list_files,
)


def ensure_session_allowed(is_owner: bool = False) -> bool:
    """Check and register session on first user action.

    Returns True if allowed. If the visitor limit is reached on first action,
    displays the visitor-limit message and returns False without performing the action.
    """
    if not check_and_register_session(st.session_state.session_id, is_owner=is_owner):
        st.info("The demo reached today's visitor limit, please come back tomorrow")
        st.markdown("[DocChat-AI on GitHub](https://github.com/abinand705/DocChat-AI#readme)")
        return False
    return True

# Custom CSS: Minimalist Modern Workspace (Linear / Perplexity inspired)
st.markdown(
    """
    <style>
    /* ========================================================================== */
    /* 1. DESIGN TOKENS & SYSTEM VARIABLES (:root)                                */
    /* ========================================================================== */
    :root {
        /* Monochrome Base (Slate / Zinc scale) */
        --bg-canvas: #fafafa;
        --bg-surface: #ffffff;
        --bg-panel: #f8fafc;
        --bg-muted: #f1f5f9;
        --bg-elevated: #ffffff;

        --border-subtle: rgba(0, 0, 0, 0.05);
        --border-default: rgba(0, 0, 0, 0.08);
        --border-strong: rgba(0, 0, 0, 0.16);

        /* Text Hierarchy */
        --text-primary: #00000a;       /* Slate 900 - Headers */
        --text-secondary: #334155;     /* Slate 700 - Body */
        --text-muted: #64748b;         /* Slate 500 - Metadata / Captions */
        --text-inverse: #ffffff;

        /* Chat Bubble Tokens (Light Mode) */
        --bg-chat-user: #0f172a;       /* Deep slate neutral */
        --text-chat-user: #f8fafc;     /* Crisp white text */
        --border-chat-user: #1e293b;

        /* Focused Accent: Electric Indigo */
        --accent-primary: #4f46e5;
        --accent-hover: #4338ca;
        --accent-subtle: rgba(79, 70, 229, 0.08);
        --accent-ring: rgba(79, 70, 229, 0.22);

        /* Semantic Colors */
        --color-success: #10b981;
        --color-error: #ef4444;
        --color-warning: #f59e0b;

        /* Typography */
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', Menlo, monospace;

        /* Spacing & Radii */
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 16px;
        --radius-pill: 9999px;

        /* Layered Soft Shadows */
        --shadow-subtle: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        --shadow-card: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
        --shadow-floating: 0 4px 20px -2px rgba(0, 0, 0, 0.06), 0 2px 6px -1px rgba(0, 0, 0, 0.04);
        --shadow-focus: 0 0 0 3px var(--accent-ring);
    }



    /* Base typography: explicitly exclude icon containers */
    html, body, input, textarea, button, select {
        font-family: var(--font-sans);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Preserve Streamlit Material Symbols & Icon rendering */
    [data-testid="stIconMaterial"],
    [data-testid*="Icon"] span,
    [data-testid*="Icon"],
    [data-testid*="Chevron"],
    .material-symbols-rounded,
    .material-icons,
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        font-feature-settings: 'liga' 1 !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        -webkit-font-smoothing: antialiased !important;
        text-rendering: optimizeLegibility !important;
    }

    /* Minimalist theme scrollbars */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-strong);
        border-radius: var(--radius-pill);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* Subtle dividers */
    hr {
        border: none !important;
        height: 1px !important;
        background-color: var(--border-default) !important;
        margin: 1.25rem 0 !important;
        opacity: 0.7;
    }

    /* ========================================================================== */
    /* 3. LAYOUT & CONTAINERS                                                     */
    /* ========================================================================== */
    .main .block-container {
        max-width: 840px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    /* Sidebar Navigation Surface */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border-default) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.5rem 1.15rem;
    }

    /* Brand Header */
    .brand-badge-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 2px;
    }

    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--text-primary);
    }

    .brand-pill {
        font-size: 0.68rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: var(--radius-pill);
        background: var(--accent-subtle);
        color: var(--accent-primary);
        border: 1px solid var(--accent-ring);
        letter-spacing: 0.02em;
    }

    .brand-subtitle {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }

    .sidebar-section-header {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-muted);
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }

    /* Main Workspace Header */
    .workspace-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    .workspace-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
        flex-wrap: wrap;
        gap: 8px;
    }

    .workspace-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--text-primary);
        margin: 0;
    }

    .workspace-meta-pills {
        display: flex;
        gap: 6px;
    }

    .meta-pill {
        font-size: 0.72rem;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: var(--radius-pill);
        background: var(--bg-muted);
        color: var(--text-muted);
        border: 1px solid var(--border-default);
    }

    .workspace-desc {
        font-size: 0.84rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* Empty state */
    .empty-state-card {
        text-align: center;
        padding: 3rem 1.5rem;
        background: var(--bg-surface);
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius-lg);
        margin: 2rem 0;
    }

    .empty-state-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }

    .empty-state-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .empty-state-desc {
        font-size: 0.84rem;
        color: var(--text-muted);
    }

    /* ========================================================================== */
    /* 4. COMPONENTS                                                              */
    /* ========================================================================== */

    /* 4.1 Buttons: Rounded modern pills */
    .stButton > button {
        border-radius: var(--radius-md) !important;
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 6px 14px !important;
        border: 1px solid var(--border-default) !important;
        background-color: var(--bg-surface) !important;
        color: var(--text-secondary) !important;
        box-shadow: var(--shadow-subtle) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: var(--bg-muted) !important;
        border-color: var(--border-strong) !important;
        color: var(--text-primary) !important;
        transform: translateY(-1px);
        box-shadow: var(--shadow-card) !important;
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Primary CTA Button */
    .stButton > button[kind="primary"] {
        background: var(--accent-primary) !important;
        color: #ffffff !important;
        border: 1px solid var(--accent-primary) !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Sidebar Example Question Chips */
    [data-testid="stSidebar"] .stButton > button {
        border-radius: var(--radius-pill) !important;
        border: 1px solid var(--border-default) !important;
        background: var(--bg-surface) !important;
        color: var(--text-secondary) !important;
        font-size: 0.77rem !important;
        font-weight: 500 !important;
        padding: 5px 12px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
        background: var(--accent-subtle) !important;
        transform: translateY(-1px);
    }

    /* Delete Button in Sidebar */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child button {
        border-radius: var(--radius-pill) !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: var(--text-muted) !important;
        padding: 4px 8px !important;
        min-height: auto !important;
        line-height: 1 !important;
        font-size: 0.75rem !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child button:hover {
        color: var(--color-error) !important;
        background: rgba(239, 68, 68, 0.08) !important;
        border-color: rgba(239, 68, 68, 0.16) !important;
        transform: none !important;
    }

    /* 4.2 Document Manager: Dropzone & File Chips */
    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed var(--border-strong) !important;
        border-radius: var(--radius-lg) !important;
        background-color: var(--bg-surface) !important;
        padding: 1rem !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-primary) !important;
        background-color: var(--accent-subtle) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        border-radius: var(--radius-pill) !important;
        font-size: 0.75rem !important;
    }

    .file-chip {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-pill);
        font-size: 0.76rem;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        box-shadow: var(--shadow-subtle);
    }

    .file-chip-icon {
        font-size: 0.78rem;
        opacity: 0.7;
    }

    .file-chip-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-weight: 500;
    }

    /* 4.3 Chat Bubbles & Avatars */
    [data-testid="stChatMessage"] {
        display: flex !important;
        align-items: flex-start !important;
        gap: 12px !important;
        padding: 12px 18px !important;
        margin-bottom: 12px !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stChatMessageContent"] {
        flex-grow: 1 !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stChatMessageContent"] p {
        margin: 0 !important;
        line-height: 1.6 !important;
    }

    /* User Message: Dark slate card / right-leaning */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: var(--bg-chat-user) !important;
        border: 1px solid var(--border-chat-user) !important;
        border-radius: 18px 18px 4px 18px !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        max-width: 82% !important;
        box-shadow: var(--shadow-card) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] span,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] div {
        color: var(--text-chat-user) !important;
    }

    /* User Avatar: Clean Slate Badge (replaces default red) */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="chatAvatarIcon-user"] {
        background-color: #334155 !important;
        color: #ffffff !important;
        border-radius: var(--radius-pill) !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        margin-top: 1px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
    }

    [data-testid="stChatMessageAvatarUser"] [data-testid="stIconMaterial"],
    [data-testid="stChatMessageAvatarUser"] svg,
    [data-testid="chatAvatarIcon-user"] svg {
        color: #ffffff !important;
        font-size: 1.15rem !important;
        width: 1.15rem !important;
        height: 1.15rem !important;
    }

    /* Assistant Message: Clean Surface Card */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 18px 18px 18px 4px !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        max-width: 95% !important;
        box-shadow: var(--shadow-subtle) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p {
        color: var(--text-secondary) !important;
    }

    /* Assistant Avatar: Electric Indigo Badge (replaces default orange) */
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: var(--accent-primary) !important;
        color: #ffffff !important;
        border-radius: var(--radius-pill) !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        margin-top: 1px !important;
        box-shadow: 0 1px 4px rgba(79, 70, 229, 0.25) !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] [data-testid="stIconMaterial"],
    [data-testid="stChatMessageAvatarAssistant"] svg,
    [data-testid="chatAvatarIcon-assistant"] svg {
        color: #ffffff !important;
        font-size: 1.15rem !important;
        width: 1.15rem !important;
        height: 1.15rem !important;
    }

    /* 4.4 Floating Bottom Input Bar */
    [data-testid="stBottom"] {
        background: transparent !important;
    }

    [data-testid="stBottom"] > div {
        background: transparent !important;
    }

    [data-testid="stChatInput"] {
        border-radius: 28px !important;
        border: 1.5px solid var(--border-default) !important;
        background: #ffffff !important;
        box-shadow: var(--shadow-floating) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px var(--accent-ring), var(--shadow-floating) !important;
    }

    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {
        background-color: transparent !important;
    }

    [data-testid="stChatInput"] textarea {
        font-family: var(--font-sans) !important;
        font-size: 0.90rem !important;
        color: #0f172a !important;
        caret-color: var(--accent-primary) !important;
        background-color: transparent !important;
        line-height: 1.5 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    /* 4.5 Collapsible Sources & Citations */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-md) !important;
        background-color: var(--bg-surface) !important;
        box-shadow: var(--shadow-subtle) !important;
        margin-top: 10px !important;
        overflow: hidden !important;
    }

    [data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        padding: 8px 12px !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease !important;
    }

    [data-testid="stExpander"] summary:hover {
        background-color: var(--bg-muted) !important;
        color: var(--accent-primary) !important;
    }

    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary svg {
        flex-shrink: 0 !important;
        width: 1.15rem !important;
        height: 1.15rem !important;
        font-size: 1.15rem !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span:not([data-testid="stIconMaterial"]) {
        margin: 0 !important;
        line-height: 1.4 !important;
    }

    .source-doc-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
        padding: 6px 2px 4px 2px;
    }

    .source-doc-page {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-muted);
        background: var(--bg-muted);
        padding: 2px 7px;
        border-radius: var(--radius-pill);
    }

    .source-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-left: 3px solid var(--accent-primary);
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: var(--radius-sm);
        box-shadow: var(--shadow-subtle);
        transition: all 0.15s ease;
    }

    .source-card:hover {
        transform: translateY(-1px);
        border-color: var(--border-strong);
        box-shadow: var(--shadow-card);
    }

    .source-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    .source-num {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--accent-primary);
    }

    .score-chip {
        font-size: 0.68rem;
        font-family: var(--font-mono);
        font-weight: 500;
        background: var(--accent-subtle);
        color: var(--accent-primary);
        padding: 1px 6px;
        border-radius: var(--radius-pill);
        border: 1px solid var(--accent-ring);
    }

    .source-passage {
        font-size: 0.8rem;
        line-height: 1.5;
        color: var(--text-secondary);
        word-break: break-word;
    }

    /* 4.6 Alerts & Notifications */
    [data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        border-width: 1px !important;
        font-size: 0.82rem !important;
        box-shadow: var(--shadow-subtle) !important;
    }

    /* 4.7 Footer */
    .app-footer {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-subtle);
        font-size: 0.74rem;
        color: var(--text-muted);
        letter-spacing: 0.02em;
    }

    /* Mobile adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        [data-testid="stChatMessage"] {
            padding: 10px 14px !important;
            max-width: 100% !important;
        }
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

    if "unreadable_pages" not in st.session_state:
        st.session_state.unreadable_pages = {}

    return session_collection


def render_sources(sources: list[dict[str, Any]], uncited: bool = False) -> None:
    """Render citations grouped by document filename and page number.

    Uses uncited flag to choose between 'Retrieved Passages' and 'Cited Sources'.
    Passage text and filenames are HTML-escaped for safety.

    Args:
        sources: List of source dictionaries containing 'file', 'page', 'score', and 'text'.
        uncited: Boolean indicating whether citations were absent or invalid.
    """
    if not sources:
        return

    expander_label = (
        f"Retrieved Passages ({len(sources)})" if uncited else f"Cited Sources ({len(sources)})"
    )

    with st.expander(expander_label, expanded=False):
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for src in sources:
            file_name = src.get("file", "Unknown document")
            page_num = src.get("page", 1)
            grouped[(file_name, page_num)].append(src)

        for (file_name, page_num), chunks in grouped.items():
            escaped_file = html.escape(str(file_name))
            st.markdown(
                f"""
                <div class="source-doc-header">
                    <span>{escaped_file}</span>
                    <span class="source-doc-page">Page {page_num}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for idx, chunk in enumerate(chunks, start=1):
                score = chunk.get("score", 0.0)
                passage = html.escape(str(chunk.get("text", "")).strip())
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-card-header">
                            <span class="source-num">Passage {idx}</span>
                            <span class="score-chip">score {score:.3f}</span>
                        </div>
                        <div class="source-passage">{passage}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_sidebar(session_collection: str, is_owner: bool = False) -> None:
    """Render sidebar controls including uploads, limits, demo loader, and file management."""
    with st.sidebar:
        # Minimalist Brand Header
        st.markdown(
            """
            <div class="brand-badge-wrap">
                <span class="brand-title">DocChat</span>
                <span class="brand-pill">v1.0</span>
            </div>
            <div class="brand-subtitle">Minimalist Document Q&amp;A Workspace</div>
            """,
            unsafe_allow_html=True,
        )

        # Privacy Notice (Requirement 6)
        st.caption("Uploaded text is sent to Google's Gemini API. Don't upload confidential documents.")

        # Show questions left caption in DEMO_MODE (Requirement 2)
        if DEMO_MODE:
            if is_owner:
                st.caption("Questions left: Unlimited (Owner)")
            else:
                questions_left = max(0, MAX_QUESTIONS_PER_SESSION - st.session_state.question_count)
                st.caption(f"Questions left: {questions_left}")

        # Check API key status
        if not os.getenv("GEMINI_API_KEY"):
            st.error("API Key missing: set GEMINI_API_KEY in .env or secrets.")

        # Quick Starts
        st.markdown('<div class="sidebar-section-header">Quick Start</div>', unsafe_allow_html=True)
        demo_path = Path("data/sample.pdf").resolve()
        if st.button("Load Sample Document", width="stretch", help="Load data/sample.pdf into this session"):
            if not ensure_session_allowed(is_owner=is_owner):
                pass
            elif not demo_path.exists():
                st.error("Demo document 'data/sample.pdf' was not found.")
            else:
                with st.spinner("Indexing sample document..."):
                    try:
                        pages = load_document(demo_path)
                        chunks = chunk_pages(pages)
                        cur_count = get_collection(session_collection).count()
                        if cur_count + len(chunks) > MAX_CHUNKS_PER_SESSION:
                            st.warning(
                                f"Cannot load sample document: adding {len(chunks)} chunks would exceed "
                                f"the limit of {MAX_CHUNKS_PER_SESSION} chunks."
                            )
                        else:
                            delete_file(demo_path.name, collection_name=session_collection)
                            add_chunks(chunks, collection_name=session_collection)
                            clear_answer_cache(collection_name=session_collection)
                            unreadable = get_unreadable_pages(demo_path.name)
                            st.session_state.unreadable_pages[demo_path.name] = unreadable
                            st.session_state.flash_message = (
                                f"Loaded sample document ({demo_path.name}) with {len(chunks)} chunks!"
                            )
                            st.rerun()
                    except Exception as demo_err:
                        logger.exception("Error loading demo document: %s", demo_err)
                        st.error(f"Failed to load demo document: {demo_err}")

        # Example question prompt starters
        indexed_files = list_files(collection_name=session_collection)
        has_documents = bool(indexed_files)
        is_at_q_limit = DEMO_MODE and not is_owner and (st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION)

        example_questions = [
            "What is the primary purpose of DocChat-AI?",
            "What vector database does DocChat-AI use?",
            "What API is used for embeddings and answer generation?",
        ]
        for idx, ex_q in enumerate(example_questions, start=1):
            if st.button(f"→  {ex_q}", key=f"ex_btn_{idx}", width="stretch", disabled=not has_documents or is_at_q_limit):
                if not ensure_session_allowed(is_owner=is_owner):
                    pass
                else:
                    st.session_state.pending_question = ex_q
                    st.rerun()

        # Document Upload Area
        st.markdown('<div class="sidebar-section-header">Upload Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            f"Select PDF or TXT files (Max {MAX_UPLOAD_MB}MB per file)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help=f"Upload documents up to {MAX_UPLOAD_MB}MB and {MAX_PAGES} pages.",
            label_visibility="collapsed",
        )

        if st.button("Process Documents", type="primary", width="stretch", disabled=not uploaded_files):
            if not ensure_session_allowed(is_owner=is_owner):
                pass
            else:
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
                            unreadable = get_unreadable_pages(filename)
                            if unreadable:
                                st.session_state.unreadable_pages[filename] = unreadable

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

        # Indexed Documents (Knowledge Base)
        st.markdown('<div class="sidebar-section-header">Indexed Documents</div>', unsafe_allow_html=True)
        is_at_q_limit = DEMO_MODE and not is_owner and (st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION)
        if indexed_files:
            # Document selector when multiple files are indexed
            if len(indexed_files) > 1:
                st.selectbox(
                    "Focus search on document",
                    ["All documents"] + indexed_files,
                    key="selected_doc_filter",
                    help="Select a specific document to search, or search all.",
                )

            for fname in indexed_files:
                col_name, col_del = st.columns([0.82, 0.18], vertical_alignment="center")
                with col_name:
                    escaped_fname = html.escape(fname)
                    st.markdown(
                        f"""
                        <div class="file-chip" title="{escaped_fname}">
                            <span class="file-chip-icon">📄</span>
                            <span class="file-chip-name">{escaped_fname}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("✕", key=f"del_{fname}", help=f"Delete {fname}"):
                        try:
                            delete_file(fname, collection_name=session_collection)
                            clear_answer_cache(collection_name=session_collection)
                            st.session_state.unreadable_pages.pop(fname, None)
                            st.session_state.flash_message = f"Removed {fname}"
                            st.rerun()
                        except Exception as del_err:
                            logger.exception("Error deleting file %s: %s", fname, del_err)
                            st.error(f"Error deleting file: {del_err}")

                # Persistent sidebar note per file (Requirement 5)
                unreadable = st.session_state.get("unreadable_pages", {}).get(fname)
                if unreadable is None:
                    unreadable = get_unreadable_pages(fname)
                    if unreadable:
                        st.session_state.unreadable_pages[fname] = unreadable
                if unreadable:
                    st.caption(format_unreadable_pages(unreadable))

                # Summarize document button per indexed file (Requirement 3)
                if st.button("Summarize document", key=f"sum_{fname}", width="stretch", disabled=is_at_q_limit):
                    if not ensure_session_allowed(is_owner=is_owner):
                        pass
                    else:
                        st.session_state.pending_summary = fname
                        st.rerun()
        else:
            st.caption("No documents in workspace.")

        # Session Controls
        st.markdown('<div class="sidebar-section-header">Session Controls</div>', unsafe_allow_html=True)
        if st.button("Clear Conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.question_count = 0
            clear_answer_cache(collection_name=session_collection)
            st.rerun()


def main() -> None:
    """Main application loop."""
    session_collection = initialize_session_state()

    # 1. Owner bypass authentication (Requirement 3)
    owner_configured_key = OWNER_KEY or get_secret("OWNER_KEY")
    is_owner = False
    if owner_configured_key:
        with st.sidebar:
            st.markdown('<div class="sidebar-section-header">Owner Access</div>', unsafe_allow_html=True)
            owner_input = st.text_input(
                "Owner key",
                type="password",
                key="owner_key_input",
                help="Enter owner key to bypass demo limits.",
            )
            if owner_input:
                if hmac.compare_digest(owner_input.strip(), owner_configured_key):
                    is_owner = True
                    st.session_state.is_owner = True
                    st.caption("🔑 Owner mode active (limits bypassed)")
                else:
                    st.session_state.is_owner = False
                    st.error("Invalid owner key.")
            else:
                st.session_state.is_owner = False
    else:
        st.session_state.is_owner = False

    # Display flash messages after rerun
    if st.session_state.get("flash_message"):
        flash_text = st.session_state.pop("flash_message")
        st.toast(flash_text, icon="✅")
        st.success(flash_text)

    render_sidebar(session_collection, is_owner=is_owner)

    # Check indexed files status
    indexed_files = list_files(collection_name=session_collection)
    has_documents = bool(indexed_files)
    file_count = len(indexed_files)
    q_count = st.session_state.question_count

    if DEMO_MODE and not is_owner:
        turns_pill = f"{q_count}/{MAX_QUESTIONS_PER_SESSION} questions"
    elif is_owner:
        turns_pill = f"{q_count} questions (Owner)"
    else:
        turns_pill = f"{q_count} questions"

    # Modern Workspace Header with metadata pills
    st.markdown(
        f"""
        <div class="workspace-header">
            <div class="workspace-title-row">
                <h1 class="workspace-title">DocChat Workspace</h1>
                <div class="workspace-meta-pills">
                    <span class="meta-pill">{file_count} {"document" if file_count == 1 else "documents"} indexed</span>
                    <span class="meta-pill">{turns_pill}</span>
                </div>
            </div>
            <p class="workspace-desc">Ask questions grounded in your private documents with verifiable citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_documents:
        st.markdown(
            """
            <div class="empty-state-card">
                <div class="empty-state-icon">⚡</div>
                <div class="empty-state-title">No documents indexed yet</div>
                <div class="empty-state-desc">
                    Upload a PDF or TXT file from the sidebar, or click <strong>Load Sample Document</strong> to test immediately.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"], uncited=message.get("uncited", False))

    # Check question limit for session
    is_at_q_limit = DEMO_MODE and not is_owner and (st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION)
    if is_at_q_limit:
        st.warning(
            f"You have reached the limit of {MAX_QUESTIONS_PER_SESSION} questions for this session. "
            "Please clear conversation or start a new session to continue."
        )

    # Check for pending summary request from sidebar (Requirement 3)
    if st.session_state.get("pending_summary"):
        target_file = st.session_state.pop("pending_summary")

        if not ensure_session_allowed(is_owner=is_owner):
            pass
        elif DEMO_MODE and not is_owner and st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
            st.warning(f"Session question limit reached ({MAX_QUESTIONS_PER_SESSION} max).")
        elif DEMO_MODE and not is_owner and not check_and_increment_daily_llm_calls():
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
        else:
            summary_query = f"Summarize document: {target_file}"
            st.session_state.messages.append({"role": "user", "content": summary_query})
            with st.chat_message("user"):
                st.markdown(summary_query)

            with st.chat_message("assistant"):
                with st.spinner(f"Generating summary for {target_file}..."):
                    try:
                        summary_res = summarize_document(
                            file=target_file,
                            collection_name=session_collection,
                        )
                        if summary_res.get("called_llm", True):
                            st.session_state.question_count += 1
                        st.markdown(summary_res["answer"])
                        if summary_res.get("sources"):
                            render_sources(
                                summary_res["sources"],
                                uncited=summary_res.get("uncited", False),
                            )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": summary_res["answer"],
                                "sources": summary_res.get("sources", []),
                                "uncited": summary_res.get("uncited", False),
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
                    except Exception as sum_exc:
                        logger.exception("Error summarizing %s: %s", target_file, sum_exc)
                        clean_error = "An error occurred while generating the summary. Please try again in a moment."
                        st.error(clean_error)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": clean_error,
                                "sources": [],
                                "uncited": False,
                            }
                        )
            st.rerun()

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

        if not ensure_session_allowed(is_owner=is_owner):
            pass
        else:
            # 1. Display user query
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # 2. Extract recent history using MAX_HISTORY_TURNS
            recent_history = st.session_state.messages[:-1][-MAX_HISTORY_TURNS:]

        # Determine target file for search/filtering
        selected_file = None
        if len(indexed_files) == 1:
            selected_file = indexed_files[0]
        elif len(indexed_files) > 1:
            doc_filter = st.session_state.get("selected_doc_filter", "All documents")
            if doc_filter != "All documents":
                selected_file = doc_filter

        # 3. Generate assistant response
        with st.chat_message("assistant"):
            try:
                # Stream answer using rag.stream_answer
                stream = stream_answer(
                    user_question,
                    history=recent_history,
                    collection_name=session_collection,
                    file=selected_file,
                )

                will_call_llm = getattr(stream, "called_llm", True)

                # Enforce limits only if the LLM is actually called
                if will_call_llm and DEMO_MODE and not is_owner:
                    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
                        limit_msg = f"You have reached the limit of {MAX_QUESTIONS_PER_SESSION} questions for this session."
                        st.warning(limit_msg)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": limit_msg,
                                "sources": [],
                                "uncited": False,
                            }
                        )
                        return

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

                full_answer = st.write_stream(stream)
                sources = getattr(stream, "sources", [])
                uncited = getattr(stream, "uncited", False)

                # Render sources expander if sources exist
                if sources:
                    render_sources(sources, uncited=uncited)

                # Increment session question count only if LLM was called
                if will_call_llm:
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
            DocChat &bull; Strictly grounded citations &bull; Isolated session storage
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
