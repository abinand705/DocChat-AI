"""Configuration constants for docchat-ai."""

import datetime
import os
from typing import Optional


def _get_int_env(key: str, default: int) -> int:
    """Helper to parse integer environment variables with a default fallback."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except (ValueError, TypeError):
        return default


def _get_bool_env(key: str, default: bool) -> bool:
    """Helper to parse boolean environment variables with a default fallback."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "t", "y")


# Chunking configuration
CHUNK_SIZE: int = _get_int_env("CHUNK_SIZE", 400)
CHUNK_OVERLAP: int = _get_int_env("CHUNK_OVERLAP", 48)

# Retrieval configuration
TOP_K: int = _get_int_env("TOP_K", 6)

# Embedding configuration
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBED_MAX_PER_MINUTE: int = _get_int_env("EMBED_MAX_PER_MINUTE", 80)
EMBED_BATCH_SIZE: int = _get_int_env("EMBED_BATCH_SIZE", 20)
EMBEDDING_BATCH_SIZE: int = EMBED_BATCH_SIZE  # Backwards compatibility

# Vector database configuration
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "docchat")

# Generation and RAG configuration
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash-lite")
RAG_MODEL: str = GEMINI_MODEL  # Backwards compatibility
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.5"))

def get_secret(key: str, default: str = "") -> str:
    """Safely get a secret from Streamlit secrets, falling back to environment variable or default."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            val = st.secrets.get(key, None)
            if val is not None and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    return default


# Application safety and UX limits (read from environment variables)
MAX_UPLOAD_MB: int = _get_int_env("MAX_UPLOAD_MB", 10)
MAX_PAGES: int = _get_int_env("MAX_PAGES", 60)
MAX_CHUNKS_PER_SESSION: int = _get_int_env("MAX_CHUNKS_PER_SESSION", 400)

MAX_QUESTIONS_PER_SESSION: int = _get_int_env("MAX_QUESTIONS_PER_SESSION", 30)
MAX_SESSIONS_PER_DAY: int = _get_int_env("MAX_SESSIONS_PER_DAY", 10)
MAX_LLM_CALLS_PER_DAY: int = _get_int_env("MAX_LLM_CALLS_PER_DAY", 20)
MAX_HISTORY_TURNS: int = _get_int_env("MAX_HISTORY_TURNS", 6)
DEMO_MODE: bool = _get_bool_env("DEMO_MODE", False)
OWNER_KEY: str = get_secret("OWNER_KEY", "").strip()


def check_and_increment_daily_llm_calls(bypass: bool = False) -> bool:
    """Check and increment the daily LLM calls counter (delegates to src.limits)."""
    from src.limits import check_and_increment_daily_llm_calls as _impl
    return _impl(bypass=bypass)


def get_daily_llm_calls_remaining(bypass: bool = False) -> int:
    """Return remaining allowed LLM calls for today (delegates to src.limits)."""
    from src.limits import get_daily_llm_calls_remaining as _impl
    return _impl(bypass=bypass)


