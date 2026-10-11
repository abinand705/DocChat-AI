"""Configuration constants for docchat-ai."""

# Chunking configuration
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 48

# Retrieval configuration
TOP_K: int = 6

# Embedding configuration
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBED_MAX_PER_MINUTE: int = 80
EMBED_BATCH_SIZE: int = 20
EMBEDDING_BATCH_SIZE: int = EMBED_BATCH_SIZE  # Backwards compatibility

# Vector database configuration
CHROMA_PATH: str = "chroma_db"
COLLECTION_NAME: str = "docchat"

# Generation and RAG configuration
GEMINI_MODEL: str = "gemini-2.5-flash"
FALLBACK_MODEL: str = "gemini-2.5-flash-lite"
RAG_MODEL: str = GEMINI_MODEL  # Backwards compatibility
MIN_SCORE: float = 0.5

# Application safety and UX limits
MAX_UPLOAD_MB: int = 10
MAX_PAGES: int = 60
MAX_CHUNKS_PER_SESSION: int = 400
MAX_QUESTIONS_PER_SESSION: int = 10
MAX_LLM_CALLS_PER_DAY: int = 15

# Module-level daily counter (resets at UTC midnight)
import datetime
from typing import Optional

_daily_call_date: Optional[datetime.date] = None
_daily_call_count: int = 0


def check_and_increment_daily_llm_calls() -> bool:
    """Check and increment the daily LLM calls counter.

    Resets at UTC midnight. Returns True if call is permitted under
    MAX_LLM_CALLS_PER_DAY, or False if daily call limit has been reached.
    """
    global _daily_call_date, _daily_call_count
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    if _daily_call_date != today_utc:
        _daily_call_date = today_utc
        _daily_call_count = 0

    if _daily_call_count >= MAX_LLM_CALLS_PER_DAY:
        return False

    _daily_call_count += 1
    return True


def get_daily_llm_calls_remaining() -> int:
    """Return remaining allowed LLM calls for today (UTC)."""
    global _daily_call_date, _daily_call_count
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    if _daily_call_date != today_utc:
        return MAX_LLM_CALLS_PER_DAY
    return max(0, MAX_LLM_CALLS_PER_DAY - _daily_call_count)
# Demo / public-deployment limits
MAX_UPLOAD_MB: int = 10
MAX_PAGES: int = 60
MAX_CHUNKS_PER_SESSION: int = 400
MAX_QUESTIONS_PER_SESSION: int = 10
MAX_LLM_CALLS_PER_DAY: int = 15
