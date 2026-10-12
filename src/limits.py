"""Rate limiting and daily usage tracking for DocChat-AI.

Manages daily distinct session counts and daily LLM call caps with thread safety
and automatic reset at UTC midnight.
"""

import datetime
import threading
from typing import Optional, Set

from src.config import (
    DEMO_MODE,
    MAX_LLM_CALLS_PER_DAY,
    MAX_SESSIONS_PER_DAY,
)

_lock = threading.Lock()
_daily_sessions_date: Optional[datetime.date] = None
_daily_sessions_seen: Set[str] = set()

_daily_call_date: Optional[datetime.date] = None
_daily_call_count: int = 0


def _get_utc_today() -> datetime.date:
    """Return today's date in UTC."""
    return datetime.datetime.now(datetime.timezone.utc).date()


def reset_daily_limits() -> None:
    """Reset all daily counters (useful for testing or manual reset)."""
    global _daily_sessions_date, _daily_sessions_seen, _daily_call_date, _daily_call_count
    with _lock:
        today_utc = _get_utc_today()
        _daily_sessions_date = today_utc
        _daily_sessions_seen.clear()
        _daily_call_date = today_utc
        _daily_call_count = 0


def check_and_register_session(
    session_id: str,
    is_owner: bool = False,
    demo_mode: Optional[bool] = None,
    max_sessions: Optional[int] = None,
) -> bool:
    """Check if a session is allowed and register it toward MAX_SESSIONS_PER_DAY.

    Should be called only on the session's first meaningful action (sample document load,
    document upload/processing, or question submission).

    Args:
        session_id: Unique identifier for the user session.
        is_owner: If True, owner bypass is applied.
        demo_mode: Optional override for DEMO_MODE (defaults to config.DEMO_MODE).
        max_sessions: Optional override for MAX_SESSIONS_PER_DAY.

    Returns:
        True if the session is allowed to proceed, False if MAX_SESSIONS_PER_DAY has
        been reached for new sessions today. Sessions already counted keep working.
    """
    effective_demo = DEMO_MODE if demo_mode is None else demo_mode
    if not effective_demo or is_owner:
        return True

    effective_limit = MAX_SESSIONS_PER_DAY if max_sessions is None else max_sessions

    global _daily_sessions_date, _daily_sessions_seen
    with _lock:
        today_utc = _get_utc_today()
        if _daily_sessions_date != today_utc:
            _daily_sessions_date = today_utc
            _daily_sessions_seen = set()

        if session_id in _daily_sessions_seen:
            return True

        if len(_daily_sessions_seen) >= effective_limit:
            return False

        _daily_sessions_seen.add(session_id)
        return True


def is_session_registered(session_id: str) -> bool:
    """Check if a session ID is already registered for today."""
    global _daily_sessions_date, _daily_sessions_seen
    with _lock:
        today_utc = _get_utc_today()
        if _daily_sessions_date != today_utc:
            return False
        return session_id in _daily_sessions_seen


def check_and_increment_daily_llm_calls(
    bypass: bool = False,
    demo_mode: Optional[bool] = None,
    max_calls: Optional[int] = None,
) -> bool:
    """Check and increment the daily LLM calls counter.

    Resets at UTC midnight. When DEMO_MODE is False or bypass is True, limits
    do not apply and this returns True.

    Args:
        bypass: If True, bypass daily LLM limit check (e.g. for owner).
        demo_mode: Optional override for DEMO_MODE.
        max_calls: Optional override for MAX_LLM_CALLS_PER_DAY.

    Returns:
        True if call is permitted under MAX_LLM_CALLS_PER_DAY, or False
        if daily call limit has been reached.
    """
    effective_demo = DEMO_MODE if demo_mode is None else demo_mode
    if not effective_demo or bypass:
        return True

    effective_limit = MAX_LLM_CALLS_PER_DAY if max_calls is None else max_calls

    global _daily_call_date, _daily_call_count
    with _lock:
        today_utc = _get_utc_today()
        if _daily_call_date != today_utc:
            _daily_call_date = today_utc
            _daily_call_count = 0

        if _daily_call_count >= effective_limit:
            return False

        _daily_call_count += 1
        return True


def get_daily_llm_calls_remaining(
    bypass: bool = False,
    demo_mode: Optional[bool] = None,
    max_calls: Optional[int] = None,
) -> int:
    """Return remaining allowed LLM calls for today (UTC)."""
    effective_demo = DEMO_MODE if demo_mode is None else demo_mode
    if not effective_demo or bypass:
        return 999999

    effective_limit = MAX_LLM_CALLS_PER_DAY if max_calls is None else max_calls

    global _daily_call_date, _daily_call_count
    with _lock:
        today_utc = _get_utc_today()
        if _daily_call_date != today_utc:
            return effective_limit
        return max(0, effective_limit - _daily_call_count)


def get_daily_sessions_count() -> int:
    """Return count of registered sessions seen today (UTC)."""
    global _daily_sessions_date, _daily_sessions_seen
    with _lock:
        today_utc = _get_utc_today()
        if _daily_sessions_date != today_utc:
            return 0
        return len(_daily_sessions_seen)


def get_daily_llm_calls_count() -> int:
    """Return count of LLM calls made today (UTC)."""
    global _daily_call_date, _daily_call_count
    with _lock:
        today_utc = _get_utc_today()
        if _daily_call_date != today_utc:
            return 0
        return _daily_call_count
