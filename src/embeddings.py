"""Embedding generation module for docchat-ai.

Generates dense vector embeddings using Google Gen AI SDK (google-genai)
and the gemini-embedding-001 model with sliding-window rate limiting,
persistent on-disk caching, and robust quota retry handling.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Try loading .env from current working directory and project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

# Import configuration constants with fallbacks
try:
    from src.config import (
        EMBED_BATCH_SIZE,
        EMBED_MAX_PER_MINUTE,
        EMBEDDING_BATCH_SIZE,
        EMBEDDING_MODEL,
    )
except ImportError:
    try:
        from config import (
            EMBED_BATCH_SIZE,
            EMBED_MAX_PER_MINUTE,
            EMBEDDING_BATCH_SIZE,
            EMBEDDING_MODEL,
        )
    except ImportError:
        EMBEDDING_MODEL = "gemini-embedding-001"
        EMBED_MAX_PER_MINUTE = 80
        EMBED_BATCH_SIZE = 20
        EMBEDDING_BATCH_SIZE = 20

logger = logging.getLogger(__name__)

# Cached client instance
_client: Optional[genai.Client] = None


class DailyQuotaExceededError(Exception):
    """Raised when the Gemini embedding daily API quota has been exhausted."""

    pass


def get_client() -> genai.Client:
    """Retrieve or initialize the Google Gen AI client using GEMINI_API_KEY.

    Returns:
        An authenticated genai.Client instance.

    Raises:
        ValueError: If GEMINI_API_KEY is missing or empty.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GEMINI_API_KEY is missing or empty. Please set GEMINI_API_KEY in your "
            ".env file or as an environment variable to use embeddings."
        )

    _client = genai.Client(api_key=api_key.strip())
    return _client


def compute_cache_key(model: str, task_type: str, text: str) -> str:
    """Compute sha256(model + task_type + text) cache key."""
    raw = f"{model}{task_type}{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DiskEmbeddingCache:
    """Persistent on-disk embedding cache using SQLite in .embed_cache/."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            db_path = project_root / ".embed_cache" / "embeddings.db"
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        cache_key TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_key ON embeddings(cache_key);"
                )
                conn.commit()

    def get(self, key: str) -> Optional[list[float]]:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute(
                    "SELECT embedding FROM embeddings WHERE cache_key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    try:
                        return json.loads(row[0])
                    except Exception:
                        return None
        return None

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        result: dict[str, list[float]] = {}
        chunk_size = 500
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                for i in range(0, len(keys), chunk_size):
                    chunk_keys = keys[i : i + chunk_size]
                    placeholders = ",".join("?" for _ in chunk_keys)
                    cursor = conn.execute(
                        f"SELECT cache_key, embedding FROM embeddings WHERE cache_key IN ({placeholders})",
                        chunk_keys,
                    )
                    for k, emb_json in cursor.fetchall():
                        try:
                            result[k] = json.loads(emb_json)
                        except Exception:
                            pass
        return result

    def set_batch(self, records: list[tuple[str, str, str, list[float]]]) -> None:
        """Insert or replace a batch of (cache_key, model, task_type, embedding)."""
        if not records:
            return
        data = [
            (key, model, task_type, json.dumps(vec))
            for key, model, task_type, vec in records
        ]
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embeddings (cache_key, model, task_type, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    data,
                )
                conn.commit()


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter tracking text count in the last 60 seconds."""

    def __init__(self, max_per_minute: int = EMBED_MAX_PER_MINUTE, window_seconds: float = 60.0):
        self.max_per_minute = max_per_minute
        self.window_seconds = window_seconds
        self.events: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self.events = [e for e in self.events if e[0] > cutoff]

    def wait_for_capacity(self, count: int) -> None:
        """Sleep until sending 'count' texts will not exceed max_per_minute in the last 60s."""
        if count <= 0:
            return

        while True:
            with self._lock:
                now = time.time()
                self._prune(now)
                current_total = sum(c for _, c in self.events)

                if current_total + count <= self.max_per_minute:
                    return

                if count > self.max_per_minute:
                    if not self.events:
                        return
                    oldest_ts = self.events[0][0]
                    sleep_duration = max(0.1, (oldest_ts + self.window_seconds) - now + 0.1)
                else:
                    excess = (current_total + count) - self.max_per_minute
                    dropped = 0
                    wake_time = now
                    for ts, c in self.events:
                        dropped += c
                        wake_time = ts + self.window_seconds
                        if dropped >= excess:
                            break
                    sleep_duration = max(0.1, wake_time - now + 0.1)

            logger.info(
                "Rate limiter: %d/%d texts in last 60s. Need %d. Sleeping %.2fs...",
                current_total,
                self.max_per_minute,
                count,
                sleep_duration,
            )
            print(
                f"Rate limit pacing: {current_total}/{self.max_per_minute} texts sent in last 60s. "
                f"Sleeping {sleep_duration:.1f}s before batch of {count}..."
            )
            time.sleep(sleep_duration)

    def record(self, count: int) -> None:
        """Record that 'count' texts were sent to the API at the current timestamp."""
        with self._lock:
            now = time.time()
            self._prune(now)
            self.events.append((now, count))


# Module singletons
_cache = DiskEmbeddingCache()
_rate_limiter = SlidingWindowRateLimiter(max_per_minute=EMBED_MAX_PER_MINUTE)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Determine if an exception represents a rate limit or transient quota exhaustion."""
    if isinstance(exc, APIError):
        if exc.code in (429, 503):
            return True
        if hasattr(exc, "status") and exc.status and "RESOURCE_EXHAUSTED" in exc.status:
            return True
    error_msg = str(exc).lower()
    return any(
        phrase in error_msg
        for phrase in ("429", "rate limit", "quota", "resource_exhausted", "resource exhausted")
    )


def _extract_retry_delay(exc: Exception) -> Optional[float]:
    """Extract retryDelay in seconds from exception details or error message."""
    if isinstance(exc, APIError) and hasattr(exc, "response_json") and isinstance(exc.response_json, dict):
        error_dict = exc.response_json.get("error", {})
        details = error_dict.get("details", []) if isinstance(error_dict, dict) else []
        if isinstance(details, list):
            for item in details:
                if isinstance(item, dict) and "retryDelay" in item:
                    rd_val = item["retryDelay"]
                    try:
                        return float(str(rd_val).rstrip("s"))
                    except (ValueError, TypeError):
                        pass

    err_str = str(exc)
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s?", err_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    match2 = re.search(r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*s?", err_str, re.IGNORECASE)
    if match2:
        try:
            return float(match2.group(1))
        except ValueError:
            pass

    return None


def _is_daily_quota_error(exc: Exception, retry_delay: Optional[float] = None) -> bool:
    """Determine whether a 429 indicates a daily quota exhaustion rather than per-minute."""
    if retry_delay is not None and retry_delay > 120.0:
        return True

    err_str = str(exc).lower()
    if isinstance(exc, APIError) and hasattr(exc, "response_json") and isinstance(exc.response_json, dict):
        try:
            err_str += " " + json.dumps(exc.response_json).lower()
        except Exception:
            pass

    if any(m in err_str for m in ["per minute", "per_minute", "perminute"]):
        return False

    daily_markers = [
        "daily",
        "per_day",
        "per day",
        "perday",
        "requests per day",
        "requests_per_day",
        "embedcontentrequestsperday",
        "generate content requests per day",
        "per project per day",
    ]
    return any(marker in err_str for marker in daily_markers)


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
    max_retries: int = 5,
) -> list[list[float]]:
    """Embed a list of text strings using Google Gen AI SDK with rate limiting and persistent caching.

    Args:
        texts: List of text strings to embed.
        task_type: Task type hint for the embedding model, e.g.,
            'RETRIEVAL_DOCUMENT' for stored chunks or 'RETRIEVAL_QUERY' for queries.
        model: Model identifier. Defaults to EMBEDDING_MODEL ('gemini-embedding-001').
        batch_size: Maximum number of texts per API request. Defaults to EMBED_BATCH_SIZE (20).
        max_retries: Maximum number of retry attempts on rate-limit errors. Defaults to 5.

    Returns:
        List of embedding vectors, each represented as a list of floats.

    Raises:
        DailyQuotaExceededError: If the daily API quota has been reached.
        ValueError: If GEMINI_API_KEY is missing.
        Exception: If the embedding request fails after all retry attempts.
    """
    if not texts:
        return []

    effective_batch_size = min(batch_size, EMBED_BATCH_SIZE)

    # 1. Compute cache keys for all input texts
    keys = [compute_cache_key(model, task_type, t) for t in texts]

    # 2. Check disk cache
    cached_map = _cache.get_many(keys)

    # Identify uncached texts (deduplicated by key to avoid redundant API embedding)
    uncached_keys: list[str] = []
    uncached_texts: list[str] = []
    seen_uncached: set[str] = set()

    for t, k in zip(texts, keys):
        if k not in cached_map and k not in seen_uncached:
            seen_uncached.add(k)
            uncached_keys.append(k)
            uncached_texts.append(t)

    cached_count = len(texts) - len(seen_uncached)
    new_count = len(seen_uncached)

    # 3. Print cached vs new count and estimated time
    est_minutes = new_count / EMBED_MAX_PER_MINUTE
    print(
        f"Embedding: {cached_count} cached, {new_count} new texts. "
        f"Estimated time: {est_minutes:.2f} min ({est_minutes * 60:.1f}s)."
    )

    # 4. If everything is cached, return immediately
    if new_count == 0:
        return [cached_map[k] for k in keys]

    # 5. Embed uncached texts in batches
    client = get_client()
    config = types.EmbedContentConfig(task_type=task_type)

    for i in range(0, len(uncached_texts), effective_batch_size):
        batch_slice_texts = uncached_texts[i : i + effective_batch_size]
        batch_slice_keys = uncached_keys[i : i + effective_batch_size]
        batch_count = len(batch_slice_texts)

        # Rate limiter: sleep if needed to respect EMBED_MAX_PER_MINUTE
        _rate_limiter.wait_for_capacity(batch_count)

        # Replace empty strings with a single space to avoid API validation errors
        sanitized_batch = [t if (t and t.strip()) else " " for t in batch_slice_texts]

        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=sanitized_batch,
                    config=config,
                )
                break
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    retry_delay = _extract_retry_delay(exc)
                    if _is_daily_quota_error(exc, retry_delay):
                        logger.error("Daily embedding quota exceeded: %s", exc)
                        raise DailyQuotaExceededError(
                            f"Gemini embedding daily API quota exceeded on model '{model}'. "
                            f"Cannot proceed today. Original error: {exc}"
                        ) from exc

                    if attempt < max_retries:
                        if retry_delay is not None and retry_delay > 0:
                            wait_time = retry_delay + 5.0
                        else:
                            wait_time = 65.0

                        logger.warning(
                            "Rate limit hit during embed_texts (attempt %d/%d). "
                            "Waiting %.1f seconds before retry...",
                            attempt,
                            max_retries,
                            wait_time,
                        )
                        print(
                            f"Rate limit hit during embed_texts (attempt {attempt}/{max_retries}). "
                            f"Waiting {wait_time:.1f} seconds before retry..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error("Embedding retries exhausted (%d attempts): %s", max_retries, exc)
                        raise
                else:
                    raise

        # Record successfully sent texts in the rate limiter
        _rate_limiter.record(batch_count)

        if not response or not response.embeddings:
            raise RuntimeError(f"No embeddings returned for batch starting at index {i}.")

        batch_vectors: list[list[float]] = []
        for item in response.embeddings:
            if item and item.values is not None:
                batch_vectors.append(list(item.values))
            else:
                batch_vectors.append([])

        # Immediately save this batch to the persistent on-disk cache
        records_to_save: list[tuple[str, str, str, list[float]]] = []
        for k, vec in zip(batch_slice_keys, batch_vectors):
            cached_map[k] = vec
            records_to_save.append((k, model, task_type, vec))

        _cache.set_batch(records_to_save)

    # 6. Reconstruct embeddings in the exact order requested
    return [cached_map[k] for k in keys]


def embed_query(
    query: str,
    model: str = EMBEDDING_MODEL,
) -> list[float]:
    """Embed a single query string for semantic retrieval.

    Args:
        query: Query string to embed.
        model: Model identifier. Defaults to EMBEDDING_MODEL ('gemini-embedding-001').

    Returns:
        Embedding vector as a list of floats.

    Raises:
        ValueError: If GEMINI_API_KEY is missing or query is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")

    clean_query = query.strip()
    key = compute_cache_key(model, "RETRIEVAL_QUERY", clean_query)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    embeddings = embed_texts([clean_query], task_type="RETRIEVAL_QUERY", model=model, batch_size=1)
    if not embeddings or not embeddings[0]:
        raise RuntimeError("Failed to generate embedding for query.")
    return embeddings[0]


if __name__ == "__main__":
    import sys

    print("--- Testing embeddings.py with disk cache & rate limiting ---")
    try:
        test_client = get_client()
        print("Google Gen AI client initialized successfully.")
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    test_query = "How does chunking work?"
    print(f"\nEmbedding query: '{test_query}' (task_type='RETRIEVAL_QUERY')...")
    q_emb = embed_query(test_query)
    print(f"Query embedding generated! Dimension: {len(q_emb)}")

    # Test cache hit on identical query
    print("\nRe-embedding identical query (should hit cache)...")
    q_emb2 = embed_query(test_query)
    assert q_emb == q_emb2, "Cached embedding mismatch!"
    print("Cache hit test passed!")

    test_docs = [
        "DocChat-AI is an intelligent document Q&A application.",
        "Chunking splits documents into overlapping boundary-aware segments.",
    ]
    print(f"\nEmbedding {len(test_docs)} documents (task_type='RETRIEVAL_DOCUMENT')...")
    doc_embs = embed_texts(test_docs, task_type="RETRIEVAL_DOCUMENT")
    print(f"Documents embedded! Count: {len(doc_embs)}, Dimension: {len(doc_embs[0])}")

    print("\nEmbeddings tests passed successfully!")
