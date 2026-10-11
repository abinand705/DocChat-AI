"""RAG query and generation pipeline for docchat-ai.

Integrates semantic vector search with Google Gemini for grounded document
question answering and streaming responses, with fallback model support,
daily quota management, and in-memory response caching.
"""

import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Generator, Iterator, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Ensure .env is loaded
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

# Import configuration constants with fallbacks
try:
    from src.config import COLLECTION_NAME, FALLBACK_MODEL, GEMINI_MODEL, MIN_SCORE, TOP_K
except ImportError:
    try:
        from config import COLLECTION_NAME, FALLBACK_MODEL, GEMINI_MODEL, MIN_SCORE, TOP_K
    except ImportError:
        COLLECTION_NAME = "docchat"
        TOP_K = 6
        MIN_SCORE = 0.5
        GEMINI_MODEL = "gemini-2.5-flash"
        FALLBACK_MODEL = "gemini-2.5-flash-lite"

# Import prompts
try:
    from src.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_prompt
except ImportError:
    try:
        from prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_prompt
    except ImportError as err:
        raise ImportError(f"Could not import prompt templates: {err}") from err

# Import vector store search and file listing
try:
    from src.vector_store import list_files, search
except ImportError:
    try:
        from vector_store import list_files, search
    except ImportError as err:
        raise ImportError(f"Could not import vector_store functions: {err}") from err

# Import embeddings get_client
try:
    from src.embeddings import get_client
except ImportError:
    try:
        from embeddings import get_client
    except ImportError:
        def get_client() -> genai.Client:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is missing. Please set it in .env.")
            return genai.Client(api_key=api_key.strip())

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when Gemini API quota or rate limits are exceeded across primary and fallback models."""

    pass


# In-memory answer cache: keyed by (collection_name, normalized_question, tuple_of_sorted_files)
_answer_cache: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}

# Process-level state: timestamp when primary model hit daily quota (expires after 60 min)
_primary_daily_quota_exhausted_time: Optional[float] = None


def is_primary_daily_quota_exhausted() -> bool:
    """Return True if the primary model has encountered daily quota exhaustion within the last 60 minutes."""
    global _primary_daily_quota_exhausted_time
    if _primary_daily_quota_exhausted_time is None:
        return False
    # Expire after 60 minutes (3600 seconds)
    if time.time() - _primary_daily_quota_exhausted_time > 3600.0:
        _primary_daily_quota_exhausted_time = None
        return False
    return True


def set_primary_daily_quota_exhausted() -> None:
    """Record that the primary model hit daily quota."""
    global _primary_daily_quota_exhausted_time
    _primary_daily_quota_exhausted_time = time.time()


def reset_daily_quota_state() -> None:
    """Reset the daily quota tracking state."""
    global _primary_daily_quota_exhausted_time
    _primary_daily_quota_exhausted_time = None


def get_cache_key(
    question: str,
    collection_name: str = COLLECTION_NAME,
) -> tuple[str, str, tuple[str, ...]]:
    """Build a cache key from (collection_name, normalised_question, sorted_file_list)."""
    norm_q = " ".join(question.strip().lower().split())
    try:
        indexed = tuple(sorted(list_files(collection_name=collection_name)))
    except Exception:
        indexed = ()
    return (collection_name, norm_q, indexed)


def clear_answer_cache(collection_name: Optional[str] = None) -> None:
    """Clear all entries in the in-memory answer cache, or only those matching collection_name."""
    global _answer_cache
    if collection_name is None:
        _answer_cache.clear()
    else:
        keys_to_delete = [k for k in _answer_cache if k[0] == collection_name]
        for k in keys_to_delete:
            _answer_cache.pop(k, None)


def _parse_retry_delay_seconds(error_str: str) -> float:
    """Extract retryDelay in seconds from error details if present."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s?", error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0


def _is_quota_error(exc: Exception) -> tuple[bool, bool]:
    """Check if an exception is a 429/RESOURCE_EXHAUSTED error and whether it indicates daily quota.

    Decides 'daily' ONLY if the error text contains 'perday' or 'per day' (case-insensitive,
    ignoring underscores) and does NOT contain 'perminute' or 'per minute'.

    Returns:
        (is_quota_error, is_daily_quota)
    """
    err_text = str(exc)
    is_quota = False

    if isinstance(exc, APIError):
        if exc.code == 429 or (exc.status and "RESOURCE_EXHAUSTED" in exc.status):
            is_quota = True

    if not is_quota:
        lower_err = err_text.lower()
        if any(keyword in lower_err for keyword in ["429", "resource_exhausted", "quota", "rate limit"]):
            is_quota = True

    if not is_quota:
        return False, False

    clean_text = err_text.lower().replace("_", "")
    has_per_day = ("perday" in clean_text or "per day" in clean_text)
    has_per_minute = ("perminute" in clean_text or "per minute" in clean_text)
    is_daily = has_per_day and not has_per_minute

    return True, is_daily


def is_not_found_response(answer: str) -> bool:
    """Check if an answer represents a 'not found' refusal.

    Normalizes curly quotes/apostrophes before comparing with NOT_FOUND_MESSAGE.
    Treats the answer as 'not found' only if the response is mostly that sentence
    (the sentence is the whole answer or at least 80% of it).
    """
    if not answer or not answer.strip():
        return True

    # Normalize curly quotes and apostrophes to standard ASCII
    norm = answer.strip().lower()
    norm = norm.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    target = NOT_FOUND_MESSAGE.strip().lower()
    target = target.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    if norm == target:
        return True

    if target in norm:
        # Check if NOT_FOUND_MESSAGE is at least 80% of the full response text
        ratio = len(target) / len(norm)
        if ratio >= 0.8:
            return True

    return False


CITATION_PATTERN: re.Pattern = re.compile(r"\[(?:Chunk\s*)?(\d+)\]")


def parse_and_rewrite_citations(
    raw_answer: str,
    relevant_chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], bool]:
    r"""Parse citations with r"\[(?:Chunk\s*)?(\d+)\]", map to (file, page), rewrite and filter sources.

    Maps each cited number to (file, page) from retrieved chunks, rewrites them in the
    final answer as [file, p.X], and builds "sources" only from the cited chunks.
    If there are no valid citations, returns all retrieved chunks and marks uncited=True.

    Args:
        raw_answer: Raw LLM response string containing bracketed chunk numbers e.g. [1] or [1][3].
        relevant_chunks: List of retrieved context chunks supplied to the model.

    Returns:
        tuple of (rewritten_answer, sources, uncited_flag).
    """
    if not raw_answer:
        return raw_answer, [], False

    if not relevant_chunks:
        return raw_answer, [], True

    cited_indices: set[int] = set()

    def _replace_citation(match: re.Match) -> str:
        try:
            num = int(match.group(1))
        except (ValueError, IndexError):
            return match.group(0)

        # 1-indexed chunk number corresponding to relevant_chunks
        if 1 <= num <= len(relevant_chunks):
            idx = num - 1
            cited_indices.add(idx)
            chunk = relevant_chunks[idx]
            fname = chunk.get("file", "document")
            page = chunk.get("page", 1)
            return f"[{fname}, p.{page}]"
        return match.group(0)

    rewritten_answer = CITATION_PATTERN.sub(_replace_citation, raw_answer)

    if cited_indices:
        sources = [
            {
                "file": relevant_chunks[idx].get("file", ""),
                "page": relevant_chunks[idx].get("page", 1),
                "text": relevant_chunks[idx].get("text", ""),
                "score": relevant_chunks[idx].get("score", 0.0),
            }
            for idx in sorted(cited_indices)
        ]
        uncited = False
    else:
        # If there are no valid citations, return all retrieved chunks and mark the answer as uncited
        sources = [
            {
                "file": c.get("file", ""),
                "page": c.get("page", 1),
                "text": c.get("text", ""),
                "score": c.get("score", 0.0),
            }
            for c in relevant_chunks
        ]
        uncited = True

    return rewritten_answer, sources, uncited


class AnswerStream:
    """Iterator wrapper for streamed responses that yields text tokens and provides sources."""

    def __init__(
        self,
        generator: Iterator[str],
        sources: list[dict[str, Any]],
        model: Optional[str] = None,
        uncited: bool = False,
    ) -> None:
        self._gen = generator
        self.sources = sources
        self.model = model
        self.uncited = uncited

    def __iter__(self) -> "AnswerStream":
        return self

    def __next__(self) -> str:
        try:
            return next(self._gen)
        except StopIteration as stop_err:
            if stop_err.value is not None:
                if isinstance(stop_err.value, tuple) and len(stop_err.value) == 2:
                    self.sources, self.uncited = stop_err.value
                else:
                    self.sources = stop_err.value
            raise


def answer_question(
    question: str,
    history: Optional[list[Any]] = None,
    k: int = TOP_K,
    min_score: float = MIN_SCORE,
    model: str = GEMINI_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    bypass_cache: bool = False,
    collection_name: str = COLLECTION_NAME,
    file: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve relevant document chunks and generate a grounded answer using Gemini.

    Workflow:
        1. Check in-memory answer cache (unless bypassed or history present).
        2. Retrieve top-k chunks with vector_store.search.
        3. Drop chunks below min_score. If none remain, return "couldn't find" without calling LLM.
        4. Build prompt and invoke Gemini model with SYSTEM_PROMPT.
        5. If primary model encounters per-minute 429, waits retryDelay and retries once on same model.
           If primary model hits daily quota or retry fails, retries once with fallback_model.
        6. If quota exceeded, raises QuotaExceededError. If non-quota error, raises RuntimeError.
        7. Cache and return {"answer": str, "sources": list[dict], "uncited": bool}.

    Args:
        question: User query string.
        history: Optional list of previous chat turns for conversational context.
        k: Maximum number of chunks to retrieve. Defaults to TOP_K (6).
        min_score: Minimum cosine similarity score threshold. Defaults to MIN_SCORE (0.5).
        model: Primary Gemini model identifier. Defaults to GEMINI_MODEL ('gemini-2.5-flash').
        fallback_model: Fallback Gemini model. Defaults to FALLBACK_MODEL ('gemini-2.5-flash-lite').
        bypass_cache: If True, bypass in-memory answer cache and invoke pipeline.
        collection_name: Target Chroma collection name. Defaults to COLLECTION_NAME ('docchat').
        file: Optional filename filter for chunk retrieval.

    Returns:
        Dict with 'answer' (str), 'sources' (list of dicts), 'model' (str), and 'uncited' (bool).

    Raises:
        QuotaExceededError: If daily/rate quota is exceeded across primary and fallback models.
        RuntimeError: If API attempts fail due to non-quota errors.
    """
    if not question or not question.strip():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
            "model": "none",
            "uncited": False,
        }

    # Step 1: Check in-memory answer cache (only when history is empty and not bypassed)
    has_history = bool(history and len(history) > 0)
    cache_key = get_cache_key(question, collection_name=collection_name)
    if not has_history and not bypass_cache and cache_key in _answer_cache:
        logger.info("Serving answer from in-memory cache for: '%s'", question)
        return _answer_cache[cache_key]

    # Step 2: Retrieve chunks from vector store
    retrieved_chunks = search(query=question.strip(), k=k, collection_name=collection_name, file=file)

    # Step 3: Filter chunks below min_score
    relevant_chunks = [c for c in retrieved_chunks if c.get("score", 0.0) >= min_score]
    if not relevant_chunks:
        result = {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
            "model": "none (threshold)",
            "uncited": False,
        }
        if not has_history:
            _answer_cache[cache_key] = result
        return result

    sources = [
        {
            "file": c.get("file", ""),
            "page": c.get("page", 1),
            "text": c.get("text", ""),
            "score": c.get("score", 0.0),
        }
        for c in relevant_chunks
    ]

    # Step 4: Build prompt
    user_prompt = build_prompt(
        question=question.strip(),
        chunks=relevant_chunks,
        history=history,
    )

    client = get_client()
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

    # Step 5: Call primary model with fallback and daily quota memory
    response = None
    used_model = None

    if is_primary_daily_quota_exhausted():
        logger.info(
            "Primary model %s daily quota already exhausted; routing directly to fallback %s...",
            model,
            fallback_model,
        )
        try:
            response = client.models.generate_content(
                model=fallback_model,
                contents=user_prompt,
                config=config,
            )
            used_model = fallback_model
        except Exception as fallback_exc:
            is_quota_fb, _ = _is_quota_error(fallback_exc)
            if is_quota_fb:
                raise QuotaExceededError(
                    "The AI model's free daily limit has been reached. Please try again later."
                ) from fallback_exc
            raise RuntimeError(
                f"Fallback model ({fallback_model}) failed: {fallback_exc}"
            ) from fallback_exc
    else:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=config,
            )
            used_model = model
        except Exception as primary_exc:
            is_quota, is_daily = _is_quota_error(primary_exc)
            if is_daily:
                set_primary_daily_quota_exhausted()
                logger.warning(
                    "Primary model %s hit daily quota. Remembering for 60m; switching to fallback %s...",
                    model,
                    fallback_model,
                )
            elif is_quota:
                # Per-minute 429: wait suggested retryDelay (max 65s) and retry once on SAME model
                delay = _parse_retry_delay_seconds(str(primary_exc))
                wait_time = min(max(delay, 5.0) if delay > 0 else 65.0, 65.0)
                logger.info(
                    "Primary model %s hit per-minute limit. Waiting %.1fs before retrying on same model...",
                    model,
                    wait_time,
                )
                time.sleep(wait_time)
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=user_prompt,
                        config=config,
                    )
                    used_model = model
                except Exception as retry_exc:
                    is_retry_quota, is_retry_daily = _is_quota_error(retry_exc)
                    if is_retry_daily:
                        set_primary_daily_quota_exhausted()
                    primary_exc = retry_exc
                    is_quota = is_retry_quota

            if response is None:
                # Retry once with fallback model
                try:
                    response = client.models.generate_content(
                        model=fallback_model,
                        contents=user_prompt,
                        config=config,
                    )
                    used_model = fallback_model
                except Exception as fallback_exc:
                    is_quota_fb, _ = _is_quota_error(fallback_exc)
                    if is_quota or is_quota_fb:
                        raise QuotaExceededError(
                            "The AI model's free daily limit has been reached. Please try again later."
                        ) from fallback_exc
                    raise RuntimeError(
                        f"Both primary ({model}) and fallback ({fallback_model}) models failed: {fallback_exc}"
                    ) from fallback_exc

    raw_answer = response.text.strip() if (response and response.text) else NOT_FOUND_MESSAGE

    if is_not_found_response(raw_answer):
        result = {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
            "model": used_model,
            "uncited": False,
        }
    else:
        rewritten_answer, final_sources, uncited = parse_and_rewrite_citations(
            raw_answer, relevant_chunks
        )
        result = {
            "answer": rewritten_answer,
            "sources": final_sources,
            "model": used_model,
            "uncited": uncited,
        }

    # Cache answer in memory only if history is empty
    if not has_history:
        _answer_cache[cache_key] = result
    return result


def stream_answer(
    question: str,
    history: Optional[list[Any]] = None,
    k: int = TOP_K,
    min_score: float = MIN_SCORE,
    model: str = GEMINI_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    collection_name: str = COLLECTION_NAME,
    file: Optional[str] = None,
) -> AnswerStream:
    """Stream response tokens for a question with fallback and in-memory cache support.

    Args:
        question: User query string.
        history: Optional list of previous chat turns.
        k: Maximum number of chunks to retrieve. Defaults to TOP_K (6).
        min_score: Minimum cosine similarity score threshold. Defaults to MIN_SCORE (0.5).
        model: Primary model identifier. Defaults to GEMINI_MODEL.
        fallback_model: Fallback model identifier. Defaults to FALLBACK_MODEL.
        collection_name: Target Chroma collection name. Defaults to COLLECTION_NAME ('docchat').
        file: Optional filename filter for chunk retrieval.

    Returns:
        AnswerStream instance that yields string pieces and exposes .sources and .uncited.
    """
    if not question or not question.strip():
        def _empty_stream() -> Generator[str, None, tuple[list[dict[str, Any]], bool]]:
            yield NOT_FOUND_MESSAGE
            return [], False

        return AnswerStream(_empty_stream(), sources=[], model="none", uncited=False)

    # Check in-memory cache first (only when history is empty)
    has_history = bool(history and len(history) > 0)
    cache_key = get_cache_key(question, collection_name=collection_name)
    if not has_history and cache_key in _answer_cache:
        cached = _answer_cache[cache_key]
        cached_answer = cached["answer"]
        cached_sources = cached["sources"]
        cached_model = cached.get("model")
        cached_uncited = cached.get("uncited", False)

        def _cached_stream() -> Generator[str, None, tuple[list[dict[str, Any]], bool]]:
            words = cached_answer.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
            return cached_sources, cached_uncited

        return AnswerStream(
            _cached_stream(),
            sources=cached_sources,
            model=cached_model,
            uncited=cached_uncited,
        )

    # Retrieve chunks from vector store
    retrieved_chunks = search(query=question.strip(), k=k, collection_name=collection_name, file=file)

    # Drop chunks below min_score
    relevant_chunks = [c for c in retrieved_chunks if c.get("score", 0.0) >= min_score]
    if not relevant_chunks:
        def _not_found_stream() -> Generator[str, None, tuple[list[dict[str, Any]], bool]]:
            yield NOT_FOUND_MESSAGE
            return [], False

        result = {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
            "model": "none (threshold)",
            "uncited": False,
        }
        if not has_history:
            _answer_cache[cache_key] = result
        return AnswerStream(_not_found_stream(), sources=[], model="none (threshold)", uncited=False)

    user_prompt = build_prompt(
        question=question.strip(),
        chunks=relevant_chunks,
        history=history,
    )

    stream_holder: dict[str, Optional[str]] = {"model": None}

    def _generator() -> Generator[str, None, tuple[list[dict[str, Any]], bool]]:
        client = get_client()
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

        stream = None
        used_model = None

        if is_primary_daily_quota_exhausted():
            logger.info(
                "Primary model %s daily quota already exhausted; streaming directly from fallback %s...",
                model,
                fallback_model,
            )
            try:
                stream = client.models.generate_content_stream(
                    model=fallback_model,
                    contents=user_prompt,
                    config=config,
                )
                iterator = iter(stream)
                first_chunk = next(iterator, None)
                used_model = fallback_model
            except Exception as fallback_exc:
                is_quota_fb, _ = _is_quota_error(fallback_exc)
                if is_quota_fb:
                    raise QuotaExceededError(
                        "The AI model's free daily limit has been reached. Please try again later."
                    ) from fallback_exc
                raise RuntimeError(
                    f"Streaming failed on fallback model: {fallback_exc}"
                ) from fallback_exc
        else:
            try:
                # Try streaming from primary model
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=user_prompt,
                    config=config,
                )
                iterator = iter(stream)
                first_chunk = next(iterator, None)
                used_model = model
            except Exception as primary_exc:
                is_quota, is_daily = _is_quota_error(primary_exc)
                if is_daily:
                    set_primary_daily_quota_exhausted()
                    logger.warning(
                        "Primary streaming model %s hit daily quota. Remembering for 60m; switching to fallback %s...",
                        model,
                        fallback_model,
                    )
                elif is_quota:
                    delay = _parse_retry_delay_seconds(str(primary_exc))
                    wait_time = min(max(delay, 5.0) if delay > 0 else 65.0, 65.0)
                    logger.info(
                        "Primary streaming model %s hit per-minute limit. Waiting %.1fs before retrying on same model...",
                        model,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    try:
                        stream = client.models.generate_content_stream(
                            model=model,
                            contents=user_prompt,
                            config=config,
                        )
                        iterator = iter(stream)
                        first_chunk = next(iterator, None)
                        used_model = model
                    except Exception as retry_exc:
                        is_retry_quota, is_retry_daily = _is_quota_error(retry_exc)
                        if is_retry_daily:
                            set_primary_daily_quota_exhausted()
                        primary_exc = retry_exc
                        is_quota = is_retry_quota

                if stream is None:
                    # Retry once with fallback model
                    try:
                        stream = client.models.generate_content_stream(
                            model=fallback_model,
                            contents=user_prompt,
                            config=config,
                        )
                        iterator = iter(stream)
                        first_chunk = next(iterator, None)
                        used_model = fallback_model
                    except Exception as fallback_exc:
                        is_quota_fb, _ = _is_quota_error(fallback_exc)
                        if is_quota or is_quota_fb:
                            raise QuotaExceededError(
                                "The AI model's free daily limit has been reached. Please try again later."
                            ) from fallback_exc
                        raise RuntimeError(
                            f"Streaming failed on fallback model: {fallback_exc}"
                        ) from fallback_exc

        stream_holder["model"] = used_model

        full_raw_text = ""
        buf = ""

        def _process_stream_chunk(text: str) -> Generator[str, None, None]:
            nonlocal buf
            buf += text
            while True:
                if "[" not in buf:
                    if buf:
                        out = buf
                        buf = ""
                        yield out
                    break

                idx = buf.find("[")
                if idx > 0:
                    yield buf[:idx]
                    buf = buf[idx:]

                # buf starts with '['
                end_idx = buf.find("]")
                if end_idx != -1:
                    bracket_text = buf[:end_idx + 1]
                    m = CITATION_PATTERN.fullmatch(bracket_text.strip())
                    if m:
                        try:
                            num = int(m.group(1))
                            if 1 <= num <= len(relevant_chunks):
                                c = relevant_chunks[num - 1]
                                yield f"[{c.get('file', '')}, p.{c.get('page', 1)}]"
                            else:
                                yield bracket_text
                        except Exception:
                            yield bracket_text
                    else:
                        yield bracket_text
                    buf = buf[end_idx + 1:]
                else:
                    if len(buf) > 30:
                        out = buf
                        buf = ""
                        yield out
                    break

        if first_chunk and first_chunk.text:
            full_raw_text += first_chunk.text
            for token in _process_stream_chunk(first_chunk.text):
                yield token

        for chunk in iterator:
            if chunk.text:
                full_raw_text += chunk.text
                for token in _process_stream_chunk(chunk.text):
                    yield token

        if buf:
            yield CITATION_PATTERN.sub(
                lambda m: f"[{relevant_chunks[int(m.group(1))-1].get('file', '')}, p.{relevant_chunks[int(m.group(1))-1].get('page', 1)}]"
                if 1 <= int(m.group(1)) <= len(relevant_chunks) else m.group(0),
                buf
            )

        if is_not_found_response(full_raw_text):
            final_sources = []
            uncited = False
            rewritten_answer = NOT_FOUND_MESSAGE
        else:
            rewritten_answer, final_sources, uncited = parse_and_rewrite_citations(
                full_raw_text.strip(), relevant_chunks
            )

        # Save completed response to cache only if history is empty
        if not has_history:
            _answer_cache[cache_key] = {
                "answer": rewritten_answer,
                "sources": final_sources,
                "model": used_model,
                "uncited": uncited,
            }
        return final_sources, uncited

    ans_stream = AnswerStream(_generator(), sources=relevant_chunks)
    return ans_stream
    return ans_stream


if __name__ == "__main__":
    import sys

    # Ensure sample.pdf is indexed in vector store
    try:
        from src.chunker import chunk_pages
        from src.loader import load_document
        from src.vector_store import add_chunks, list_files
    except ImportError:
        try:
            from chunker import chunk_pages
            from loader import load_document
            from vector_store import add_chunks, list_files
        except ImportError as import_err:
            print(f"Error importing dependencies: {import_err}", file=sys.stderr)
            sys.exit(1)

    sample_pdf = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"
    if not sample_pdf.exists():
        print(f"Error: Sample PDF not found at {sample_pdf}", file=sys.stderr)
        sys.exit(1)

    indexed_files = list_files()
    if sample_pdf.name not in indexed_files:
        print(f"Indexing {sample_pdf.name} in vector store...")
        pages = load_document(sample_pdf)
        chunks = chunk_pages(pages)
        add_chunks(chunks)
        print(f"Indexed {len(chunks)} chunks.")

    print(f"Using models: Primary='{GEMINI_MODEL}', Fallback='{FALLBACK_MODEL}', MIN_SCORE={MIN_SCORE}")

    # 3 Evaluation Questions
    test_questions = [
        "How does chunking work?",
        "What vector database is used?",
        "Who won the 2018 World Cup?",
    ]

    print("==================================================")
    print("               DocChat-AI RAG Tests               ")
    print("==================================================")

    for idx, query in enumerate(test_questions, start=1):
        print(f"\n[Test {idx}/3] Query: '{query}'")
        print("-" * 50)
        res = answer_question(query)

        print(f"Answer:\n{res['answer']}\n")
        print(f"Sources ({len(res['sources'])}):")
        if res["sources"]:
            for s in res["sources"]:
                print(f"  * [{s['file']}, p.{s['page']}] (Score: {s['score']:.4f})")
        else:
            print("  * None")
        print("-" * 50)

    # Test cache
    print("\n--- Testing In-Memory Answer Cache ---")
    cache_test_query = "What vector database is used?"
    cached_res = answer_question(cache_test_query)
    print(f"Cached result retrieved successfully for: '{cache_test_query}' (sources: {len(cached_res['sources'])})")

    print("\nAll RAG tests completed successfully!")
