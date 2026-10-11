"""Evaluation runner for DocChat-AI.

Supports:
1. --retrieval-only: Evaluates Hit@K, MRR, and threshold filtering without LLM calls.
2. Full RAG pipeline: Evaluates answer accuracy, citation correctness, and refusal rate.
Includes resume support, rate limit backoff, progress checkpoints, and REPORT.md generation.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional

# Add project root to sys.path so 'src' can be imported cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chunker import chunk_pages
from src.config import FALLBACK_MODEL, GEMINI_MODEL, MIN_SCORE, TOP_K
from src.loader import load_document
from src.rag import QuotaExceededError, answer_question, is_primary_daily_quota_exhausted
from src.vector_store import add_chunks, list_files, search

EVAL_DIR = project_root / "eval"
DEFAULT_QUESTIONS_FILE = EVAL_DIR / "questions.json"
DEFAULT_PDF_FILE = project_root / "data" / "sample.pdf"


def ensure_index_ready(pdf_path: Path = DEFAULT_PDF_FILE) -> None:
    """Ensure the specified PDF is loaded and indexed in the vector store."""
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    indexed = list_files()
    if pdf_path.name not in indexed:
        print(f"Indexing {pdf_path.name} in vector store...")
        pages = load_document(pdf_path)
        chunks = chunk_pages(pages)
        add_chunks(chunks)
        print(f"Indexed {len(chunks)} chunks.")


def parse_citations(text: str) -> list[tuple[str, int]]:
    """Parse citations like [sample.pdf, p.2] from answer text."""
    pattern = r"([a-zA-Z0-9_\-\. ]+?)\s*,\s*p\.?\s*(\d+)"
    citations: list[tuple[str, int]] = []
    seen = set()

    for match in re.finditer(pattern, text):
        fname = match.group(1).strip().lstrip("[").rstrip("]").strip()
        try:
            page = int(match.group(2))
        except ValueError:
            continue
        if fname and (fname, page) not in seen:
            seen.add((fname, page))
            citations.append((fname, page))

    return citations


def extract_cited_pages(text: str) -> list[int]:
    """Extract all cited page numbers from answer text."""
    pages: list[int] = []
    # Match bracketed citations e.g. [sample.pdf, p.1, p.2] or [sample.pdf, p.4]
    bracket_matches = re.findall(r"\[([^\]]+)\]", text)
    for bracket_content in bracket_matches:
        for p_match in re.finditer(r"\b(?:p\.?|page)\s*(\d+)\b", bracket_content, re.IGNORECASE):
            try:
                pages.append(int(p_match.group(1)))
            except ValueError:
                pass
    # If none found in brackets, check any p.X / page X pattern in text
    if not pages:
        for p_match in re.finditer(r"\b(?:p\.?|page)\s*(\d+)\b", text, re.IGNORECASE):
            try:
                pages.append(int(p_match.group(1)))
            except ValueError:
                pass
    return sorted(list(set(pages)))


_pdf_pages_cache: dict[str, dict[int, str]] = {}


def get_pdf_pages_text(pdf_path: Optional[Path] = None) -> dict[int, str]:
    """Retrieve mapping of page number -> extracted text for the specified PDF."""
    if pdf_path is None:
        pdf_path = DEFAULT_PDF_FILE
    pdf_key = str(pdf_path.resolve())
    if pdf_key not in _pdf_pages_cache:
        if pdf_path.exists():
            pages = load_document(pdf_path)
            _pdf_pages_cache[pdf_key] = {p["page"]: p["text"] for p in pages}
        else:
            _pdf_pages_cache[pdf_key] = {}
    return _pdf_pages_cache[pdf_key]


def normalize_match_text(text: str) -> str:
    """Normalize text for substring matching:
    1. Lowercase
    2. Replace newlines and runs of whitespace with a single space
    3. Remove spaces around hyphens ("real -time" -> "real-time")
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()


def text_contains_any(haystack: str, needles: list[str]) -> bool:
    """Check if normalized haystack contains any normalized needle string."""
    if not haystack or not needles:
        return False
    norm_haystack = normalize_match_text(haystack)
    for needle in needles:
        norm_needle = normalize_match_text(needle)
        if norm_needle and norm_needle in norm_haystack:
            return True
    return False


def check_citation(
    answer: str,
    expected_any: list[str],
    pdf_pages: Optional[dict[int, str]] = None,
    pdf_path: Optional[Path] = None,
) -> bool:
    """Check whether ANY cited page's text (from the PDF pages) contains one of expected_any."""
    if not expected_any:
        return False
    cited = extract_cited_pages(answer)
    if not cited:
        return False
    if pdf_pages is None:
        pdf_pages = get_pdf_pages_text(pdf_path)
    for page_num in cited:
        page_text = pdf_pages.get(page_num, "")
        if text_contains_any(page_text, expected_any):
            return True
    return False


def check_exact_page_citation(answer: str, expected_page: Optional[int]) -> bool:
    """Check whether expected_page is cited in the answer."""
    if not expected_page:
        return False
    cited = extract_cited_pages(answer)
    if expected_page in cited:
        return True
    pattern = rf"(?:\[.*?\b)?(?:p\.?|page)\s*{expected_page}\b"
    return bool(re.search(pattern, answer, re.IGNORECASE))


def load_questions(questions_path: Path = DEFAULT_QUESTIONS_FILE) -> list[dict[str, Any]]:
    """Load evaluation questions from the specified JSON file."""
    if not questions_path.exists():
        print(f"Error: Questions file not found at {questions_path}", file=sys.stderr)
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load previously saved results from results_path if present."""
    if not results_path.exists():
        # Fallback to legacy results.json if evaluating default questions and results_questions.json does not yet exist
        if results_path.name == "results_questions.json":
            legacy_file = results_path.parent / "results.json"
            if legacy_file.exists():
                try:
                    with open(legacy_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return []
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_results(results: list[dict[str, Any]], results_path: Path) -> None:
    """Save results atomically to results_path."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = results_path.parent / f"{results_path.name}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    temp_file.replace(results_path)


def backfill_record_fields(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    pdf_path: Path,
    results_path: Path,
) -> list[dict[str, Any]]:
    """Ensure older records have chunk_hit_at_k, model, cited_pages, and citation check fields populated."""
    q_map = {q["id"]: q for q in questions}
    pdf_pages = get_pdf_pages_text(pdf_path)
    modified = False

    for r in results:
        qid = r.get("id")
        q_item = q_map.get(qid)

        # 1. Backfill model
        if "model" not in r:
            modified = True
            if r.get("mode") == "retrieval_only":
                r["model"] = "none (retrieval-only)"
            else:
                if "couldn't find" in r.get("actual_answer", "").lower() and r.get("top_score", 0.0) < MIN_SCORE:
                    r["model"] = "none (threshold)"
                else:
                    r["model"] = GEMINI_MODEL

        # 2. Backfill cited_pages
        if r.get("mode") == "full" and "cited_pages" not in r:
            modified = True
            r["cited_pages"] = extract_cited_pages(r.get("actual_answer", ""))

        # 3. Backfill/update chunk_hit_at_k for answerable questions
        if r.get("type") == "answerable":
            exp_any = r.get("expected_any") or (q_item.get("expected_any") if q_item else [])
            q_text = (q_item.get("question") if q_item else r.get("question", ""))
            if exp_any and q_text:
                retrieved = search(q_text, k=TOP_K, file=pdf_path.name if pdf_path else None)
                new_chunk_hit = any(
                    text_contains_any(c.get("text", ""), exp_any)
                    for c in retrieved
                )
                if r.get("chunk_hit_at_k") != new_chunk_hit:
                    modified = True
                    r["chunk_hit_at_k"] = new_chunk_hit

                if r.get("mode") == "retrieval_only":
                    if r.get("is_correct") != new_chunk_hit:
                        modified = True
                        r["is_correct"] = new_chunk_hit
                    if "retrieved_chunks" not in r:
                        modified = True
                        r["retrieved_chunks"] = [
                            {
                                "page": c.get("page", 1),
                                "score": round(float(c.get("score", 0.0)), 4),
                                "text": c.get("text", "")[:100],
                            }
                            for c in retrieved
                        ]
            elif "chunk_hit_at_k" not in r:
                r["chunk_hit_at_k"] = False

        # 4. Re-evaluate citation and answer metrics according to updated requirements
        if r.get("mode") == "full" and r.get("type") == "answerable":
            exp_any = r.get("expected_any") or (q_item.get("expected_any") if q_item else [])
            exp_page = r.get("expected_page") or (q_item.get("expected_page") if q_item else None)
            act_ans = r.get("actual_answer", "")

            new_cite_correct = check_citation(act_ans, exp_any, pdf_pages=pdf_pages)
            new_cite_exact = check_exact_page_citation(act_ans, exp_page)
            new_ans_correct = text_contains_any(act_ans, exp_any)
            if (
                r.get("citation_correct") != new_cite_correct
                or r.get("citation_exact_page") != new_cite_exact
                or r.get("answer_correct") != new_ans_correct
                or r.get("is_correct") != new_ans_correct
            ):
                modified = True
                r["citation_correct"] = new_cite_correct
                r["citation_exact_page"] = new_cite_exact
                r["answer_correct"] = new_ans_correct
                r["is_correct"] = new_ans_correct

    if modified:
        save_results(results, results_path)
    return results


def debug_question(
    qid: str,
    all_questions: list[dict[str, Any]],
    pdf_path: Optional[Path] = None,
) -> None:
    """Debug a specific question ID: print top-k retrieved chunks, threshold status, and final answer."""
    target_q = None
    for q in all_questions:
        if q["id"].lower() == qid.strip().lower():
            target_q = q
            break

    if not target_q:
        print(f"Error: Question ID '{qid}' not found in questions.json", file=sys.stderr)
        sys.exit(1)

    q_id = target_q["id"]
    q_type = target_q.get("type", "unknown")
    question = target_q["question"]
    expected_any = target_q.get("expected_any", [])
    expected_page = target_q.get("expected_page")

    print("\n" + "=" * 70)
    print(f"                      DEBUGGING QUESTION: {q_id}")
    print("=" * 70)
    print(f"Question      : {question}")
    print(f"Type          : {q_type}")
    print(f"Expected Any  : {expected_any}")
    print(f"Expected Page : {expected_page}")
    print(f"MIN_SCORE     : {MIN_SCORE}")
    print(f"TOP_K         : {TOP_K}")
    print("-" * 70)

    file_filter = pdf_path.name if pdf_path else None

    # Retrieve top-k chunks
    retrieved_chunks = search(question, k=TOP_K, file=file_filter)
    print(f"\nRetrieved Chunks ({len(retrieved_chunks)}):")

    chunk_hit_found = False
    passed_chunks_count = 0

    for idx, c in enumerate(retrieved_chunks, start=1):
        c_page = c.get("page", 1)
        c_score = c.get("score", 0.0)
        c_text = c.get("text", "")
        c_snippet = c_text[:150].replace("\n", " ")

        passed_min = c_score >= MIN_SCORE
        if passed_min:
            passed_chunks_count += 1

        matches_expected = text_contains_any(c_text, expected_any) if expected_any else False
        if matches_expected:
            chunk_hit_found = True

        print(f"\n  [Chunk {idx}]")
        print(f"    Page                  : {c_page}")
        print(f"    Score                 : {c_score:.4f}")
        print(f"    Passed MIN_SCORE      : {'YES' if passed_min else 'NO'} ({c_score:.4f} >= {MIN_SCORE})")
        print(f"    Contains expected_any : {'YES' if matches_expected else 'NO'}")
        print(f"    First 150 chars       : {c_snippet!r}")

    # Retrieval diagnostic summary
    page_hit = (expected_page in [c.get("page") for c in retrieved_chunks]) if expected_page else False
    print("\n" + "-" * 70)
    print("Retrieval Diagnostic:")
    print(f"  Page Hit@{TOP_K}  : {'YES' if page_hit else 'NO'}")
    print(f"  Chunk Hit@{TOP_K} : {'YES' if chunk_hit_found else 'NO'}")
    print(f"  Passed Threshold: {passed_chunks_count}/{len(retrieved_chunks)} chunk(s)")

    # Final answer from RAG pipeline
    print("-" * 70)
    print("Running Full RAG Pipeline...")
    rag_res = answer_question(question, bypass_cache=True, file=file_filter)
    final_answer = rag_res.get("answer", "")
    sources = rag_res.get("sources", [])
    model_used = rag_res.get("model", "N/A")

    print(f"Model Used       : {model_used}")
    print(f"Sources Returned : {len(sources)}")
    if sources:
        for s in sources:
            print(f"  * [{s.get('file')}, p.{s.get('page')}] (Score: {s.get('score', 0):.4f})")
    print(f"\nFinal Answer:\n{final_answer}")
    print("=" * 70 + "\n")


def evaluate_retrieval(question_item: dict[str, Any], pdf_path: Optional[Path] = None) -> dict[str, Any]:
    """Evaluate retrieval without calling the LLM."""
    q_id = question_item["id"]
    q_type = question_item["type"]
    question = question_item["question"]
    expected_page = question_item.get("expected_page")
    expected_any = question_item.get("expected_any", [])
    file_filter = pdf_path.name if pdf_path else None

    start_time = time.time()
    results = search(question, k=TOP_K, file=file_filter)
    latency = time.time() - start_time

    top_score = results[0]["score"] if results else 0.0
    retrieved_pages = [r.get("page") for r in results]
    retrieved_chunk_summaries = [
        {
            "page": r.get("page", 1),
            "score": round(float(r.get("score", 0.0)), 4),
            "text": r.get("text", "")[:100],
        }
        for r in results
    ]

    if q_type == "answerable":
        hit_at_k = expected_page in retrieved_pages if expected_page else False
        chunk_hit_at_k = any(
            text_contains_any(r.get("text", ""), expected_any)
            for r in results
        ) if expected_any else False

        rank = None
        reciprocal_rank = 0.0
        if expected_page:
            for idx, p in enumerate(retrieved_pages, start=1):
                if p == expected_page:
                    rank = idx
                    reciprocal_rank = 1.0 / idx
                    break

        is_correct = chunk_hit_at_k
        return {
            "id": q_id,
            "type": q_type,
            "question": question,
            "mode": "retrieval_only",
            "model": "none (retrieval-only)",
            "expected_page": expected_page,
            "expected_any": expected_any,
            "retrieved_pages": retrieved_pages,
            "retrieved_chunks": retrieved_chunk_summaries,
            "hit_at_k": hit_at_k,
            "chunk_hit_at_k": chunk_hit_at_k,
            "first_correct_rank": rank,
            "reciprocal_rank": reciprocal_rank,
            "top_score": top_score,
            "is_correct": is_correct,
            "latency_seconds": round(latency, 3),
            "timestamp": datetime.now().isoformat(),
        }
    else:
        # Unanswerable: expected to be blocked by MIN_SCORE threshold
        blocked_by_threshold = (top_score < MIN_SCORE)
        is_correct = blocked_by_threshold
        return {
            "id": q_id,
            "type": q_type,
            "question": question,
            "mode": "retrieval_only",
            "model": "none (retrieval-only)",
            "top_score": top_score,
            "retrieved_chunks": retrieved_chunk_summaries,
            "blocked_by_threshold": blocked_by_threshold,
            "is_correct": is_correct,
            "latency_seconds": round(latency, 3),
            "timestamp": datetime.now().isoformat(),
        }


def evaluate_full(question_item: dict[str, Any], pdf_path: Optional[Path] = None) -> dict[str, Any]:
    """Evaluate full RAG pipeline with LLM answer and citation generation."""
    q_id = question_item["id"]
    q_type = question_item["type"]
    question = question_item["question"]
    expected_any = question_item.get("expected_any", [])
    expected_page = question_item.get("expected_page")
    file_filter = pdf_path.name if pdf_path else None

    start_time = time.time()
    # Evaluate retrieval for hit@k, chunk_hit@k, and MRR tracking
    retrieval_res = search(question, k=TOP_K, file=file_filter)
    retrieved_pages = [r.get("page") for r in retrieval_res]
    top_score = retrieval_res[0]["score"] if retrieval_res else 0.0

    hit_at_k = False
    chunk_hit_at_k = False
    rank = None
    reciprocal_rank = 0.0
    if expected_page:
        hit_at_k = expected_page in retrieved_pages
        for idx, p in enumerate(retrieved_pages, start=1):
            if p == expected_page:
                rank = idx
                reciprocal_rank = 1.0 / idx
                break

    if expected_any:
        chunk_hit_at_k = any(
            text_contains_any(r.get("text", ""), expected_any)
            for r in retrieval_res
        )

    # Call RAG pipeline (bypassing answer cache)
    rag_res = answer_question(question, bypass_cache=True, file=file_filter)
    latency = time.time() - start_time

    answer = rag_res.get("answer", "")
    sources = rag_res.get("sources", [])
    model_used = rag_res.get("model")
    if not model_used:
        model_used = FALLBACK_MODEL if is_primary_daily_quota_exhausted() else GEMINI_MODEL
    cited_pages = extract_cited_pages(answer)

    if q_type == "answerable":
        # Check if any expected string appears in answer (normalized)
        answer_correct = text_contains_any(answer, expected_any) if expected_any else False
        pdf_pages = get_pdf_pages_text(pdf_path)
        citation_correct = check_citation(answer, expected_any, pdf_pages=pdf_pages)
        citation_exact_page = check_exact_page_citation(answer, expected_page)
        is_correct = answer_correct

        return {
            "id": q_id,
            "type": q_type,
            "question": question,
            "mode": "full",
            "model": model_used,
            "expected_any": expected_any,
            "expected_page": expected_page,
            "actual_answer": answer,
            "cited_pages": cited_pages,
            "answer_correct": answer_correct,
            "citation_correct": citation_correct,
            "citation_exact_page": citation_exact_page,
            "is_correct": is_correct,
            "hit_at_k": hit_at_k,
            "chunk_hit_at_k": chunk_hit_at_k,
            "first_correct_rank": rank,
            "reciprocal_rank": reciprocal_rank,
            "top_score": top_score,
            "latency_seconds": round(latency, 3),
            "timestamp": datetime.now().isoformat(),
        }
    else:
        # Unanswerable: correct if answer contains "couldn't find"
        refusal_correct = "couldn't find" in answer.lower() or "could not find" in answer.lower()
        is_correct = refusal_correct

        return {
            "id": q_id,
            "type": q_type,
            "question": question,
            "mode": "full",
            "model": model_used,
            "actual_answer": answer,
            "cited_pages": cited_pages,
            "refusal_correct": refusal_correct,
            "is_correct": is_correct,
            "top_score": top_score,
            "latency_seconds": round(latency, 3),
            "timestamp": datetime.now().isoformat(),
        }


def print_summary(results: list[dict[str, Any]], mode: str) -> None:
    """Print evaluation report metrics, per-model breakdown, citation failures, and failed questions."""
    mode_results = [r for r in results if r.get("mode") == mode]
    if not mode_results:
        print("No evaluation records found to display.")
        return

    answerable = [r for r in mode_results if r.get("type") == "answerable"]
    unanswerable = [r for r in mode_results if r.get("type") == "unanswerable"]

    total = len(mode_results)
    total_correct = sum(1 for r in mode_results if r.get("is_correct"))
    overall_acc = (total_correct / total) * 100 if total else 0.0

    avg_latency = sum(r.get("latency_seconds", 0.0) for r in mode_results) / total if total else 0.0

    print("\n" + "=" * 70)
    print(f"            DocChat-AI Evaluation Summary [{mode.upper()}]")
    print("=" * 70)
    print(f"Total Evaluated Questions : {total}")
    print(f"Overall Accuracy          : {overall_acc:.1f}% ({total_correct}/{total})")
    print(f"Average Latency           : {avg_latency:.2f}s")

    hits = [r.get("hit_at_k") for r in answerable if "hit_at_k" in r]
    chunk_hits = [r.get("chunk_hit_at_k") for r in answerable if "chunk_hit_at_k" in r]
    mrrs = [r.get("reciprocal_rank", 0.0) for r in answerable if "reciprocal_rank" in r]
    hit_at_k_pct = (sum(1 for h in hits if h) / len(hits)) * 100 if hits else 0.0
    chunk_hit_pct = (sum(1 for ch in chunk_hits if ch) / len(chunk_hits)) * 100 if chunk_hits else 0.0
    mrr_score = sum(mrrs) / len(mrrs) if mrrs else 0.0

    if mode == "full":
        ans_correct = sum(1 for r in answerable if r.get("answer_correct"))
        ans_acc = (ans_correct / len(answerable)) * 100 if answerable else 0.0
        cite_correct = sum(1 for r in answerable if r.get("citation_correct"))
        cite_acc = (cite_correct / len(answerable)) * 100 if answerable else 0.0
        exact_cite_correct = sum(1 for r in answerable if r.get("citation_exact_page"))
        exact_cite_acc = (exact_cite_correct / len(answerable)) * 100 if answerable else 0.0
        ref_correct = sum(1 for r in unanswerable if r.get("refusal_correct"))
        ref_rate = (ref_correct / len(unanswerable)) * 100 if unanswerable else 0.0

        print(f"Answerable Accuracy       : {ans_acc:.1f}% ({ans_correct}/{len(answerable)})")
        print(f"Citation Accuracy (Valid) : {cite_acc:.1f}% ({cite_correct}/{len(answerable)})")
        print(f"Exact Page Citation Match : {exact_cite_acc:.1f}% ({exact_cite_correct}/{len(answerable)})")
        print(f"Unanswerable Refusal Rate : {ref_rate:.1f}% ({ref_correct}/{len(unanswerable)})")
        print(f"Chunk Hit@{TOP_K} (Headline)       : {chunk_hit_pct:.1f}% ({sum(1 for ch in chunk_hits if ch)}/{len(chunk_hits)})")
        print(f"Page Hit@{TOP_K} (Secondary)      : {hit_at_k_pct:.1f}% ({sum(1 for h in hits if h)}/{len(hits)})")
        print(f"MRR (Mean Reciprocal Rank): {mrr_score:.4f}")
    else:
        # Retrieval-only mode metrics
        chunk_hits_count = sum(1 for r in answerable if r.get("chunk_hit_at_k"))
        chunk_hit_rate = (chunk_hits_count / len(answerable)) * 100 if answerable else 0.0
        page_hits_count = sum(1 for r in answerable if r.get("hit_at_k"))
        page_hit_rate = (page_hits_count / len(answerable)) * 100 if answerable else 0.0
        blocked_count = sum(1 for r in unanswerable if r.get("blocked_by_threshold"))
        blocked_rate = (blocked_count / len(unanswerable)) * 100 if unanswerable else 0.0

        print(f"Chunk Hit@{TOP_K} (Headline)       : {chunk_hit_rate:.1f}% ({chunk_hits_count}/{len(answerable)})")
        print(f"Page Hit@{TOP_K} (Secondary)      : {page_hit_rate:.1f}% ({page_hits_count}/{len(answerable)})")
        print(f"MRR (Mean Reciprocal Rank): {mrr_score:.4f}")
        print(f"Blocked by threshold (not the real refusal rate): {blocked_rate:.1f}% ({blocked_count}/{len(unanswerable)})")
        print("  * Note: Near-topic unanswerable questions are expected to pass the threshold and must be evaluated in full mode.")

    # Per-Model Breakdown
    print("-" * 70)
    print("Per-Model Breakdown:")
    models_seen: dict[str, dict[str, int]] = {}
    for r in mode_results:
        m = r.get("model", "unknown")
        if m not in models_seen:
            models_seen[m] = {"total": 0, "correct": 0}
        models_seen[m]["total"] += 1
        if r.get("is_correct"):
            models_seen[m]["correct"] += 1

    for m, stats in sorted(models_seen.items()):
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0.0
        print(f"  * {m:24s} : {stats['total']:2d} queries | Accuracy: {acc:5.1f}% ({stats['correct']}/{stats['total']})")

    # Full-mode Citation Failures
    if mode == "full":
        citation_failures = [r for r in answerable if not r.get("citation_correct")]
        print("-" * 70)
        print(f"Citation Failures ({len(citation_failures)}):")
        if not citation_failures:
            print("  None! All answerable questions cited pages containing expected answers.")
        else:
            for idx, cf in enumerate(citation_failures, start=1):
                cited = cf.get("cited_pages")
                if cited is None:
                    cited = extract_cited_pages(cf.get("actual_answer", ""))
                cited_str = ", ".join(f"p.{p}" for p in cited) if cited else "None"
                print(f"  [{idx}] ID: {cf['id']} | Expected: p.{cf.get('expected_page')} | Actually Cited: {cited_str}")
                print(f"      Question: {cf['question']}")
                ans_preview = cf.get("actual_answer", "").replace("\n", " ")[:120]
                print(f"      Answer  : {ans_preview}...")

    # List Failed Questions
    failed = [r for r in mode_results if not r.get("is_correct")]
    print("-" * 70)
    print(f"Failed Questions ({len(failed)}):")
    if not failed:
        print("  None! All questions passed successfully.")
    else:
        for idx, f in enumerate(failed, start=1):
            print(f"  [{idx}] ID: {f['id']} | Type: {f['type']}")
            print(f"      Question: {f['question']}")
            if f["type"] == "answerable":
                expected = f.get("expected_any") or f"Page {f.get('expected_page')}"
                print(f"      Expected Any    : {expected}")
                if mode == "retrieval_only":
                    print(f"      Chunk Hit@{TOP_K}     : NO (Missed at chunk level)")
                    print(f"      Page Hit@{TOP_K}      : {'YES' if f.get('hit_at_k') else 'NO'}")
                    print(f"      Top Score       : {f.get('top_score')}")

                    retrieved_chunks = f.get("retrieved_chunks")
                    if not retrieved_chunks:
                        try:
                            fetched = search(f["question"], k=TOP_K)
                            retrieved_chunks = [
                                {
                                    "page": c.get("page", 1),
                                    "score": round(float(c.get("score", 0.0)), 4),
                                    "text": c.get("text", "")[:100],
                                }
                                for c in fetched
                            ]
                        except Exception:
                            retrieved_chunks = []

                    if retrieved_chunks:
                        print(f"      Retrieved Chunks ({len(retrieved_chunks)}):")
                        for c_idx, c in enumerate(retrieved_chunks, start=1):
                            snip = (c.get("text") or c.get("snippet", ""))[:100].replace("\n", " ")
                            p = c.get("page", 1)
                            s = c.get("score", 0.0)
                            print(f"        [{c_idx}] Page {p} (Score: {s:.4f}): {snip!r}")
                else:
                    actual = f.get("actual_answer", f"Top score: {f.get('top_score')}")
                    print(f"      Actual          : {actual}")
            else:
                expected = "Blocked by threshold (top score < MIN_SCORE)"
                actual = f"Top score: {f.get('top_score')}"
                print(f"      Expected        : {expected}")
                print(f"      Actual          : {actual}")
                if mode == "retrieval_only":
                    print("      Note            : Near-topic unanswerable questions are expected to pass the threshold and must be evaluated in full mode.")
    print("=" * 70 + "\n")


def generate_report(
    results: list[dict[str, Any]],
    mode: str,
    report_file: Path,
    pdf_path: Optional[Path] = None,
) -> None:
    """Generate markdown evaluation report summarizing the evaluation run."""
    mode_results = [r for r in results if r.get("mode") == mode]
    if not mode_results:
        return

    answerable = [r for r in mode_results if r.get("type") == "answerable"]
    unanswerable = [r for r in mode_results if r.get("type") == "unanswerable"]

    total = len(mode_results)
    total_correct = sum(1 for r in mode_results if r.get("is_correct"))
    overall_acc = (total_correct / total) * 100 if total else 0.0
    avg_latency = sum(r.get("latency_seconds", 0.0) for r in mode_results) / total if total else 0.0

    ans_correct = sum(1 for r in answerable if r.get("answer_correct" if mode == "full" else "chunk_hit_at_k"))
    ans_acc = (ans_correct / len(answerable)) * 100 if answerable else 0.0

    cite_correct = sum(1 for r in answerable if r.get("citation_correct"))
    cite_acc = (cite_correct / len(answerable)) * 100 if answerable else 0.0

    exact_cite_correct = sum(1 for r in answerable if r.get("citation_exact_page"))
    exact_cite_acc = (exact_cite_correct / len(answerable)) * 100 if answerable else 0.0

    ref_correct = sum(1 for r in unanswerable if r.get("refusal_correct" if mode == "full" else "blocked_by_threshold"))
    ref_rate = (ref_correct / len(unanswerable)) * 100 if unanswerable else 0.0

    hits = [r.get("hit_at_k") for r in answerable if "hit_at_k" in r]
    chunk_hits = [r.get("chunk_hit_at_k") for r in answerable if "chunk_hit_at_k" in r]
    mrrs = [r.get("reciprocal_rank", 0.0) for r in answerable if "reciprocal_rank" in r]
    hit_at_k_pct = (sum(1 for h in hits if h) / len(hits)) * 100 if hits else 0.0
    chunk_hit_pct = (sum(1 for ch in chunk_hits if ch) / len(chunk_hits)) * 100 if chunk_hits else 0.0
    mrr_score = sum(mrrs) / len(mrrs) if mrrs else 0.0

    # Per-model metrics
    models_seen: dict[str, dict[str, int]] = {}
    for r in mode_results:
        m = r.get("model", "unknown")
        if m not in models_seen:
            models_seen[m] = {"total": 0, "correct": 0}
        models_seen[m]["total"] += 1
        if r.get("is_correct"):
            models_seen[m]["correct"] += 1

    failed = [r for r in mode_results if not r.get("is_correct")]

    pdf_name = pdf_path.name if pdf_path else "sample.pdf"
    lines = [
        f"# DocChat-AI Evaluation Report ({report_file.stem})",
        "",
        f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Document:** `{pdf_name}`  ",
        f"> **Evaluation Mode:** `{mode}`  ",
        f"> **Primary Model:** `{GEMINI_MODEL}` | **Fallback Model:** `{FALLBACK_MODEL}`  ",
        f"> **Retrieval Top-K:** `{TOP_K}` | **Min Score Threshold:** `{MIN_SCORE}`  ",
        "",
        "## Performance Metrics",
        "",
        "| Metric | Result | Description |",
        "| :--- | :--- | :--- |",
    ]

    if mode == "retrieval_only":
        lines.extend([
            f"| **Chunk Hit@{TOP_K} (Headline)** | **`{chunk_hit_pct:.1f}%`** | Expected answer text contained in top {TOP_K} chunks |",
            f"| **Page Hit@{TOP_K} (Secondary)** | **`{hit_at_k_pct:.1f}%`** | Target page retrieved within top {TOP_K} chunks |",
            f"| **Mean Reciprocal Rank (MRR)** | **`{mrr_score:.4f}`** | Average reciprocal rank of the first relevant passage |",
            f"| **Blocked by threshold (not the real refusal rate)** | **`{ref_rate:.1f}%`** | Queries scoring below {MIN_SCORE} threshold |",
            f"| **Overall Accuracy** | **`{overall_acc:.1f}%`** | Total correct evaluations out of {total} queries |",
            f"| **Average Latency** | **`{avg_latency:.2f}s`** | Average retrieval time per query |",
            "",
            "> **Note on Blocked by threshold:** Near-topic unanswerable questions are expected to pass the threshold and must be evaluated in full mode.",
            "",
        ])
    else:
        lines.extend([
            f"| **Overall Accuracy** | **`{overall_acc:.1f}%`** | Total correct answers / refusals out of {total} queries |",
            f"| **Answerable Accuracy** | **`{ans_acc:.1f}%`** | Expected facts successfully included in responses |",
            f"| **Citation Accuracy** | **`{cite_acc:.1f}%`** | Cited page(s) actually contain expected answer text |",
            f"| **Exact Page Citation Match** | **`{exact_cite_acc:.1f}%`** | Response explicitly cited the exact expected page |",
            f"| **Unanswerable Refusal Rate** | **`{ref_rate:.1f}%`** | Clean refusals on out-of-domain / unanswerable questions |",
            f"| **Chunk Hit@{TOP_K} (Headline)** | **`{chunk_hit_pct:.1f}%`** | Expected answer text contained in top {TOP_K} chunks |",
            f"| **Page Hit@{TOP_K} (Secondary)** | **`{hit_at_k_pct:.1f}%`** | Target page retrieved within top {TOP_K} chunks |",
            f"| **Mean Reciprocal Rank (MRR)** | **`{mrr_score:.4f}`** | Average reciprocal rank of the first relevant passage |",
            f"| **Average Latency** | **`{avg_latency:.2f}s`** | Average end-to-end processing time per query |",
            "",
        ])

    lines.extend([
        "## Per-Model Breakdown",
        "",
        "| Model | Evaluated Queries | Correct | Accuracy |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for m, stats in sorted(models_seen.items()):
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0.0
        lines.append(f"| `{m}` | {stats['total']} | {stats['correct']} | **`{acc:.1f}%`** |")

    if mode == "full":
        citation_failures = [r for r in answerable if not r.get("citation_correct")]
        if citation_failures:
            lines.extend([
                "",
                "## Citation Failures",
                "",
                "| ID | Question | Expected Page | Actually Cited | Answer Snippet |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for cf in citation_failures:
                q_text = cf["question"].replace("|", "\\|")
                cited = cf.get("cited_pages")
                if cited is None:
                    cited = extract_cited_pages(cf.get("actual_answer", ""))
                cited_str = ", ".join(f"p.{p}" for p in cited) if cited else "None"
                ans_snip = cf.get("actual_answer", "")[:80].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{cf['id']}` | {q_text} | p.{cf.get('expected_page')} | {cited_str} | {ans_snip}... |")

    lines.extend([
        "",
        "## Evaluation Highlights",
        "",
        "- **Grounding:** The assistant declines to answer queries when required information is not found in documents.",
        f"- **Retrieval Performance:** Relevant passages are surfaced with a `{chunk_hit_pct:.1f}%` Chunk Hit@{TOP_K} headline rate (`{hit_at_k_pct:.1f}%` Page Hit@{TOP_K} secondary), and MRR `{mrr_score:.4f}`.",
    ])

    if mode == "full":
        lines.append(f"- **Accurate Sourcing:** Model consistently provides bracketed page citations `{cite_acc:.1f}%` of the time.")

    if failed:
        lines.extend([
            "",
            "## Failed Cases",
            "",
        ])
        if mode == "retrieval_only":
            lines.extend([
                "| ID | Type | Question | Expected Any | Top Score | Status |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for f in failed:
                q_text = f["question"].replace("|", "\\|")
                exp = str(f.get("expected_any", f"Page {f.get('expected_page')}")).replace("|", "\\|")
                ts = f.get("top_score", 0.0)
                status_str = "Missed at chunk level" if f["type"] == "answerable" else f"Passed threshold ({ts} >= {MIN_SCORE})"
                lines.append(f"| `{f['id']}` | {f['type']} | {q_text} | {exp} | {ts} | {status_str} |")

            chunk_misses = [f for f in failed if f["type"] == "answerable"]
            if chunk_misses:
                lines.extend([
                    "",
                    "### Answerable Questions Missed at Chunk Level",
                    "",
                ])
                for cm in chunk_misses:
                    lines.append(f"#### `{cm['id']}`: {cm['question']}")
                    lines.append(f"- **Expected Any:** `{cm.get('expected_any')}`")
                    lines.append(f"- **Expected Page:** `p.{cm.get('expected_page')}`")
                    r_chunks = cm.get("retrieved_chunks") or []
                    if r_chunks:
                        lines.append("- **Retrieved Chunks:**")
                        for ci, c in enumerate(r_chunks, start=1):
                            snip = (c.get("text") or c.get("snippet", ""))[:100].replace("\n", " ")
                            lines.append(f"  {ci}. **Page {c.get('page')}** (Score: `{c.get('score', 0.0):.4f}`): `{snip}`")
                    lines.append("")
        else:
            lines.extend([
                "| ID | Type | Question | Expected | Actual |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for f in failed:
                q_text = f["question"].replace("|", "\\|")
                exp = str(f.get("expected_any", f"Page {f.get('expected_page')}"))
                act = str(f.get("actual_answer", f"Score: {f.get('top_score')}"))[:80].replace("|", "\\|")
                lines.append(f"| `{f['id']}` | {f['type']} | {q_text} | {exp} | {act} |")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report written to: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on DocChat-AI.")
    parser.add_argument(
        "--only-type",
        choices=["answerable", "unanswerable"],
        default=None,
        help="Evaluate only a specific question type ('answerable' or 'unanswerable').",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval metrics (Hit@K, MRR, threshold) without calling LLM.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset and start evaluation from scratch, ignoring previous results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pending questions to evaluate in this run.",
    )
    parser.add_argument(
        "--debug",
        type=str,
        default=None,
        metavar="QID",
        help="Debug a specific question ID (e.g. q11) by printing retrieved chunks, score threshold status, chunk hit info, and final answer.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(DEFAULT_QUESTIONS_FILE),
        help="Path to questions JSON file (default: eval/questions.json).",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(DEFAULT_PDF_FILE),
        help="Path to target document PDF file (default: data/sample.pdf).",
    )

    args = parser.parse_args()
    mode = "retrieval_only" if args.retrieval_only else "full"

    questions_path = Path(args.questions).resolve()
    pdf_path = Path(args.pdf).resolve()

    q_name = questions_path.stem
    results_path = EVAL_DIR / f"results_{q_name}.json"
    report_path = EVAL_DIR / f"REPORT_{q_name}.md"

    ensure_index_ready(pdf_path)

    all_questions = load_questions(questions_path)
    if args.only_type:
        all_questions = [q for q in all_questions if q.get("type") == args.only_type]
        print(f"Filtered to {len(all_questions)} '{args.only_type}' questions from {questions_path.name}.")
    else:
        print(f"Loaded {len(all_questions)} evaluation questions from {questions_path.name}.")

    if args.debug:
        debug_question(args.debug, all_questions, pdf_path=pdf_path)
        return

    existing_results: list[dict[str, Any]] = []
    if not args.reset:
        existing_results = load_results(results_path)
        existing_results = backfill_record_fields(existing_results, all_questions, pdf_path, results_path)
        if not results_path.exists() and existing_results:
            save_results(existing_results, results_path)
        # Keep results from other modes intact
        completed_ids = {
            r["id"] for r in existing_results if r.get("mode") == mode
        }
    else:
        completed_ids = set()
        if args.only_type:
            existing_results = [
                r for r in load_results(results_path)
                if r.get("mode") != mode or r.get("type") != args.only_type
            ]
        else:
            existing_results = [r for r in load_results(results_path) if r.get("mode") != mode]
        existing_results = backfill_record_fields(existing_results, all_questions, pdf_path, results_path)
        save_results(existing_results, results_path)
        print(f"Reset flag provided. Starting evaluation fresh for this mode{' (' + args.only_type + ' only)' if args.only_type else ''}.")

    pending_questions = [q for q in all_questions if q["id"] not in completed_ids]
    print(f"Mode: '{mode}' | Target PDF: {pdf_path.name} | Results: {results_path.name} | Report: {report_path.name}")
    print(f"Total questions: {len(all_questions)} | Already completed: {len(completed_ids)} | Pending: {len(pending_questions)}")

    if args.limit is not None and args.limit > 0:
        pending_questions = pending_questions[:args.limit]
        print(f"Limiting execution to next {len(pending_questions)} pending question(s).")

    if not pending_questions:
        print("All questions have already been evaluated for this mode.")
        print_summary(existing_results, mode)
        generate_report(existing_results, mode, report_path, pdf_path=pdf_path)
        return

    print("\nStarting evaluation...")
    for idx, q_item in enumerate(pending_questions, start=1):
        q_id = q_item["id"]
        q_text = q_item["question"]
        print(f"[{idx}/{len(pending_questions)}] Processing {q_id}: \"{q_text}\"...", end="", flush=True)

        try:
            if args.retrieval_only:
                res_record = evaluate_retrieval(q_item, pdf_path=pdf_path)
            else:
                res_record = evaluate_full(q_item, pdf_path=pdf_path)

            # Update results and save checkpoint immediately
            existing_results = [r for r in existing_results if not (r["id"] == q_id and r.get("mode") == mode)]
            existing_results.append(res_record)
            save_results(existing_results, results_path)

            status = "PASS" if res_record.get("is_correct") else "FAIL"
            print(f" -> {status} ({res_record.get('latency_seconds', 0)}s)")

            # Sleep 4 seconds between LLM calls in full mode
            if not args.retrieval_only and idx < len(pending_questions):
                time.sleep(4)

        except QuotaExceededError as q_err:
            remaining = len(pending_questions) - idx + 1
            print(f"\n\n[!] QuotaExceededError encountered: {q_err}")
            print(f"[!] Evaluation stopped cleanly. Progress saved to {results_path.name}.")
            print(f"[!] {remaining} question(s) remain pending. Run again later to resume.")
            break
        except Exception as exc:
            print(f" -> ERROR: {exc}")
            # Save error record
            err_record = {
                "id": q_id,
                "type": q_item.get("type", "unknown"),
                "question": q_text,
                "mode": mode,
                "is_correct": False,
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
            existing_results.append(err_record)
            save_results(existing_results, results_path)

    # Print summary and generate markdown report
    current_results = load_results(results_path)
    print_summary(current_results, mode)
    generate_report(current_results, mode, report_path, pdf_path=pdf_path)


if __name__ == "__main__":
    main()
