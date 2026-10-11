"""Threshold calibration script for DocChat-AI.

Runs semantic vector searches across answerable and unanswerable questions
from an evaluation dataset to evaluate similarity scores and help calibrate
the optimal MIN_SCORE threshold.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Add project root to sys.path so 'src' can be imported cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chunker import chunk_pages
from src.config import MIN_SCORE, TOP_K
from src.loader import load_document
from src.vector_store import add_chunks, list_files, search


def ensure_data_indexed(pdf_path: Path) -> None:
    """Ensure the specified PDF document is indexed in the vector store."""
    if not pdf_path.exists():
        print(f"Error: Target PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    indexed = list_files()
    if pdf_path.name not in indexed:
        print(f"Indexing {pdf_path.name} in vector store...")
        pages = load_document(pdf_path)
        chunks = chunk_pages(pages)
        add_chunks(chunks)
        print(f"Indexed {len(chunks)} chunks.")


def load_questions_file(questions_path: Path) -> list[dict[str, Any]]:
    """Load question items from a JSON file."""
    if not questions_path.exists():
        print(f"Error: Questions file not found at {questions_path}", file=sys.stderr)
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, list):
        return data

    raise ValueError(f"Unrecognized question format in {questions_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate MIN_SCORE threshold by evaluating retrieval similarity scores."
    )
    fallback_pdf = project_root / "data" / "sample.pdf"
    fallback_q = project_root / "eval" / "questions.json"

    parser.add_argument(
        "--pdf",
        type=str,
        default=str(fallback_pdf),
        help=f"Path to PDF document to index and test against (default: {fallback_pdf}).",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(fallback_q),
        help=f"Path to questions JSON file (default: {fallback_q}).",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    questions_path = Path(args.questions)

    ensure_data_indexed(pdf_path)
    questions = load_questions_file(questions_path)

    answerable = [q for q in questions if q.get("type") == "answerable"]
    unanswerable = [q for q in questions if q.get("type") == "unanswerable"]

    # Fallback if no questions are categorized by type
    if not answerable and not unanswerable:
        answerable = questions

    print("=" * 80)
    print("             DocChat-AI: Similarity Score Threshold Calibration")
    print("=" * 80)
    print(f"Target Document              : {pdf_path.name}")
    print(f"Questions File               : {questions_path.name} (Total: {len(questions)})")
    print(f"Current Configured MIN_SCORE : {MIN_SCORE}")
    print(f"Retrieval TOP_K              : {TOP_K}\n")

    # Evaluate Answerable Questions
    print(f"--- [1] Answerable Questions (n={len(answerable)}) ---")
    ans_scores: list[float] = []
    for q in answerable:
        q_id = q.get("id", "?")
        q_text = q.get("question", "")
        results = search(q_text, k=TOP_K, file=pdf_path.name)
        top_score = results[0]["score"] if results else 0.0
        best_match = (
            f"[{results[0]['file']}, p.{results[0]['page']}]" if results else "N/A"
        )
        ans_scores.append(top_score)
        status = "PASS" if top_score >= MIN_SCORE else "BLOCKED"
        print(f"  [{q_id:>3}] Score: {top_score:.4f} [{status:<7}] | Best: {best_match:<24} | \"{q_text}\"")

    # Evaluate Unanswerable Questions
    print(f"\n--- [2] Unanswerable Questions (n={len(unanswerable)}) ---")
    unans_scores: list[float] = []
    for q in unanswerable:
        q_id = q.get("id", "?")
        q_text = q.get("question", "")
        results = search(q_text, k=TOP_K, file=pdf_path.name)
        top_score = results[0]["score"] if results else 0.0
        best_match = (
            f"[{results[0]['file']}, p.{results[0]['page']}]" if results else "N/A"
        )
        unans_scores.append(top_score)
        status = "BLOCKED" if top_score < MIN_SCORE else "PASSED"
        print(f"  [{q_id:>3}] Score: {top_score:.4f} [{status:<7}] | Best: {best_match:<24} | \"{q_text}\"")

    # List answerable questions blocked at current MIN_SCORE
    blocked_answerable = [
        (q, score)
        for q, score in zip(answerable, ans_scores)
        if score < MIN_SCORE
    ]

    print("\n" + "=" * 80)
    print(f"Blocked Answerable Questions at Current MIN_SCORE ({MIN_SCORE:.2f})")
    print("=" * 80)
    if blocked_answerable:
        print(f"Found {len(blocked_answerable)} answerable question(s) blocked by MIN_SCORE:")
        for q, score in blocked_answerable:
            q_id = q.get("id", "?")
            print(f"  - [{q_id}] Score: {score:.4f} (< {MIN_SCORE}) | \"{q.get('question')}\"")
    else:
        print("  None. All answerable questions pass the current MIN_SCORE threshold!")

    # Statistical Summary
    print("\n" + "=" * 80)
    print("                               Summary Statistics")
    print("=" * 80)
    if ans_scores:
        ans_min = min(ans_scores)
        ans_max = max(ans_scores)
        ans_avg = sum(ans_scores) / len(ans_scores)
        print(f"Answerable Questions   (n={len(ans_scores):>2}): Min = {ans_min:.4f} | Max = {ans_max:.4f} | Avg = {ans_avg:.4f}")
    else:
        ans_min, ans_max, ans_avg = 0.0, 0.0, 0.0
        print("Answerable Questions   : None evaluated")

    if unans_scores:
        unans_min = min(unans_scores)
        unans_max = max(unans_scores)
        unans_avg = sum(unans_scores) / len(unans_scores)
        print(f"Unanswerable Questions (n={len(unans_scores):>2}): Min = {unans_min:.4f} | Max = {unans_max:.4f} | Avg = {unans_avg:.4f}")
    else:
        unans_min, unans_max, unans_avg = 0.0, 0.0, 0.0
        print("Unanswerable Questions : None evaluated")

    print("-" * 80)
    print(f"Current Configured MIN_SCORE: {MIN_SCORE:.2f}")
    if ans_scores and unans_scores:
        recommended_mid = round((unans_max + ans_min) / 2, 2)
        if unans_max < MIN_SCORE <= ans_min:
            print(f"[*] Optimal: MIN_SCORE ({MIN_SCORE:.2f}) separates all unanswerable (< {unans_max:.4f}) and answerable (>= {ans_min:.4f}).")
        elif MIN_SCORE <= unans_max:
            print(f"[!] Note: {sum(1 for s in unans_scores if s >= MIN_SCORE)} unanswerable question(s) pass MIN_SCORE (max unans score: {unans_max:.4f}).")
            print("    These will be evaluated and grounded by the LLM refusal prompt.")
        elif MIN_SCORE > ans_min:
            print(f"[!] Warning: {len(blocked_answerable)} answerable question(s) are blocked by MIN_SCORE ({MIN_SCORE:.2f} > min answerable {ans_min:.4f}).")
            print(f"    Consider adjusting MIN_SCORE closer to {recommended_mid}.")
    print("=" * 80)


if __name__ == "__main__":
    main()
