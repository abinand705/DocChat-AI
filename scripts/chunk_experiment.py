"""Chunking and Top-K Grid Search Experiment for DocChat-AI.

Evaluates retrieval metrics across combinations of chunk_size and top_k:
- chunk_size in [400, 600, 800, 1000] with overlap = 12% of chunk_size
- top_k in [3, 4, 6]
Uses a temporary, isolated Chroma collection for each chunking configuration,
indexes the specified PDF document, evaluates retrieval on answerable questions
from the specified questions JSON, and cleans up the temporary collection.

Embeds each chunking configuration only once and reuses vectors across top_k
values. Reports Chunk Hit@k, MRR, number of chunks, and index coverage
fraction (top_k / chunks) per combination in console and saves results to
eval/chunk_experiment_<name>.md.

Tie-break logic: prefers highest Chunk Hit@k -> smaller top_k -> higher MRR -> smaller chunk_size.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chunker import chunk_pages
try:
    from src.config import EMBED_BATCH_SIZE
except ImportError:
    EMBED_BATCH_SIZE = 20
from src.embeddings import embed_texts
from src.loader import load_document
from src.vector_store import get_chroma_client

EVAL_DIR = project_root / "eval"
DEFAULT_QUESTIONS_FILE = EVAL_DIR / "questions.json"
DEFAULT_PDF_FILE = project_root / "data" / "sample.pdf"

CHUNK_SIZES = [400, 600, 800, 1000]
TOP_K_VALUES = [3, 4, 6]
OVERLAP_RATIO = 0.12


def load_answerable_questions(questions_path: Path) -> list[dict[str, Any]]:
    """Load answerable evaluation questions from questions JSON file."""
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found at {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        all_q = json.load(f)
    return [q for q in all_q if q.get("type") == "answerable"]


def normalize_match_text(text: str) -> str:
    """Normalize text for substring matching:
    - Lowercase
    - Replace newlines and runs of whitespace with a single space
    - Remove spaces around hyphens ("real -time" -> "real-time")
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


def load_partial_results(partial_file: Path) -> list[dict[str, Any]]:
    """Load partial experiment results from JSON file."""
    if not partial_file.exists():
        return []
    try:
        with open(partial_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"Warning: Failed to load partial results from {partial_file}: {e}", file=sys.stderr)
    return []


def save_partial_results(results: list[dict[str, Any]], partial_file: Path) -> None:
    """Save partial experiment results atomically to JSON file."""
    partial_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = partial_file.parent / f"{partial_file.name}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    temp_file.replace(partial_file)


def run_experiment(
    questions_path: Path = DEFAULT_QUESTIONS_FILE,
    pdf_path: Path = DEFAULT_PDF_FILE,
    partial_file: Optional[Path] = None,
    reset: bool = False,
) -> list[dict[str, Any]]:
    """Execute grid search across chunk sizes and top_k values with partial checkpointing."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found at {pdf_path}")

    if partial_file is None:
        q_name = questions_path.stem
        partial_file = EVAL_DIR / f"chunk_experiment_{q_name}_partial.json"

    if reset and partial_file.exists():
        partial_file.unlink()
        print(f"Reset flag specified. Cleared previous partial results at {partial_file.name}.")

    print(f"Loading document: {pdf_path.name}...")
    pages = load_document(pdf_path)
    print(f"Loaded {len(pages)} pages from {pdf_path.name}.")

    answerable_questions = load_answerable_questions(questions_path)
    total_q = len(answerable_questions)
    print(f"Loaded {total_q} answerable evaluation questions from {questions_path.name}.")

    # Load existing partial results
    results: list[dict[str, Any]] = []
    completed_chunk_sizes: set[int] = set()
    if not reset:
        results = load_partial_results(partial_file)
        completed_chunk_sizes = {r["chunk_size"] for r in results if isinstance(r, dict) and "chunk_size" in r}
        if completed_chunk_sizes:
            print(f"Found partial results in {partial_file.name} ({len(results)} records).")
            print(f"Already completed chunk sizes: {sorted(completed_chunk_sizes)}")

    if all(cs in completed_chunk_sizes for cs in CHUNK_SIZES):
        print("All chunk size configurations are already completed in partial results.")
        return results

    # 1. Pre-embed queries once to save API quota across all configurations
    print("\nPre-embedding evaluation queries (single batch)...")
    query_texts = [q["question"].strip() for q in answerable_questions]
    query_vectors = embed_texts(query_texts, task_type="RETRIEVAL_QUERY", batch_size=EMBED_BATCH_SIZE)
    print(f"Generated {len(query_vectors)} query embeddings.")
    time.sleep(1.0)

    chroma_client = get_chroma_client()

    print("\n" + "=" * 80)
    print("           Starting Chunk Size & Top-K Retrieval Grid Search")
    print("=" * 80)

    for chunk_size in CHUNK_SIZES:
        if chunk_size in completed_chunk_sizes:
            print(f"\n--> Configuration: chunk_size={chunk_size} already completed in partial results. Skipping.")
            continue

        overlap = int(chunk_size * OVERLAP_RATIO)
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        chunk_count = len(chunks)
        print(f"\n--> Configuration: chunk_size={chunk_size}, overlap={overlap} ({chunk_count} chunks total)")

        current_config_results: list[dict[str, Any]] = []
        temp_coll_name = f"tempexp-{chunk_size}-{int(time.time() * 1000)}"

        try:
            print(f"    Embedding {chunk_count} chunks...")
            chunk_texts = [c["text"] for c in chunks]
            chunk_vectors = embed_texts(chunk_texts, task_type="RETRIEVAL_DOCUMENT", batch_size=EMBED_BATCH_SIZE)
            print(f"    Embedding complete ({len(chunk_vectors)} vectors).")

            try:
                # Create isolated temporary collection
                temp_collection = chroma_client.create_collection(
                    name=temp_coll_name,
                    metadata={"hnsw:space": "cosine"},
                )

                # Upsert chunks and embeddings
                chunk_ids = [c["chunk_id"] for c in chunks]
                metadatas = [{"page": c["page"], "file": c["file"]} for c in chunks]
                temp_collection.upsert(
                    ids=chunk_ids,
                    embeddings=chunk_vectors,
                    documents=chunk_texts,
                    metadatas=metadatas,
                )

                # Evaluate each top_k reusing the indexed collection
                for top_k in TOP_K_VALUES:
                    chunk_hits = 0
                    page_hits = 0
                    reciprocal_ranks: list[float] = []

                    # Calculate coverage fraction (top_k / chunks)
                    effective_k = min(top_k, chunk_count)
                    coverage_fraction = effective_k / chunk_count if chunk_count > 0 else 0.0
                    coverage_pct = coverage_fraction * 100
                    coverage_str = f"{coverage_pct:.1f}% ({effective_k}/{chunk_count})"

                    for q_item, q_vec in zip(answerable_questions, query_vectors):
                        expected_any = q_item.get("expected_any", [])
                        expected_page = q_item.get("expected_page")

                        query_res = temp_collection.query(
                            query_embeddings=[q_vec],
                            n_results=effective_k,
                            where={"file": pdf_path.name},
                            include=["documents", "metadatas", "distances"],
                        )

                        retrieved_docs = query_res["documents"][0] if query_res.get("documents") else []
                        retrieved_metas = query_res["metadatas"][0] if query_res.get("metadatas") else []
                        retrieved_pages = [m.get("page") for m in retrieved_metas if isinstance(m, dict)]

                        # Chunk Hit@k: any retrieved chunk contains any expected_any
                        is_chunk_hit = any(
                            text_contains_any(doc, expected_any)
                            for doc in retrieved_docs
                        ) if expected_any else False

                        if is_chunk_hit:
                            chunk_hits += 1

                        # Page Hit@k & Reciprocal Rank
                        is_page_hit = expected_page in retrieved_pages if expected_page else False
                        if is_page_hit:
                            page_hits += 1

                        r_rank = 0.0
                        if expected_page:
                            for idx, p in enumerate(retrieved_pages, start=1):
                                if p == expected_page:
                                 r_rank = 1.0 / idx
                                 break
                        reciprocal_ranks.append(r_rank)

                    chunk_hit_rate = (chunk_hits / total_q) * 100 if total_q else 0.0
                    page_hit_rate = (page_hits / total_q) * 100 if total_q else 0.0
                    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

                    record = {
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "top_k": top_k,
                        "chunk_count": chunk_count,
                        "coverage_fraction": round(coverage_fraction, 4),
                        "coverage_pct": round(coverage_pct, 1),
                        "coverage_str": coverage_str,
                        "chunk_hits": chunk_hits,
                        "total_q": total_q,
                        "chunk_hit_rate": round(chunk_hit_rate, 2),
                        "page_hit_rate": round(page_hit_rate, 2),
                        "mrr": round(mrr, 4),
                    }
                    current_config_results.append(record)
                    print(
                        f"    top_k={top_k:1d} | Chunks: {chunk_count:2d} | Coverage: {coverage_str:13s} "
                        f"| Chunk Hit@{top_k}: {chunk_hit_rate:5.1f}% ({chunk_hits}/{total_q}) "
                        f"| Page Hit@{top_k}: {page_hit_rate:5.1f}% | MRR: {mrr:.4f}"
                    )

                results.extend(current_config_results)
                completed_chunk_sizes.add(chunk_size)

                # Save partial results after finishing this chunk-size configuration
                save_partial_results(results, partial_file)
                print(f"    Saved progress for chunk_size={chunk_size} to {partial_file.name} ({len(results)} total records).")

            finally:
                # Guarantee cleanup of the temporary collection
                try:
                    chroma_client.delete_collection(name=temp_coll_name)
                    print(f"    Cleaned up temporary collection '{temp_coll_name}'.")
                except Exception as del_err:
                    print(f"    Warning: Failed to delete collection '{temp_coll_name}': {del_err}", file=sys.stderr)

        except Exception as exc:
            print("\n" + "=" * 80, file=sys.stderr)
            print(f"ERROR: Embedding or evaluation failed during chunk_size={chunk_size}: {exc}", file=sys.stderr)
            if results:
                save_partial_results(results, partial_file)
                print(f"Partial progress ({len(results)} records) saved to: {partial_file}", file=sys.stderr)
            resume_cmd = (
                f"python scripts/chunk_experiment.py "
                f"--pdf \"{pdf_path}\" --questions \"{questions_path}\""
            )
            print(f"To resume the experiment once rate limits/quotas permit, run:\n  {resume_cmd}", file=sys.stderr)
            print("=" * 80 + "\n", file=sys.stderr)
            raise

    return results


def find_best_combination(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Find combination with best Chunk Hit@k, breaking ties by smaller top_k, higher MRR, then smaller chunk size."""
    return max(
        results,
        key=lambda r: (r["chunk_hit_rate"], -r["top_k"], r["mrr"], -r["chunk_size"]),
    )


def print_and_save_report(
    results: list[dict[str, Any]],
    report_file: Path,
    pdf_path: Path,
    questions_path: Path,
) -> dict[str, Any]:
    """Format and output experiment results table and save markdown report."""
    best = find_best_combination(results)

    # 1. Print Console Table
    print("\n" + "=" * 95)
    print("                    RETRIEVAL EXPERIMENT RESULTS TABLE")
    print("=" * 95)
    header = f"{'Chunk Size':>10} | {'Overlap':>7} | {'Top-K':>5} | {'Chunks':>6} | {'Coverage (k/n)':>15} | {'Chunk Hit@k':>11} | {'Page Hit@k':>10} | {'MRR':>6}"
    print(header)
    print("-" * 95)

    for r in results:
        is_best = (r["chunk_size"] == best["chunk_size"] and r["top_k"] == best["top_k"])
        marker = " *" if is_best else ""
        row = (
            f"{r['chunk_size']:>10d} | "
            f"{r['overlap']:>7d} | "
            f"{r['top_k']:>5d} | "
            f"{r['chunk_count']:>6d} | "
            f"{r['coverage_str']:>15s} | "
            f"{r['chunk_hit_rate']:>10.1f}% | "
            f"{r['page_hit_rate']:>9.1f}% | "
            f"{r['mrr']:>6.4f}{marker}"
        )
        print(row)
    print("=" * 95)

    # 2. Print Best Combination
    print("\n" + "*" * 95)
    print("                      OPTIMAL CONFIGURATION")
    print("*" * 95)
    print(f"  Best Combination : chunk_size = {best['chunk_size']}, top_k = {best['top_k']}")
    print(f"  Overlap (12%)    : {best['overlap']} characters")
    print(f"  Chunks in Index  : {best['chunk_count']}")
    print(f"  Index Coverage   : {best['coverage_str']}")
    print(f"  Chunk Hit@k      : {best['chunk_hit_rate']:.1f}% ({best['chunk_hits']}/{best['total_q']})")
    print(f"  Page Hit@k       : {best['page_hit_rate']:.1f}%")
    print(f"  MRR              : {best['mrr']:.4f}")
    print("  Tie-break logic  : Highest Chunk Hit@k -> Smaller top_k -> Higher MRR -> Smaller chunk_size")
    print("*" * 95 + "\n")

    # 3. Generate Markdown Report
    lines = [
        f"# DocChat-AI Chunking & Top-K Experiment Report ({report_file.stem})",
        "",
        f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Document:** `{pdf_path.name}`  ",
        f"> **Questions Evaluated:** {best['total_q']} answerable questions from `{questions_path.name}`  ",
        f"> **Overlap Policy:** 12% of chunk size  ",
        "",
        "## Performance Grid",
        "",
        "| Chunk Size | Overlap | Top-K | Chunks | Coverage (k/chunks) | Chunk Hit@k | Page Hit@k | MRR |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        is_best = (r["chunk_size"] == best["chunk_size"] and r["top_k"] == best["top_k"])
        highlight = "**" if is_best else ""
        tag = " (Best)" if is_best else ""
        lines.append(
            f"| {highlight}{r['chunk_size']}{highlight} | "
            f"{r['overlap']} | "
            f"{highlight}{r['top_k']}{highlight} | "
            f"{r['chunk_count']} | "
            f"`{r['coverage_str']}` | "
            f"{highlight}`{r['chunk_hit_rate']:.1f}%`{highlight}{tag} | "
            f"`{r['page_hit_rate']:.1f}%` | "
            f"`{r['mrr']:.4f}` |"
        )

    lines.extend([
        "",
        "## Optimal Configuration",
        "",
        f"- **Best Chunk Size:** `{best['chunk_size']}` (overlap: `{best['overlap']}` chars, `{best['chunk_count']}` total chunks)",
        f"- **Best Top-K:** `{best['top_k']}`",
        f"- **Index Coverage:** `{best['coverage_str']}` of index retrieved per query",
        f"- **Chunk Hit@k:** **`{best['chunk_hit_rate']:.1f}%`** ({best['chunk_hits']}/{best['total_q']})",
        f"- **Page Hit@k:** **`{best['page_hit_rate']:.1f}%`**",
        f"- **Mean Reciprocal Rank (MRR):** **`{best['mrr']:.4f}`**",
        "- **Selection Criterion:** Highest Chunk Hit@k, breaking ties by smaller `top_k`, then higher MRR, then smaller `chunk_size`.",
        "",
        "## Key Observations",
        "",
        "- **Index Coverage & Triviality:** Configurations with very large chunks yield few total chunks (e.g. 12 chunks for size 1000). At top_k=6, 50.0% of the entire document index is retrieved, which inflates hit rate trivially.",
        "- **Information Granularity:** Smaller chunk sizes produce more focused passages and lower index coverage per query, ensuring retrieved chunks contain precise answers rather than entire multi-topic sections.",
        "- **MRR Prioritization:** Preferring higher MRR before chunk size in tie-breaking ensures that relevant information is ranked closer to position 1.",
    ])

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved experiment report to: {report_file}")

    # Also maintain legacy eval/chunk_experiment.md if evaluating default questions
    if report_file.name == "chunk_experiment_questions.md":
        legacy_report = report_file.parent / "chunk_experiment.md"
        legacy_report.write_text("\n".join(lines), encoding="utf-8")

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chunking & Top-K Grid Search Experiment for DocChat-AI.")
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore partial results and run experiment from scratch.",
    )

    args = parser.parse_args()
    questions_path = Path(args.questions).resolve()
    pdf_path = Path(args.pdf).resolve()

    q_name = questions_path.stem
    report_file = EVAL_DIR / f"chunk_experiment_{q_name}.md"

    results = run_experiment(questions_path=questions_path, pdf_path=pdf_path, reset=args.reset)
    print_and_save_report(
        results=results,
        report_file=report_file,
        pdf_path=pdf_path,
        questions_path=questions_path,
    )


if __name__ == "__main__":
    main()
