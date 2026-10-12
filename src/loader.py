"""Document loader module for docchat-ai.

Extracts text from PDF and TXT files, returning structured page data.
"""

from collections import Counter
from pathlib import Path
import re
from typing import Any, BinaryIO, Optional, Union
from pypdf import PdfReader


def _extract_filename(file: Union[str, Path, Any]) -> str:
    """Extract a clean filename from a path or file-like object."""
    if isinstance(file, (str, Path)):
        return Path(file).name
    if hasattr(file, "name") and file.name:
        return Path(file.name).name
    return "document"


def remove_headers_footers(
    pages_data: list[dict[str, Any]],
    threshold_ratio: float = 0.30,
    min_pages: int = 5,
    return_removed: bool = False,
) -> Union[list[dict[str, Any]], tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Remove running headers and footers before chunking.

    For documents with at least min_pages (default 5):
    - Normalizes whitespace for line matching.
    - Drops any line that appears on at least threshold_ratio of the pages (default 30%).
    - Drops any line that contains only digits (e.g. page numbers).

    Args:
        pages_data: List of page dictionaries with 'text', 'file', and 'page'.
        threshold_ratio: Threshold frequency across pages (default 0.30).
        min_pages: Minimum page count required to filter headers/footers (default 5).
        return_removed: If True, returns a tuple of (cleaned_pages, removed_records).

    Returns:
        Cleaned page dictionaries (or tuple with removed records if return_removed=True).
    """
    if len(pages_data) < min_pages:
        if return_removed:
            return pages_data, []
        return pages_data

    # Count how many distinct pages each normalized line appears on
    page_counts: Counter[str] = Counter()
    for page in pages_data:
        seen_on_page: set[str] = set()
        for raw_line in page.get("text", "").splitlines():
            norm = re.sub(r"\s+", " ", raw_line).strip()
            if norm:
                seen_on_page.add(norm)
        for norm in seen_on_page:
            page_counts[norm] += 1

    total_pages = len(pages_data)
    min_threshold_count = threshold_ratio * total_pages
    frequent_lines = {
        line for line, count in page_counts.items()
        if count >= min_threshold_count
    }

    removed_records: list[dict[str, Any]] = []
    cleaned_pages: list[dict[str, Any]] = []

    for page in pages_data:
        kept_lines: list[str] = []
        page_num = page.get("page", 1)
        for raw_line in page.get("text", "").splitlines():
            norm = re.sub(r"\s+", " ", raw_line).strip()
            if not norm:
                kept_lines.append("")
                continue
            if norm in frequent_lines:
                removed_records.append({
                    "line": norm,
                    "reason": f"appears on {page_counts[norm]}/{total_pages} pages ({page_counts[norm] / total_pages * 100:.1f}%)",
                    "page": page_num,
                })
                continue
            if norm.isdigit():
                removed_records.append({
                    "line": norm,
                    "reason": "digit-only line (page number)",
                    "page": page_num,
                })
                continue
            kept_lines.append(norm)

        cleaned_text = "\n".join(kept_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
        cleaned_page = dict(page)
        cleaned_page["text"] = cleaned_text
        cleaned_pages.append(cleaned_page)

    if return_removed:
        return cleaned_pages, removed_records
    return cleaned_pages


_unreadable_pages_by_file: dict[str, list[int]] = {}


def get_unreadable_pages(filename: Union[str, Path, Any]) -> list[int]:
    """Return recorded unreadable pages for a given filename."""
    fname = _extract_filename(filename)
    return list(_unreadable_pages_by_file.get(fname, []))


def set_unreadable_pages(filename: Union[str, Path, Any], pages: list[int]) -> None:
    """Record unreadable pages for a given filename."""
    fname = _extract_filename(filename)
    _unreadable_pages_by_file[fname] = sorted(set(pages))


def format_unreadable_pages(pages: list[int]) -> str:
    """Format a list of page numbers into a persistent note string.

    Example:
        format_unreadable_pages([4, 21, 22, 23, 24, 25])
        -> 'Pages 4, 21-25 have no readable text (images are not read).'
    """
    if not pages:
        return ""
    sorted_pages = sorted(set(pages))
    ranges: list[str] = []
    start = sorted_pages[0]
    prev = sorted_pages[0]

    for p in sorted_pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = p
            prev = p

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    range_str = ", ".join(ranges)
    prefix = "Page" if (len(sorted_pages) == 1 and "-" not in ranges[0]) else "Pages"
    verb = "has" if prefix == "Page" else "have"
    return f"{prefix} {range_str} {verb} no readable text (images are not read)."


def load_document(
    file: Union[str, Path, BinaryIO, Any],
    return_unreadable: bool = False,
) -> Union[list[dict[str, Any]], tuple[list[dict[str, Any]], list[int]]]:
    """Load and extract text from a PDF or TXT document.

    Records pages whose extracted text is empty or under 20 characters.
    Stores 'total_pages' in each extracted page dictionary.

    Args:
        file: A file path (str or Path) or an uploaded file object
            (such as a Streamlit UploadedFile or file-like object).
        return_unreadable: If True, returns a tuple of (pages_data, unreadable_pages).

    Returns:
        A list of dictionaries with keys:
            - 'text' (str): Extracted and cleaned text content.
            - 'file' (str): Name of the file.
            - 'page' (int): Page number (1-indexed for PDF, 1 for TXT).
            - 'total_pages' (int): Total page count of the document.
        Or a tuple (pages_data, unreadable_pages) if return_unreadable=True.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If the file format is unsupported or the file is corrupted.
    """
    filename = _extract_filename(file)
    ext = Path(filename).suffix.lower()

    if ext not in (".pdf", ".txt"):
        raise ValueError(
            f"Unsupported file format '{ext}' for file '{filename}'. "
            "Only PDF (.pdf) and TXT (.txt) files are supported."
        )

    if isinstance(file, (str, Path)):
        path_obj = Path(file)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: '{file}'")
        if not path_obj.is_file():
            raise ValueError(f"Path is not a regular file: '{file}'")

    if ext == ".pdf":
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            reader = PdfReader(file)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise ValueError(
                        f"PDF file '{filename}' is encrypted and could not be decrypted."
                    )

            total_pages = len(reader.pages)
            pages_data: list[dict[str, Any]] = []
            unreadable_pages: list[int] = []

            for page_idx, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text()
                cleaned = extracted.strip() if extracted else ""
                if len(cleaned) < 20:
                    unreadable_pages.append(page_idx)
                if cleaned:
                    pages_data.append({
                        "text": cleaned,
                        "file": filename,
                        "page": page_idx,
                        "total_pages": total_pages,
                        "unreadable_pages": unreadable_pages,
                    })

            set_unreadable_pages(filename, unreadable_pages)
            pages_data = remove_headers_footers(pages_data)
            if return_unreadable:
                return pages_data, unreadable_pages
            return pages_data
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Corrupted or invalid PDF file '{filename}': {exc}"
            ) from exc

    # Handle TXT files
    try:
        if isinstance(file, (str, Path)):
            content_bytes = Path(file).read_bytes()
        elif hasattr(file, "getvalue"):
            content_bytes = file.getvalue()
        elif hasattr(file, "read"):
            if hasattr(file, "seek"):
                file.seek(0)
            raw = file.read()
            content_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        else:
            raise ValueError(f"Unable to read content from file '{filename}'.")

        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("latin-1")
            except Exception as dec_err:
                raise ValueError(
                    f"Failed to decode text file '{filename}': {dec_err}"
                ) from dec_err

        cleaned = text.strip()
        total_pages = 1
        unreadable_pages = [1] if len(cleaned) < 20 else []
        set_unreadable_pages(filename, unreadable_pages)

        if not cleaned:
            if return_unreadable:
                return [], unreadable_pages
            return []

        pages_data = [{
            "text": cleaned,
            "file": filename,
            "page": 1,
            "total_pages": total_pages,
            "unreadable_pages": unreadable_pages,
        }]
        pages_data = remove_headers_footers(pages_data)
        if return_unreadable:
            return pages_data, unreadable_pages
        return pages_data
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Failed to read text file '{filename}': {exc}"
        ) from exc


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test document loading and header/footer removal.")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file to test.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if args.pdf:
        target_path = Path(args.pdf)
    else:
        target_path = project_root / "data" / "sample.pdf"

    if not target_path.exists():
        print(f"Error: Target file not found at {target_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading document: {target_path}")

    # Extract raw pages without header/footer removal first to inspect differences
    reader = PdfReader(target_path)
    raw_pages: list[dict[str, Any]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text()
        if extracted and extracted.strip():
            raw_pages.append({
                "text": extracted.strip(),
                "file": target_path.name,
                "page": page_idx,
            })

    cleaned_pages, removed_records = remove_headers_footers(raw_pages, return_removed=True)

    print("\n" + "=" * 70)
    print("           Header & Footer Removal Results")
    print("=" * 70)
    print(f"Total Pages Analyzed : {len(raw_pages)}")
    print(f"Total Lines Removed   : {len(removed_records)}")

    if removed_records:
        # Group removed lines by line text
        grouped_removed: dict[str, list[dict[str, Any]]] = {}
        for rec in removed_records:
            grouped_removed.setdefault(rec["line"], []).append(rec)

        print(f"\nUnique Lines Removed  : {len(grouped_removed)}\n")
        print("Removed Lines Detail:")
        for line_text, occurrences in grouped_removed.items():
            reason = occurrences[0]["reason"]
            pages_list = sorted({o["page"] for o in occurrences})
            pages_str = ", ".join(str(p) for p in pages_list[:10])
            if len(pages_list) > 10:
                pages_str += f"... (+{len(pages_list) - 10} more)"
            print(f"  - {repr(line_text)}")
            print(f"    Reason : {reason}")
            print(f"    Pages  : [{pages_str}] (count: {len(occurrences)})")
    else:
        print("\nNo running headers, footers, or digit-only lines detected.")

    try:
        from src.chunker import chunk_pages
    except ImportError:
        try:
            from chunker import chunk_pages
        except ImportError:
            chunk_pages = None

    if chunk_pages:
        chunks = chunk_pages(cleaned_pages)
        print(f"\nChunks generated after cleaning: {len(chunks)}")
        if chunks:
            print(f"Sample First Chunk Preview:\n{chunks[0]['text'][:200]}...")

