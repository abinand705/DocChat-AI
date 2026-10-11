"""Document chunking module for docchat-ai.

Splits loaded pages into overlapping text chunks with boundary awareness.
"""

from pathlib import Path
from typing import Any

try:
    from src.config import CHUNK_OVERLAP, CHUNK_SIZE
except ImportError:
    try:
        from config import CHUNK_OVERLAP, CHUNK_SIZE
    except ImportError:
        CHUNK_SIZE = 800
        CHUNK_OVERLAP = 100


def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks, respecting sentence and paragraph boundaries.

    Args:
        text: Input string to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        List of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    # Boundary delimiters in descending order of semantic structure
    delimiters = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    while start < len(text):
        # If remaining text fits within chunk_size, finalize
        if len(text) - start <= chunk_size:
            final_chunk = text[start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break

        target_end = start + chunk_size
        # Look backwards from target_end for a natural boundary
        lookback = max(overlap, int(chunk_size * 0.35))
        min_end = max(start + 1, target_end - lookback)
        search_slice = text[min_end:target_end]

        cut_idx = target_end
        for delim in delimiters:
            pos = search_slice.rfind(delim)
            if pos != -1:
                if delim in (". ", "? ", "! "):
                    # Include the punctuation mark in current chunk
                    cut_idx = min_end + pos + 1
                elif delim in ("\n\n", "\n"):
                    cut_idx = min_end + pos
                else:
                    cut_idx = min_end + pos
                break

        current_chunk = text[start:cut_idx].strip()
        if current_chunk:
            chunks.append(current_chunk)

        # Calculate next start position using overlap
        target_start = max(start + 1, cut_idx - overlap)
        if target_start < len(text):
            # Advance past partial words to avoid starting mid-word
            while target_start < cut_idx and text[target_start] not in " \t\r\n":
                target_start += 1
            # Skip any leading whitespace characters
            while target_start < cut_idx and text[target_start] in " \t\r\n":
                target_start += 1
            # Fallback if snapping moved beyond cut_idx
            if target_start >= cut_idx:
                target_start = cut_idx

        # Guarantee forward progress
        start = max(start + 1, target_start)

    return chunks


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split extracted document pages into overlapping chunks.

    Args:
        pages: List of page dictionaries from load_document,
            each containing 'text', 'file', and 'page'.
        chunk_size: Maximum character length for each chunk. Defaults to CHUNK_SIZE (800).
        overlap: Character overlap between consecutive chunks. Defaults to CHUNK_OVERLAP (100).

    Returns:
        A list of dictionaries with keys:
            - 'text' (str): The chunked text content.
            - 'file' (str): The original filename.
            - 'page' (int): The page number.
            - 'chunk_id' (str): Unique chunk identifier formatted as '<filename>_p<page>_c<chunk>'.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    chunked_results: list[dict[str, Any]] = []

    for page_data in pages:
        text = page_data.get("text", "")
        file_name = page_data.get("file", "document")
        page_num = page_data.get("page", 1)

        # Base name for chunk_id (e.g., 'sample' from 'sample.pdf')
        base_name = Path(file_name).stem if file_name else "document"

        text_pieces = split_text(text, chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, piece in enumerate(text_pieces, start=1):
            chunk_id = f"{base_name}_p{page_num}_c{chunk_idx}"
            chunked_results.append({
                "text": piece,
                "file": file_name,
                "page": page_num,
                "chunk_id": chunk_id,
            })

    return chunked_results


if __name__ == "__main__":
    import sys

    try:
        from src.loader import load_document
    except ImportError:
        try:
            from loader import load_document
        except ImportError:
            load_document = None

    if load_document is None:
        print("Error: Could not import load_document.")
        sys.exit(1)

    sample_pdf_path = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"
    if not sample_pdf_path.exists():
        print(f"Sample PDF not found at {sample_pdf_path}")
        sys.exit(1)

    pages = load_document(sample_pdf_path)
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    print(f"Number of pages: {len(pages)}")
    print(f"Number of chunks: {len(chunks)}")
    if chunks:
        print(f"First chunk:\n{chunks[0]}")
