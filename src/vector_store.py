"""Vector store module for docchat-ai.

Manages persistent ChromaDB vector storage with cosine distance similarity,
providing methods to upsert chunk embeddings, search semantically, list files,
and delete chunks associated with a specific file.
"""

import os
from pathlib import Path
import time
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv

# Ensure .env is loaded
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

# Import configuration constants with fallbacks
try:
    from src.config import CHROMA_PATH, COLLECTION_NAME, TOP_K
except ImportError:
    try:
        from config import CHROMA_PATH, COLLECTION_NAME, TOP_K
    except ImportError:
        CHROMA_PATH = "chroma_db"
        COLLECTION_NAME = "docchat"
        TOP_K = 6

# Import embedding functions with fallbacks
try:
    from src.embeddings import embed_query, embed_texts
except ImportError:
    try:
        from embeddings import embed_query, embed_texts
    except ImportError as err:
        raise ImportError(f"Could not import embedding functions: {err}") from err

# Module-level cache for client and collections keyed by collection name
_chroma_client: Optional[chromadb.PersistentClient] = None
_collections: dict[str, chromadb.Collection] = {}


def get_chroma_client(path: str = CHROMA_PATH) -> chromadb.PersistentClient:
    """Retrieve or create the persistent ChromaDB client.

    Args:
        path: Filesystem directory path for the persistent Chroma database.

    Returns:
        chromadb.PersistentClient instance.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=path)
    return _chroma_client


def get_collection(
    name: str = COLLECTION_NAME,
    path: str = CHROMA_PATH,
) -> chromadb.Collection:
    """Retrieve or create the ChromaDB collection configured with cosine distance.

    Caches collection instances in a dict keyed by collection name. Stores 'created_at'
    timestamp in collection metadata.

    Args:
        name: Name of the collection. Defaults to COLLECTION_NAME ('docchat').
        path: Path to the Chroma database directory.

    Returns:
        chromadb.Collection instance configured with cosine distance.
    """
    global _collections
    if name not in _collections:
        client = get_chroma_client(path=path)
        _collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "created_at": time.time()},
            embedding_function=None,
        )
    return _collections[name]


def cleanup_old_collections(
    max_session_collections: int = 20,
    path: str = CHROMA_PATH,
) -> int:
    """If more than max_session_collections 's_*' collections exist, delete the oldest ones.

    Uses 'created_at' in collection metadata to determine age.

    Args:
        max_session_collections: Maximum number of session collections to retain (default 20).
        path: Path to Chroma database.

    Returns:
        Number of collections deleted.
    """
    global _collections
    client = get_chroma_client(path=path)
    try:
        all_cols = client.list_collections()
    except Exception:
        return 0

    session_cols: list[tuple[float, str]] = []
    for c in all_cols:
        c_name = getattr(c, "name", str(c))
        if c_name.startswith("s_"):
            meta = getattr(c, "metadata", {}) or {}
            created_at = meta.get("created_at", 0.0) if isinstance(meta, dict) else 0.0
            session_cols.append((float(created_at), c_name))

    deleted_count = 0
    if len(session_cols) > max_session_collections:
        # Sort ascending by created_at (oldest first)
        session_cols.sort(key=lambda item: item[0])
        num_to_delete = len(session_cols) - max_session_collections
        for _, c_name in session_cols[:num_to_delete]:
            try:
                client.delete_collection(name=c_name)
                _collections.pop(c_name, None)
                deleted_count += 1
            except Exception:
                pass

    return deleted_count


def add_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = 500,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Embed and store document chunks in the vector collection using upsert.

    Embeds chunk texts with task_type 'RETRIEVAL_DOCUMENT' and stores them
    with chunk_id as the vector ID, preserving file and page as metadata.
    Uses upsert to prevent duplicates when re-uploading existing files.

    Args:
        chunks: List of chunk dictionaries containing 'text', 'file',
            'page', and 'chunk_id'.
        batch_size: Number of records to upsert per ChromaDB batch. Defaults to 500.
        collection_name: Target collection name. Defaults to COLLECTION_NAME ('docchat').

    Raises:
        ValueError: If GEMINI_API_KEY is missing.
    """
    if not chunks:
        return

    collection = get_collection(name=collection_name)

    ids = [str(c["chunk_id"]) for c in chunks]
    texts = [str(c.get("text", "")) for c in chunks]
    metadatas = [
        {
            "file": str(c.get("file", "")),
            "page": int(c.get("page", 1)),
        }
        for c in chunks
    ]

    # Generate embeddings for all chunk texts
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    # Upsert in batches to respect database batch limits
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def search(
    query: str,
    k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
    file: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Perform semantic similarity search on stored chunks.

    Args:
        query: User question or search query string.
        k: Maximum number of top matching chunks to retrieve. Defaults to TOP_K.
        collection_name: Target collection name. Defaults to COLLECTION_NAME ('docchat').
        file: Optional filename filter to restrict search to a specific document.

    Returns:
        List of dictionaries with keys:
            - 'text' (str): Chunk text content.
            - 'file' (str): Source filename.
            - 'page' (int): Source page number.
            - 'score' (float): Cosine similarity score (1 - cosine distance).

    Raises:
        ValueError: If GEMINI_API_KEY is missing.
    """
    if not query or not query.strip():
        return []

    collection = get_collection(name=collection_name)
    count = collection.count()
    if count == 0:
        return []

    n_results = min(k, count)
    if n_results <= 0:
        return []

    query_vector = embed_query(query.strip())

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if file:
        query_kwargs["where"] = {"file": file}

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        return []

    hits: list[dict[str, Any]] = []
    docs_list = results.get("documents", [[]])
    if not docs_list or not docs_list[0]:
        return hits

    documents = docs_list[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for i, doc in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
        dist = distances[i] if i < len(distances) else 0.0
        # For cosine distance: cosine_similarity = 1 - cosine_distance
        score = round(float(1.0 - dist), 4)

        hits.append({
            "text": doc,
            "file": meta.get("file", ""),
            "page": meta.get("page", 1),
            "score": score,
        })

    return hits


def list_files(collection_name: str = COLLECTION_NAME) -> list[str]:
    """Retrieve a sorted list of unique filenames currently stored in the collection.

    Args:
        collection_name: Target collection name. Defaults to COLLECTION_NAME ('docchat').

    Returns:
        Sorted list of unique filename strings.
    """
    collection = get_collection(name=collection_name)
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    unique_files = {
        meta["file"]
        for meta in metadatas
        if isinstance(meta, dict) and "file" in meta and meta["file"]
    }
    return sorted(unique_files)


def delete_file(filename: str, collection_name: str = COLLECTION_NAME) -> None:
    """Delete all chunks belonging to a specific filename from the collection.

    Args:
        filename: Name of the file whose chunks should be removed.
        collection_name: Target collection name. Defaults to COLLECTION_NAME ('docchat').
    """
    if not filename:
        return
    collection = get_collection(name=collection_name)
    collection.delete(where={"file": filename})


if __name__ == "__main__":
    import sys

    # Check for GEMINI_API_KEY before running test
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        print(
            "Error: GEMINI_API_KEY is not set.\n"
            "Please create a .env file in the project root containing:\n"
            "  GEMINI_API_KEY=your_gemini_api_key_here\n"
            "or set the GEMINI_API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Locate sample PDF at data/sample.pdf relative to project root
    sample_pdf_path = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"
    if not sample_pdf_path.exists():
        print(f"Error: Sample PDF not found at {sample_pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Resilient imports for loader and chunker
    try:
        from src.loader import load_document
        from src.chunker import chunk_pages
    except ImportError:
        try:
            from loader import load_document
            from chunker import chunk_pages
        except ImportError as exc:
            print(f"Error importing loader or chunker: {exc}", file=sys.stderr)
            sys.exit(1)

    print("==================================================")
    print("           DocChat-AI Vector Store Test           ")
    print("==================================================")

    # 1. Load document
    print(f"\n[1/4] Loading document from: {sample_pdf_path.name}...")
    pages = load_document(sample_pdf_path)
    print(f"      Loaded {len(pages)} pages successfully.")

    # 2. Chunk document
    print(f"\n[2/4] Chunking pages...")
    chunks = chunk_pages(pages)
    print(f"      Generated {len(chunks)} chunks.")

    # 3. Store chunks in vector store
    print(f"\n[3/4] Adding chunks to Chroma collection '{COLLECTION_NAME}'...")
    add_chunks(chunks)
    stored_files = list_files()
    print(f"      Stored successfully! Unique files in index: {stored_files}")

    # 4. Search
    test_query = "How does chunking work?"
    print(f"\n[4/4] Searching for: '{test_query}' (Top {TOP_K})...\n")
    search_results = search(test_query, k=TOP_K)

    print(f"Retrieved {len(search_results)} relevant chunks:")
    print("--------------------------------------------------")
    for rank, result in enumerate(search_results, start=1):
        print(f"Rank {rank} | Score: {result['score']:.4f} | File: {result['file']} | Page: {result['page']}")
        # Print snippet of chunk text
        snippet = result["text"].replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(f"Snippet: {snippet}")
        print("--------------------------------------------------")

    print("\nVector store test completed successfully!")
