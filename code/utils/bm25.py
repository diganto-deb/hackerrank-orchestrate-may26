from __future__ import annotations

import pickle

from rank_bm25 import BM25Okapi


def build_bm25(
    texts: list[str], doc_ids: list[str]
) -> tuple[BM25Okapi, list[str]]:
    """Build a BM25 index from texts and document IDs.

    Args:
        texts: List of document texts to index.
        doc_ids: List of document identifiers corresponding to texts.

    Returns:
        Tuple of (BM25Okapi index, doc_ids list).
    """
    tokenized = [text.lower().split() for text in texts]
    return BM25Okapi(tokenized), doc_ids


def save_bm25(index: BM25Okapi, doc_ids: list[str], path: str) -> None:
    """Serialize and save BM25 index and document IDs to disk.

    Args:
        index: BM25Okapi index to save.
        doc_ids: Document ID list to save.
        path: File path to save pickle to.
    """
    with open(path, "wb") as f:
        pickle.dump({"index": index, "doc_ids": doc_ids}, f)


def load_bm25(path: str) -> tuple[BM25Okapi, list[str]]:
    """Load a BM25 index and document IDs from disk.

    Args:
        path: File path to load pickle from.

    Returns:
        Tuple of (BM25Okapi index, doc_ids list).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["doc_ids"]


def search_bm25(
    index: BM25Okapi, doc_ids: list[str], query: str, top_k: int = 5
) -> list[tuple[str, float]]:
    """Search the BM25 index for documents matching a query.

    Args:
        index: BM25Okapi index to search.
        doc_ids: Document ID list corresponding to index.
        query: Search query text.
        top_k: Maximum number of results to return.

    Returns:
        List of (doc_id, score) tuples sorted by relevance descending,
        capped at top_k, with zero-score results excluded.
    """
    tokens = query.lower().split()
    scores = index.get_scores(tokens)
    ranked = sorted(
        zip(doc_ids, scores), key=lambda x: x[1], reverse=True
    )
    return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0.0]
