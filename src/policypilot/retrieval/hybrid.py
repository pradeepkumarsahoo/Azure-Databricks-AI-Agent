"""Reciprocal Rank Fusion for combining dense and sparse retrieval results."""
from __future__ import annotations

from policypilot.retrieval.base import SearchResult


def rrf_fuse(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    *,
    k: int = 60,
) -> list[SearchResult]:
    """Combine two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i) over all lists where the document appears.
    k=60 is the standard value from the original RRF paper (Cormack et al. 2009).
    Deduplication uses exact text match — sufficient for chunked filings.
    """
    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(dense, start=1):
        key = r.text
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        result_map[key] = r

    for rank, r in enumerate(sparse, start=1):
        key = r.text
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in result_map:
            result_map[key] = r

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        SearchResult(text=key, score=scores[key], metadata=result_map[key].metadata)
        for key in sorted_keys
    ]
