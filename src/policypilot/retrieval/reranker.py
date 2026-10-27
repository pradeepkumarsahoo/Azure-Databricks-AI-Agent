"""Cross-encoder reranker for precision re-scoring after initial retrieval."""
from __future__ import annotations

from policypilot.config import RERANKER_MODEL
from policypilot.retrieval.base import SearchResult


class CrossEncoderReranker:
    """Reranks candidates using a cross-encoder (query, passage) scorer.

    Call rerank() after an initial broad retrieval (top_k_retrieval) to select
    the top_k_rerank results with highest cross-attention relevance scores.
    """

    def __init__(self, model: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return results
        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        out = []
        for rank, (result, score) in enumerate(ranked[:top_k], start=1):
            result.score = float(score)
            result.rank = rank
            out.append(result)
        return out
