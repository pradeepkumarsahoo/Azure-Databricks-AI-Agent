# Author: Pradeep Sahoo
"""Local dev VectorStore backend: Chroma (dense) + BM25 (sparse) with RRF fusion."""
from __future__ import annotations

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from policypilot.config import CHROMA_DIR, EMBEDDING_MODEL
from policypilot.retrieval.base import SearchResult
from policypilot.retrieval.hybrid import rrf_fuse

_COLLECTION_NAME = "policypilot_filings"


class LocalChromaVectorStore:
    def __init__(self, persist_dir: str | None = None):
        persist_dir = persist_dir or str(CHROMA_DIR)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(_COLLECTION_NAME)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[str] = []
        self._bm25_metas: list[dict] = []
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        data = self._collection.get(include=["documents", "metadatas"])
        docs: list[str] = data.get("documents") or []
        metas: list[dict] = data.get("metadatas") or []
        self._bm25_corpus = docs
        self._bm25_metas = metas
        if docs:
            self._bm25 = BM25Okapi([d.lower().split() for d in docs])
        else:
            self._bm25 = None

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()
        self._collection.upsert(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )
        self._rebuild_bm25()

    def _dense_search(self, query: str, k: int, ticker: str | None) -> list[SearchResult]:
        query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()
        query_kwargs: dict = {"query_embeddings": query_embedding, "n_results": k}
        if ticker:
            query_kwargs["where"] = {"ticker": ticker}
        results = self._collection.query(**query_kwargs)
        out: list[SearchResult] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]
        for text, meta, dist in zip(docs[0], metas[0], dists[0], strict=False):
            out.append(SearchResult(text=text, score=1 - dist, metadata=meta))
        return out

    def _sparse_search(self, query: str, k: int, ticker: str | None) -> list[SearchResult]:
        if self._bm25 is None:
            return []
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        indices: list[int] | range = range(len(self._bm25_corpus))
        if ticker:
            indices = [i for i in indices if self._bm25_metas[i].get("ticker") == ticker]
        sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)[:k]
        return [
            SearchResult(
                text=self._bm25_corpus[i],
                score=float(scores[i]),
                metadata=self._bm25_metas[i],
            )
            for i in sorted_indices
            if scores[i] > 0
        ]

    def search(self, query: str, k: int = 5, ticker: str | None = None) -> list[SearchResult]:
        candidates = max(k * 4, 20)
        dense = self._dense_search(query, k=candidates, ticker=ticker)
        sparse = self._sparse_search(query, k=candidates, ticker=ticker)
        fused = rrf_fuse(dense, sparse)
        return fused[:k]

    def count(self) -> int:
        return self._collection.count()
