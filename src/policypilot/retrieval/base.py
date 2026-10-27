# Author: Pradeep Sahoo
"""VectorStore and Reranker protocols — swap implementations via config."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    rank: int = 0


class VectorStore(Protocol):
    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None: ...

    def search(self, query: str, k: int = 5, ticker: str | None = None) -> list[SearchResult]: ...


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
