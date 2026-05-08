"""Wires corpus + keyword index + vector index + hybrid fusion.

`SearchEngine` loads documents once, builds TF-IDF and FAISS-backed indexes,
and dispatches `keyword` / `vector` / `hybrid` queries. The process-wide
singleton is set via `init_engine` from the FastAPI lifespan hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.search.corpus import Document, load_corpus
from app.search.hybrid import reciprocal_rank_fusion
from app.search.inverted_index import TfidfInvertedIndex
from app.search.vector_store import VectorIndex

SearchMode = Literal["keyword", "vector", "hybrid"]


@dataclass
class SearchResultItem:
    """One ranked hit: primary `score` from the active retriever; `rrf_score` only for hybrid."""

    doc_id: str
    title: str
    snippet: str
    score: float
    rrf_score: float | None


class SearchEngine:
    def __init__(self, corpus_path: Path) -> None:
        self._corpus_path = corpus_path
        self._documents: list[Document] = load_corpus(corpus_path)
        self._by_id: dict[str, Document] = {d.doc_id: d for d in self._documents}
        # Both indexes are built eagerly; vector path may download the embedding model.
        self._keyword = TfidfInvertedIndex(self._documents)
        self._vector = VectorIndex(self._documents)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        top_k: int = 10,
    ) -> list[SearchResultItem]:
        """Return up to `top_k` hits; empty query yields no results."""
        q = query.strip()
        if not q:
            return []

        if mode == "keyword":
            hits = self._keyword.search(q, top_k=top_k)
            return [
                SearchResultItem(
                    doc_id=h.doc_id,
                    title=self._by_id[h.doc_id].title,
                    snippet=self._keyword.snippet(h.doc_id, q),
                    score=h.score,
                    rrf_score=None,
                )
                for h in hits
            ]

        if mode == "vector":
            hits = self._vector.search(q, top_k=top_k)
            return [
                SearchResultItem(
                    doc_id=h.doc_id,
                    title=self._by_id[h.doc_id].title,
                    snippet=self._vector.snippet(h.doc_id, q),
                    score=h.score,
                    rrf_score=None,
                )
                for h in hits
            ]

        # Hybrid: retrieve a wider candidate pool from each channel, then RRF re-rank to top_k.
        kw = self._keyword.search(q, top_k=max(top_k * 3, 20))
        vec = self._vector.search(q, top_k=max(top_k * 3, 20))
        fused = reciprocal_rank_fusion(kw, vec, top_k=top_k)
        items: list[SearchResultItem] = []
        for f in fused:
            # Prefer keyword-highlight-style snippet; fall back to vector store snippet.
            snip = self._keyword.snippet(f.doc_id, q) or self._vector.snippet(
                f.doc_id, q
            )
            items.append(
                SearchResultItem(
                    doc_id=f.doc_id,
                    title=self._by_id[f.doc_id].title,
                    snippet=snip,
                    score=f.score,
                    rrf_score=f.rrf_score,
                )
            )
        return items


# Single shared instance for the running process (set before handling requests).
_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    """Return the initialized engine; raises if `init_engine` has not run."""
    global _engine
    if _engine is None:
        raise RuntimeError("Search engine not initialized")
    return _engine


def init_engine(corpus_path: Path) -> SearchEngine:
    """Build indexes from `corpus_path` and store as the global engine."""
    global _engine
    _engine = SearchEngine(corpus_path)
    return _engine
