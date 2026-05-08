"""Dense embeddings + FAISS inner-product search (vectors L2-normalized => cosine).

The SentenceTransformer model and FAISS index are built lazily on first search so
importing the app does not download weights until vector or hybrid mode is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.search.corpus import Document


@dataclass
class VectorHit:
    doc_id: str
    score: float


class VectorIndex:
    def __init__(
        self,
        documents: list[Document],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._docs = documents
        self._doc_ids = [d.doc_id for d in documents]
        self._bodies = [d.body for d in documents]
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._dim: int = 0

    def _ensure_built(self) -> None:
        """Load model if needed, embed all documents, build a flat inner-product index."""
        if self._index is not None:
            return
        self._model = SentenceTransformer(self._model_name)
        embeddings = self._model.encode(
            self._bodies,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(embeddings, dtype=np.float32)
        self._dim = arr.shape[1]
        # IndexFlatIP + unit vectors => cosine similarity.
        index = faiss.IndexFlatIP(self._dim)
        index.add(arr)
        self._index = index

    def search(self, query: str, top_k: int = 10) -> list[VectorHit]:
        """Embed query once and run k-NN in embedding space."""
        self._ensure_built()
        assert self._model is not None and self._index is not None
        q = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        qv = np.asarray(q, dtype=np.float32)
        scores, indices = self._index.search(qv, min(top_k, len(self._doc_ids)))
        out: list[VectorHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            out.append(VectorHit(doc_id=self._doc_ids[int(idx)], score=float(score)))
        return out

    def snippet(self, doc_id: str, query: str, window: int = 20) -> str:
        """Cheap excerpt: first substring match on a word, else start of document."""
        try:
            idx = self._doc_ids.index(doc_id)
        except ValueError:
            return ""
        text = self._bodies[idx]
        words = text.split()
        if not words:
            return ""
        q_lower = query.lower()
        best = 0
        for i, w in enumerate(words):
            if q_lower in w.lower():
                best = i
                break
        lo = max(0, best - 3)
        hi = min(len(words), lo + window)
        chunk = words[lo:hi]
        prefix = "… " if lo > 0 else ""
        suffix = " …" if hi < len(words) else ""
        return prefix + " ".join(chunk) + suffix
