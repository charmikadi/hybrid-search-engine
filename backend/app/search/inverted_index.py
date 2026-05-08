"""Classic inverted index with TF-IDF scoring (cosine on weighted term vectors).

Tokenization is alphanumeric runs (lowercased). Query--document similarity uses
dot product of TF-IDF weights divided by query and document L2 norms (cosine).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from app.search.corpus import Document

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens only (no stemming)."""
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class Hit:
    doc_id: str
    score: float


class TfidfInvertedIndex:
    """
    In-memory inverted index: term -> list of (doc_index, term_frequency).
    IDF uses smoothed log: log((N + 1) / (df + 1)) + 1.
    Query scoring: sum(tf * idf) for overlapping terms (BM25-like intuition, TF-IDF weighting).
    """

    def __init__(self, documents: list[Document]) -> None:
        self._docs = documents
        self._doc_ids = [d.doc_id for d in documents]
        self._bodies = [d.body for d in documents]
        n = len(documents)
        self._tfs: list[dict[str, int]] = []
        inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: dict[str, int] = defaultdict(int)

        for i, body in enumerate(self._bodies):
            tokens = tokenize(body)
            tf: dict[str, int] = defaultdict(int)
            seen_terms: set[str] = set()
            for t in tokens:
                tf[t] += 1
            for t in tf:
                seen_terms.add(t)
            # Document frequency: count each term at most once per document.
            for t in seen_terms:
                df[t] += 1
            self._tfs.append(dict(tf))
            for term, c in tf.items():
                inverted[term].append((i, c))

        self._inverted = dict(inverted)
        self._idf: dict[str, float] = {}
        for term, dfi in df.items():
            self._idf[term] = math.log((n + 1) / (dfi + 1)) + 1.0

        # Per-document ||tf-idf|| for cosine denominator.
        self._doc_norm: list[float] = []
        for i in range(n):
            s = 0.0
            for term, c in self._tfs[i].items():
                w = c * self._idf.get(term, 0.0)
                s += w * w
            self._doc_norm.append(math.sqrt(s) if s > 0 else 1.0)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        """Score all docs that share at least one query term; return top_k by cosine similarity."""
        q_terms = tokenize(query)
        if not q_terms:
            return []

        q_tf: dict[str, int] = defaultdict(int)
        for t in q_terms:
            q_tf[t] += 1

        scores: dict[int, float] = defaultdict(float)
        q_weight_sq = 0.0
        for term, qc in q_tf.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            qw = qc * idf
            q_weight_sq += qw * qw
            postings = self._inverted.get(term)
            if not postings:
                continue
            for doc_i, tf in postings:
                dw = tf * idf
                scores[doc_i] += qw * dw

        q_norm = math.sqrt(q_weight_sq) if q_weight_sq > 0 else 1.0
        ranked: list[tuple[int, float]] = []
        for doc_i, raw in scores.items():
            sim = raw / (q_norm * self._doc_norm[doc_i])
            ranked.append((doc_i, sim))

        ranked.sort(key=lambda x: x[1], reverse=True)
        out: list[Hit] = []
        for doc_i, score in ranked[:top_k]:
            out.append(Hit(doc_id=self._doc_ids[doc_i], score=float(score)))
        return out

    def snippet(self, doc_id: str, query: str, window: int = 24) -> str:
        """Return a short window of token-aligned text with maximal query-term overlap."""
        try:
            idx = self._doc_ids.index(doc_id)
        except ValueError:
            return ""
        body = self._bodies[idx]
        tokens = tokenize(body)
        q_set = set(tokenize(query))
        if not tokens:
            return ""
        best_start = 0
        best_hits = -1
        for start in range(max(1, len(tokens) - window + 1)):
            chunk = tokens[start : start + window]
            hits = sum(1 for t in chunk if t in q_set)
            if hits > best_hits:
                best_hits = hits
                best_start = start
        # Align to raw word spans from the original body (tokenization differs slightly from split).
        words = _TOKEN.findall(body)
        lo = max(0, best_start)
        hi = min(len(words), best_start + window)
        snippet_words = words[lo:hi]
        prefix = "… " if lo > 0 else ""
        suffix = " …" if hi < len(words) else ""
        return prefix + " ".join(snippet_words) + suffix
