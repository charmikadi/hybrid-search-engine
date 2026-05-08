"""Merge keyword and vector ranked lists with Reciprocal Rank Fusion (RRF).

RRF combines two orderings without requiring scores to be on the same scale.
We keep per-channel scores only to build a small tie-breaker for the API ``score`` field.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.search.inverted_index import Hit
from app.search.vector_store import VectorHit


@dataclass
class FusedHit:
    """After fusion: ``rrf_score`` is the pure RRF sum; ``score`` may add a tiny lexical+vector nudge."""

    doc_id: str
    score: float
    rrf_score: float


def reciprocal_rank_fusion(
    keyword_hits: list[Hit],
    vector_hits: list[VectorHit],
    k: int = 60,
    top_k: int = 10,
) -> list[FusedHit]:
    """
    RRF: score(d) = sum 1/(k + rank(d)) across lists where d appears.

    ``k`` is the usual RRF constant (higher k dampens rank differences).
    """
    scores: dict[str, float] = {}
    # Remember original retriever scores for optional blending in the output.
    kw_score: dict[str, float] = {}
    vec_score: dict[str, float] = {}

    for rank, h in enumerate(keyword_hits, start=1):
        scores[h.doc_id] = scores.get(h.doc_id, 0.0) + 1.0 / (k + rank)
        kw_score[h.doc_id] = h.score
    for rank, h in enumerate(vector_hits, start=1):
        scores[h.doc_id] = scores.get(h.doc_id, 0.0) + 1.0 / (k + rank)
        vec_score[h.doc_id] = h.score

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out: list[FusedHit] = []
    for doc_id, rrf in ordered:
        # Primary signal is RRF; add a small weighted sum of both channel scores when both exist.
        ks = kw_score.get(doc_id)
        vs = vec_score.get(doc_id)
        blend = rrf
        if ks is not None and vs is not None:
            blend = rrf + 0.01 * (ks + vs)
        out.append(FusedHit(doc_id=doc_id, score=float(blend), rrf_score=float(rrf)))
    return out
