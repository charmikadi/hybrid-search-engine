"""FastAPI: hybrid keyword + dense vector search with RRF fusion.

Loads the JSONL corpus once at startup, builds keyword + vector indexes in memory,
and exposes /health and /search.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.engine import SearchEngine, get_engine, init_engine
from app.schemas import HealthResponse, SearchHit, SearchRequest, SearchResponse

# Default corpus: repo root is two levels above this file (backend/app/main.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CORPUS = _REPO_ROOT / "data" / "corpus.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: resolve corpus path, fail fast if missing, then build SearchEngine.
    corpus = Path(os.environ.get("HYBRID_CORPUS_PATH", str(_DEFAULT_CORPUS)))
    if not corpus.is_file():
        raise RuntimeError(f"Corpus not found: {corpus}")
    init_engine(corpus)
    yield
    # Shutdown: nothing to tear down (indexes live in process memory).


app = FastAPI(
    title="Hybrid Search Engine",
    description="Inverted-index TF-IDF keyword search + sentence embeddings (FAISS) + RRF hybrid fusion.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the Vite dev server (or any origin) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check and document count after indexes are ready."""
    eng: SearchEngine = get_engine()
    return HealthResponse(status="ok", documents=eng.document_count)


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    """Run keyword, vector, or hybrid search; return ranked hits and wall time."""
    eng = get_engine()
    t0 = time.perf_counter()
    try:
        rows = eng.search(body.query, mode=body.mode, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    took = (time.perf_counter() - t0) * 1000.0

    # Map internal row objects to API response models (Pydantic).
    hits = [
        SearchHit(
            doc_id=r.doc_id,
            title=r.title,
            snippet=r.snippet,
            score=r.score,
            rrf_score=r.rrf_score,
        )
        for r in rows
    ]
    return SearchResponse(
        query=body.query,
        mode=body.mode,
        hits=hits,
        took_ms=round(took, 2),
    )
