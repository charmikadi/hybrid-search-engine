from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SearchMode = Literal["keyword", "vector", "hybrid"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: SearchMode = "hybrid"
    top_k: int = Field(10, ge=1, le=50)


class SearchHit(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float
    rrf_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    hits: list[SearchHit]
    took_ms: float


class HealthResponse(BaseModel):
    status: str
    documents: int
