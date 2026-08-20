"""Pydantic request/response models for the FastAPI app."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to answer and verify.")


class RetrievedSource(BaseModel):
    chunk_id: str
    source: str
    score: float


class VerifyResponse(BaseModel):
    query: str
    answer: str
    status: Literal["answered", "insufficient_evidence"]
    confidence_verbalized: float
    confidence_engineered: float
    revision_count: int
    sources: list[RetrievedSource]


class HealthResponse(BaseModel):
    status: str
    corpus_chunk_count: int
    model: str
