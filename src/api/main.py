"""FastAPI app exposing the verification pipeline over HTTP.

Kept deliberately minimal: one endpoint that runs a query through the full
Generator -> Critic -> Coordinator graph and returns the final answer,
status, and both confidence scores, plus a health check. The retriever
(and the sentence-transformers embedding model it loads) is constructed
once at import time, not per-request — reloading it on every call would
dominate latency.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, RetrievedSource, VerifyRequest, VerifyResponse
from verification.config import settings
from verification.graph import run_verification
from verification.retrieval.vector_store import ChromaRetriever

app = FastAPI(
    title="Adversarial Verification API",
    description=(
        "Multi-agent (Generator/Critic/Coordinator) verification pipeline "
        "over a retrieval-augmented corpus."
    ),
    version="0.1.0",
)

_retriever = ChromaRetriever()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        corpus_chunk_count=_retriever.count(),
        model=settings.openai_model,
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest) -> VerifyResponse:
    if _retriever.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Corpus index is empty — run `python scripts/build_index.py` before querying.",
        )

    state = run_verification(request.query, _retriever)

    sources = [
        RetrievedSource(chunk_id=c["chunk_id"], source=c["source"], score=c["score"])
        for c in state.get("retrieved_chunks", [])
    ]

    return VerifyResponse(
        query=request.query,
        answer=state["final_answer"],
        status=state["final_status"],
        confidence_verbalized=state["confidence_verbalized"],
        confidence_engineered=state["confidence_engineered"],
        revision_count=state.get("revision_count", 0),
        sources=sources,
    )
