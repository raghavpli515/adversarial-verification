# Shared state schema for the verification graph.

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    source: str
    score: float


class CriticFinding(TypedDict):
    claim: str
    issue_type: Literal["unsupported", "contradiction", "overconfident"]
    explanation: str


class CriticReport(TypedDict):
    findings: list[CriticFinding]
    supported_claim_count: int
    flagged_claim_count: int


CoordinatorDecision = Literal["accept", "revise", "escalate"]
FinalStatus = Literal["answered", "insufficient_evidence"]


class VerificationState(TypedDict, total=False):
    # --- input ---
    query: str

    # --- retrieval ---
    retrieved_chunks: list[RetrievedChunk]

    # --- generator ---
    generator_answer: str
    generator_citations: list[str]  # chunk_ids the generator claims to rely on

    # --- critic ---
    critic_report: CriticReport

    # --- coordinator ---
    coordinator_decision: CoordinatorDecision
    revision_count: int  # bumped each time coordinator routes back to generate

    # --- final output ---
    final_answer: str
    final_status: FinalStatus
    confidence_verbalized: float  # coordinator's self-reported 0-1 confidence
    confidence_engineered: float  # formula over retrieval/critic signals — see confidence.py

    # --- observability (append-only across the whole run) ---
    trace: Annotated[list[dict], operator.add]
