"""Coordinator agent: arbitrates between the Generator's draft and the
Critic's findings.

Deliberately NOT another open-ended LLM conversation — accept/revise/escalate
is pure rule-based logic over the Critic's finding count and the revision
budget (deterministic, auditable). The one LLM call here is reserved for
something that needs judgment: self-reported confidence, used only as the
naive baseline the eval compares against the engineered score (`confidence.py`).

Decision rules (see `_decide`):
  - no evidence retrieved at all -> escalate
  - critic found nothing -> accept
  - critic found problems, revision available -> revise
  - critic found problems, revision budget exhausted -> escalate
"""

from __future__ import annotations

from verification.config import settings
from verification.confidence import get_engineered_confidence, get_verbalized_confidence
from verification.state import CoordinatorDecision, VerificationState

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence in the corpus to answer this question reliably."
)


def _decide(state: VerificationState) -> CoordinatorDecision:
    chunks = state.get("retrieved_chunks", [])
    report = state.get("critic_report", {"findings": []})
    flagged = len(report.get("findings", []))
    revision_count = state.get("revision_count", 0)

    if not chunks:
        return "escalate"
    if flagged == 0:
        return "accept"
    if revision_count < settings.max_revision_cycles:
        return "revise"
    return "escalate"


def coordinate_node(state: VerificationState) -> dict:
    decision = _decide(state)

    if decision == "revise":
        return {
            "coordinator_decision": "revise",
            "revision_count": state.get("revision_count", 0) + 1,
            "trace": [{"node": "coordinate", "decision": "revise"}],
        }

    final_status = "answered" if decision == "accept" else "insufficient_evidence"
    final_answer = (
        state.get("generator_answer", "") if decision == "accept" else INSUFFICIENT_EVIDENCE_MESSAGE
    )

    # Confidence is computed against the state as it will be returned (i.e.
    # with `final_answer` populated), so get_verbalized_confidence sees the
    # actual final answer rather than an in-progress draft.
    scored_state: VerificationState = {**state, "final_answer": final_answer}
    verbalized, verbalized_usage = get_verbalized_confidence(scored_state)
    engineered = get_engineered_confidence(state)

    return {
        "coordinator_decision": decision,
        "final_status": final_status,
        "final_answer": final_answer,
        "confidence_verbalized": verbalized,
        "confidence_engineered": engineered,
        "trace": [{"node": "coordinate", "decision": decision, **verbalized_usage}],
    }
