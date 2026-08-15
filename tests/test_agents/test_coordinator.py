"""Unit tests for the Coordinator's accept/revise/escalate decision logic.

The decision itself is deliberately rule-based (see coordinator.py's module
docstring for why), which makes it fully testable without touching the
network — every branch of `_decide` is exercised directly, and
`coordinate_node`'s LLM call (verbalized confidence) is mocked out so these
tests run offline.
"""

from __future__ import annotations

from unittest.mock import patch

from verification.agents.coordinator import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    _decide,
    coordinate_node,
)


def _state(**overrides):
    base = {
        "query": "q",
        "retrieved_chunks": [{"chunk_id": "a::0", "text": "t", "source": "a.md", "score": 0.8}],
        "generator_answer": "The answer.",
        "generator_citations": ["a::0"],
        "critic_report": {"findings": [], "supported_claim_count": 1, "flagged_claim_count": 0},
        "revision_count": 0,
    }
    base.update(overrides)
    return base


def _with_findings(**state_overrides):
    return _state(
        critic_report={
            "findings": [{"claim": "x", "issue_type": "unsupported", "explanation": "e"}],
            "supported_claim_count": 0,
            "flagged_claim_count": 1,
        },
        **state_overrides,
    )


def test_decide_escalates_when_nothing_retrieved():
    assert _decide(_state(retrieved_chunks=[])) == "escalate"


def test_decide_accepts_when_critic_finds_nothing():
    assert _decide(_state()) == "accept"


def test_decide_revises_when_findings_and_budget_remains():
    assert _decide(_with_findings(revision_count=0)) == "revise"


def test_decide_escalates_when_findings_and_budget_exhausted():
    # settings.max_revision_cycles defaults to 1 (see config.py) — a state
    # arriving with revision_count=1 has already used its one revision.
    assert _decide(_with_findings(revision_count=1)) == "escalate"


def test_no_retrieval_escalates_even_if_critic_somehow_found_nothing():
    # Belt-and-suspenders: empty retrieval must win over a clean critic
    # report, since a clean report over zero evidence is meaningless.
    state = _state(retrieved_chunks=[], critic_report={"findings": [], "supported_claim_count": 0})
    assert _decide(state) == "escalate"


def test_coordinate_node_revise_bumps_revision_count_and_skips_confidence_calls():
    with patch("verification.agents.coordinator.get_verbalized_confidence") as mock_verbalized:
        result = coordinate_node(_with_findings(revision_count=0))

    mock_verbalized.assert_not_called()
    assert result["coordinator_decision"] == "revise"
    assert result["revision_count"] == 1
    assert "final_answer" not in result


def test_coordinate_node_accept_produces_final_answer_and_both_confidences():
    with (
        patch("verification.agents.coordinator.get_verbalized_confidence", return_value=(0.9, {})),
        patch("verification.agents.coordinator.get_engineered_confidence", return_value=0.85),
    ):
        result = coordinate_node(_state())

    assert result["coordinator_decision"] == "accept"
    assert result["final_status"] == "answered"
    assert result["final_answer"] == "The answer."
    assert result["confidence_verbalized"] == 0.9
    assert result["confidence_engineered"] == 0.85


def test_coordinate_node_escalate_uses_fixed_insufficient_evidence_message():
    with (
        patch("verification.agents.coordinator.get_verbalized_confidence", return_value=(0.2, {})),
        patch("verification.agents.coordinator.get_engineered_confidence", return_value=0.1),
    ):
        result = coordinate_node(_state(retrieved_chunks=[]))

    assert result["coordinator_decision"] == "escalate"
    assert result["final_status"] == "insufficient_evidence"
    assert result["final_answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
