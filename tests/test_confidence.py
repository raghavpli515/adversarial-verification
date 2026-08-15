"""Unit tests for the engineered confidence formula — the core "measurable
calibration" logic the project's headline claim rests on. Pure function,
no network calls, tested directly against synthetic state.
"""

from __future__ import annotations

from verification.confidence import get_engineered_confidence


def _state(**overrides):
    base = {
        "retrieved_chunks": [
            {"chunk_id": "a::0", "text": "...", "source": "a.md", "score": 0.9},
            {"chunk_id": "a::1", "text": "...", "source": "a.md", "score": 0.6},
        ],
        "generator_citations": ["a::0"],
        "critic_report": {"findings": [], "supported_claim_count": 3, "flagged_claim_count": 0},
        "revision_count": 0,
    }
    base.update(overrides)
    return base


def test_engineered_confidence_high_when_clean():
    assert 0.8 <= get_engineered_confidence(_state()) <= 1.0


def test_engineered_confidence_drops_with_unsupported_finding():
    clean = get_engineered_confidence(_state())
    flawed = get_engineered_confidence(
        _state(
            critic_report={
                "findings": [{"claim": "x", "issue_type": "unsupported", "explanation": "e"}],
                "supported_claim_count": 2,
                "flagged_claim_count": 1,
            }
        )
    )
    assert flawed < clean


def test_overconfident_finding_penalized_less_than_unsupported():
    unsupported = get_engineered_confidence(
        _state(
            critic_report={
                "findings": [{"claim": "x", "issue_type": "unsupported", "explanation": "e"}],
                "supported_claim_count": 2,
                "flagged_claim_count": 1,
            }
        )
    )
    overconfident = get_engineered_confidence(
        _state(
            critic_report={
                "findings": [{"claim": "x", "issue_type": "overconfident", "explanation": "e"}],
                "supported_claim_count": 2,
                "flagged_claim_count": 1,
            }
        )
    )
    assert overconfident > unsupported


def test_confidence_collapses_when_citation_does_not_match_retrieved_chunks():
    # Generator cited a chunk_id that wasn't actually retrieved — retrieval
    # quality can't vouch for the answer, so this should score lower than a
    # matching citation, not silently average over unrelated chunks.
    mismatched = get_engineered_confidence(_state(generator_citations=["nonexistent::0"]))
    matched = get_engineered_confidence(_state())
    assert mismatched < matched


def test_revision_nudges_confidence_down_even_if_final_answer_is_clean():
    not_revised = get_engineered_confidence(_state(revision_count=0))
    revised = get_engineered_confidence(_state(revision_count=1))
    assert revised < not_revised


def test_confidence_bounded_0_to_1_in_worst_case():
    state = _state(
        retrieved_chunks=[{"chunk_id": "a::0", "text": "...", "source": "a.md", "score": 0.01}],
        generator_citations=["a::0"],
        critic_report={
            "findings": [
                {"claim": f"c{i}", "issue_type": "unsupported", "explanation": "e"} for i in range(5)
            ],
            "supported_claim_count": 0,
            "flagged_claim_count": 5,
        },
        revision_count=1,
    )
    score = get_engineered_confidence(state)
    assert 0.0 <= score <= 1.0
