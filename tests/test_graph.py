"""Integration test for the compiled LangGraph pipeline.

Node-level LLM calls are mocked at exactly three patch points — one per
call site (generator, critic, confidence) — which is possible only because
llm.py centralizes every API call behind `call_structured` (see llm.py's
docstring). That's the payoff of the wrapper: this test exercises REAL
control flow — does `coordinate` actually route back to `generate` on a
"revise" decision, does the loop stop after `max_revision_cycles`, does an
empty retrieval short-circuit straight to escalation — without hitting the
network at all.
"""

from __future__ import annotations

from unittest.mock import patch

from verification.graph import run_verification
from verification.llm import StructuredCallResult
from verification.retrieval.base import Retriever
from verification.state import RetrievedChunk


class FakeRetriever(Retriever):
    """A Retriever that returns a fixed, pre-set list of chunks — no Chroma,
    no embeddings, no I/O. This is the payoff of `Retriever` being an
    abstract interface (see retrieval/base.py): tests don't need a real
    vector store to exercise graph control flow."""

    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self._chunks


def _chunk(chunk_id: str = "c::0", score: float = 0.9) -> RetrievedChunk:
    return {
        "chunk_id": chunk_id,
        "text": "Neil Armstrong commanded Apollo 11.",
        "source": "c.md",
        "score": score,
    }


def _structured(data: dict) -> StructuredCallResult:
    return StructuredCallResult(data=data, input_tokens=10, output_tokens=10, latency_ms=5.0)


def test_graph_accept_path_no_revision_needed():
    retriever = FakeRetriever([_chunk()])

    generate_result = _structured({"answer": "Neil Armstrong.", "cited_chunk_ids": ["c::0"]})
    critic_result = _structured({"findings": [], "supported_claim_count": 1})
    confidence_result = _structured({"confidence": 0.9})

    with (
        patch("verification.agents.generator.call_structured", return_value=generate_result) as gen_mock,
        patch("verification.agents.critic.call_structured", return_value=critic_result) as critic_mock,
        patch("verification.confidence.call_structured", return_value=confidence_result),
    ):
        final_state = run_verification("Who commanded Apollo 11?", retriever)

    assert final_state["coordinator_decision"] == "accept"
    assert final_state["final_status"] == "answered"
    assert final_state["final_answer"] == "Neil Armstrong."
    assert final_state["revision_count"] == 0
    assert gen_mock.call_count == 1
    assert critic_mock.call_count == 1


def test_graph_revises_once_then_accepts():
    retriever = FakeRetriever([_chunk()])

    generate_results = [
        _structured({"answer": "Buzz Aldrin.", "cited_chunk_ids": ["c::0"]}),  # wrong on first pass
        _structured({"answer": "Neil Armstrong.", "cited_chunk_ids": ["c::0"]}),  # fixed on revision
    ]
    critic_results = [
        _structured(
            {
                "findings": [
                    {
                        "claim": "Buzz Aldrin commanded Apollo 11.",
                        "issue_type": "contradiction",
                        "explanation": "evidence says Armstrong commanded Apollo 11",
                    }
                ],
                "supported_claim_count": 0,
            }
        ),
        _structured({"findings": [], "supported_claim_count": 1}),
    ]
    confidence_result = _structured({"confidence": 0.85})

    with (
        patch("verification.agents.generator.call_structured", side_effect=generate_results) as gen_mock,
        patch("verification.agents.critic.call_structured", side_effect=critic_results) as critic_mock,
        patch("verification.confidence.call_structured", return_value=confidence_result),
    ):
        final_state = run_verification("Who commanded Apollo 11?", retriever)

    assert final_state["revision_count"] == 1
    assert final_state["coordinator_decision"] == "accept"
    assert final_state["final_answer"] == "Neil Armstrong."
    assert gen_mock.call_count == 2
    assert critic_mock.call_count == 2


def test_graph_escalates_once_revision_budget_is_exhausted():
    retriever = FakeRetriever([_chunk()])

    # Every draft comes back flawed, even after the one allowed revision —
    # the loop must stop and escalate rather than retry forever.
    generate_result = _structured({"answer": "Buzz Aldrin.", "cited_chunk_ids": ["c::0"]})
    critic_result = _structured(
        {
            "findings": [
                {
                    "claim": "Buzz Aldrin commanded Apollo 11.",
                    "issue_type": "contradiction",
                    "explanation": "evidence says Armstrong commanded Apollo 11",
                }
            ],
            "supported_claim_count": 0,
        }
    )
    confidence_result = _structured({"confidence": 0.3})

    with (
        patch("verification.agents.generator.call_structured", return_value=generate_result) as gen_mock,
        patch("verification.agents.critic.call_structured", return_value=critic_result),
        patch("verification.confidence.call_structured", return_value=confidence_result),
    ):
        final_state = run_verification("Who commanded Apollo 11?", retriever)

    # max_revision_cycles defaults to 1 (config.py): one revise, then escalate.
    assert final_state["revision_count"] == 1
    assert final_state["coordinator_decision"] == "escalate"
    assert final_state["final_status"] == "insufficient_evidence"
    assert gen_mock.call_count == 2  # original draft + the one revision


def test_graph_escalates_immediately_when_nothing_retrieved():
    retriever = FakeRetriever([])  # empty corpus match — nothing to ground an answer in

    generate_result = _structured({"answer": "I don't know.", "cited_chunk_ids": []})
    critic_result = _structured({"findings": [], "supported_claim_count": 0})
    confidence_result = _structured({"confidence": 0.1})

    with (
        patch("verification.agents.generator.call_structured", return_value=generate_result),
        patch("verification.agents.critic.call_structured", return_value=critic_result),
        patch("verification.confidence.call_structured", return_value=confidence_result),
    ):
        final_state = run_verification("What did the Apollo 18 crew find?", retriever)

    assert final_state["coordinator_decision"] == "escalate"
    assert final_state["final_status"] == "insufficient_evidence"
    assert final_state["revision_count"] == 0  # no evidence short-circuits straight to escalation
