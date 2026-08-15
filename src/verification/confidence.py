"""Two confidence signals, deliberately computed two different ways, so the
eval harness can compare them:

1. `get_verbalized_confidence` — just ask the LLM to self-report a 0-1
   number. This is the naive baseline. Verbalized confidence is known in
   the literature to correlate poorly with actual correctness (models tend
   to sound confident regardless of whether they're right), and one of this
   project's eval results is measuring exactly how poorly, on this system.

2. `get_engineered_confidence` — a hand-tuned formula over signals the
   *system* can observe directly, without asking the model to introspect:
   how well-matched the cited evidence was, how many problems the Critic
   found relative to total claims, and whether a revision was needed at
   all. Nothing here asks "how sure are you" — everything here is either
   a retrieval score or a count.

Both get logged to `VerificationState` (`confidence_verbalized`,
`confidence_engineered`) and both get scored against the LLM-judge's
correctness label in `eval/metrics.py` via Expected Calibration Error (ECE).
The claim in the README ("naive confidence is poorly calibrated, ours is
better") is only honest if this formula was written *before* looking at
eval results and tuned by held-out testing — not reverse-engineered from
the eval set after the fact. Keep it that way when iterating.
"""

from __future__ import annotations

from verification.llm import call_structured
from verification.state import VerificationState

VERBALIZED_CONFIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {
            "type": "number",
            "description": (
                "Self-reported confidence from 0.0 (no confidence) to 1.0 (certain) "
                "that the final answer is fully correct and fully supported by the "
                "evidence it was checked against."
            ),
        },
    },
    "required": ["confidence"],
    "additionalProperties": False,
}

VERBALIZED_SYSTEM_PROMPT = """You are asked to self-report how confident you are that \
a final answer is correct and fully supported by the evidence it was checked against. \
Report a single number from 0.0 to 1.0 reflecting your genuine confidence."""


def get_verbalized_confidence(state: VerificationState) -> tuple[float, dict]:
    report = state.get("critic_report", {"findings": [], "supported_claim_count": 0})
    user = (
        f"Question: {state['query']}\n"
        f"Final answer: {state.get('final_answer', '')}\n"
        f"Critic findings against this answer: {report.get('findings', [])}"
    )
    result = call_structured(
        system=VERBALIZED_SYSTEM_PROMPT, user=user, schema=VERBALIZED_CONFIDENCE_SCHEMA
    )
    usage = {
        "verbalized_input_tokens": result.input_tokens,
        "verbalized_output_tokens": result.output_tokens,
        "verbalized_latency_ms": round(result.latency_ms, 1),
    }
    confidence = max(0.0, min(1.0, float(result.data["confidence"])))
    return confidence, usage


def get_engineered_confidence(state: VerificationState) -> float:
    chunks = state.get("retrieved_chunks", [])
    citations = set(state.get("generator_citations", []))
    cited_scores = [c["score"] for c in chunks if c["chunk_id"] in citations]
    # If the generator cited nothing (or cited chunks that weren't actually
    # retrieved — a malformed citation), retrieval quality can't vouch for
    # the answer, so this component collapses to 0 rather than silently
    # falling back to "average over all retrieved chunks."
    retrieval_component = sum(cited_scores) / len(cited_scores) if cited_scores else 0.0

    report = state.get("critic_report", {"findings": [], "supported_claim_count": 0})
    findings = report.get("findings", [])
    supported = report.get("supported_claim_count", 0)
    total_claims = max(supported + len(findings), 1)

    # Overconfident findings are a softer signal than an outright unsupported
    # or contradicted claim, so they only cost half the penalty.
    penalty = sum(0.5 if f["issue_type"] == "overconfident" else 1.0 for f in findings)
    critic_component = max(0.0, 1.0 - (penalty / total_claims))

    # Needing a revision at all is mild evidence the first draft had a real
    # problem, so nudge the ceiling down slightly even if the revised
    # answer came back clean.
    revision_penalty = 0.1 if state.get("revision_count", 0) > 0 else 0.0

    score = 0.5 * retrieval_component + 0.5 * critic_component - revision_penalty
    return max(0.0, min(1.0, score))
