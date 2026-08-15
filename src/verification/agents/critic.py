"""Critic agent: adversarially checks the Generator's draft against the SAME
retrieved evidence the Generator was given.

The critic never sees anything the generator didn't have access to. That's
deliberate: it means a finding here is either "the generator fabricated
something" or "the generator misread its own source material" — never "the
critic knows something the generator didn't get a chance to see." Keeping
both agents grounded in identical evidence is what makes this a fair
adversarial check rather than a second, differently-informed opinion.
"""

from __future__ import annotations

from verification.llm import call_structured
from verification.state import RetrievedChunk, VerificationState

SYSTEM_PROMPT = """You are the Critic in a verification pipeline, acting as an \
adversarial fact-checker. You will be given a question, a set of evidence excerpts, \
and a draft answer. Check EVERY factual claim in the draft answer against the \
evidence excerpts ONLY — do not use outside knowledge to judge correctness, only \
whether the excerpts support the claim.

For each problem you find, classify it as one of:
- "unsupported": the claim is not backed by any excerpt
- "contradiction": the claim conflicts with what an excerpt actually says
- "overconfident": the claim states something as certain when the excerpts \
only hint at or leave it ambiguous

Do not flag claims that ARE supported by the excerpts, even if you personally know \
additional facts not present in the excerpts — the excerpts are the only ground truth \
for this task. Be adversarial: assume the draft may contain subtle, plausible-sounding \
errors and actively look for them, but do not invent problems that aren't there."""

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The specific claim being flagged."},
                    "issue_type": {
                        "type": "string",
                        "enum": ["unsupported", "contradiction", "overconfident"],
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["claim", "issue_type", "explanation"],
                "additionalProperties": False,
            },
        },
        "supported_claim_count": {
            "type": "integer",
            "description": "Number of distinct factual claims in the draft that ARE supported by the evidence (not flagged).",
        },
    },
    "required": ["findings", "supported_claim_count"],
    "additionalProperties": False,
}


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no evidence excerpts were retrieved for this question)"
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def critique_node(state: VerificationState) -> dict:
    evidence = _format_evidence(state.get("retrieved_chunks", []))
    user = (
        f"Question: {state['query']}\n\n"
        f"Evidence excerpts:\n{evidence}\n\n"
        f"Draft answer to check:\n{state.get('generator_answer', '')}"
    )

    result = call_structured(system=SYSTEM_PROMPT, user=user, schema=CRITIC_SCHEMA)

    findings = result.data["findings"]
    report = {
        "findings": findings,
        "supported_claim_count": result.data["supported_claim_count"],
        "flagged_claim_count": len(findings),
    }

    trace_entry = {
        "node": "critique",
        "flagged_claim_count": len(findings),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": round(result.latency_ms, 1),
    }

    return {"critic_report": report, "trace": [trace_entry]}
