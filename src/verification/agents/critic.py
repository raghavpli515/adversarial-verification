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

A claim counts as "unsupported" only when the excerpts contain NO information that \
would let a careful reader conclude it — not when the excerpts merely fail to spell \
out a fine-grained distinction the claim never actually made. Reasonable paraphrasing, \
reasonable causal framing consistent with the excerpts, and combining several \
sentences from the excerpts into one summary sentence are NOT grounds for a finding.

Example: if an excerpt says "A fire broke out during a launch pad test, fueled by the \
pure-oxygen atmosphere; the crew could not escape because the inward-opening hatch was \
held shut by cabin pressure," a draft claim that "the fire was caused by a launch pad \
test and pure-oxygen atmosphere, and the crew couldn't escape because the hatch opened \
inward and was held shut by pressure" is FULLY SUPPORTED. Do not flag it just because \
the excerpt doesn't explicitly rank which detail is "the" cause versus a contributing \
factor, or because the draft combined multiple sentences from the excerpt into one.

Do not flag claims that ARE supported by the excerpts, even if you personally know \
additional facts not present in the excerpts — the excerpts are the only ground truth \
for this task. Your job is precise verification, not skepticism for its own sake: read \
the draft calmly and check whether each claim is actually backed by the excerpts. Do \
not manufacture a technicality to justify a finding — a claim that is a reasonable, \
accurate paraphrase or synthesis of the excerpts is correct, full stop, even if the \
excerpts don't use identical wording or an explicit causal marker like "because". Only \
report a finding when you can point to a genuine gap or conflict between the draft and \
the excerpts — never merely a difference in phrasing or level of directness.

CRITICAL: `findings` must contain ONLY claims that have an actual problem. If a claim \
is fully supported by the excerpts, do not add an entry for it to `findings` at all — \
writing an entry whose explanation says something like "this is supported" or "this \
claim is correct" is WRONG, even if you set issue_type to one of the three labels. \
There is no way to represent "no problem" inside a findings entry, so the only correct \
way to report a fully-supported claim is to leave it out of `findings` entirely. A \
completely accurate draft answer should produce an empty findings array."""

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
