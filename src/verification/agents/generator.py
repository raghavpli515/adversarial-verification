"""Generator agent: drafts an answer to the query from retrieved evidence.

On the first pass this is a plain "answer from context" call. On a revision
pass (state["revision_count"] > 0), this same node runs again, but the
prompt now also includes the Critic's findings from the previous round —
the generator is asked to specifically fix flagged claims rather than
starting over from a blank slate. That's what makes a "revision cycle"
meaningfully different from just retrying the same call.
"""

from __future__ import annotations

from verification.llm import call_structured
from verification.state import RetrievedChunk, VerificationState

SYSTEM_PROMPT = """You are the Generator in a verification pipeline. Answer the user's \
question using ONLY the provided evidence excerpts. For every factual claim in your \
answer, you must be able to point to a specific excerpt that supports it — if the \
evidence does not support a claim, do not make it. If the evidence does not contain \
enough information to answer the question at all, say so plainly rather than filling \
the gap from general knowledge, even if you happen to know the answer from elsewhere.

Cite the excerpt IDs your answer actually relies on."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the user's question, grounded in the evidence excerpts.",
        },
        "cited_chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of the evidence excerpts the answer actually relies on.",
        },
    },
    "required": ["answer", "cited_chunk_ids"],
    "additionalProperties": False,
}


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no evidence excerpts were retrieved for this question)"
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def generate_node(state: VerificationState) -> dict:
    evidence = _format_evidence(state.get("retrieved_chunks", []))
    revision_count = state.get("revision_count", 0)

    user_parts = [f"Question: {state['query']}", f"Evidence excerpts:\n{evidence}"]

    if revision_count > 0:
        report = state.get("critic_report", {"findings": []})
        findings_text = "\n".join(
            f'- [{f["issue_type"]}] "{f["claim"]}": {f["explanation"]}'
            for f in report.get("findings", [])
        ) or "(no specific findings recorded)"
        user_parts.append(
            "Your previous answer was reviewed and the following issues were found. "
            "Revise your answer to address every issue below — remove or fix any "
            "unsupported claim, and resolve any flagged contradiction, without "
            "introducing new unsupported claims:\n" + findings_text
        )

    result = call_structured(
        system=SYSTEM_PROMPT,
        user="\n\n".join(user_parts),
        schema=ANSWER_SCHEMA,
    )

    trace_entry = {
        "node": "generate",
        "revision_count": revision_count,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": round(result.latency_ms, 1),
    }

    return {
        "generator_answer": result.data["answer"],
        "generator_citations": result.data["cited_chunk_ids"],
        "trace": [trace_entry],
    }
