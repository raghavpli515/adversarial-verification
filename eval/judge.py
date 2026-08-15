"""LLM-judge grading: scores a system's final answer/status against the
gold label for one eval item.

This is a separate call from any of the three pipeline agents and from the
grading critic pass used for unsupported-claim measurement — an independent
grading function with its own prompt, not a self-report from the system
being graded. It uses the same underlying model as the pipeline (there's no
separate "judge model" budgeted for this MVP), so it isn't independent in
the strongest possible sense — a stretch goal worth naming honestly in the
README's Limitations section is cross-validating with a second model.
"""

from __future__ import annotations

from verification.llm import call_structured

JUDGE_SYSTEM_PROMPT = """You are grading whether an AI system's answer to a question is \
correct, given a gold-standard reference answer. You are not grading style, tone, or \
phrasing — only factual correctness and whether the system reached the right \
conclusion.

Grade "correct" as true if the system's answer conveys the same key facts as the gold \
answer, even if worded very differently. Grade "correct" as false if the system's \
answer contains a factual error, omits a key fact the gold answer states, asserts \
something the gold answer contradicts, or accepts a false premise that the gold \
answer explicitly corrects.

Separately, grade "status_correct" as true if the system's status ("answered" vs \
"insufficient_evidence") matches the expected status. A system that says \
"insufficient evidence" when the question was actually answerable from the evidence \
is a false escalation — grade status_correct as false. A system that gives a \
confident-sounding answer when the expected status is "insufficient_evidence" is \
answering from outside knowledge instead of admitting a gap — also grade \
status_correct as false."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "status_correct": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["correct", "status_correct", "rationale"],
    "additionalProperties": False,
}


def grade(
    query: str,
    gold_answer: str,
    expected_status: str,
    system_answer: str,
    system_status: str,
) -> dict:
    user = (
        f"Question: {query}\n\n"
        f"Gold answer: {gold_answer}\n"
        f"Expected status: {expected_status}\n\n"
        f"System's answer: {system_answer}\n"
        f"System's status: {system_status}"
    )
    result = call_structured(system=JUDGE_SYSTEM_PROMPT, user=user, schema=JUDGE_SCHEMA)
    return {
        "correct": result.data["correct"],
        "status_correct": result.data["status_correct"],
        "rationale": result.data["rationale"],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
