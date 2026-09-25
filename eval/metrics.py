"""Metrics computed from eval run records.

Two design choices worth being able to defend: unsupported-claim rate uses
an INDEPENDENT grading pass — `run_eval.py` calls the Critic's
`critique_node` directly against the final answer, after the fact, rather
than reading the pipeline's internal `critic_report` — reading the internal
report would make "accept" decisions trivially show zero unsupported
claims by construction, a circular metric. And ECE is computed only over
items the pipeline actually answered, not escalated — "insufficient
evidence" has no truth value to calibrate confidence against the way an
answer does, so mixing the two would conflate two different questions.
"""

from __future__ import annotations

from dataclasses import dataclass

# gpt-4o pricing at time of writing: $2.50 / $10.00 per million input /
# output tokens (see config.py OPENAI_MODEL). This is a point-in-time
# figure, not fetched live — verify against OpenAI's current pricing page
# before trusting the eval report's cost numbers for anything beyond a rough
# order-of-magnitude estimate. Update these if the model or pricing changes.
INPUT_COST_PER_MTOK = 2.50
OUTPUT_COST_PER_MTOK = 10.00


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * INPUT_COST_PER_MTOK + (
        output_tokens / 1_000_000
    ) * OUTPUT_COST_PER_MTOK


def unsupported_claim_rate(grading_reports: list[dict]) -> float:
    """Fraction of items where the independent grading critic found at
    least one 'unsupported' or 'contradiction' finding. 'overconfident'
    findings are excluded from this headline number — they're a softer
    signal (a true-but-overstated claim), not a hallucination."""
    if not grading_reports:
        return 0.0
    flagged = sum(
        1
        for report in grading_reports
        if any(f["issue_type"] in ("unsupported", "contradiction") for f in report.get("findings", []))
    )
    return flagged / len(grading_reports)


def false_escalation_rate(records: list[dict], status_key: str) -> float:
    """Of items where the gold expected_status is 'answered', what fraction
    did the system escalate to 'insufficient_evidence' anyway?"""
    answerable = [r for r in records if r["expected_status"] == "answered"]
    if not answerable:
        return 0.0
    escalated = sum(1 for r in answerable if r[status_key] == "insufficient_evidence")
    return escalated / len(answerable)


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Standard binned ECE: partition [0, 1] into n_bins, and within each
    bin compare the average confidence to the actual accuracy. ECE is the
    size-weighted average absolute gap across bins — 0 is perfectly
    calibrated, higher is worse."""
    if not confidences:
        return 0.0
    n = len(confidences)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, is_correct in zip(confidences, correct):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, is_correct))

    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bucket_conf = sum(c for c, _ in bucket) / len(bucket)
        bucket_acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / n) * abs(bucket_conf - bucket_acc)
    return ece


def reliability_diagram_data(confidences: list[float], correct: list[bool], n_bins: int = 10) -> list[dict]:
    """Per-bin (mean_confidence, accuracy, count) — feed this to a chart for
    the README's reliability diagram. Empty bins are kept (with null
    values) so the x-axis stays evenly spaced when plotted."""
    n_effective = max(n_bins, 1)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_effective)]
    for conf, is_correct in zip(confidences, correct):
        idx = min(int(conf * n_effective), n_effective - 1)
        bins[idx].append((conf, is_correct))

    rows = []
    for i, bucket in enumerate(bins):
        row = {"bin": i, "bin_range": (i / n_effective, (i + 1) / n_effective), "count": len(bucket)}
        if bucket:
            row["mean_confidence"] = sum(c for c, _ in bucket) / len(bucket)
            row["accuracy"] = sum(1 for _, ok in bucket if ok) / len(bucket)
        else:
            row["mean_confidence"] = None
            row["accuracy"] = None
        rows.append(row)
    return rows


@dataclass
class LatencyCostSummary:
    p50_latency_ms: float
    mean_latency_ms: float
    p50_cost_usd: float
    mean_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def summarize_latency_cost(
    per_query_latency_ms: list[float],
    per_query_cost_usd: list[float],
    total_input_tokens: int,
    total_output_tokens: int,
) -> LatencyCostSummary:
    return LatencyCostSummary(
        p50_latency_ms=_p50(per_query_latency_ms),
        mean_latency_ms=(sum(per_query_latency_ms) / len(per_query_latency_ms)) if per_query_latency_ms else 0.0,
        p50_cost_usd=_p50(per_query_cost_usd),
        mean_cost_usd=(sum(per_query_cost_usd) / len(per_query_cost_usd)) if per_query_cost_usd else 0.0,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )
