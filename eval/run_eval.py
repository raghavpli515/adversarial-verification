"""Full eval harness: runs every prompt in
eval/dataset/adversarial_prompts.jsonl through both the single-pass baseline
and the full verification pipeline, grades both with an independent LLM
judge, and computes the metrics reported in the README.

Usage:
    python eval/run_eval.py                          # full 50-prompt run
    python eval/run_eval.py --limit 5                 # quick smoke test
    python eval/run_eval.py --categories out_of_corpus # one category only
    python eval/run_eval.py --no-mlflow                # skip MLflow logging
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# eval/ itself (for the sibling baseline/judge/metrics modules) and src/
# (for the verification package) both need to be importable regardless of
# how this script is invoked (direct run, pytest, from repo root, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import metrics  # noqa: E402
from baseline import run_baseline  # noqa: E402
from judge import grade  # noqa: E402

from verification.agents.critic import critique_node  # noqa: E402
from verification.graph import run_verification  # noqa: E402
from verification.retrieval.vector_store import ChromaRetriever  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "adversarial_prompts.jsonl"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_dataset(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _grading_critic_report(query: str, chunks: list, answer: str) -> dict:
    """Independent measurement pass — see metrics.py docstring for why this
    can't reuse the pipeline's own in-loop critic_report."""
    result = critique_node({"query": query, "retrieved_chunks": chunks, "generator_answer": answer})
    return result["critic_report"]


def _tokens(trace: list[dict]) -> tuple[int, int]:
    inp = sum(t.get("input_tokens", 0) + t.get("verbalized_input_tokens", 0) for t in trace)
    out = sum(t.get("output_tokens", 0) + t.get("verbalized_output_tokens", 0) for t in trace)
    return inp, out


def _latency(trace: list[dict]) -> float:
    return sum(t.get("latency_ms", 0) + t.get("verbalized_latency_ms", 0) for t in trace)


def run_one(item: dict, retriever: ChromaRetriever) -> dict:
    query = item["query"]

    baseline_state = run_baseline(query, retriever)
    pipeline_state = run_verification(query, retriever)

    baseline_grading = _grading_critic_report(
        query, baseline_state["retrieved_chunks"], baseline_state["final_answer"]
    )
    pipeline_grading = _grading_critic_report(
        query, pipeline_state["retrieved_chunks"], pipeline_state["final_answer"]
    )

    baseline_judge = grade(
        query=query,
        gold_answer=item["gold_answer"],
        expected_status=item["expected_status"],
        system_answer=baseline_state["final_answer"],
        system_status=baseline_state["final_status"],
    )
    pipeline_judge = grade(
        query=query,
        gold_answer=item["gold_answer"],
        expected_status=item["expected_status"],
        system_answer=pipeline_state["final_answer"],
        system_status=pipeline_state["final_status"],
    )

    b_in, b_out = _tokens(baseline_state["trace"])
    p_in, p_out = _tokens(pipeline_state["trace"])

    return {
        "id": item["id"],
        "category": item["category"],
        "query": query,
        "expected_status": item["expected_status"],
        "gold_answer": item["gold_answer"],
        # baseline
        "baseline_answer": baseline_state["final_answer"],
        "baseline_status": baseline_state["final_status"],
        "baseline_confidence": baseline_state["confidence_verbalized"],
        "baseline_correct": baseline_judge["correct"],
        "baseline_status_correct": baseline_judge["status_correct"],
        "baseline_grading_findings": baseline_grading["findings"],
        "baseline_input_tokens": b_in,
        "baseline_output_tokens": b_out,
        "baseline_latency_ms": _latency(baseline_state["trace"]),
        "baseline_cost_usd": metrics.estimate_cost_usd(b_in, b_out),
        # pipeline
        "pipeline_answer": pipeline_state["final_answer"],
        "pipeline_status": pipeline_state["final_status"],
        "pipeline_confidence_verbalized": pipeline_state["confidence_verbalized"],
        "pipeline_confidence_engineered": pipeline_state["confidence_engineered"],
        "pipeline_correct": pipeline_judge["correct"],
        "pipeline_status_correct": pipeline_judge["status_correct"],
        "pipeline_grading_findings": pipeline_grading["findings"],
        "pipeline_revision_count": pipeline_state.get("revision_count", 0),
        "pipeline_input_tokens": p_in,
        "pipeline_output_tokens": p_out,
        "pipeline_latency_ms": _latency(pipeline_state["trace"]),
        "pipeline_cost_usd": metrics.estimate_cost_usd(p_in, p_out),
    }


def aggregate(records: list[dict]) -> dict:
    baseline_grading_reports = [{"findings": r["baseline_grading_findings"]} for r in records]
    pipeline_grading_reports = [{"findings": r["pipeline_grading_findings"]} for r in records]

    # ECE only over items the pipeline actually answered — see metrics.py.
    pipeline_answered = [r for r in records if r["pipeline_status"] == "answered"]
    verbalized_conf = [r["pipeline_confidence_verbalized"] for r in pipeline_answered]
    engineered_conf = [r["pipeline_confidence_engineered"] for r in pipeline_answered]
    correctness = [r["pipeline_correct"] for r in pipeline_answered]

    return {
        "n_items": len(records),
        "n_pipeline_answered": len(pipeline_answered),
        "baseline_unsupported_claim_rate": metrics.unsupported_claim_rate(baseline_grading_reports),
        "pipeline_unsupported_claim_rate": metrics.unsupported_claim_rate(pipeline_grading_reports),
        "baseline_accuracy": sum(r["baseline_correct"] for r in records) / len(records),
        "pipeline_accuracy": sum(r["pipeline_correct"] for r in records) / len(records),
        "pipeline_false_escalation_rate": metrics.false_escalation_rate(records, "pipeline_status"),
        "ece_verbalized": metrics.expected_calibration_error(verbalized_conf, correctness),
        "ece_engineered": metrics.expected_calibration_error(engineered_conf, correctness),
        "reliability_verbalized": metrics.reliability_diagram_data(verbalized_conf, correctness),
        "reliability_engineered": metrics.reliability_diagram_data(engineered_conf, correctness),
        "baseline_latency": metrics.summarize_latency_cost(
            [r["baseline_latency_ms"] for r in records],
            [r["baseline_cost_usd"] for r in records],
            sum(r["baseline_input_tokens"] for r in records),
            sum(r["baseline_output_tokens"] for r in records),
        ).__dict__,
        "pipeline_latency": metrics.summarize_latency_cost(
            [r["pipeline_latency_ms"] for r in records],
            [r["pipeline_cost_usd"] for r in records],
            sum(r["pipeline_input_tokens"] for r in records),
            sum(r["pipeline_output_tokens"] for r in records),
        ).__dict__,
    }


def _log_to_mlflow(summary: dict, n_items: int) -> None:
    import mlflow

    from verification.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model", settings.anthropic_model)
        mlflow.log_param("max_revision_cycles", settings.max_revision_cycles)
        mlflow.log_param("retrieval_top_k", settings.retrieval_top_k)
        mlflow.log_param("n_items", n_items)

        mlflow.log_metrics(
            {
                "baseline_unsupported_claim_rate": summary["baseline_unsupported_claim_rate"],
                "pipeline_unsupported_claim_rate": summary["pipeline_unsupported_claim_rate"],
                "baseline_accuracy": summary["baseline_accuracy"],
                "pipeline_accuracy": summary["pipeline_accuracy"],
                "pipeline_false_escalation_rate": summary["pipeline_false_escalation_rate"],
                "ece_verbalized": summary["ece_verbalized"],
                "ece_engineered": summary["ece_engineered"],
                "baseline_p50_latency_ms": summary["baseline_latency"]["p50_latency_ms"],
                "pipeline_p50_latency_ms": summary["pipeline_latency"]["p50_latency_ms"],
                "baseline_mean_cost_usd": summary["baseline_latency"]["mean_cost_usd"],
                "pipeline_mean_cost_usd": summary["pipeline_latency"]["mean_cost_usd"],
            }
        )
    print("Logged run to MLflow.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--categories", type=str, default=None, help="comma-separated category filter")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(DATASET_PATH)
    if args.categories:
        wanted = set(args.categories.split(","))
        dataset = [item for item in dataset if item["category"] in wanted]
    if args.limit:
        dataset = dataset[: args.limit]

    retriever = ChromaRetriever()
    if retriever.count() == 0:
        print("WARNING: Chroma index is empty. Run `python scripts/build_index.py` first.")

    records = []
    for i, item in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {item['id']} ({item['category']}) ...")
        records.append(run_one(item, retriever))

    summary = aggregate(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"eval_run_{timestamp}.json"
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    if not args.no_mlflow:
        _log_to_mlflow(summary, len(dataset))


if __name__ == "__main__":
    main()
