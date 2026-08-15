"""Single-pass baseline: Generator only, no Critic loop, no Coordinator
arbitration. This is what the eval harness compares the full pipeline
against — same retrieval, same generator prompt, same underlying model.
The only difference from the pipeline is that there is no verification
step before the answer is returned, and the baseline never has the option
to say "insufficient evidence" — it always answers.
"""

from __future__ import annotations

from verification.agents.generator import generate_node
from verification.confidence import get_verbalized_confidence
from verification.retrieval.base import Retriever
from verification.state import VerificationState


def run_baseline(query: str, retriever: Retriever) -> VerificationState:
    chunks = retriever.query(query)
    state: VerificationState = {
        "query": query,
        "retrieved_chunks": chunks,
        "revision_count": 0,
        "trace": [],
    }

    gen_update = generate_node(state)
    state = {**state, **gen_update}
    state["final_answer"] = state["generator_answer"]
    state["final_status"] = "answered"  # baseline has no escalation path

    verbalized, usage = get_verbalized_confidence(state)
    state["confidence_verbalized"] = verbalized
    state["trace"].append({"node": "baseline_confidence", **usage})

    return state
