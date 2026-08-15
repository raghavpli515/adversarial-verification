"""LangGraph wiring for the verification pipeline.

Graph shape:

    retrieve -> generate -> critique -> coordinate --(revise)--> generate
                                             |
                                    (accept | escalate)
                                             v
                                            END

Design notes:

- Retrieval runs exactly once per query, not once per revision. A revision
  cycle asks the Generator to produce a *better* answer from the *same*
  evidence — it's the answer that was wrong, not the evidence that changed.
  Re-retrieving on every revision would also make it impossible to tell
  whether a revision helped, since the Critic would be checking against a
  moving target.

- The conditional edge out of `coordinate` is the entire reason this is a
  graph and not a linear chain: `coordinate` can route back to `generate`
  (a cycle) or forward to `END`, and that choice is made from state at
  runtime, not fixed at graph-construction time. LangGraph implements this
  via `add_conditional_edges`, which takes a routing function that reads the
  state and returns the name of the next node (or the special `END` marker).

- Two independent loop guards exist and answer different questions:
  `settings.max_revision_cycles` (enforced inside `coordinator.py`) is the
  intentional business-logic cap — "how many times should we let the
  generator try again before giving up." `recursion_limit`, passed to
  `graph.invoke(..., config=...)` below, is a hard backstop against the
  graph looping more than expected at all, e.g. from a bug in the routing
  function — it protects the process, not the answer-quality/cost tradeoff.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from verification.agents.coordinator import coordinate_node
from verification.agents.critic import critique_node
from verification.agents.generator import generate_node
from verification.retrieval.base import Retriever
from verification.state import VerificationState


def _make_retrieve_node(retriever: Retriever):
    def retrieve_node(state: VerificationState) -> dict:
        chunks = retriever.query(state["query"])
        return {
            "retrieved_chunks": chunks,
            "trace": [{"node": "retrieve", "chunk_count": len(chunks)}],
        }

    return retrieve_node


def _route_after_coordinate(state: VerificationState) -> str:
    return "generate" if state["coordinator_decision"] == "revise" else END


def build_graph(retriever: Retriever):
    graph = StateGraph(VerificationState)

    graph.add_node("retrieve", _make_retrieve_node(retriever))
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("coordinate", coordinate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "coordinate")
    graph.add_conditional_edges(
        "coordinate",
        _route_after_coordinate,
        {"generate": "generate", END: END},
    )

    return graph.compile()


def run_verification(query: str, retriever: Retriever) -> dict[str, Any]:
    """Convenience entry point used by the API and the eval harness."""
    compiled_graph = build_graph(retriever)
    initial_state: VerificationState = {
        "query": query,
        "revision_count": 0,
        "trace": [],
    }
    # recursion_limit is generous relative to the worst case (retrieve +
    # (generate, critique, coordinate) x (1 + max_revision_cycles)) — it's a
    # backstop, not a tuned budget.
    return compiled_graph.invoke(initial_state, config={"recursion_limit": 50})
