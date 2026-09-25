"""LangGraph wiring for the verification pipeline.

Graph shape:

    retrieve -> generate -> critique -> coordinate --(revise)--> generate
                                             |
                                    (accept | escalate)
                                             v
                                            END

Retrieval runs once per query, not once per revision — a revision asks the
Generator for a *better* answer from the *same* evidence, not new evidence,
and re-retrieving would make it impossible to tell whether the revision
itself helped. The conditional edge out of `coordinate`
(`add_conditional_edges`, routing on state at runtime) is why this is a
graph rather than a linear chain. Two independent loop guards exist:
`settings.max_revision_cycles` is the intentional business-logic cap on
revision attempts; `recursion_limit` below is a hard backstop against the
graph looping more than expected at all, e.g. from a routing bug.
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
