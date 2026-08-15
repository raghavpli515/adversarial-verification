"""Retriever interface.

This is the "swappable retrieval layer" from the project brief — deliberately
kept to a single method. The MVP ships exactly one implementation
(ChromaRetriever). The point of the abstraction isn't that we build multiple
backends now; it's that the graph nodes (generator.py) depend on this
interface, not on Chroma directly, so swapping in a hybrid BM25+vector
retriever or a hosted vector DB later is a new class, not a rewrite of the
agents. Don't over-build this — a second implementation only gets written if
there's time left after the eval harness works end-to-end.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from verification.state import RetrievedChunk


class Retriever(ABC):
    @abstractmethod
    def query(self, text: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to top_k chunks relevant to `text`, ranked by score desc."""
        raise NotImplementedError
