"""Tests for HybridRetriever's Reciprocal Rank Fusion (RRF).

Most tests here use fake sub-retrievers with fixed, hand-picked orderings —
fast and precise, since they test the fusion MATH directly rather than
depending on what a real embedding model or BM25 happens to return. The one
exception is the regression test at the bottom, which uses the real corpus
and both real retrievers to confirm the specific bug this was built to fix
(see corpus_loader.py's module docstring) stays fixed.
"""

from __future__ import annotations

from verification.retrieval.base import Retriever
from verification.retrieval.bm25_retriever import BM25Retriever
from verification.retrieval.hybrid_retriever import HybridRetriever
from verification.retrieval.vector_store import ChromaRetriever
from verification.state import RetrievedChunk


class FakeRetriever(Retriever):
    """Returns a fixed, pre-ordered list of chunks regardless of query text —
    lets a test dictate exactly what "rank" (and, optionally, what raw
    score) each retriever assigns. Rank is what RRF fusion looks at; score
    matters separately for testing which chunks HybridRetriever preserves
    the real score for vs. which get a rank-derived fallback (see
    hybrid_retriever.py's module docstring for why that distinction
    exists). Pass either a list of chunk_ids (all get score=1.0) or a list
    of (chunk_id, score) tuples when a test needs a distinctive score."""

    def __init__(self, ordered_chunks: list[str | tuple[str, float]], count: int = 0):
        self._ordered_chunks = [(c, 1.0) if isinstance(c, str) else c for c in ordered_chunks]
        self._count = count
        self.last_top_k: int | None = None

    def count(self) -> int:
        return self._count

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.last_top_k = top_k
        entries = self._ordered_chunks[:top_k] if top_k else self._ordered_chunks
        return [
            RetrievedChunk(chunk_id=cid, text=f"text for {cid}", source="doc.md", score=score)
            for cid, score in entries
        ]


def test_hybrid_fuses_via_reciprocal_rank_not_raw_score():
    # Dense ranks A, B, C; sparse ranks C, A, B. Naive concatenation would
    # favor whichever list is processed first — RRF instead rewards a chunk
    # for ranking well across BOTH lists. A appears at (rank0, rank1) across
    # the two lists — the best combined position — so it must win even
    # though it isn't ranked first by both retrievers individually.
    dense = FakeRetriever(["A", "B", "C"])
    sparse = FakeRetriever(["C", "A", "B"])
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    results = hybrid.query("query", top_k=3)
    fused_order = [c["chunk_id"] for c in results]

    assert fused_order == ["A", "C", "B"]


def test_hybrid_preserves_real_dense_score_for_chunks_dense_found():
    # The actual bug this pair of tests guards against: an earlier version
    # discarded every chunk's real score and replaced it uniformly with a
    # rank-derived 1/(1+rank) value. That pinned every rank-0 result at
    # EXACTLY 1.0 regardless of how strong the match really was — measured
    # directly, this collapsed the project's calibration result (ECE
    # jumped from ~0.06 to 0.23, worse than the naive baseline it was
    # supposed to beat). A chunk dense actually found must keep its real,
    # graded score — not a rank(0) => 1.0 placeholder.
    dense = FakeRetriever([("A", 0.73), ("B", 0.68)])
    sparse = FakeRetriever([("A", 5.1), ("B", 4.9)])
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    results = {c["chunk_id"]: c["score"] for c in hybrid.query("query", top_k=2)}
    assert results["A"] == 0.73
    assert results["B"] == 0.68


def test_hybrid_falls_back_to_rank_based_score_for_bm25_only_chunks():
    # A chunk BM25 found that dense never surfaced at all has no comparable
    # real score to fall back on — BM25's own score is unbounded and
    # corpus-size-dependent, not on the 0-1 scale confidence.py expects.
    # Only THIS case should get the synthetic 1/(1+rank) estimate.
    dense = FakeRetriever([("A", 0.9)])
    sparse = FakeRetriever([("A", 5.0), ("B", 4.0)])  # B is BM25-only
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    results = {c["chunk_id"]: c["score"] for c in hybrid.query("query", top_k=2)}
    assert results["A"] == 0.9  # real dense score, untouched
    assert results["B"] == 1.0 / (1.0 + 1)  # rank-1 fallback, not B's raw BM25 score of 4.0


def test_hybrid_requests_more_candidates_than_final_top_k():
    # Fusion needs a wider candidate pool than the final result size, or a
    # chunk that ranks just outside a narrow top_k on ONE retriever never
    # gets a chance to be rescued by a strong rank on the other.
    dense = FakeRetriever([f"chunk-{i}" for i in range(50)])
    sparse = FakeRetriever([f"chunk-{i}" for i in range(50)])
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    hybrid.query("query", top_k=5)

    assert dense.last_top_k > 5
    assert sparse.last_top_k > 5


def test_hybrid_count_delegates_to_dense_retriever():
    # Not part of the Retriever ABC — see hybrid_retriever.py's count() for
    # why this specifically checks the dense side (BM25 has no separate
    # build step and is never "empty" the way an unbuilt Chroma index is).
    dense = FakeRetriever([], count=133)
    sparse = FakeRetriever([], count=999)
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    assert hybrid.count() == 133


def test_hybrid_retrieval_finds_context_dependent_fact_regression():
    # The real bug, reproduced end to end: this exact sentence never
    # mentions "Apollo 13" or "13" — that context is established several
    # sentences earlier in the source document, before sentence-level
    # chunking split them apart. Dense embeddings alone could infer
    # relevance from surrounding vocabulary; BM25 alone could not (measured
    # at rank 44 of 133 before the title-prefix fix, dragging the fused
    # rank below the retrieval cutoff even though dense alone found it at
    # rank 11). See corpus_loader.py's module docstring for the full story.
    retriever = HybridRetriever(dense=ChromaRetriever(), sparse=BM25Retriever())

    results = retriever.query("What nickname was given to the Apollo 13 Lunar Module?")

    matches = [c for c in results if "Aquarius" in c["text"]]
    assert matches, "the Aquarius chunk must be retrieved, not just present in the corpus"
    assert results[0] in matches, "title-prefixing should make this the top hybrid result"
