"""Hybrid retriever: fuses ChromaRetriever's dense results with
BM25Retriever's sparse results via Reciprocal Rank Fusion (RRF) rather
than a weighted score sum — BM25 and cosine similarity live on
incomparable scales, and RRF only needs each retriever's RANK,
`1/(k+rank)` summed across retrievers, avoiding a normalization constant
that would silently go stale as either scorer changes.

A real bug this file used to have: an earlier version replaced every
chunk's real score with a rank-derived `1/(1+rank)` estimate, which is
EXACTLY 1.0 for every rank-0 result regardless of match strength — this
pinned 15 of 50 eval items at 1.0 engineered confidence (several wrong),
regressing calibration to ECE 0.233. Fixed by keeping each chunk's real
dense score whenever one exists (`confidence.py` depends on that scale)
and using the rank-derived estimate only as a fallback for BM25-only
chunks, which have no comparable real score.
"""

from __future__ import annotations

from verification.config import settings
from verification.retrieval.base import Retriever
from verification.state import RetrievedChunk

RRF_K = 60  # standard default from the original RRF paper; not tuned further here
CANDIDATE_MULTIPLIER = 3  # pull more candidates from each retriever than top_k before fusing


class HybridRetriever(Retriever):
    def __init__(self, dense: Retriever, sparse: Retriever):
        self._dense = dense
        self._sparse = sparse

    def count(self) -> int:
        # Not part of the Retriever ABC (see base.py) — a ChromaRetriever
        # convenience method the API uses for a health check and an
        # empty-index guard. Delegated to the dense side specifically:
        # BM25Retriever builds in-memory from the corpus dir with no
        # separate build step, so it's never "empty" the way an unbuilt
        # Chroma index can be — the dense count is the one that actually
        # signals whether `scripts/build_index.py` has been run.
        return self._dense.count()

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or settings.retrieval_top_k
        candidate_k = k * CANDIDATE_MULTIPLIER

        dense_results = self._dense.query(text, top_k=candidate_k)
        sparse_results = self._sparse.query(text, top_k=candidate_k)

        found_by_dense = {c["chunk_id"] for c in dense_results}

        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, RetrievedChunk] = {}
        for ranked_list in (dense_results, sparse_results):
            for rank, chunk in enumerate(ranked_list):
                chunk_id = chunk["chunk_id"]
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                # dense-first iteration order + setdefault: a chunk found by
                # both retrievers keeps its real dense score here, not a
                # synthetic one — see module docstring for why that matters.
                chunk_by_id.setdefault(chunk_id, chunk)

        fused_order = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        results: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(fused_order):
            base = chunk_by_id[chunk_id]
            score = (
                base["score"]
                if chunk_id in found_by_dense
                else 1.0 / (1.0 + rank)  # BM25-only match; no comparable real score exists
            )
            results.append(
                RetrievedChunk(chunk_id=base["chunk_id"], text=base["text"], source=base["source"], score=score)
            )
        return results
