"""Hybrid retriever: fuses ChromaRetriever's dense results with
BM25Retriever's sparse results via Reciprocal Rank Fusion (RRF), rather
than combining raw scores directly.

RRF instead of a weighted score sum: BM25 scores and cosine similarity live
on completely different, incomparable scales — BM25 is unbounded and
corpus-size-dependent, while the dense score here is 1/(1 + L2 distance),
roughly 0-1. Averaging or weighting two incomparable scales requires a
normalization constant that would silently go stale the moment either
underlying scorer changes. RRF sidesteps this: it only uses each
retriever's RANK position, `1 / (k + rank)` for a small constant k, summed
across retrievers. Simple, one well-understood parameter, and what most
production hybrid-search systems use for exactly this reason.

Why this exists at all: dense embeddings pool a whole chunk into one
vector, so a fact can lose a similarity race to an unrelated chunk that
just happens to share more surface-level vocabulary structure — this is
the exact, diagnosed bug in corpus_loader.py's module docstring (a chunk
ranking 29th of 46 for a query it directly answered). BM25 has no such
failure mode for exact term matches. The two retrievers fail differently,
which is the actual argument for combining them, not a claim that either
one is better in general.

IMPORTANT interaction with confidence.py, and a real bug this file used to
have: `get_engineered_confidence` averages `retrieved_chunks[i]["score"]`
for cited chunks and treats it as a genuinely graded, roughly 0-1
similarity signal — calibrated against the dense retriever's
`1/(1+distance)` scale, which the project's headline calibration result
(ECE ~0.05-0.09 across three runs) was measured against. An earlier version
of this file discarded each chunk's real score and replaced it uniformly
with a rank-derived `1/(1+rank)` pseudo-similarity. That seemed safe (same
rough scale) but wasn't: `1/(1+rank)` is EXACTLY 1.0 for every rank-0
result regardless of how strong the match actually was, collapsing genuine
gradation into a few fixed values. Measured directly, this pinned 15 of 50
eval items at exactly 1.0 engineered confidence — several of which were
wrong — which is worse calibration (ECE 0.23) than the naive verbalized
baseline it was supposed to beat.

The fix: keep each chunk's REAL score whenever one exists on the scale
confidence.py expects. Dense results are processed first and `chunk_by_id`
uses `setdefault`, so a chunk found by both retrievers already retains its
genuine dense cosine-similarity score internally — the bug was in the
final result construction, which overwrote that good value with the
synthetic one regardless. The rank-derived `1/(1+rank)` estimate is now
used ONLY as a fallback for chunks BM25 found that dense did not surface at
all, since BM25's own score is unbounded and corpus-size-dependent and has
no directly comparable meaning on the 0-1 scale confidence.py needs.
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
