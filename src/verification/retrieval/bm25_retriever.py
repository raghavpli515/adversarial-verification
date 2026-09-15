"""BM25 (sparse, exact-term) retriever — the counterpart to ChromaRetriever's
dense embedding search. See hybrid_retriever.py for how the two get combined.

BM25 does keyword/term-frequency matching, not semantic similarity — it has
no idea "Lunar Module" and "spacecraft" might be related, but it also has no
way to dilute an exact term match the way a pooled sentence embedding can
(see corpus_loader.py's module docstring for the dilution bug this is meant
to complement, not replace). "Aquarius" scores highly here for a query
containing "Aquarius" regardless of what else is in the chunk or how many
other chunks share a similar surface template.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from verification.config import settings
from verification.retrieval.base import Retriever
from verification.retrieval.corpus_loader import RawChunk, load_corpus_dir
from verification.state import RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever(Retriever):
    """In-memory BM25 index built directly from the corpus directory — the
    same source of truth ChromaRetriever indexes via build_index.py, so the
    two retrievers always work from identical chunk boundaries. No
    persistence: at ~150 chunks, rebuilding at process startup is
    sub-second, so there's no build_index.py-equivalent step for this one."""

    def __init__(self, corpus_dir: str | Path | None = None):
        self._chunks: list[RawChunk] = load_corpus_dir(corpus_dir or settings.corpus_dir)
        tokenized_corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def count(self) -> int:
        return len(self._chunks)

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or settings.retrieval_top_k
        scores = self._bm25.get_scores(_tokenize(text))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[i].chunk_id,
                text=self._chunks[i].text,
                source=self._chunks[i].source,
                score=float(scores[i]),
            )
            for i in ranked_indices
        ]
