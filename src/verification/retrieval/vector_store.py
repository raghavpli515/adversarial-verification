"""Chroma-backed implementation of the Retriever interface.

Uses Chroma's local persistent client — no separate database service to run,
which is why the MVP Docker setup doesn't need docker-compose. Embeddings are
computed with sentence-transformers (all-MiniLM-L6-v2 by default: fast, small,
good enough for a single-domain corpus this size — swapping to a larger
embedding model is a one-line config change, not a code change).
"""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from verification.config import settings
from verification.retrieval.base import Retriever
from verification.state import RetrievedChunk

COLLECTION_NAME = "corpus"


class ChromaRetriever(Retriever):
    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir or settings.chroma_persist_dir)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model or settings.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
        )

    def upsert(self, chunk_id: str, text: str, source: str) -> None:
        """Add or replace a single chunk. Used by scripts/build_index.py."""
        self._collection.upsert(
            ids=[chunk_id],
            documents=[text],
            metadatas=[{"source": source}],
        )

    def count(self) -> int:
        return self._collection.count()

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or settings.retrieval_top_k
        result = self._collection.query(query_texts=[text], n_results=k)

        chunks: list[RetrievedChunk] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]  # Chroma returns distance, not similarity

        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
            # Chroma's default space is L2 distance for this embedding function;
            # convert to a bounded, higher-is-better score for the confidence
            # formula in confidence.py (which expects "similarity-like" inputs).
            score = 1.0 / (1.0 + distance)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc,
                    source=meta.get("source", "unknown"),
                    score=score,
                )
            )
        return chunks
