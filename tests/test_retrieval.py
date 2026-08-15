"""Tests for the retrieval layer: chunking is pure and fully offline; the
Chroma round trip is real but needs no Anthropic API key, since embeddings
are computed locally via sentence-transformers.
"""

from __future__ import annotations

from verification.retrieval.corpus_loader import chunk_text
from verification.retrieval.vector_store import ChromaRetriever


def test_chunk_text_merges_short_paragraphs_into_one_chunk():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, source="doc.md", max_chars=1000)
    assert len(chunks) == 1
    assert "Para one." in chunks[0].text
    assert "Para three." in chunks[0].text


def test_chunk_text_splits_when_over_max_chars():
    long_para = "x" * 500
    text = f"{long_para}\n\n{long_para}\n\n{long_para}"
    chunks = chunk_text(text, source="doc.md", max_chars=800)
    assert len(chunks) == 3
    assert all(len(c.text) <= 800 for c in chunks)


def test_chunk_ids_are_unique_and_source_scoped():
    text = "Para one.\n\nPara two."
    chunks = chunk_text(text, source="doc.md", max_chars=5)  # forces one chunk per paragraph
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("doc::") for cid in ids)


def test_chroma_retriever_round_trip(tmp_path):
    retriever = ChromaRetriever(persist_dir=str(tmp_path / "chroma"))
    retriever.upsert(
        "doc::0",
        "Neil Armstrong commanded Apollo 11 and was the first person to walk on the Moon.",
        "doc.md",
    )
    retriever.upsert(
        "doc::1",
        "The Saturn V rocket had five F-1 engines in its first stage.",
        "doc.md",
    )

    assert retriever.count() == 2

    results = retriever.query("Who was the first person on the Moon?", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc::0"
    assert results[0]["source"] == "doc.md"
    assert 0.0 < results[0]["score"] <= 1.0
