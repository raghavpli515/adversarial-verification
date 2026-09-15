"""Tests for the retrieval layer: chunking and BM25 are pure and fully
offline; the Chroma round trip is real but needs no API key, since
embeddings are computed locally via sentence-transformers.
"""

from __future__ import annotations

from verification.retrieval.bm25_retriever import BM25Retriever
from verification.retrieval.corpus_loader import chunk_text
from verification.retrieval.vector_store import ChromaRetriever


def test_chunk_text_merges_short_sentences_within_one_paragraph():
    # Sentences are the merge unit, not paragraphs — three short sentences in
    # ONE paragraph should still combine into a single chunk under a
    # generous max_chars.
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_text(text, source="doc.md", max_chars=1000)
    assert len(chunks) == 1
    assert "First sentence." in chunks[0].text
    assert "Third sentence." in chunks[0].text


def test_chunk_text_never_merges_across_paragraph_boundaries():
    # Paragraphs are a hard boundary — even a generous max_chars must not
    # combine two separate source paragraphs into one chunk. This is the
    # behavior that fixes the real bug this chunker was rewritten for: a
    # paragraph mixing several distinct facts diluted any single fact's
    # embedding similarity enough that the correct chunk ranked 29th out of
    # 46 for a query it directly answered.
    text = "Paragraph one is short.\n\nParagraph two is also short."
    chunks = chunk_text(text, source="doc.md", max_chars=1000)
    assert len(chunks) == 2
    assert chunks[0].text == "Paragraph one is short."
    assert chunks[1].text == "Paragraph two is also short."


def test_chunk_text_splits_within_a_paragraph_when_sentences_exceed_max_chars():
    # The actual regression case: one paragraph, multiple distinct-fact
    # sentences that together exceed max_chars, must split into more than
    # one chunk rather than diluting every fact into a single chunk.
    text = (
        'The Lunar Module was nicknamed "Aquarius." '
        "Mission Control improvised procedures to keep the crew alive. "
        "All three astronauts returned safely to Earth."
    )
    chunks = chunk_text(text, source="doc.md", max_chars=90)
    assert len(chunks) > 1
    assert all(len(c.text) <= 90 for c in chunks)
    # the nickname fact must survive intact in some single chunk, not be
    # split mid-sentence or merged away with unrelated later sentences
    assert any('nicknamed "Aquarius."' in c.text for c in chunks)


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


def test_chunk_text_extracts_header_as_title_and_drops_it_as_a_chunk():
    # The regression case for the BM25/hybrid retrieval bug: a sentence deep
    # in a document ("...the crew...") often never restates the document's
    # subject, which BM25 can't infer from context the way dense embeddings
    # can. The header's title is prepended to every chunk instead of being
    # dropped or kept as a chunk of its own.
    text = "# Apollo 13\n\nThe crew used the Lunar Module as a lifeboat."
    chunks = chunk_text(text, source="apollo_13.md")
    assert len(chunks) == 1  # header contributes no chunk of its own
    assert chunks[0].text == "Apollo 13: The crew used the Lunar Module as a lifeboat."


def test_chunk_text_title_prefix_applies_to_every_chunk_in_the_document():
    text = "# Apollo 13\n\nFirst fact.\n\nSecond fact."
    chunks = chunk_text(text, source="apollo_13.md")
    assert len(chunks) == 2
    assert all(c.text.startswith("Apollo 13: ") for c in chunks)


def test_chunk_text_without_a_header_has_no_title_prefix():
    # Backward-compatible: text with no leading "# Title" paragraph chunks
    # exactly as before, no prefix invented from nothing.
    text = "Just a plain paragraph, no header."
    chunks = chunk_text(text, source="doc.md")
    assert chunks[0].text == "Just a plain paragraph, no header."


def test_bm25_retriever_finds_exact_term_match(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.md").write_text(
        "# Doc A\n\nThe Lunar Module was nicknamed Aquarius.", encoding="utf-8"
    )
    (corpus_dir / "b.md").write_text(
        "# Doc B\n\nThe Command Module was nicknamed Odyssey.", encoding="utf-8"
    )

    retriever = BM25Retriever(corpus_dir=corpus_dir)
    assert retriever.count() == 2

    results = retriever.query("Aquarius", top_k=1)
    assert len(results) == 1
    assert results[0]["source"] == "a.md"


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
