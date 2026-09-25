"""Load and chunk the raw corpus documents for indexing.

Chunks split on paragraph boundaries, then greedily merge by SENTENCE (not
paragraph) up to `max_chars` — an earlier paragraph-level chunker let one
multi-fact paragraph dilute a single fact's embedding similarity enough to
rank 29th of 46 for a query it directly answered. Markdown header lines are
excluded from the chunk stream (bare headers clustered in embedding space
and crowded out real content), but each header's title is prepended to
every other chunk in that document, since BM25 needs the entity name a
sentence may not restate on its own (a chunk ranked 11th of 133 for dense
retrieval but 44th for BM25 without it — see `bm25_retriever.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# One or more sentence-ending punctuation marks, optionally followed by a
# single closing quote (straight or curly, single or double), then a
# lookahead (not consumed) for whitespace or end of string.
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’]?(?=\s|$)")

# A paragraph that is ONLY a markdown header line (nothing else) — see the
# module docstring for why these are dropped rather than kept as chunks.
_HEADER_ONLY = re.compile(r"^#{1,6}\s+\S.*$")


@dataclass(frozen=True)   # frozen=True makes the dataclass immutable, meaning its attributes cannot be modified after creation
class RawChunk:
    chunk_id: str
    text: str
    source: str  # filename, used as the citation the generator/critic point to


def _split_sentences(paragraph: str) -> list[str]:
    paragraph = paragraph.strip()
    sentences = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        end = match.end()
        sentences.append(paragraph[start:end].strip())
        start = end
    remainder = paragraph[start:].strip()
    if remainder:
        sentences.append(remainder)
    return [s for s in sentences if s]


def chunk_text(text: str, source: str, max_chars: int = 300) -> list[RawChunk]:
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # The first header-only paragraph becomes the document's title, prepended
    # to every other chunk below — see the module docstring for why. Any
    # header-only paragraph (first or otherwise) is excluded from the body,
    # so it never becomes a chunk of its own.
    title: str | None = None
    paragraphs: list[str] = []
    for p in raw_paragraphs:
        if _HEADER_ONLY.fullmatch(p):
            if title is None:
                title = re.sub(r"^#{1,6}\s+", "", p).strip()
            continue
        paragraphs.append(p)

    chunks: list[RawChunk] = []
    for para in paragraphs:
        buffer = ""
        for sentence in _split_sentences(para):
            candidate = f"{buffer} {sentence}" if buffer else sentence
            if len(candidate) > max_chars and buffer:
                chunks.append(_make_chunk(buffer, source, len(chunks), title))
                buffer = sentence
            else:
                buffer = candidate
        if buffer:
            chunks.append(_make_chunk(buffer, source, len(chunks), title))
    return chunks


def _make_chunk(text: str, source: str, index: int, title: str | None = None) -> RawChunk:
    stem = Path(source).stem
    full_text = f"{title}: {text}" if title else text
    return RawChunk(chunk_id=f"{stem}::{index}", text=full_text, source=source)


def load_corpus_dir(corpus_dir: str | Path) -> list[RawChunk]:
    """Read every .txt/.md file in corpus_dir and return chunked documents."""
    corpus_dir = Path(corpus_dir)
    all_chunks: list[RawChunk] = []
    for path in sorted(corpus_dir.glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, source=path.name))
    return all_chunks
