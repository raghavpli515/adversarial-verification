"""Load and chunk the raw corpus documents for indexing.

Chunking strategy: split on paragraph boundaries, then greedily merge
adjacent paragraphs up to `max_chars` so a chunk is never a single sentence
fragment (bad for embedding quality) or an entire multi-topic article (bad
for retrieval precision — the critic needs to point at a specific supporting
chunk, not "somewhere in this 2000-word article"). This is intentionally
simple; a semantic chunker is stretch-goal territory, not needed for a
~20-document corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)   # frozen=True makes the dataclass immutable, meaning its attributes cannot be modified after creation
class RawChunk:
    chunk_id: str
    text: str
    source: str  # filename, used as the citation the generator/critic point to


def chunk_text(text: str, source: str, max_chars: int = 800) -> list[RawChunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[RawChunk] = []
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) > max_chars and buffer:
            chunks.append(_make_chunk(buffer, source, len(chunks)))
            buffer = para
        else:
            buffer = candidate
    if buffer:
        chunks.append(_make_chunk(buffer, source, len(chunks)))
    return chunks


def _make_chunk(text: str, source: str, index: int) -> RawChunk:
    stem = Path(source).stem
    return RawChunk(chunk_id=f"{stem}::{index}", text=text, source=source)


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
