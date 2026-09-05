"""Load and chunk the raw corpus documents for indexing.

Chunking strategy: split on paragraph boundaries (a hard boundary — chunks
never merge across two different source paragraphs, since two adjacent
paragraphs are usually different sub-topics), then WITHIN each paragraph,
greedily merge SENTENCES up to `max_chars`.

Sentences, not paragraphs, are the atomic merge unit. This came from a real
bug: an earlier version merged at the paragraph level, so a single
paragraph covering several distinct facts (e.g. "the Lunar Module was
nicknamed X... Mission Control did Y... the crew later did Z...") became
one chunk. Dense sentence embeddings pool the whole chunk into one vector,
so a fact buried in a topically broad chunk gets diluted and can lose a
similarity race to a short, single-topic chunk from a completely different,
wrong document. In production this showed up as a chunk ranking 29th out of
46 for a query it directly answered. Splitting on sentences keeps each
chunk closer to "one fact," which is also what the critic needs anyway — it
points at a specific supporting chunk, not "somewhere in this paragraph."

The sentence splitter matches on a run of `.`/`!`/`?`, optionally followed by
a single closing quote mark, requiring whitespace (or end of string) right
after that — using a lookahead rather than a lookbehind, since Python's `re`
only allows fixed-width lookbehind and the optional quote makes the "did we
just end a sentence" check variable-width. The optional-quote handling
matters a lot on this corpus specifically: sentences ending in a quoted word
(`nicknamed "Casper."`) are common, and a naive "period then whitespace"
check never fires there — the period is immediately followed by the closing
quote, not whitespace — silently merging that sentence with whatever comes
next. That's exactly the dilution bug this chunker exists to avoid, so it's
worth handling rather than shrugging off as cosmetic.

Not handled, and still a cosmetic-only gap: abbreviations like "Jr." or
"U.S." can still cause a false split, since the splitter has no dictionary
of abbreviations. In practice the following fragment is short enough that it
re-merges into the same chunk during the greedy merge step below, so this
particular gap costs an odd chunk boundary, not lost information. A robust
NLP sentence tokenizer is stretch-goal territory for a ~20-document
hand-authored corpus.

Markdown header paragraphs (e.g. "# Apollo 13") are dropped entirely before
chunking, not just left small. Every document in this corpus opens with one,
and since sentence-level chunking makes each its own tiny chunk, ~20 nearly
identical bare-title chunks ("# Apollo 11", "# Apollo 12", ...) cluster
together in embedding space and can crowd every substantive chunk in a
document out of the top-k — this was observed directly: a query about Apollo
1 retrieved ten other missions' bare headers and zero Apollo 1 content
chunks. Dropping them costs nothing, since the mission name is always
restated in the first sentence of the body text anyway (and the citation
already carries the source filename).
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
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip() and not _HEADER_ONLY.fullmatch(p.strip())
    ]

    chunks: list[RawChunk] = []
    for para in paragraphs:
        buffer = ""
        for sentence in _split_sentences(para):
            candidate = f"{buffer} {sentence}" if buffer else sentence
            if len(candidate) > max_chars and buffer:
                chunks.append(_make_chunk(buffer, source, len(chunks)))
                buffer = sentence
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
