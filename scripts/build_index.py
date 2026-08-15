"""Build (or rebuild) the Chroma index from eval/dataset/corpus/*.txt.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verification.retrieval.corpus_loader import load_corpus_dir  # noqa: E402
from verification.retrieval.vector_store import ChromaRetriever  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parents[1] / "eval" / "dataset" / "corpus"


def main() -> None: # -> None: means the function does not return any value
    chunks = load_corpus_dir(CORPUS_DIR)
    if not chunks:
        print(f"No .txt/.md files found in {CORPUS_DIR} — nothing to index.")
        return

    retriever = ChromaRetriever()
    for chunk in chunks:
        retriever.upsert(chunk_id=chunk.chunk_id, text=chunk.text, source=chunk.source)

    print(f"Indexed {len(chunks)} chunks from {CORPUS_DIR} into Chroma "
          f"(collection now has {retriever.count()} chunks total).")


if __name__ == "__main__":
    main()
