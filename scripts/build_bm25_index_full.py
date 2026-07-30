"""Build a BM25 index over the FULL ~276k-article corpus and save it to disk.

Follow-up to scripts/build_bm25_index.py's 20k-article pilot index.

Run once from the repo root, after scripts/build_corpus_full.py:
`python scripts/build_bm25_index_full.py`
"""

import time
from pathlib import Path

import pandas as pd

from llmsci.retrieve.sparse import BM25Index

DATA = Path("data")
CHUNK_DIR = DATA / "corpus_chunks_full"
INDEX_DIR = DATA / "bm25_index_full"


def main() -> None:
    start = time.time()
    shard_paths = sorted(CHUNK_DIR.glob("shard-*.parquet"))
    dfs = [pd.read_parquet(p, columns=["chunk_id", "title", "text"]) for p in shard_paths]
    chunks = pd.concat(dfs, ignore_index=True)
    print(f"loaded {len(chunks)} chunks from {len(shard_paths)} shard(s) ({time.time() - start:.0f}s)")

    index = BM25Index(chunks["text"].tolist())
    print(f"indexed ({time.time() - start:.0f}s total)")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_DIR)
    chunks[["chunk_id", "title", "text"]].to_parquet(INDEX_DIR / "chunk_texts.parquet", index=False)
    print(f"saved BM25 index -> {INDEX_DIR}/ ({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()
