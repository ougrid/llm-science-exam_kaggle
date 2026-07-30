"""Build a BM25 index over the general-Wikipedia sampled corpus and save it to disk.

Mirrors scripts/build_bm25_index_full.py, pointed at the ~14%-sampled general
corpus from scripts/build_corpus_general.py instead of the full mbanaei
corpus -- see that script's docstring for why a sample rather than all
~21.56M chunks (memory).

Measured directly: indexing all 16 shards (3,020,431 chunks) pushed this
box's RSS to 12.45 GB with under 250 MB free and climbing into swap -- an
OOM in progress, cut short only by an unrelated machine reboot. SHARD_STEP=2
below indexes every other shard (evenly spaced, not a contiguous block, to
keep the original alphabetical/topical spread) for ~1.6M chunks, a safer
margin under this box's 15 GiB total RAM.

Run once from the repo root, after scripts/build_corpus_general.py:
`python scripts/build_bm25_index_general.py`
"""

import time
from pathlib import Path

import pandas as pd

from llmsci.retrieve.sparse import BM25Index

DATA = Path("data")
CHUNK_DIR = DATA / "corpus_chunks_general"
INDEX_DIR = DATA / "bm25_index_general"
SHARD_STEP = 2


def main() -> None:
    start = time.time()
    shard_paths = sorted(CHUNK_DIR.glob("shard-*.parquet"))[::SHARD_STEP]
    print(f"using {len(shard_paths)}/{len(sorted(CHUNK_DIR.glob('shard-*.parquet')))} shards (SHARD_STEP={SHARD_STEP})")
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
