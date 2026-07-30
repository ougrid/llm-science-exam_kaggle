"""Streaming chunker for the general jjinho/wikipedia-20230701 corpus.

Motivated by the top-2 writeups (see PLAN.md's "Last-day pivot" section):
mbanaei's STEM-only corpus structurally cannot contain non-STEM entities
(e.g. the "Big Mama Thornton" hand-inspection finding in reports/ablation_table.md).
This corpus is a general (non-STEM-filtered) Wikipedia dump, sharded by first
letter, schema (id, title, text, categories) -- verified directly against
one shard before writing this, per CLAUDE.md's "verify, don't assume".

Measured directly by streaming every shard's `text` column once (not
estimated): the full corpus is 6,284,571 articles / ~21.56M chunks / ~23.5 GB
of raw chunked text at MAX_CHUNK_WORDS=200. That alone exceeds this box's
15 GiB total RAM before any pandas/bm25s overhead -- bm25s needs the whole
corpus in memory at once to compute global IDF, so there is no streaming
workaround for the *indexing* step regardless of how the chunking step is
written. SAMPLE_FRACTION below draws a reproducible, order-independent random
sample of articles (~14%, proportional across all 28 shards for topical
diversity) targeting a chunk count close to the ~2.1M-chunk scale of the
mbanaei corpus it's replacing -- a deliberate, documented scope cut, not a
silently-smaller "full corpus".
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from llmsci.corpus import chunk_paragraph_text

RAW_DIR = Path("data/wikipedia_general_raw")
OUT_DIR = Path("data/corpus_chunks_general")
SAMPLE_FRACTION = 0.14
SEED = 42
SHARD_ROWS = 200_000
PROGRESS_EVERY = 200_000


def iter_sampled_articles() -> Iterator[dict]:
    """Yield a reproducible ~SAMPLE_FRACTION random sample of articles, streamed shard by shard."""
    rng = random.Random(SEED)
    files = sorted(f for f in RAW_DIR.glob("*.parquet") if f.name != "wiki_2023_index.parquet")
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["id", "title", "text"], batch_size=20_000):
            for id_, title, text in zip(
                batch.column("id").to_pylist(),
                batch.column("title").to_pylist(),
                batch.column("text").to_pylist(),
            ):
                if not text:
                    continue
                if rng.random() < SAMPLE_FRACTION:
                    yield {"id": id_, "title": title, "text": text}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("chunk_id", pa.int64()),
            ("article_id", pa.int64()),
            ("title", pa.string()),
            ("section", pa.string()),
            ("text", pa.string()),
        ]
    )
    title_to_id: dict[str, int] = {}
    buffer: list[dict] = []
    shard_idx = 0
    chunk_id = 0
    n_articles = 0
    start = time.time()
    for article in iter_sampled_articles():
        title = article["title"]
        article_id = title_to_id.setdefault(title, len(title_to_id))
        n_articles += 1
        for piece in chunk_paragraph_text(article["text"]):
            buffer.append(
                {
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "title": title,
                    "section": "",
                    "text": f"{title}: {piece}",
                }
            )
            chunk_id += 1
        if len(buffer) >= SHARD_ROWS:
            pq.write_table(pa.Table.from_pylist(buffer, schema=schema), OUT_DIR / f"shard-{shard_idx:05d}.parquet")
            shard_idx += 1
            buffer = []
        if n_articles % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            print(f"  ...{n_articles} sampled articles, {chunk_id} chunks so far ({elapsed:.0f}s)", flush=True)
    if buffer:
        pq.write_table(pa.Table.from_pylist(buffer, schema=schema), OUT_DIR / f"shard-{shard_idx:05d}.parquet")
    print(f"sampled {n_articles} articles -> {chunk_id} chunks -> {OUT_DIR}/ ({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()
