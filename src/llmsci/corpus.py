"""Streaming chunker for the mbanaei/all-paraphs-parsed-expanded corpus.

The source dataset (2,101,279 rows across 276,559 STEM Wikipedia articles) is
already paragraph-parsed: each row is one paragraph under (title, section).
Paragraph length is already close to a good chunk size (median 537 chars /
~130 tokens per a sample check), so v1 chunking only splits the long tail
(paragraphs over `MAX_CHUNK_WORDS` words) into non-overlapping word windows,
rather than PLAN.md's fuller "3-sentence window, stride 1" recipe. This is a
deliberate simplification for time, not an oversight -- revisit with a real
sentence splitter (e.g. blingfire) if Day-3 attribution work shows chunk
boundaries are costing recall.

Every chunk is prefixed with its article title (1st place's trick from
PLAN.md), since questions often reference the article topic only implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

MAX_CHUNK_WORDS = 200


def chunk_paragraph_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split `text` into non-overlapping windows of at most `max_words` words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def iter_chunks(ds: Dataset) -> Iterator[dict]:
    """Yield one dict per chunk: chunk_id, article_id, title, section, text.

    `text` is the title-prefixed chunk body, ready to embed/index. `article_id`
    is a stable per-title integer, used for the article-level leakage checks
    and for hierarchical article->chunk retrieval.
    """
    title_to_id: dict[str, int] = {}
    chunk_id = 0
    for row in ds:
        title = row["title"]
        article_id = title_to_id.setdefault(title, len(title_to_id))
        for piece in chunk_paragraph_text(row["text"]):
            yield {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "title": title,
                "section": row["section"],
                "text": f"{title}: {piece}",
            }
            chunk_id += 1


def slice_by_article_count(ds: Dataset, n_articles: int) -> Dataset:
    """Return the subset of rows belonging to the first `n_articles` unique titles.

    Row order in the source dataset groups by title already (adjacent rows
    share a title in practice for this corpus), so this is a single linear
    scan rather than a full unique-then-filter pass.
    """
    seen: set[str] = set()
    keep_idx = []
    for i, title in enumerate(ds["title"]):
        if title not in seen:
            if len(seen) >= n_articles:
                break
            seen.add(title)
        keep_idx.append(i)
    return ds.select(keep_idx)


def write_chunk_shards(ds: Dataset, out_dir: Path, shard_rows: int = 200_000) -> int:
    """Stream chunks to parquet shards under `out_dir`. Returns total chunk count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("chunk_id", pa.int64()),
            ("article_id", pa.int64()),
            ("title", pa.string()),
            ("section", pa.string()),
            ("text", pa.string()),
        ]
    )
    buffer: list[dict] = []
    shard_idx = 0
    total = 0
    for chunk in iter_chunks(ds):
        buffer.append(chunk)
        total += 1
        if len(buffer) >= shard_rows:
            _write_shard(buffer, out_dir / f"shard-{shard_idx:05d}.parquet", schema)
            shard_idx += 1
            buffer = []
    if buffer:
        _write_shard(buffer, out_dir / f"shard-{shard_idx:05d}.parquet", schema)
    return total


def _write_shard(rows: list[dict], path: Path, schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)
