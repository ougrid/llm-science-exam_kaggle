"""Index the FULL general corpus, not half of it. The last untested score lever.

Why this matters for the leaderboard. data/bm25_index_general holds 1,600,063
chunks -- every SECOND shard, because scripts/build_bm25_index_general.py sets
SHARD_STEP=2 after a full build hit 12.45 GB RSS and started swapping. So the
submitted pipeline searches HALF the corpus it could. Our answer-support recall@5
is 0.6183 [0.5800,0.6567] against cdeotte's 0.6667 [0.6283,0.7050] on the same
rows -- a resolved 4.8-point deficit -- and missing coverage is the obvious
suspect.

This is the one remaining lever that is ours rather than borrowed, and the only one
that should help the HIDDEN test more than the gold 200: 4,000 diverse questions
are far likelier to touch topics absent from a half-corpus than 200 are. It also
plausibly explains part of the 0.8600-local / 0.761131-LB gap, since a coverage
hole surfaces as retrieval misses on unseen topics rather than as reader error. And
it improves every reader, including the public checkpoint already in the
submission, with no training.

WHY THE ORIGINAL OOM'd, and it was not the corpus size. The old script does:

    dfs = [pd.read_parquet(p, columns=[...]) for p in shard_paths]   # 16 frames
    chunks = pd.concat(dfs, ignore_index=True)                       # + a full copy
    index = BM25Index(chunks["text"].tolist())                       # + a third copy

Three simultaneous copies of ~3M rows. CLAUDE.md's rule is explicit -- "never
pd.concat the corpus" -- and the script it points at breaks it. This version:

  * reads one shard at a time and extends a single text list, freeing each frame
  * streams chunk_id/title/text straight to parquet with a ParquetWriter, so the
    metadata never accumulates in RAM
  * prints RSS and system-available memory after every shard, and ABORTS at a
    declared floor rather than letting the OOM killer decide

That last point is the difference between a build that fails informatively and one
that takes the machine down. The previous attempt was "cut short only by an
unrelated machine reboot", i.e. nobody was watching the number that mattered.

Run: python scripts/build_bm25_index_general_full.py [--min-available-gb 2.0]
"""

from __future__ import annotations

import argparse
import gc
import resource
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from llmsci.retrieve.sparse import BM25Index

DATA = Path("data")
CHUNK_DIR = DATA / "corpus_chunks_general"
OUT_DIR = DATA / "bm25_index_general_full"


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-available-gb", type=float, default=2.0,
                    help="abort if system available memory drops below this")
    ap.add_argument("--max-shards", type=int, default=None)
    # The shipped index took shards [::2] (0,2,4,...). Its COMPLEMENT (1,3,5,...) is
    # 1.42M chunks -- smaller than the 1.6M that builds fine -- so two half-indexes
    # fused at query time reach full coverage at half the peak memory, instead of one
    # 3.02M index that does not fit this box at all. Corpus diversity through
    # multiple indexes rather than one bigger one is also what 1st place did.
    ap.add_argument("--shard-step", type=int, default=1)
    ap.add_argument("--shard-offset", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    start = time.time()
    shards = sorted(CHUNK_DIR.glob("shard-*.parquet"))[args.shard_offset :: args.shard_step]
    if args.max_shards:
        shards = shards[: args.max_shards]
    print(f"full build over {len(shards)} shards (the shipped index used every 2nd)")
    print(f"abort floor: {args.min_available_gb:.1f} GB available | "
          f"start: RSS {rss_gb():.2f} GB, available {available_gb():.2f} GB", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    writer = None
    try:
        for i, p in enumerate(shards, 1):
            df = pd.read_parquet(p, columns=["chunk_id", "title", "text"])
            texts.extend(df["text"].tolist())
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_dir / "chunk_texts.parquet", table.schema)
            writer.write_table(table)
            del df, table
            gc.collect()
            avail = available_gb()
            print(f"  shard {i}/{len(shards)}: {len(texts):>9,} chunks | "
                  f"RSS {rss_gb():.2f} GB | available {avail:.2f} GB "
                  f"[{time.time()-start:.0f}s]", flush=True)
            if avail < args.min_available_gb:
                print(f"  ** ABORTING at shard {i}: available {avail:.2f} GB is below the "
                      f"{args.min_available_gb:.1f} GB floor. The partial parquet is valid but "
                      f"the index was never built -- rerun with fewer shards.", flush=True)
                return
    finally:
        if writer is not None:
            writer.close()

    print(f"\nloaded {len(texts):,} chunks, RSS {rss_gb():.2f} GB, "
          f"available {available_gb():.2f} GB -- tokenising (the real memory peak)", flush=True)
    # The previous attempt died HERE, not during loading, and took the terminal with
    # it: the OOM killer picks whatever victim it likes. The per-shard floor above
    # never fired because tokenisation is a single call with no checkpoint inside it.
    # Two mitigations. First, run this script under `ulimit -v` so an over-budget
    # allocation raises MemoryError in THIS process instead of the kernel choosing a
    # casualty. Second, catch it and say what to retry with, so a failure is a
    # measurement of the ceiling rather than a lost session.
    try:
        index = BM25Index(texts)
    except MemoryError:
        n = len(texts)
        print(f"  ** MemoryError tokenising {n:,} chunks. This is the ceiling for this box.")
        print(f"     Retry with --max-shards {max(1, len(shards) - 2)} "
              f"(~{n * (len(shards) - 2) // len(shards):,} chunks).")
        print("     The shipped 1,600,063-chunk index is untouched and still usable.")
        return
    print(f"indexed | RSS {rss_gb():.2f} GB | available {available_gb():.2f} GB "
          f"[{time.time()-start:.0f}s]", flush=True)
    index.save(out_dir)
    print(f"saved -> {out_dir}/ [{time.time()-start:.0f}s]")
    print(f"\npeak RSS {rss_gb():.2f} GB (the old pd.concat build hit 12.45 GB on HALF this)")
    print("next: scripts/compare_corpus_recall_paired.py to check the recall gain is real")


if __name__ == "__main__":
    main()
