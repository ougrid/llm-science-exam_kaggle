"""Build the FULL corpus (all ~276k articles) into chunk parquet shards.

Follow-up to scripts/build_corpus_slice.py's 20k-article pilot slice: that
pilot's BM25 index only reached ~0.39 MAP@3 when used for a fully
self-consistent open-book training run (vs. 0.61 for a cdeotte-context run
matched to the same source), and the leading explanation is retrieval
recall -- the 20k-article slice covers only ~7% of the corpus, so most T1
source articles simply aren't retrievable. This tests whether real recall
(the full corpus) closes that gap.

Run once from the repo root: `python scripts/build_corpus_full.py`
"""

import time
from pathlib import Path

from datasets import load_from_disk

from llmsci.corpus import write_chunk_shards

DATA = Path("data")


def main() -> None:
    start = time.time()
    ds = load_from_disk(DATA / "corpus_raw")
    print(f"full corpus: {len(ds)} paragraph rows")

    out_dir = DATA / "corpus_chunks_full"
    total_chunks = write_chunk_shards(ds, out_dir)
    print(f"wrote {total_chunks} chunks -> {out_dir}/ ({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()
