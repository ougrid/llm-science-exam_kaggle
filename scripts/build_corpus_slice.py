"""Build a 20k-article slice of the corpus into chunk parquet shards.

PLAN.md Day 2 item 1: start retrieval iteration on a small slice so a full
retrieve-and-score cycle takes seconds, before scaling to the full ~270k
articles. Source: data/corpus_raw/ (mbanaei/all-paraphs-parsed-expanded,
downloaded via `kaggle datasets download`).

Run once from the repo root: `python scripts/build_corpus_slice.py`
"""

from pathlib import Path

from datasets import load_from_disk

from llmsci.corpus import slice_by_article_count, write_chunk_shards

DATA = Path("data")
N_ARTICLES = 20_000


def main() -> None:
    ds = load_from_disk(DATA / "corpus_raw")
    print(f"full corpus: {len(ds)} paragraph rows")

    subset = slice_by_article_count(ds, N_ARTICLES)
    print(f"20k-article slice: {len(subset)} paragraph rows")

    out_dir = DATA / "corpus_chunks_20k"
    total_chunks = write_chunk_shards(subset, out_dir)
    print(f"wrote {total_chunks} chunks -> {out_dir}/")


if __name__ == "__main__":
    main()
