"""Build a fully self-consistent open-book dataset using OUR OWN retriever.

Today's debugging found that cdeotte's `all_12_with_context2.csv` mixes 12
source datasets with different context "flavors" -- T1 happened to draw 93%
from one source while training drew mostly from others, and that mismatch
alone was enough to hide a real learning signal. This sidesteps the whole
problem: retrieve context for train_pool.csv AND t1_dev.csv with the exact
same BM25 index (data/bm25_index_20k, the 20k-article slice), so there is by
construction no train/eval retriever or source inconsistency -- exactly
CLAUDE.md's non-negotiable rule, satisfied structurally instead of by luck.

The 20k-article slice covers only ~7% of the full corpus, so absolute
retrieval recall will be mediocre -- that's fine for this purpose. The
point is internal consistency, not retrieval quality; a future pass with
the full ~270k-article corpus is the natural upgrade once this is proven
out.

Run once from the repo root: `python scripts/build_own_retrieval_context.py`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
INDEX_DIR = DATA / "bm25_index_20k"
TOP_K = 5
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


def attach_context(df: pd.DataFrame, index: BM25Index, chunks: pd.DataFrame) -> pd.DataFrame:
    queries = [build_query(row["prompt"], [row[c] for c in OPTION_COLUMNS]) for _, row in df.iterrows()]
    results = index.search_batch(queries, k=TOP_K)
    df = df.copy()
    df["context"] = [" ".join(chunks.iloc[i]["text"] for i, _ in r) for r in results]
    return df


def main() -> None:
    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet")
    index = BM25Index.load(INDEX_DIR, chunks["text"].tolist())

    train_pool = pd.read_csv(DATA / "train_pool.csv")
    t1 = pd.read_csv(DATA / "t1_dev.csv")

    # A handful of train_pool rows have a null option (same class of bug found
    # in cdeotte's data today, much rarer here: 4/4982 rows). Drop rather than
    # let it silently become the literal string "nan".
    null_option_rows = train_pool[OPTION_COLUMNS].isna().any(axis=1)
    if null_option_rows.any():
        print(f"dropping {int(null_option_rows.sum())} train_pool rows with a null option")
        train_pool = train_pool.loc[~null_option_rows].reset_index(drop=True)

    train_pool_own_context = attach_context(train_pool, index, chunks)
    t1_own_context = attach_context(t1, index, chunks)

    train_pool_own_context.to_parquet(DATA / "train_pool_own_context.parquet", index=False)
    t1_own_context.to_parquet(DATA / "t1_dev_own_context.parquet", index=False)
    print(f"train_pool_own_context: {len(train_pool_own_context)} rows -> data/train_pool_own_context.parquet")
    print(f"t1_dev_own_context: {len(t1_own_context)} rows -> data/t1_dev_own_context.parquet")
    print("example context:", train_pool_own_context["context"].iloc[0][:200])


if __name__ == "__main__":
    main()
