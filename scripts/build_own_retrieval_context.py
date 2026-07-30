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

import argparse
from pathlib import Path

import pandas as pd

from llmsci.retrieve.rerank import rerank_by_phrase_match
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
TOP_K = 5
RERANK_POOL_K = 50  # candidate pool size before reranking down to TOP_K
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


def attach_context(
    df: pd.DataFrame, index: BM25Index, chunks: pd.DataFrame, chunk_texts: list[str], rerank: bool
) -> pd.DataFrame:
    queries = [build_query(row["prompt"], [row[c] for c in OPTION_COLUMNS]) for _, row in df.iterrows()]
    pool_k = RERANK_POOL_K if rerank else TOP_K
    results = index.search_batch(queries, k=pool_k)
    df = df.copy()
    contexts = []
    for query, candidates in zip(queries, results):
        if rerank:
            reranked = rerank_by_phrase_match(query, candidates, chunk_texts)
            top = [(idx, score) for idx, score, _pm in reranked[:TOP_K]]
        else:
            top = candidates[:TOP_K]
        contexts.append(" ".join(chunks.iloc[i]["text"] for i, _ in top))
    df["context"] = contexts
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="bm25_index_full", help="index dir name under data/")
    parser.add_argument("--suffix", default="full", help="output filename suffix, e.g. train_pool_own_context_<suffix>.parquet")
    parser.add_argument("--rerank", action="store_true", help="rerank top-50 BM25 candidates by phrase-match count before taking top-5")
    args = parser.parse_args()
    index_dir = DATA / args.index_dir

    chunk_texts_series = pd.read_parquet(index_dir / "chunk_texts.parquet")
    chunk_texts = chunk_texts_series["text"].tolist()
    index = BM25Index.load(index_dir, chunk_texts)

    train_pool = pd.read_csv(DATA / "train_pool.csv")
    t1 = pd.read_csv(DATA / "t1_dev.csv")

    # A handful of train_pool rows have a null option (same class of bug found
    # in cdeotte's data today, much rarer here: 4/4982 rows). Drop rather than
    # let it silently become the literal string "nan".
    null_option_rows = train_pool[OPTION_COLUMNS].isna().any(axis=1)
    if null_option_rows.any():
        print(f"dropping {int(null_option_rows.sum())} train_pool rows with a null option")
        train_pool = train_pool.loc[~null_option_rows].reset_index(drop=True)

    train_pool_own_context = attach_context(train_pool, index, chunk_texts_series, chunk_texts, args.rerank)
    t1_own_context = attach_context(t1, index, chunk_texts_series, chunk_texts, args.rerank)

    train_out = DATA / f"train_pool_own_context_{args.suffix}.parquet"
    t1_out = DATA / f"t1_dev_own_context_{args.suffix}.parquet"
    train_pool_own_context.to_parquet(train_out, index=False)
    t1_own_context.to_parquet(t1_out, index=False)
    print(f"train_pool_own_context: {len(train_pool_own_context)} rows -> {train_out}")
    print(f"t1_dev_own_context: {len(t1_own_context)} rows -> {t1_out}")
    print("example context:", train_pool_own_context["context"].iloc[0][:200])


if __name__ == "__main__":
    main()
