"""Retrieval eval harness for the general-Wikipedia corpus, mirroring
scripts/run_retrieval_eval.py exactly but pointed at data/bm25_index_general
(scripts/build_bm25_index_general.py) instead of the mbanaei full corpus.

Purpose: does swapping to a general (non-STEM-filtered) Wikipedia corpus
actually raise answer-support recall, motivated by the hand-inspection
finding that the STEM-only mbanaei corpus cannot contain certain non-STEM
entities that T1 questions reference (see reports/ablation_table.md).

Run once from the repo root, after scripts/build_bm25_index_general.py:
`python scripts/run_retrieval_eval_general.py`
"""

from pathlib import Path

import pandas as pd

from llmsci.experiment import git_sha, log_experiment
from llmsci.retrieve.eval import evaluate_retrieval
from llmsci.retrieve.sparse import BM25Index

DATA = Path("data")
INDEX_DIR = DATA / "bm25_index_general"


def report(name: str, df: pd.DataFrame, index: BM25Index, chunk_texts: list[str]) -> None:
    result = evaluate_retrieval(df, index, chunk_texts)
    print(f"\n=== {name} (n={len(df)}) ===")
    print(result.to_string(index=False))

    git_sha_val = git_sha()
    for _, row in result.iterrows():
        log_experiment(
            {
                "date": pd.Timestamp.now().isoformat(timespec="seconds"),
                "git_sha": git_sha_val,
                "config": f"retrieval_eval_GENERAL-corpus-bm25_answer-support-proxy_{row['metric']}",
                "tier": name,
                "n": len(df),
                "map3_mean": round(row["mean"], 4),
                "map3_ci_lower": round(row["ci_lower"], 4),
                "map3_ci_upper": round(row["ci_upper"], 4),
                "random_baseline": "",
                "train_seconds": "",
                "eval_seconds": "",
                "hypothesis": (
                    "does swapping mbanaei's STEM-only corpus for a general (non-STEM-filtered) "
                    "Wikipedia corpus raise answer-support recall, motivated by the hand-inspection "
                    "finding that the STEM-only corpus structurally cannot contain some non-STEM "
                    "entities T1 questions reference"
                ),
                "notes": (
                    f"General corpus: jjinho/wikipedia-20230701, a ~14%-sampled / 8-of-16-shard subset "
                    f"(1,600,063 chunks) of the full 6.28M-article dump -- the full dump's ~21.56M "
                    f"chunks / ~23.5GB raw text don't fit this box's 15 GiB RAM for BM25 indexing (see "
                    f"scripts/build_corpus_general.py, scripts/build_bm25_index_general.py). Compare "
                    f"against the mbanaei-corpus row from the same metric/tier in this log "
                    f"(retrieval_eval_full-corpus-bm25_answer-support-proxy_{row['metric']}) -- same "
                    f"proxy, same T1/T3 rows, different corpus, so directly comparable despite the "
                    f"smaller sample size and different corpus domain."
                ),
            }
        )


def main() -> None:
    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet")
    index = BM25Index.load(INDEX_DIR, chunks["text"].tolist())
    chunk_texts = chunks["text"].tolist()

    t1 = pd.read_csv(DATA / "t1_dev.csv")
    report("T1", t1, index, chunk_texts)

    t3_path = DATA / "t3_ood.parquet"
    if t3_path.exists():
        t3 = pd.read_parquet(t3_path)
        import json

        t3 = t3.copy()
        options_expanded = t3["options"].apply(json.loads)
        for i, col in enumerate(["A", "B", "C", "D", "E"]):
            t3[col] = options_expanded.apply(lambda opts, i=i: opts[i] if i < len(opts) else "")
        t3["prompt"] = t3["question"]
        report("T3-OOD", t3, index, chunk_texts)


if __name__ == "__main__":
    main()
