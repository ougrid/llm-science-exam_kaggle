"""Day-3 retrieval eval harness: recall@k / MRR / nDCG for the full-corpus
BM25 index against T1 (and T3 if present), using the answer-support-recall
proxy (src/llmsci/retrieve/eval.py) since no source-article ground truth
exists in any of our data tiers.

Run once from the repo root, after scripts/build_bm25_index_full.py:
`python scripts/run_retrieval_eval.py`
"""

from pathlib import Path

import pandas as pd

from llmsci.experiment import git_sha, log_experiment
from llmsci.retrieve.eval import evaluate_retrieval
from llmsci.retrieve.sparse import BM25Index

DATA = Path("data")
INDEX_DIR = DATA / "bm25_index_full"


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
                "config": f"retrieval_eval_full-corpus-bm25_answer-support-proxy_{row['metric']}",
                "tier": name,
                "n": len(df),
                "map3_mean": round(row["mean"], 4),
                "map3_ci_lower": round(row["ci_lower"], 4),
                "map3_ci_upper": round(row["ci_upper"], 4),
                "random_baseline": "",
                "train_seconds": "",
                "eval_seconds": "",
                "hypothesis": "Day-3 retrieval eval harness: how good is our full-corpus BM25 retriever on its own, independent of any reader",
                "notes": (
                    f"Proxy B (answer-support recall): a retrieved chunk counts as a hit if it contains "
                    f"a content word (len>=5) present in the correct option but absent from all four "
                    f"distractors. No source-article ground truth exists in T1/T3/gold-200 (checked "
                    f"directly), so PLAN.md's Proxy A (source-article recall) isn't computable -- this is "
                    f"the deliberate fallback PLAN.md itself names as usable even on the gold 200. "
                    f"map3_mean/ci columns hold this metric's value, not a MAP@3 score."
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
