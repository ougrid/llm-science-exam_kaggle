"""Day-3 attribution: oracle-context ceiling + 2x2 failure decomposition.

Uses the row-7 checkpoint (deberta-v3-base, full-corpus own-retrieval,
trained on Kaggle but re-scored locally at 0.4297 -- see DEVLOG.md's
Kaggle-vs-local environment entry) as "the reader" for this analysis.

No source-article ground truth exists in T1 (checked directly earlier
today), so the oracle passage is approximated as: search a larger pool
(top-100) via the same BM25 index, and take the first chunk that is an
answer-support hit (src/llmsci/retrieve/eval.py's proxy) as the "oracle"
context for that row. Rows with no hit anywhere in the top-100 have no
oracle available and are excluded from the oracle-ceiling number (counted
and reported, not silently dropped).

Split into two phases so the (CPU-only) retrieval phase can run while the
GPU is busy with another training job: run with --phase retrieve first,
then --phase eval once the GPU is free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from llmsci.experiment import git_sha, log_experiment
from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, logits_to_ranked_labels
from llmsci.retrieve.eval import distinctive_keywords, is_answer_support_hit
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
INDEX_DIR = DATA / "bm25_index_full"
CHECKPOINT_DIR = Path("/tmp/kaggle_v5_output/deberta-v3-base-open-book-own-retrieval-best")
ORACLE_POOL_K = 100
MAX_LENGTH = 384
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]
INTERMEDIATE_PATH = DATA / "t1_oracle_context.parquet"


def build_oracle_contexts() -> None:
    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet")
    chunk_texts = chunks["text"].tolist()
    index = BM25Index.load(INDEX_DIR, chunk_texts)

    t1 = pd.read_csv(DATA / "t1_dev.csv")
    queries = [build_query(row["prompt"], [row[c] for c in OPTION_COLUMNS]) for _, row in t1.iterrows()]
    results = index.search_batch(queries, k=ORACLE_POOL_K)

    oracle_contexts = []
    oracle_available = []
    for (_, row), candidates in zip(t1.iterrows(), results):
        kws = distinctive_keywords(row)
        found = None
        for idx, _score in candidates:
            if is_answer_support_hit(chunk_texts[idx], kws):
                found = chunk_texts[idx]
                break
        oracle_contexts.append(found if found is not None else "")
        oracle_available.append(found is not None)

    t1 = t1.copy()
    t1["oracle_context"] = oracle_contexts
    t1["oracle_available"] = oracle_available
    t1.to_parquet(INTERMEDIATE_PATH, index=False)
    n_avail = sum(oracle_available)
    print(f"oracle context available for {n_avail}/{len(t1)} rows ({100 * n_avail / len(t1):.1f}%)")
    print(f"saved -> {INTERMEDIATE_PATH}")


def run_eval() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForMultipleChoice.from_pretrained(CHECKPOINT_DIR).to(device)
    model.eval()
    collator = DataCollatorForMultipleChoice(tokenizer)

    t1_oracle = pd.read_parquet(INTERMEDIATE_PATH)
    t1_own = pd.read_parquet(DATA / "t1_dev_own_context_full.parquet")
    baseline = random_baseline_map_at_k()

    def infer(df: pd.DataFrame, context_col: str) -> np.ndarray:
        ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col=context_col)
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collator)
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                all_logits.append(out.logits.float().cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    # 1. Actual pipeline predictions (top-5 concatenated own-retrieval context)
    logits_actual = infer(t1_own, "context")
    y_pred_actual = logits_to_ranked_labels(logits_actual, k=3)
    scores_actual = average_precision_scores(t1_own["answer"].tolist(), y_pred_actual, k=3)
    reader_correct = np.array([1 if p[0] == a else 0 for p, a in zip(y_pred_actual, t1_own["answer"])])

    # 2. Oracle-context predictions, only on rows where an oracle passage was found
    available_mask = t1_oracle["oracle_available"].values
    t1_oracle_avail = t1_oracle.loc[available_mask].reset_index(drop=True)
    logits_oracle = infer(t1_oracle_avail, "oracle_context")
    y_pred_oracle = logits_to_ranked_labels(logits_oracle, k=3)
    scores_oracle = average_precision_scores(t1_oracle_avail["answer"].tolist(), y_pred_oracle, k=3)
    oracle_mean, oracle_lower, oracle_upper = bootstrap_ci(scores_oracle, n_resamples=10_000, seed=0)

    actual_mean, actual_lower, actual_upper = bootstrap_ci(scores_actual, n_resamples=10_000, seed=0)
    print(f"actual pipeline (top-5 own-retrieval context): {actual_mean:.4f} [{actual_lower:.4f},{actual_upper:.4f}]")
    print(f"oracle context (n={len(t1_oracle_avail)}, {available_mask.mean()*100:.1f}% of T1): "
          f"{oracle_mean:.4f} [{oracle_lower:.4f},{oracle_upper:.4f}]")
    print(f"retrieval-attributable loss (oracle - actual, on the same {len(t1_oracle_avail)}-row subset): "
          f"{oracle_mean - actual_mean:.4f}")

    # 3. 2x2 failure decomposition: reader_correct x gold_retrieved (was there a hit in the ACTUAL top-5)
    kws_per_row = [distinctive_keywords(row) for _, row in t1_own.iterrows()]
    gold_retrieved = np.array([
        1 if is_answer_support_hit(ctx, kws) else 0
        for ctx, kws in zip(t1_own["context"], kws_per_row)
    ])
    table = pd.crosstab(
        pd.Series(reader_correct, name="reader_correct"),
        pd.Series(gold_retrieved, name="gold_retrieved"),
    )
    print("\n2x2 failure decomposition (rows=reader_correct, cols=gold_retrieved):")
    print(table)
    print("\nas percentages of n=%d:" % len(t1_own))
    print((table / len(t1_own) * 100).round(1))

    log_experiment({
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": "attribution_oracle_ceiling_and_2x2_decomposition",
        "tier": "T1",
        "n": len(t1_oracle_avail),
        "map3_mean": round(oracle_mean, 4),
        "map3_ci_lower": round(oracle_lower, 4),
        "map3_ci_upper": round(oracle_upper, 4),
        "random_baseline": round(baseline, 4),
        "train_seconds": "",
        "eval_seconds": "",
        "hypothesis": "oracle-context ceiling (best-available answer-support chunk from top-100) and 2x2 reader/retrieval failure decomposition, using the row-7 checkpoint scored locally",
        "notes": (
            f"Oracle context approximated as the first answer-support hit within a top-100 BM25 "
            f"pool (no source-article ground truth exists in T1, checked directly earlier today). "
            f"Available for {available_mask.sum()}/{len(t1_oracle)} rows ({available_mask.mean()*100:.1f}%); "
            f"rows without an oracle hit anywhere in top-100 are excluded from this number, not "
            f"imputed. Actual pipeline (same checkpoint, real top-5 concatenated context): "
            f"{actual_mean:.4f} [{actual_lower:.4f},{actual_upper:.4f}] (n={len(t1_own)}). Oracle vs "
            f"actual gap on the same {len(t1_oracle_avail)}-row subset: {oracle_mean - actual_mean:.4f} "
            f"-- this is the retrieval-attributable loss PLAN.md's attribution table calls for. "
            f"2x2 table (reader_correct x gold_retrieved, gold_retrieved = actual top-5 context "
            f"contains an answer-support hit): {table.values.tolist()}."
        ),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["retrieve", "eval"], required=True)
    args = parser.parse_args()
    if args.phase == "retrieve":
        build_oracle_contexts()
    else:
        run_eval()


if __name__ == "__main__":
    main()
