"""Paired version of the Day-2 mismatch row.

run_mismatch_row.py reported context vs no-context as two separate CIs,
which is exactly the comparison CLAUDE.md says not to make this way. This
script evaluates the SAME closed-book-best checkpoint on the SAME 1,500 T1
rows under both conditions (context="" and context=our BM25 top-5), keeps
per-row AP@3 for each, and reports a paired bootstrap CI on the difference.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from llmsci.experiment import git_sha, log_experiment
from llmsci.metrics import average_precision_scores, paired_bootstrap, random_baseline_map_at_k
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, logits_to_ranked_labels
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-closed-book-best"
INDEX_DIR = DATA / "bm25_index_20k"
MAX_LENGTH = 256
TOP_K = 5
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


def run_condition(model, tokenizer, df, context_col, device):
    collator = DataCollatorForMultipleChoice(tokenizer)
    ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col=context_col)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collator)
    n_batches = len(loader)
    all_logits = []
    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (n_batches - i - 1) / rate if rate > 0 else float("nan")
                print(
                    f"  [{context_col}] batch {i + 1}/{n_batches} "
                    f"({elapsed:.0f}s elapsed, {rate:.2f} batch/s, ETA {eta:.0f}s)"
                )
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits_to_ranked_labels(logits, k=3)
    return average_precision_scores(df["answer"].tolist(), y_pred, k=3)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet")
    index = BM25Index.load(INDEX_DIR, chunks["text"].tolist())

    t1 = pd.read_csv(DATA / "t1_dev.csv")
    queries = [build_query(row["prompt"], [row[c] for c in OPTION_COLUMNS]) for _, row in t1.iterrows()]
    results = index.search_batch(queries, k=TOP_K)
    t1 = t1.copy()
    t1["bm25_context"] = [" ".join(chunks.iloc[i]["text"] for i, _ in r) for r in results]
    t1["no_context"] = ""

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForMultipleChoice.from_pretrained(CHECKPOINT_DIR).to(device)
    model.eval()

    scores_no_ctx = run_condition(model, tokenizer, t1, "no_context", device)
    scores_ctx = run_condition(model, tokenizer, t1, "bm25_context", device)

    baseline = random_baseline_map_at_k()
    print(f"no-context   mean AP@3: {np.mean(scores_no_ctx):.4f}")
    print(f"bm25-context mean AP@3: {np.mean(scores_ctx):.4f}")

    # paired_bootstrap(a, b) computes mean(b - a) per its docstring -- pass
    # (no_ctx, ctx) so the result is (context - no-context), matching the
    # "context minus no-context" label below. Got this backward on the first
    # run (passed (ctx, no_ctx)), which silently inverted the conclusion to
    # "context helps" when the raw means (no-context 0.5641 > context 0.5003)
    # already showed the opposite -- see the corrected experiments/log.csv row.
    diff_mean, diff_lower, diff_upper = paired_bootstrap(scores_no_ctx, scores_ctx, n_resamples=10_000, seed=0)
    print(
        f"paired bootstrap delta (context - no-context): {diff_mean:.4f} "
        f"[{diff_lower:.4f},{diff_upper:.4f}] (baseline {baseline:.4f})"
    )
    resolved = not (diff_lower <= 0 <= diff_upper)
    print(f"CI excludes 0 (resolved, not noise-band): {resolved}")

    log_experiment(
        {
            "date": pd.Timestamp.now().isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "config": "deberta-v3-base_closed-book-best-ckpt_PAIRED_context-vs-no-context_NOT-RETRAINED",
            "tier": "T1",
            "n": len(t1),
            "map3_mean": round(float(diff_mean), 4),
            "map3_ci_lower": round(float(diff_lower), 4),
            "map3_ci_upper": round(float(diff_upper), 4),
            "random_baseline": round(baseline, 4),
            "train_seconds": "",
            "eval_seconds": "",
            "hypothesis": "mismatch row (paired): does feeding unfamiliar retrieved context to the closed-book-best checkpoint change MAP@3, measured as a per-row paired difference rather than two separate CIs",
            "notes": (
                f"map3_mean/ci columns here hold the PAIRED DELTA (context minus no-context), "
                f"not an absolute MAP@3 -- no-context mean {np.mean(scores_no_ctx):.4f}, "
                f"bm25-context mean {np.mean(scores_ctx):.4f}, same checkpoint "
                f"({CHECKPOINT_DIR.name}), same 1500 T1 rows, same MAX_LENGTH=256. "
                f"CI {'excludes' if resolved else 'includes'} 0 -> "
                f"{'context measurably hurts this untrained-for-context checkpoint' if resolved and diff_mean < 0 else 'not resolved by this eval' if not resolved else 'context measurably helps (unexpected for a mismatch row)'}"
                ". Supersedes the unpaired run_mismatch_row.py comparison (0.5641 closed-book-best "
                "vs 0.4999 with context) which compared two independent CIs, not a per-row paired "
                "difference -- exactly what CLAUDE.md's non-negotiable rules warn against."
            ),
        }
    )


if __name__ == "__main__":
    main()
