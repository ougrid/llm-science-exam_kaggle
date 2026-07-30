"""Day-2 mismatch row: feed retrieved context into the closed-book reader
WITHOUT retraining it, and confirm MAP@3 goes flat or down.

Uses our own BM25 index over the 20k-article corpus slice (not cdeotte's
context) -- retrieval quality doesn't matter for this diagnostic, since the
point is testing whether a reader trained on context="" degrades when handed
an unfamiliar non-empty-context input format, not whether the retrieved
passages are the right ones. Uses the Day-1 best closed-book checkpoint
(data/checkpoints/deberta-v3-base-closed-book-best) at the same MAX_LENGTH=256
it was trained with, so any change is attributable to the input-format shift
alone, not to a wider context window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from llmsci.experiment import git_sha, log_experiment
from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, logits_to_ranked_labels
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-closed-book-best"
INDEX_DIR = DATA / "bm25_index_20k"
MAX_LENGTH = 256  # unchanged from closed-book training -- isolates format shift, not context length
TOP_K = 5
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet")
    index = BM25Index.load(INDEX_DIR, chunks["text"].tolist())

    t1 = pd.read_csv(DATA / "t1_dev.csv")
    queries = [build_query(row["prompt"], [row[c] for c in OPTION_COLUMNS]) for _, row in t1.iterrows()]
    results = index.search_batch(queries, k=TOP_K)
    t1 = t1.copy()
    t1["context"] = [" ".join(chunks.iloc[i]["text"] for i, _ in row_results) for row_results in results]
    print(f"retrieved top-{TOP_K} context for {len(t1)} T1 rows")
    print("example context:", t1["context"].iloc[0][:200])

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForMultipleChoice.from_pretrained(CHECKPOINT_DIR).to(device)
    model.eval()

    collator = DataCollatorForMultipleChoice(tokenizer)
    ds = MultipleChoiceDataset(t1, tokenizer, max_length=MAX_LENGTH, context_col="context")
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collator)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits_to_ranked_labels(logits, k=3)
    scores = average_precision_scores(t1["answer"].tolist(), y_pred, k=3)
    mean, lower, upper = bootstrap_ci(scores, n_resamples=10_000, seed=0)
    baseline = random_baseline_map_at_k()
    print(f"T1 + our-BM25-top{TOP_K}-context, closed-book-best checkpoint (not retrained): "
          f"MAP@3 {mean:.4f} [{lower:.4f},{upper:.4f}] (baseline {baseline:.4f})")

    log_experiment(
        {
            "date": pd.Timestamp.now().isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "config": "deberta-v3-base_closed-book-best-ckpt_fed_bm25-top5-context_NOT-RETRAINED",
            "tier": "T1",
            "n": len(t1),
            "map3_mean": round(mean, 4),
            "map3_ci_lower": round(lower, 4),
            "map3_ci_upper": round(upper, 4),
            "random_baseline": round(baseline, 4),
            "train_seconds": "",
            "eval_seconds": "",
            "hypothesis": "mismatch row: does the closed-book reader benefit from unfamiliar retrieved context without being retrained on it",
            "notes": (
                "PLAN.md's Day-2 mismatch row. Loaded the Day-1 best closed-book checkpoint "
                "(never saw non-empty context during training) and fed it top-5 BM25 chunks "
                "from our own 20k-article corpus slice, at the same MAX_LENGTH=256 it was "
                "trained with. Retrieval quality is irrelevant here -- the slice covers only "
                "~7% of the full 270k-article corpus, so many T1 source articles aren't even "
                "present -- the point is testing input-format shock, not retrieval recall. "
                f"Compare against the closed-book-best result logged earlier ({CHECKPOINT_DIR.name})."
            ),
        }
    )


if __name__ == "__main__":
    main()
