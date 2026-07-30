"""Reference measurement: what does a known-good PUBLIC checkpoint score on our
clean gold 200?

WHY THIS IS THE RIGHT INSTRUMENT, AND WHY T1 IS NOT.
`reference_reproduction/RESULTS.md` established that
`mgoksu/llm-science-run-context-2` scores 0.9170 [0.9056, 0.9282] on T1 but is
**100% contaminated against it** -- audited at file level, its training files
(`6000_train_examples.csv` + `extra_train_set.csv`) cover all 1,500 T1 prompts.
That T1 number is meaningless and must never be reported as a pipeline result.

The gold 200 (`data/holdout_gold.csv`, the official `train.csv`) is a **clean**
surface for exactly this checkpoint: `scripts/build_context_train_pool.py`
asserts zero prompt overlap between the public cdeotte pool and the gold 200,
and `reference_reproduction/RESULTS.md` independently names the gold 200 and T3
as "the only clean evaluation surfaces for a public checkpoint".

WHAT THIS NUMBER IS, PRECISELY:
- It is a **calibration anchor / ceiling reference**: what a 2023 top-solution
  ensemble leg achieves on our own clean holdout, using OUR retrieval.
- It is **NOT this project's pipeline score**, and must be reported in a
  separate, explicitly-labelled row. Conflating it with an own-pipeline number
  would be the single fastest way to fail an interview on this project, since
  the provenance is discoverable in about ninety seconds.
- It is a **lower bound on that checkpoint's true ability**, because it reads
  context from our BM25 retriever rather than the retriever it was trained
  against -- a train/inference retriever mismatch of exactly the kind
  CLAUDE.md warns about, here working against the checkpoint.

Uses part 2's real inference format (`ctx[:1750] + " #### " + prompt`, then
`tokenizer(first, option, truncation=True)`), transcribed from
`reference_reproduction/reproduced/eval_shipped.py` so the checkpoint is scored
the way it was actually run rather than under our own tokenization.

Spends ONE of the project's ~8 budgeted gold-set evaluations. Logged with a
date and reason per CLAUDE.md.

Run from the repo root:
  python scripts/eval_public_checkpoint_on_gold.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from llmsci.experiment import git_sha, log_experiment
from llmsci.gpu_guard import cap_memory_fraction
from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k
from llmsci.reader.mc import DataCollatorForMultipleChoice, logits_to_ranked_labels
from llmsci.retrieve.sparse import BM25Index, build_query

DATA = Path("data")
INDEX_DIR = DATA / "bm25_index_general"
MODEL_DIR = Path("reference_reproduction/models/mgoksu-run-context-2")
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]
TOP_K = 5
CONTEXT_CHAR_CLIP = 1750  # part 2, cell 27 -- the real limiter in its inference path
BATCH_SIZE = 2


class Part2InferenceDataset(Dataset):
    """Part 2's inference tokenization (see module docstring)."""

    def __init__(self, df, tokenizer, context_col="context", char_clip=CONTEXT_CHAR_CLIP):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.context_col = context_col
        self.char_clip = char_clip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ctx = str(row[self.context_col])[: self.char_clip]
        first = f"{ctx} #### {row['prompt']}"
        enc = self.tokenizer([first] * 5, [str(row[c]) for c in OPTION_COLUMNS], truncation=True)
        item = dict(enc)
        if row["answer"] in OPTION_COLUMNS:
            item["label"] = OPTION_COLUMNS.index(row["answer"])
        return item


def main() -> None:
    if not MODEL_DIR.exists():
        raise SystemExit(f"public checkpoint not found at {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)

    gold = pd.read_csv(DATA / "holdout_gold.csv")
    print(f"gold set: {len(gold)} rows")

    chunks = pd.read_parquet(INDEX_DIR / "chunk_texts.parquet")
    chunk_texts = chunks["text"].tolist()
    index = BM25Index.load(INDEX_DIR, chunk_texts)
    queries = [build_query(r["prompt"], [r[c] for c in OPTION_COLUMNS]) for _, r in gold.iterrows()]
    results = index.search_batch(queries, k=TOP_K)
    gold = gold.copy()
    gold["context"] = [" ".join(chunk_texts[i] for i, _ in r) for r in results]
    print(f"attached our own general-corpus BM25 top-{TOP_K} context "
          f"(median {gold['context'].str.len().median():.0f} chars, clipped to {CONTEXT_CHAR_CLIP})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    loader = DataLoader(
        Part2InferenceDataset(gold, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DataCollatorForMultipleChoice(tokenizer),
    )
    out, start = [], time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            out.append(model(**batch).logits.float().cpu().numpy())
            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                el = time.time() - start
                print(f"  batch {i + 1}/{len(loader)} ({el:.0f}s)", flush=True)
    logits = np.concatenate(out, axis=0)
    scores = average_precision_scores(gold["answer"].tolist(), logits_to_ranked_labels(logits, k=3), k=3)
    mean, lo, hi = bootstrap_ci(scores, n_resamples=10_000, seed=0)
    baseline = random_baseline_map_at_k()
    eval_seconds = time.time() - start

    print()
    print(f"PUBLIC CHECKPOINT (mgoksu/llm-science-run-context-2) on the CLEAN gold 200")
    print(f"  MAP@3 {mean:.4f} [{lo:.4f},{hi:.4f}]   (random baseline {baseline:.4f}, n={len(gold)})")
    print(f"  NOT this project's pipeline score -- reference/ceiling anchor only.")
    print(f"  For contrast, the same weights on CONTAMINATED T1: 0.9170 [0.9056,0.9282].")

    log_experiment({
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": "REFERENCE-NOT-OURS_public-checkpoint_mgoksu-run-context-2_our-general-corpus-context",
        "tier": "T2-gold-200",
        "n": len(gold),
        "map3_mean": round(mean, 4),
        "map3_ci_lower": round(lo, 4),
        "map3_ci_upper": round(hi, 4),
        "random_baseline": round(baseline, 4),
        "train_seconds": "",
        "eval_seconds": round(eval_seconds, 1),
        "hypothesis": (
            "what does a known-good PUBLIC 2023 checkpoint score on our CLEAN gold 200 (as a "
            "calibration anchor / reader ceiling), given that its 0.9170 on T1 is 100% "
            "contaminated and therefore uninterpretable"
        ),
        "notes": (
            f"GOLD-SET EVALUATION #1 spent deliberately; reason: establish a clean calibration "
            f"anchor for the final report, which T1 cannot provide for this checkpoint. THIS IS NOT "
            f"AN OWN-PIPELINE RESULT and must be reported only in a row explicitly labelled as "
            f"containing a public checkpoint (see PLAN.md 'Final-day score strategy', item S3). "
            f"Checkpoint: mgoksu/llm-science-run-context-2, the second leg of cdeotte part 2's "
            f"50/50 ensemble behind the published 0.823761 public LB. Clean here because "
            f"scripts/build_context_train_pool.py asserts zero prompt overlap between the public "
            f"cdeotte pool and the gold 200, and reference_reproduction/RESULTS.md independently "
            f"identifies the gold 200 and T3 as the only clean surfaces for this checkpoint. Scored "
            f"in part 2's native inference format (ctx[:1750] + ' #### ' + prompt, then "
            f"tokenizer(first, option, truncation=True)), transcribed from "
            f"reference_reproduction/reproduced/eval_shipped.py. Context is OUR general-corpus BM25 "
            f"top-{TOP_K}, NOT the retriever it was trained against, so this is a LOWER BOUND on "
            f"the checkpoint's true ability -- a train/inference retriever mismatch working against "
            f"it. Same weights on contaminated T1: 0.9170 [0.9056,0.9282]."
        ),
    })
    print("\nlogged to experiments/log.csv")


if __name__ == "__main__":
    main()
