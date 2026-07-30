"""Decisive follow-up: train and eval on the SAME context source (source=2).

Two prior diagnostics on the pilot open-book run left an open question:
- T1's matched context is 93% source=2, but the 15k-row training subset was
  88% sources 3/6/4/9/8/7 -- a real train/eval context-distribution mismatch.
- But evaluating on in-distribution held-out data (matching the TRAINING
  mix) still showed only a marginal lift over baseline, not PLAN.md's
  ~0.82-0.86 -- so the mismatch doesn't fully explain the flatness either.

This removes the mismatch variable entirely: train on source=2 rows only
(4,586 of them, excluding T1's 1,500), eval on T1 (93% source=2). If this
still shows no real lift, the problem is upstream of source-mismatch --
in the data quality, the pipeline, or the config itself. Evaluates every 15
optimizer steps (finer than the 50-step granularity used for the 15k-row
pilot) specifically to avoid missing a Day-1-style narrow transient spike.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from llmsci.experiment import git_sha, log_experiment
from llmsci.gpu_guard import cap_memory_fraction, probe_training_speed
from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, logits_to_ranked_labels

DATA = Path("data")
BEST_CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-open-book-source2-best"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 384
MAX_CONTEXT_CHARS = 8_000
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 16
EVAL_BATCH_SIZE = 8
EPOCHS = 3
EVAL_EVERY_STEPS = 15
LR = 5e-6
SEED = 42
MAX_MS_PER_STEP = 800


def evaluate(model, tokenizer, df, collator, device):
    model.eval()
    ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
    all_logits = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())
    seconds = time.time() - start
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits_to_ranked_labels(logits, k=3)
    scores = average_precision_scores(df["answer"].tolist(), y_pred, k=3)
    mean, lower, upper = bootstrap_ci(scores, n_resamples=10_000, seed=0)
    model.train()
    return mean, lower, upper, seconds


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
        probe_training_speed(MODEL_NAME, BATCH_SIZE, 5, MAX_LENGTH, device, MAX_MS_PER_STEP)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME).to(device)

    full = pd.read_parquet(DATA / "train_pool_context.parquet")
    train_df = full[full["source"] == 2].reset_index(drop=True)
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_df = pd.read_parquet(DATA / "t1_dev_context.parquet")
    t1_df["context"] = t1_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    print(f"train (source=2 only): {len(train_df)} rows, T1+context: {len(t1_df)} rows "
          f"({(t1_df.merge(train_df[['prompt']].assign(_x=1), on='prompt', how='left')['_x'].notna().sum())} "
          "prompt-overlap sanity check, should be 0)")

    collator = DataCollatorForMultipleChoice(tokenizer)
    train_ds = MultipleChoiceDataset(train_df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-6)
    num_optim_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_optim_steps), num_training_steps=num_optim_steps
    )
    print(f"optimizer steps/epoch: {len(train_loader) // GRAD_ACCUM_STEPS}, total: {num_optim_steps}")

    best_mean, best_lower, best_upper, best_optim_step = -1.0, None, None, -1
    baseline = random_baseline_map_at_k()

    model.train()
    train_start = time.time()
    optim_step = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1} step {step}")
            (loss / GRAD_ACCUM_STEPS).backward()
            total_loss += loss.item()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1
                if optim_step % EVAL_EVERY_STEPS == 0:
                    mean, lower, upper, _ = evaluate(model, tokenizer, t1_df, collator, device)
                    print(
                        f"epoch {epoch + 1} optim_step {optim_step} T1+ctx MAP@3 {mean:.4f} "
                        f"[{lower:.4f},{upper:.4f}] (baseline {baseline:.4f})"
                    )
                    if mean > best_mean:
                        best_mean, best_lower, best_upper = mean, lower, upper
                        best_optim_step = optim_step
                        BEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(BEST_CHECKPOINT_DIR)
                        tokenizer.save_pretrained(BEST_CHECKPOINT_DIR)
        print(f"epoch {epoch + 1}/{EPOCHS} mean loss: {total_loss / len(train_loader):.4f}")
    train_seconds = time.time() - train_start

    final_mean, final_lower, final_upper, eval_seconds = evaluate(model, tokenizer, t1_df, collator, device)
    print(f"FINAL T1+ctx MAP@3: {final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}]")
    print(f"BEST (optim_step {best_optim_step}) T1+ctx MAP@3: {best_mean:.4f} "
          f"[{best_lower:.4f},{best_upper:.4f}] (baseline {baseline:.4f})")

    log_experiment({
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": f"deberta-v3-base_open-book_SOURCE2-MATCHED_maxlen{MAX_LENGTH}_{EPOCHS}ep_lr{LR}_eps1e-6_n{len(train_df)}",
        "tier": "T1+cdeotte-context",
        "n": len(t1_df),
        "map3_mean": round(best_mean, 4),
        "map3_ci_lower": round(best_lower, 4),
        "map3_ci_upper": round(best_upper, 4),
        "random_baseline": round(baseline, 4),
        "train_seconds": round(train_seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "hypothesis": "decisive test: does open-book training show real signal when train/eval context source is matched (removes the mismatch confound)",
        "notes": (
            f"Train and eval both source=2 (T1 is 93% source=2). Final end-of-training: "
            f"{final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}]. Eval every {EVAL_EVERY_STEPS} "
            "optim steps (finer than the 15k-row pilot's 50) to catch a possible narrow transient "
            "spike. If this is still flat/marginal, the source-mismatch hypothesis is ruled out as "
            "the primary explanation and the issue is upstream (data quality or pipeline/config)."
        ),
    })


if __name__ == "__main__":
    main()
