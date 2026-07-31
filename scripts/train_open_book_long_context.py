"""Context-length sweep: does raising MAX_LENGTH recover more signal than
reranking did?

Motivated by two independent, quantified findings: (1) this project's own
full-corpus own-retrieval context is a median 681 tokens (91.7% exceed the
384-token budget used everywhere so far), and (2) the reference_reproduction
comparison track measured that on cdeotte's context, truncation costs ~9.1
MAP@3 points where reranking only recovers ~1.5 -- context length, not
ranking, looks like the bigger lever.

Trained and evaluated ENTIRELY LOCALLY, deliberately, after finding that
Kaggle's environment scores the identical checkpoint differently (and once
in the opposite direction) from local evaluation -- see DEVLOG.md's "the
reranked retrain landed" entry. Comparing this run's result against row 7
of reports/ablation_table.md (0.4297 [0.4099,0.4496], same data minus the
context-length change, also scored locally) is a clean, same-environment
comparison.

MAX_LENGTH=768 (not 1024): 1024 OOMs even at batch=1 on the local 8GB card
(see the gpu_guard probe run before this script was written); 768 fits at
batch=1 only, so effective batch 32 needs grad_accum=32.
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
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, logits_to_ranked_labels, assert_trainable_dtype

DATA = Path("data")
BEST_CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-open-book-long-context-best"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 768
MAX_CONTEXT_CHARS = 8_000
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 32  # effective batch 32; batch=1 is the max that fits len=768 locally
EVAL_BATCH_SIZE = 2
EPOCHS = 3
EVAL_EVERY_STEPS = 15
LR = 5e-6
SEED = 42
MAX_MS_PER_STEP = 900  # measured ~371 ms/microbatch for batch=1,len=768; see src/llmsci/gpu_guard.py


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
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    assert_trainable_dtype(model)  # fp16 params silently cannot train; see mc.py

    train_df = pd.read_parquet(DATA / "train_pool_own_context_full.parquet")
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_df = pd.read_parquet(DATA / "t1_dev_own_context_full.parquet")
    t1_df["context"] = t1_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    print(f"train: {len(train_df)} rows, T1+own-context: {len(t1_df)} rows, MAX_LENGTH={MAX_LENGTH}")

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
                        f"epoch {epoch + 1} optim_step {optim_step} T1+own-ctx MAP@3 {mean:.4f} "
                        f"[{lower:.4f},{upper:.4f}] (baseline {baseline:.4f})",
                        flush=True,
                    )
                    if mean > best_mean:
                        best_mean, best_lower, best_upper = mean, lower, upper
                        best_optim_step = optim_step
                        BEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(BEST_CHECKPOINT_DIR)
                        tokenizer.save_pretrained(BEST_CHECKPOINT_DIR)
            if step % 500 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        print(f"epoch {epoch + 1}/{EPOCHS} mean loss: {total_loss / len(train_loader):.4f}", flush=True)
    train_seconds = time.time() - train_start

    final_mean, final_lower, final_upper, eval_seconds = evaluate(model, tokenizer, t1_df, collator, device)
    print(f"FINAL T1+own-ctx MAP@3: {final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}]")
    print(f"BEST (optim_step {best_optim_step}) T1+own-ctx MAP@3: {best_mean:.4f} "
          f"[{best_lower:.4f},{best_upper:.4f}] (baseline {baseline:.4f})")
    print(f"training took {train_seconds:.1f}s")

    log_experiment({
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": f"deberta-v3-base_open-book_OWN-RETRIEVAL-FULL-CORPUS_LONG-CONTEXT_maxlen{MAX_LENGTH}_{EPOCHS}ep_lr{LR}_eps1e-6_n{len(train_df)}_LOCAL-ONLY",
        "tier": "T1+own-BM25-full-corpus-context",
        "n": len(t1_df),
        "map3_mean": round(best_mean, 4),
        "map3_ci_lower": round(best_lower, 4),
        "map3_ci_upper": round(best_upper, 4),
        "random_baseline": round(baseline, 4),
        "train_seconds": round(train_seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "hypothesis": "does raising MAX_LENGTH 384->768 recover more signal than phrase-match reranking did, per the reference_reproduction finding that truncation costs more than reranking recovers",
        "notes": (
            f"Trained AND evaluated entirely locally (no Kaggle) to avoid the Kaggle-vs-local "
            f"evaluation-environment discrepancy found in the reranking experiment -- directly "
            f"comparable to row 7 of reports/ablation_table.md (0.4297 [0.4099,0.4496], same "
            f"train_pool_own_context_full/t1_dev_own_context_full data, MAX_LENGTH=384, also "
            f"scored locally). Final end-of-training: {final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}]. "
            f"batch_size=1 (not 2) because len=768 OOMs at batch=2 on the local 8GB card even "
            f"with gpu_guard's memory cap active; grad_accum=32 to keep effective batch 32."
        ),
    })


if __name__ == "__main__":
    main()
