"""Day-1 closed-book baseline: DeBERTa-v3-base, no retrieved context.

Trains on data/train_pool.csv (the synthetic pool minus T1 and minus
near-duplicates of the gold 200), evaluates MAP@3 on T1 with a bootstrap CI,
and appends a row to experiments/log.csv. This number is the project's null
hypothesis -- the parametric-knowledge plateau every later retrieval gain
is measured against.
"""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, logits_to_ranked_labels

DATA = Path("data")
CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-closed-book"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # effective batch 32, matching PLAN.md's cited community recipe
EVAL_BATCH_SIZE = 16
EPOCHS = 8  # generous on purpose: eval every epoch to see whether this is undertrained
LR = 5e-6  # low end of PLAN.md's cited 4e-6-8e-7 range; uniform, see note below
SEED = 42


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


LOG_FIELDS = [
    "date",
    "git_sha",
    "config",
    "tier",
    "n",
    "map3_mean",
    "map3_ci_lower",
    "map3_ci_upper",
    "random_baseline",
    "train_seconds",
    "eval_seconds",
    "hypothesis",
    "notes",
]


def log_experiment(row: dict) -> None:
    log_path = Path("experiments/log.csv")
    log_path.parent.mkdir(exist_ok=True)
    is_new = not log_path.exists()
    row = {**{k: "" for k in LOG_FIELDS}, **row}  # ensure every field is present
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def evaluate(model, tokenizer, df, collator, device) -> tuple[float, float, float, float]:
    """Return (map3_mean, ci_lower, ci_upper, seconds) on `df`."""
    model.eval()
    ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH)
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
    y_true = df["answer"].tolist()
    y_pred = logits_to_ranked_labels(logits, k=3)
    scores = average_precision_scores(y_true, y_pred, k=3)
    mean, lower, upper = bootstrap_ci(scores, n_resamples=10_000, seed=0)
    model.train()
    return mean, lower, upper, seconds


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Plain fp32, no autocast. Two things were tried and both broke:
    # (1) casting the whole model to bf16 and optimizing it directly made
    #     AdamW's updates round to zero against bf16's ~3-digit mantissa at
    #     this LR -- loss never moved off ln(5) across 2 full epochs.
    # (2) fp32 weights + bf16 autocast hit DeBERTa-v3's documented
    #     layer_norm_eps=1e-7 NaN instability (PLAN.md flagged this for fp16
    #     on T4; it turns out bf16 autocast triggers the same failure
    #     locally) -- every tensor in the saved checkpoint was NaN by the
    #     end of epoch 1. At 184M params this model has no real memory
    #     pressure on 8GB VRAM, so reduced precision was buying nothing
    #     here -- it was premature optimization that cost two broken runs.
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME).to(device)

    train_df = pd.read_csv(DATA / "train_pool.csv")
    t1_df = pd.read_csv(DATA / "t1_dev.csv")
    print(f"train_pool: {len(train_df)} rows, T1 dev: {len(t1_df)} rows")

    collator = DataCollatorForMultipleChoice(tokenizer)
    train_ds = MultipleChoiceDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    # eps=1e-6, not PyTorch AdamW's default 1e-8: DeBERTa-v3 reliably NaNs
    # within the first 1-2 optimizer steps at the default eps, reproduced
    # even on a single repeated synthetic batch with a perfectly ordinary
    # pre-clip grad norm (~3.7) -- this is Adam's well-documented first-step
    # instability (near-zero bias-corrected second moment on step 1 makes
    # the effective update size behave unpredictably for small-gradient
    # parameters), and eps=1e-6 is Microsoft's own documented DeBERTa fix
    # for exactly this. Confirmed fp32 vs bf16-autocast made no difference
    # -- this was never a precision bug.
    #
    # Discriminative LR (encoder 1e-5, head 1e-4) was tried and made things
    # *worse*: a small isolated diagnostic (same 128 real rows, no warmup)
    # showed loss climbing monotonically at every head_lr from 1e-5 to 5e-5,
    # reaching 40+ at 5e-5 -- real divergence, not just noise. Dropped back
    # to a single uniform LR at the low end of PLAN.md's own cited
    # community-converged range (4e-6 to 8e-7) for this exact model/task,
    # plus gradient accumulation to reach their effective batch of 32.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-6)
    num_optim_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_optim_steps), num_training_steps=num_optim_steps
    )

    model.train()
    train_start = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss ({loss.item()}) at epoch {epoch + 1} step {step} -- "
                    "stopping now instead of training to completion on corrupted weights"
                )
            (loss / GRAD_ACCUM_STEPS).backward()
            total_loss += loss.item()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        print(f"epoch {epoch + 1}/{EPOCHS} mean loss: {total_loss / len(train_loader):.4f}")

        mean, lower, upper, _ = evaluate(model, tokenizer, t1_df, collator, device)
        print(f"  -> T1 MAP@3 after epoch {epoch + 1}: {mean:.4f}  95% CI [{lower:.4f}, {upper:.4f}]")
    train_seconds = time.time() - train_start
    print(f"training took {train_seconds:.1f}s")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    # Final report uses the last epoch's number -- no picking the best epoch
    # after the fact, that's exactly the best-of-N optimism PLAN.md warns
    # about even at N as small as 8.
    mean, lower, upper, eval_seconds = evaluate(model, tokenizer, t1_df, collator, device)
    baseline = random_baseline_map_at_k()
    print(f"FINAL T1 MAP@3: {mean:.4f}  95% CI [{lower:.4f}, {upper:.4f}]  (random baseline {baseline:.4f})")

    log_experiment(
        {
            "date": pd.Timestamp.now().isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "config": f"deberta-v3-base_closed-book_maxlen256_{EPOCHS}ep_lr5e-6_eps1e-6_accum4",
            "tier": "T1",
            "n": len(t1_df),
            "map3_mean": round(mean, 4),
            "map3_ci_lower": round(lower, 4),
            "map3_ci_upper": round(upper, 4),
            "random_baseline": round(baseline, 4),
            "train_seconds": round(train_seconds, 1),
            "eval_seconds": round(eval_seconds, 1),
            "hypothesis": "closed-book plateau: no retrieved context, prompt+option only",
        }
    )


if __name__ == "__main__":
    main()
