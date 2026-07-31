"""Day-2 open-book baseline: DeBERTa-v3-base with retrieved context.

Trains on data/train_pool_context.parquet -- cdeotte's `60k-data-with-context-v2`
pre-retrieved-context dataset, filtered to remove any row that exact-matches a
T1 row (see scripts/build_context_train_pool.py) -- and evaluates on
data/t1_dev_context.parquet, which carries context from that *same* retriever
for the same 1,500 T1 questions. Using cdeotte's context on both sides keeps
train/test context from the same retriever, per CLAUDE.md's non-negotiable
rule; it is a placeholder until src/llmsci/retrieve/ exists, at which point T1
should be re-evaluated with our own retriever's context instead.

Reuses the closed-book run's hard-won hyperparameters (lr=5e-6, eps=1e-6,
gradient accumulation to effective batch 32) and its best-checkpoint-by-
validation-score fix from the start, rather than rediscovering both lessons
against a second model.
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
CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-open-book-final"
BEST_CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-open-book-best"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 384
MAX_CONTEXT_CHARS = 8_000  # pre-tokenizer clip; truncation="only_first" makes this a speed guard only
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 16  # effective batch 32, matching the closed-book recipe
EVAL_BATCH_SIZE = 8
EPOCHS = 2
EVAL_EVERY_STEPS = 50
LR = 5e-6
SEED = 42
N_TRAIN_SUBSET = 15_000  # pilot run on a random subset; see DEVLOG.md for why
MAX_MS_PER_STEP = 800  # measured ~310 ms/microbatch for this config; see src/llmsci/gpu_guard.py


def evaluate(model, tokenizer, df, collator, device) -> tuple[float, float, float, float]:
    """Return (map3_mean, ci_lower, ci_upper, seconds) on `df`."""
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
    if device.type == "cuda":
        # See src/llmsci/gpu_guard.py for why both of these exist. WSL2's CUDA
        # driver silently backs allocations past the physical VRAM limit with
        # system RAM ("shared GPU memory"), which is 15-25x slower than GDDR,
        # not a hard OOM -- a batch=4/max_length=384 config here once ran 5
        # hours for 200/936 optimizer steps before this was caught, because
        # nothing errored. The memory cap converts that into an immediate OOM;
        # the speed probe catches any other silent slowdown the cap wouldn't
        # (thermal throttling, CPU contention, a driver hiccup).
        cap_memory_fraction(0.975)
        probe_training_speed(MODEL_NAME, BATCH_SIZE, 5, MAX_LENGTH, device, MAX_MS_PER_STEP)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    assert_trainable_dtype(model)  # fp16 params silently cannot train; see mc.py

    train_df = pd.read_parquet(DATA / "train_pool_context.parquet")
    train_df = train_df.sample(n=N_TRAIN_SUBSET, random_state=SEED).reset_index(drop=True)
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_df = pd.read_parquet(DATA / "t1_dev_context.parquet")
    t1_df["context"] = t1_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    print(f"train subset: {len(train_df)} rows (of {N_TRAIN_SUBSET} requested), T1+context: {len(t1_df)} rows")

    collator = DataCollatorForMultipleChoice(tokenizer)
    train_ds = MultipleChoiceDataset(train_df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-6)
    num_optim_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_optim_steps), num_training_steps=num_optim_steps
    )

    best_mean, best_lower, best_upper = -1.0, None, None
    best_optim_step = -1
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
            if step % 200 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        print(f"epoch {epoch + 1}/{EPOCHS} mean loss: {total_loss / len(train_loader):.4f}")
    train_seconds = time.time() - train_start
    print(f"training took {train_seconds:.1f}s")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    final_mean, final_lower, final_upper, eval_seconds = evaluate(model, tokenizer, t1_df, collator, device)
    print(f"FINAL (end-of-training) T1+ctx MAP@3: {final_mean:.4f}  95% CI [{final_lower:.4f}, {final_upper:.4f}]")
    print(
        f"BEST (optim_step {best_optim_step}) T1+ctx MAP@3: {best_mean:.4f}  "
        f"95% CI [{best_lower:.4f}, {best_upper:.4f}]  (random baseline {baseline:.4f})"
    )

    log_experiment(
        {
            "date": pd.Timestamp.now().isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "config": (
                f"deberta-v3-base_open-book-cdeotte-context_maxlen{MAX_LENGTH}_"
                f"{EPOCHS}ep_lr{LR}_eps1e-6_accum{GRAD_ACCUM_STEPS}_n{N_TRAIN_SUBSET}_BEST-CKPT"
            ),
            "tier": "T1+cdeotte-context",
            "n": len(t1_df),
            "map3_mean": round(best_mean, 4),
            "map3_ci_lower": round(best_lower, 4),
            "map3_ci_upper": round(best_upper, 4),
            "random_baseline": round(baseline, 4),
            "train_seconds": round(train_seconds, 1),
            "eval_seconds": round(eval_seconds, 1),
            "hypothesis": "the jump: does retrieved context (from a public pre-retrieval pipeline, not yet ours) lift MAP@3 over the closed-book plateau",
            "notes": (
                f"Pilot run on a random {N_TRAIN_SUBSET}-row subset of the 52,923-row "
                "cdeotte-context train pool (not the full set yet -- fast-iteration pilot "
                "first, matching the closed-book debugging lesson to eval frequently before "
                "trusting an epoch-level number). Best checkpoint at optim_step "
                f"{best_optim_step} (of {optim_step} total). Final end-of-training state: "
                f"{final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}]. Context is from "
                "cdeotte's public 60k-data-with-context-v2 pipeline, not our own retriever "
                "(src/llmsci/retrieve/ doesn't exist yet) -- train and eval both use that "
                "same source, satisfying the train/test-retriever-match rule, but this "
                "number will need re-measuring once our own retrieval pipeline exists, since "
                "our retriever's recall/precision profile will likely differ from cdeotte's."
            ),
        }
    )


if __name__ == "__main__":
    main()
