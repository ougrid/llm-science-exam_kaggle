"""Day-1 closed-book baseline: DeBERTa-v3-base, no retrieved context.

Trains on data/train_pool.csv (the synthetic pool minus T1 and minus
near-duplicates of the gold 200), evaluates MAP@3 on T1 with a bootstrap CI,
and appends a row to experiments/log.csv. This number is the project's null
hypothesis -- the parametric-knowledge plateau every later retrieval gain
is measured against.
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
CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-closed-book-final"
BEST_CHECKPOINT_DIR = DATA / "checkpoints" / "deberta-v3-base-closed-book-best"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # effective batch 32, matching PLAN.md's cited community recipe
EVAL_BATCH_SIZE = 16
EPOCHS = 3
EVAL_EVERY_STEPS = 10  # optimizer steps, not micro-batches -- see note below
LR = 5e-6  # low end of PLAN.md's cited 4e-6-8e-7 range; uniform, see note below
SEED = 42
MAX_MS_PER_STEP = 350  # measured ~128 ms/microbatch for this config; see src/llmsci/gpu_guard.py


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
    if device.type == "cuda":
        # See src/llmsci/gpu_guard.py for why both of these exist: WSL2's CUDA
        # driver silently backs over-VRAM allocations with system RAM instead
        # of raising OOM, which reads as fine but runs 15-25x slower. This
        # run's peak usage (~7902/8151 MiB) already sits right at that edge.
        cap_memory_fraction(0.975)
        probe_training_speed(MODEL_NAME, BATCH_SIZE, 5, MAX_LENGTH, device, MAX_MS_PER_STEP)

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
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    assert_trainable_dtype(model)  # fp16 params silently cannot train; see mc.py

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

    # Evaluate every EVAL_EVERY_STEPS *optimizer* steps, not per epoch, and
    # keep the best checkpoint by T1 MAP@3. Once-per-epoch checkpointing
    # missed the real story entirely: a finer-grained diagnostic showed the
    # model reaching T1 MAP@3 ~0.51 after only ~38 optimizer steps (~25% of
    # one epoch), then collapsing back toward the 0.3667 random baseline by
    # 50% of that same epoch -- every prior run measured only after the
    # collapse. This is standard best-checkpoint-by-validation-score
    # selection within a single run's own trajectory (what
    # Trainer(load_best_model_at_end=True) does by default), not the
    # best-of-N-independent-configs cherry-picking PLAN.md warns against.
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
                        f"epoch {epoch + 1} optim_step {optim_step} T1 MAP@3 {mean:.4f} "
                        f"[{lower:.4f},{upper:.4f}] (baseline {baseline:.4f})"
                    )
                    if mean > best_mean:
                        best_mean, best_lower, best_upper = mean, lower, upper
                        best_optim_step = optim_step
                        BEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(BEST_CHECKPOINT_DIR)
                        tokenizer.save_pretrained(BEST_CHECKPOINT_DIR)
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        print(f"epoch {epoch + 1}/{EPOCHS} mean loss: {total_loss / len(train_loader):.4f}")
    train_seconds = time.time() - train_start
    print(f"training took {train_seconds:.1f}s")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    final_mean, final_lower, final_upper, eval_seconds = evaluate(model, tokenizer, t1_df, collator, device)
    print(f"FINAL (end-of-training) T1 MAP@3: {final_mean:.4f}  95% CI [{final_lower:.4f}, {final_upper:.4f}]")
    print(
        f"BEST (optim_step {best_optim_step}) T1 MAP@3: {best_mean:.4f}  "
        f"95% CI [{best_lower:.4f}, {best_upper:.4f}]  (random baseline {baseline:.4f})"
    )

    log_experiment(
        {
            "date": pd.Timestamp.now().isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "config": f"deberta-v3-base_closed-book_maxlen256_{EPOCHS}ep_lr5e-6_eps1e-6_accum4_BEST-CKPT",
            "tier": "T1",
            "n": len(t1_df),
            "map3_mean": round(best_mean, 4),
            "map3_ci_lower": round(best_lower, 4),
            "map3_ci_upper": round(best_upper, 4),
            "random_baseline": round(baseline, 4),
            "train_seconds": round(train_seconds, 1),
            "eval_seconds": round(eval_seconds, 1),
            "hypothesis": "closed-book plateau: no retrieved context, prompt+option only",
            "notes": (
                f"Best checkpoint at optim_step {best_optim_step} (of "
                f"{optim_step} total), selected by T1 MAP@3 during training, "
                f"not the final end-of-training state (which was "
                f"{final_mean:.4f} [{final_lower:.4f},{final_upper:.4f}] -- "
                "collapsed back toward baseline, consistent with every "
                "prior run tonight). See DEVLOG.md for the full story."
            ),
        }
    )


if __name__ == "__main__":
    main()
