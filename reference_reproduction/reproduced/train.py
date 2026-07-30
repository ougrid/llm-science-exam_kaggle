"""Train cdeotte's open-book reader recipe and score it on our T1 dev set.

Reimplemented from ../NOTES.md. Uses a hand-rolled loop rather than HF
Trainer: the notebook's `TrainingArguments` keywords (`evaluation_strategy`,
`Trainer(tokenizer=)`) were renamed in transformers 5.x, and an explicit loop
makes the recipe auditable against the notes instead of hiding it behind
argument defaults that have themselves changed since 4.31.

Recipe held fixed from the notebook: deberta-v3-large, embeddings + first 18
of 24 layers frozen, MAX_INPUT=256, truncation='only_first', lr 2e-5, cosine
schedule, warmup_ratio 0.1, weight_decay 0.01 (excluding bias/LayerNorm, as
HF Trainer does), max_grad_norm 1.0, effective batch 16, 2 epochs, and
NO best-checkpoint selection (cdeotte sets load_best_model_at_end=False, so
the final weights are what gets scored).

Deviations, all documented in ../NOTES.md: bf16 instead of fp16 (Blackwell has
bf16; DeBERTa-v3's layer_norm_eps=1e-7 is a known fp16 NaN source), one GPU
instead of 2xT4 with accumulation raised to keep the effective batch at 16,
and the in-training eval set is a held-out slice of the training pool rather
than the sacred official 200.

Usage:
    python train.py --tag faithful   --context-col context
    python train.py --tag reranked   --context-col context_reranked
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from common import (
    ADAM_EPS,
    EPOCHS,
    GRAD_ACCUM_STEPS,
    LR,
    MAX_INPUT,
    MICRO_BATCH,
    MODEL_NAME,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    MCCollator,
    OpenBookMCDataset,
    build_model,
    logits_to_ranked_labels,
    predict_logits,
)
from llmsci.gpu_guard import assert_step_speed, cap_memory_fraction
from llmsci.metrics import (
    average_precision_scores,
    bootstrap_ci,
    random_baseline_map_at_k,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"

SEED = 42
EVAL_EVERY = 16  # optimizer steps; monitoring only, nothing is selected on it
# Measured on this box by bench_step.py on 2026-07-30: 190 ms/micro-step for
# batch=1 x 5 choices x 256 tokens, frozen deberta-v3-large, bf16 autocast,
# peak 5.40 GB / 8.55 GB. Guard set at ~2.5x to catch the WSL2 shared-memory
# fallback (15-25x slower) without tripping on normal variance.
MAX_MS_PER_STEP = 475


def decayed_and_undecayed(model):
    """HF Trainer's grouping: no weight decay on bias or LayerNorm."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def score(df, logits, seed: int = 0):
    y_pred = logits_to_ranked_labels(logits, k=3)
    ap = average_precision_scores(df["answer"].tolist(), y_pred, k=3)
    mean, lo, hi = bootstrap_ci(ap, n_resamples=10_000, seed=seed)
    return ap, mean, lo, hi


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tag", required=True, help="run label, e.g. faithful / reranked")
    ap_.add_argument("--context-col", default="context")
    ap_.add_argument("--save-model", action="store_true",
                     help="save final weights (needed to re-score under part-2's inference format)")
    ap_.add_argument("--train-file", default="train_1024.parquet")
    ap_.add_argument("--epochs", type=int, default=EPOCHS)
    ap_.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    args = ap_.parse_args()
    epochs = args.epochs
    eval_every = args.eval_every

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  tag: {args.tag}  context column: {args.context_col}")
    if device.type == "cuda":
        cap_memory_fraction(0.975)

    train_df = pd.read_parquet(DATA / args.train_file)
    monitor_df = pd.read_parquet(DATA / "monitor_500.parquet")
    t1_df = pd.read_parquet(DATA / "t1_eval_cdeotte_ctx.parquet")
    for name, d in [("train", train_df), ("monitor", monitor_df), ("t1", t1_df)]:
        if args.context_col not in d.columns:
            raise KeyError(f"{name} frame has no column {args.context_col!r}; run rerank.py first")
        d[args.context_col] = d[args.context_col].fillna("").astype(str)
    print(f"train {len(train_df)} | monitor {len(monitor_df)} | T1 {len(t1_df)}")

    # Leakage guard restated at training time, not just at prep time.
    assert not set(train_df["prompt"]) & set(t1_df["prompt"]), "T1 prompt in training set"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(device)
    model.train()

    decay, no_decay = decayed_and_undecayed(model)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
        eps=ADAM_EPS,
    )

    # WSL2 guard on the real model/optimizer at the production batch shape.
    ids = torch.randint(0, 1000, (MICRO_BATCH, 5, MAX_INPUT), device=device)
    mask, tt = torch.ones_like(ids), torch.zeros_like(ids)
    lab = torch.randint(0, 5, (MICRO_BATCH,), device=device)

    def probe():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=mask, token_type_ids=tt, labels=lab)
        (out.loss / GRAD_ACCUM_STEPS).backward()

    if device.type == "cuda":
        assert_step_speed(probe, MAX_MS_PER_STEP,
                          label=f"micro-step (bs={MICRO_BATCH} x5 x{MAX_INPUT})")
    optimizer.zero_grad(set_to_none=True)
    del ids, mask, tt, lab

    collator = MCCollator(tokenizer)
    train_ds = OpenBookMCDataset(train_df, tokenizer, MAX_INPUT, args.context_col)
    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True,
                              collate_fn=collator, drop_last=True)

    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )
    print(f"optimizer steps: {steps_per_epoch}/epoch, {total_steps} total "
          f"(effective batch {MICRO_BATCH * GRAD_ACCUM_STEPS})")

    baseline = random_baseline_map_at_k()
    curve = []
    optim_step = 0
    t_start = time.time()

    for epoch in range(epochs):
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch+1} micro-step {i}")
            (loss / GRAD_ACCUM_STEPS).backward()
            running += loss.item()
            seen += 1
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1
                if optim_step % eval_every == 0 or optim_step == total_steps:
                    lg = predict_logits(model, tokenizer, monitor_df, device, batch_size=4,
                                        max_length=MAX_INPUT, context_col=args.context_col,
                                        label="monitor")
                    _, m, lo, hi = score(monitor_df, lg)
                    el = time.time() - t_start
                    eta = el / optim_step * (total_steps - optim_step)
                    print(f"[step {optim_step}/{total_steps}] loss {running/max(seen,1):.4f} "
                          f"monitor MAP@3 {m:.4f} [{lo:.4f},{hi:.4f}] "
                          f"elapsed {el/60:.1f}m eta {eta/60:.1f}m", flush=True)
                    curve.append({"step": optim_step, "loss": running / max(seen, 1),
                                  "monitor_map3": m})
                    running, seen = 0.0, 0
        print(f"epoch {epoch+1}/{epochs} done ({(time.time()-t_start)/60:.1f}m)")

    train_seconds = time.time() - t_start

    # Final model (no checkpoint selection, matching load_best_model_at_end=False)
    print("\nscoring final model on T1 dev (1500 rows)...")
    t0 = time.time()
    t1_logits = predict_logits(model, tokenizer, t1_df, device, batch_size=4,
                               max_length=MAX_INPUT, context_col=args.context_col, label="T1")
    eval_seconds = time.time() - t0
    ap, mean, lo, hi = score(t1_df, t1_logits)

    mon_logits = predict_logits(model, tokenizer, monitor_df, device, batch_size=4,
                                max_length=MAX_INPUT, context_col=args.context_col,
                                label="monitor-final")
    _, mm, mlo, mhi = score(monitor_df, mon_logits)

    print(f"\n=== {args.tag} ===")
    print(f"T1 dev MAP@3 (n={len(t1_df)}): {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"monitor MAP@3 (n={len(monitor_df)}): {mm:.4f}  95% CI [{mlo:.4f}, {mhi:.4f}]")
    print(f"random baseline: {baseline:.4f}")
    print(f"train {train_seconds/60:.1f} min, T1 inference {eval_seconds/60:.1f} min")
    print(f"peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if args.save_model:
        ckpt = HERE.parent / "models" / f"reproduced-{args.tag}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"saved checkpoint to {ckpt}")

    np.save(RESULTS / f"t1_logits_{args.tag}.npy", t1_logits)
    np.save(RESULTS / f"t1_ap_{args.tag}.npy", ap)
    with open(RESULTS / f"summary_{args.tag}.json", "w") as f:
        json.dump({
            "tag": args.tag,
            "context_col": args.context_col,
            "model": MODEL_NAME,
            "n_train": len(train_df),
            "n_t1": len(t1_df),
            "max_input": MAX_INPUT,
            "epochs": epochs,
            "lr": LR,
            "effective_batch": MICRO_BATCH * GRAD_ACCUM_STEPS,
            "t1_map3": mean, "t1_ci_lower": lo, "t1_ci_upper": hi,
            "monitor_map3": mm, "monitor_ci_lower": mlo, "monitor_ci_upper": mhi,
            "random_baseline": baseline,
            "train_seconds": round(train_seconds, 1),
            "eval_seconds": round(eval_seconds, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "curve": curve,
        }, f, indent=2)
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
