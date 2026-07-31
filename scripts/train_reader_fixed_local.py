"""The first training run in this project with BOTH known faults fixed.

Every own-model number in this repo so far is invalid, for two independent reasons
that happened to share one signature (train_loss pinned at ln(5)=1.6094 with
perfectly healthy gradients, optimizer, labels, context and input format):

  1. fp16 PARAMETERS. transformers 5.x follows the checkpoint dtype and deberta-v3
     ships fp16, so AdamW was updating half-precision weights in place. An update
     of lr=2e-5 is ~1.3 ULP for a weight near 0.03 and exactly zero for any weight
     >= 0.1. Fixed by loading fp32 and using autocast for compute only.
  2. lr=2e-5 with freeze 18/24. Inherited from cdeotte part 2 (~60k rows, 435M
     model) and never measured on our 4,586 rows / 77.2M trainable parameters.

The second was found with a 2x2 (scripts/sweep_lr_freeze_local.py) on the full
4,586-row pool, and it is an INTERACTION -- exactly one corner of four learns:

    lr 2e-5, freeze 9/12  ->  1.6111 -> 1.6117   null
    lr 1e-4, freeze 9/12  ->  1.6116 -> 1.4372   LEARNS
    lr 2e-5, freeze 0/12  ->  1.6112 -> 1.5947   null
    lr 1e-4, freeze 0/12  ->  1.6113 -> 1.6129   null, and the WORST corner

The frozen lower layers are what make the higher LR usable. Both single-knob
changes fail. The recipe we inherited sat at the worst corner for three days.

WHY deberta-v3-BASE AND NOT -LARGE. Not a fallback, a measured choice. base is
where the winning corner is demonstrated; large has not yet been shown to train at
all under this compute budget (a 300-step confirmation is running separately). A
184M reader that trains beats a 435M one stuck at ln(5), and base fits the local
8 GB card at fp32, so this run costs no Kaggle quota. If large confirms, it gets
its own run; this one stands on its own either way.

WHAT TO READ. train_loss must fall clearly below 1.6094 -- on the full pool the
base winner is still flat at step 100 and breaks away between 125 and 150, so do
not judge before ~200. MAP@3 on T1 is reported with a bootstrap CI against the
0.3667 analytic baseline; the honest ceiling on this eval set with this retrieval
is 0.7970 (the known-good public reader), and the invalidated previous best was
0.6086.

Checkpoints every eval, so a crash or a closed lid costs one eval interval.

Run: python scripts/train_reader_fixed_local.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from llmsci.gpu_guard import cap_memory_fraction, probe_training_speed
from llmsci.metrics import average_precision_scores, bootstrap_ci, random_baseline_map_at_k
from llmsci.reader.mc import (
    DataCollatorForMultipleChoice,
    MultipleChoiceDataset,
    assert_trainable_dtype,
    logits_to_ranked_labels,
)

DATA = Path("data")
TRAIN_SRC2 = DATA / "train_pool_own_context_src2.parquet"
T1_CTX = DATA / "t1_dev_own_context_general_big.parquet"
OUT = DATA / "checkpoints" / "base-fixed-lr1e-4"

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 384          # matches the eval/submission format
MAX_CONTEXT_CHARS = 8_000
MICRO_BATCH = 2
GRAD_ACCUM = 8            # effective batch 16
EVAL_BATCH = 4            # fp32 weights + 8 GB card; autocast keeps compute fp16
LR = 1e-4                 # MEASURED (2x2 above), not inherited
N_FROZEN = 9              # of 12; the interaction partner of the LR above
FREEZE_EMBEDDINGS = True
EPOCHS = 8
EVAL_EVERY = 200          # base breaks away between 125 and 150, so 200 is the first honest read
EVAL_SUBSET = 500         # selection only; +-0.032 CI half-width vs +-0.018 on the full 1,500
SEED = 42
TIME_BUDGET_S = 100 * 60
RANDOM_LOSS = math.log(5)
BASELINE = random_baseline_map_at_k()


def evaluate(model, tok, df, collator, device, batch_size=EVAL_BATCH):
    model.eval()
    loader = DataLoader(
        MultipleChoiceDataset(df, tok, max_length=MAX_LENGTH, context_col="context"),
        batch_size=batch_size, shuffle=False, collate_fn=collator,
    )
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                         enabled=(device.type == "cuda")):
        for b in loader:
            b.pop("labels", None)
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu().numpy())
    logits = np.concatenate(out, axis=0)
    scores = average_precision_scores(df["answer"].tolist(), logits_to_ranked_labels(logits, k=3), k=3)
    model.train()
    return bootstrap_ci(scores, n_resamples=10_000, seed=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    # The source-2-only restriction (4,586 of 39,249 rows) came from hypothesis 4,
    # the train/eval generator mismatch -- which cell B of
    # diagnose_train_data_and_format.py falsified: a known-good reader scores
    # 0.8037 on the TRAIN file vs 0.7970 on the eval file, so the two are equally
    # readable and generator shift was never a mechanism. With that gone there is
    # no measured reason to discard 88% of the pool, so the file is now an argument.
    ap.add_argument("--train-file", default=str(TRAIN_SRC2))
    ap.add_argument("--out-suffix", default="", help="appended to the checkpoint dir name")
    ap.add_argument("--time-budget-min", type=float, default=TIME_BUDGET_S / 60)
    args = ap.parse_args()
    train_path = Path(args.train_file)
    out_dir = Path(str(OUT) + args.out_suffix)
    time_budget = args.time_budget_min * 60

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    torch.manual_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    # CLAUDE.md requires BOTH gpu_guard calls before loading the real model. The
    # first big-pool attempt called only cap_memory_fraction() and then died
    # silently at step 250 after degrading 2.17 -> 2.69 s/step, which is exactly
    # the class of failure the speed probe exists to catch up front.
    # Threshold from measurement, not guesswork: the 4,586-row run held
    # 2.16-2.31 s/step for 2,288 steps at this batch shape, so 4,000 ms is ~1.7x
    # the observed steady state -- loose enough not to false-trip on a cold cache,
    # tight enough to catch a PCIe-backed spill (which costs 15-25x, not 70%).
    if device.type == "cuda":
        probe_training_speed(
            MODEL_NAME, batch_size=MICRO_BATCH, num_choices=5, seq_len=MAX_LENGTH,
            device=device, max_ms_per_step=4000.0,
            # These three MUST mirror the real run below. A probe that optimises
            # all 184.4M params instead of the 22.2M trainable ones needs ~8x the
            # optimizer state and OOMs where production would not -- which is
            # exactly what happened on the first attempt at this run.
            n_frozen_layers=N_FROZEN, freeze_embeddings=FREEZE_EMBEDDINGS,
            autocast_dtype=torch.float16,
        )
    print(f"device {device} | {MODEL_NAME} | lr={args.lr:.0e} | freeze {N_FROZEN}/12 "
          f"| ln(5)={RANDOM_LOSS:.4f} | baseline={BASELINE:.4f}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32)
    assert_trainable_dtype(model)          # the fp16 guard; refuses sub-fp32 params
    model = model.to(device)
    if FREEZE_EMBEDDINGS:
        for p in model.deberta.embeddings.parameters():
            p.requires_grad = False
    for layer in model.deberta.encoder.layer[:N_FROZEN]:
        for p in layer.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M of {total/1e6:.1f}M ({100*trainable/total:.1f}%)")

    train_df = pd.read_parquet(train_path)
    train_df["context"] = train_df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1 = pd.read_parquet(T1_CTX)
    t1["context"] = t1["context"].str.slice(0, MAX_CONTEXT_CHARS)
    t1_sel = t1.sample(n=min(EVAL_SUBSET, len(t1)), random_state=SEED).reset_index(drop=True)
    print(f"train {len(train_df)} rows from {train_path.name} | T1 full {len(t1)} | T1 selection {len(t1_sel)}")

    collator = DataCollatorForMultipleChoice(tok)
    loader = DataLoader(
        MultipleChoiceDataset(train_df, tok, max_length=MAX_LENGTH, context_col="context"),
        batch_size=MICRO_BATCH, shuffle=True, collate_fn=collator,
        generator=torch.Generator().manual_seed(SEED),
    )
    steps_per_epoch = len(loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, eps=1e-6, weight_decay=0.01
    )
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    print(f"optim steps/epoch {steps_per_epoch}, total {total_steps}, "
          f"budget {time_budget/60:.0f} min", flush=True)

    best = (-1.0, 0.0, 0.0)
    best_step = -1
    step, t0, stopped, window = 0, time.time(), False, []
    first_rate = 1e9  # set at step 100; guards the watchdog before then
    model.train()

    for epoch in range(args.epochs):
        if stopped:
            break
        it = iter(loader)
        for micro in range(len(loader)):
            try:
                b = next(it)
            except StopIteration:
                break
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                loss = model(**b).loss
            if not torch.isfinite(loss):
                print(f"NON-FINITE loss at epoch {epoch+1} micro {micro} -- aborting", flush=True)
                stopped = True
                break
            scaler.scale(loss / GRAD_ACCUM).backward()
            window.append(loss.item())
            if (micro + 1) % GRAD_ACCUM:
                continue

            scaler.unscale_(opt)                      # unscale BEFORE clipping, or the
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # threshold is meaningless
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 25 == 0:
                m = sum(window) / len(window)
                el = time.time() - t0
                # The probe validates the start of a run; this catches degradation
                # DURING it. The first big-pool attempt slid 2.17 -> 2.69 s/step
                # before dying, and nothing would have flagged that.
                if step >= 100 and el / step > 2.0 * first_rate:
                    print(f"  ** STEP TIME DEGRADED: {el/step:.2f}s/step vs "
                          f"{first_rate:.2f}s/step early -- suspect a memory spill; "
                          f"stopping so the checkpoint is not lost", flush=True)
                    stopped = True
                    break
                if step == 100:
                    first_rate = el / step
                print(f"  ep{epoch+1} step {step}/{total_steps} train_loss {m:.4f} "
                      f"(ln5 {RANDOM_LOSS:.4f}) lr {sched.get_last_lr()[0]:.2e} "
                      f"[{el:.0f}s, {el/step:.2f}s/step, eta {(total_steps-step)*el/step/60:.0f}m]",
                      flush=True)
                window = []

            if step % EVAL_EVERY == 0:
                mean, lo, hi = evaluate(model, tok, t1_sel, collator, device)
                sig = "ABOVE baseline" if lo > BASELINE else "not resolved vs baseline"
                print(f"  >> step {step}: T1({len(t1_sel)}) MAP@3 {mean:.4f} [{lo:.4f},{hi:.4f}] "
                      f"base {BASELINE:.4f} -- {sig}", flush=True)
                if mean > best[0]:
                    best, best_step = (mean, lo, hi), step
                    model.save_pretrained(out_dir)
                    tok.save_pretrained(out_dir)
                    (out_dir / "progress.json").write_text(json.dumps({
                        "best_step": best_step, "best_map3_subset": best,
                        "lr": args.lr, "n_frozen": N_FROZEN, "elapsed_s": time.time() - t0,
                    }, indent=2))
                    print(f"     saved checkpoint (best so far) -> {out_dir}", flush=True)

            if time.time() - t0 > time_budget:
                print(f"TIME BUDGET reached at step {step} -- stopping cleanly", flush=True)
                stopped = True
                break

    train_s = time.time() - t0
    print(f"\ntraining stopped at step {step}/{total_steps} after {train_s/60:.1f} min")

    # Headline number: the BEST-by-subset checkpoint, rescored on the FULL 1,500.
    if best_step > 0:
        del model, opt, sched, scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = AutoModelForMultipleChoice.from_pretrained(out_dir, dtype=torch.float32).to(device)
        mean, lo, hi = evaluate(model, tok, t1, collator, device)
        print(f"BEST(step {best_step}) on FULL T1 ({len(t1)}): MAP@3 {mean:.4f} [{lo:.4f},{hi:.4f}]")
        print(f"  its {EVAL_SUBSET}-row selection score was {best[0]:.4f} "
              f"[{best[1]:.4f},{best[2]:.4f}]")
        print(f"\n  baseline (analytic)                 {BASELINE:.4f}")
        print(f"  previous best own-model (INVALID)   0.6086   <- fp16 + wrong LR")
        print(f"  known-good public reader (ceiling)  0.7970 [0.7687,0.8247]")
        verdict = ("clears the invalidated 0.6086" if lo > 0.6086 else
                   "above baseline but below the old invalid 0.6086" if lo > BASELINE else
                   "NOT resolved above baseline")
        print(f"  => {verdict}")
        (out_dir / "result_summary.json").write_text(json.dumps({
            "model": MODEL_NAME, "train_file": train_path.name, "train_rows": len(train_df),
            "lr": args.lr, "n_frozen": N_FROZEN,
            "max_length": MAX_LENGTH, "effective_batch": MICRO_BATCH * GRAD_ACCUM,
            "steps_done": step, "steps_planned": total_steps, "train_minutes": train_s / 60,
            "best_step": best_step, "map3_full_t1": [mean, lo, hi],
            "map3_selection_subset": list(best), "baseline": BASELINE,
            "ceiling_known_good_reader": 0.7970, "verdict": verdict,
        }, indent=2))
        print(f"wrote {out_dir/'result_summary.json'}")
    else:
        print("no eval reached -- nothing saved")


if __name__ == "__main__":
    main()
