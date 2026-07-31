"""Is the recipe's LR or its layer freezing the binding constraint?

Context. The fp16-parameter bug (DEVLOG, "the fifth hypothesis") was real and is
fixed: verified fp32 weights, and a 16-row overfit that goes 1.5687 -> 0.1013 on
that single variable. It did NOT fix the real run. With fp32 confirmed in the
Kaggle log, train_loss still sits at 1.617-1.627 against ln(5)=1.6094 after 400
optimizer steps, and every MAP@3 CI still contains the 0.3667 baseline.

So the fp16 fix was necessary, not sufficient, and the 16-row overfit proved less
than I read into it: memorising 16 rows at maxlen 128 mostly exercises the
randomly-initialised head. It established that updates survive rounding. It did
not establish that this recipe can LEARN the task.

What differs between the test that worked and the run that doesn't: 16 rows vs
4,586, maxlen 128 vs 384, no warmup vs 103 warmup steps, and -- structurally --
a fresh head at lr=2e-5 on top of an encoder with 18 of 24 layers frozen.

This sweep isolates the two structural suspects instead of guessing between them.
Deliberately NOT another single-hypothesis test: it is a 2x2 that can come back
"neither", which is the property my first five hypotheses all lacked.

  * LR: 2e-5 (cdeotte part 2's value, which we inherited) vs 1e-4. A randomly
    initialised classification head on a mostly-frozen encoder is the textbook
    case for a higher LR, and we never measured it -- we adopted 2e-5 from a
    recipe that used it on 60k rows with a different freezing depth.
  * FREEZING: proportional freezing vs none. If the frozen bottom cannot adapt
    its features to retrieved-context input, no LR fixes it.

Run on deberta-v3-BASE, not large: 12 layers fits the local 8 GB card at fp32 and
runs ~6x faster, so all four arms finish in minutes rather than burning Kaggle
quota. That is a real limitation -- base is not large -- so this sweep is for
DIRECTION only. A positive arm gets confirmed on large before any quota is spent.

Pass criterion, declared before running: train_loss must fall clearly below
ln(5)=1.6094 and keep falling. Anything that merely hovers is a null, however
much the MAP@3 wiggles -- per CLAUDE.md, train_loss is the tell, and a model with
this many trainable parameters should be able to overfit 1,024 rows.

Run: python scripts/sweep_lr_freeze_local.py
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from llmsci.gpu_guard import cap_memory_fraction
from llmsci.reader.mc import (
    DataCollatorForMultipleChoice,
    MultipleChoiceDataset,
    assert_trainable_dtype,
)

DATA = Path("data")
TRAIN_SRC2 = DATA / "train_pool_own_context_src2.parquet"

MODEL_NAME = "microsoft/deberta-v3-base"
N_LAYERS = 12
MAX_LENGTH = 256
MAX_CONTEXT_CHARS = 8_000
MICRO_BATCH = 2
GRAD_ACCUM = 8  # effective 16, matching the kernel
N_ROWS = 1_024
STEPS = 150
SEED = 42
RANDOM_LOSS = math.log(5)

# (label, lr, n_frozen_layers). 18/24 on large is 75%, so 9/12 on base matches.
ARMS = [
    ("lr2e-5 freeze9/12  (the kernel's recipe)", 2e-5, 9),
    ("lr1e-4 freeze9/12", 1e-4, 9),
    ("lr2e-5 freeze0/12", 2e-5, 0),
    ("lr1e-4 freeze0/12", 1e-4, 0),
]


def run_arm(label: str, lr: float, n_frozen: int, df, tok, device) -> list[float]:
    torch.manual_seed(SEED)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32)
    assert_trainable_dtype(model)
    model = model.to(device)
    if n_frozen:
        for p in model.deberta.embeddings.parameters():
            p.requires_grad = False
        for layer in model.deberta.encoder.layer[:n_frozen]:
            for p in layer.parameters():
                p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    collator = DataCollatorForMultipleChoice(tok)
    loader = DataLoader(
        MultipleChoiceDataset(df, tok, max_length=MAX_LENGTH, context_col="context"),
        batch_size=MICRO_BATCH, shuffle=True, collate_fn=collator,
        generator=torch.Generator().manual_seed(SEED),
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, eps=1e-6, weight_decay=0.01
    )
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * STEPS), STEPS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"\n  {label}   ({trainable/1e6:.1f}M trainable)", flush=True)
    model.train()
    losses, window, step, t0 = [], [], 0, time.time()
    it = iter(loader)
    while step < STEPS:
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                loss = model(**b).loss
            scaler.scale(loss / GRAD_ACCUM).backward()
            window.append(loss.item())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 25 == 0:
            m = sum(window) / len(window)
            losses.append(m)
            window = []
            flag = "  <- below ln(5)" if m < RANDOM_LOSS else ""
            print(f"    step {step:3d}/{STEPS}  train_loss {m:.4f}{flag}  [{time.time()-t0:.0f}s]",
                  flush=True)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=N_ROWS)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    print(f"device {device}  |  {MODEL_NAME}  |  ln(5)={RANDOM_LOSS:.4f}")
    print(f"{args.rows} source-matched rows, maxlen {MAX_LENGTH}, "
          f"micro-batch {MICRO_BATCH} x accum {GRAD_ACCUM}, {STEPS} optim steps")
    print("pass criterion (declared up front): train_loss clearly below ln(5) and still falling")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_parquet(TRAIN_SRC2).head(args.rows).reset_index(drop=True)
    df["context"] = df["context"].str.slice(0, MAX_CONTEXT_CHARS)

    results = {}
    for label, lr, nf in ARMS:
        results[label] = run_arm(label, lr, nf, df, tok, device)

    print("\n" + "=" * 78)
    print(f"{'arm':<42} {'first':>8} {'last':>8} {'delta':>8}  verdict")
    for label, losses in results.items():
        if not losses:
            continue
        d = losses[-1] - losses[0]
        v = "LEARNS" if losses[-1] < RANDOM_LOSS - 0.05 else "null (at ln5)"
        print(f"{label:<42} {losses[0]:8.4f} {losses[-1]:8.4f} {d:+8.4f}  {v}")
    print("=" * 78)
    winners = [k for k, v in results.items() if v and v[-1] < RANDOM_LOSS - 0.05]
    if not winners:
        print("ALL ARMS NULL. Neither LR nor freezing depth is the binding constraint, so")
        print("the fault is upstream of both -- in what the model is being shown or asked.")
        print("Do NOT spend Kaggle quota on another LR/freezing variant.")
    else:
        print("arms that learn:", *winners, sep="\n  ")
        print("\nConfirm on deberta-v3-large before spending quota: base is not large,")
        print("and this sweep is for direction only.")


if __name__ == "__main__":
    main()
