"""Is the retrieved context HELPING the reader learn, or swamping the signal?

The question this exists to answer, and why it is the right one to ask next.

A known-good reader scores 0.7947 on our eval set using OUR exact training input
format and OUR context. So the input format carries the signal and the task is
learnable from it. Yet training from the base checkpoint on 4,586 correctly
labelled rows moves nothing: train_loss hovers at ln(5)=1.6094 in both fp16 and
fp32, on both a T4 and locally, at two learning rates and two freezing depths.

Everything on the data side has been cleared by measurement (labels, alignment,
context relevance, input format -- see scripts/diagnose_train_data_and_format.py)
and everything on the optimizer side too (grad norms, weight deltas, dtype). What
has NEVER been tested is whether the context, as a training input, makes the task
harder to LEARN even though it makes the task possible to SOLVE.

The mechanism that would do that: at maxlen 384 the model sees ~340 tokens of
context against ~50 tokens of prompt+option. Answer-support recall is 0.62, so
roughly 38% of rows contain no supporting evidence at all -- for those, the
correct label is unpredictable from the input, and the gradient they contribute is
noise pointing in an arbitrary direction. A fresh classification head has to find
a faint signal underneath that. A reader already fine-tuned on 60k rows does not:
it arrives knowing how to read context and simply ignores the useless ones.

If that is what is happening, LESS context should train BETTER early on, which is
the opposite of the usual assumption and the reason this is worth measuring rather
than reasoning about.

Arms, all else held identical:

  * no context (closed book). If this learns while open-book does not, the
    context is actively obstructing learning -- decisive.
  * 500 chars (~120 tokens). Signal-to-noise strongly favours prompt+option.
  * 2000 chars (~480 tokens, truncated to fit). Middle.
  * 8000 chars -- what the kernel currently does.

Declared before running, so the result cannot be reinterpreted afterwards:

  * Closed book learning while 8000 stays flat  => context is the obstruction.
    Fix the training curriculum (shorter context, or filter rows with no answer
    support), not the LR.
  * ALL arms flat => the fault is not the context either. At that point the
    remaining suspect is data SCALE (4,586 rows to bootstrap open-book reading
    from a fresh head, against the 60k the public checkpoints used), and the
    honest move is to say the recipe needs the bigger pool rather than keep
    testing variants. Do not spend Kaggle quota on another context variant.
  * A short arm winning => report it as a curriculum finding, and confirm on
    deberta-v3-large before spending quota.

deberta-v3-base and 1,024 rows for speed: all four arms in ~6 minutes locally, no
Kaggle quota. base is not large, so this is DIRECTION only.

Run: python scripts/sweep_context_budget_local.py [--lr 2e-5]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
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
MAX_LENGTH = 256
MICRO_BATCH = 2
GRAD_ACCUM = 8
N_ROWS = 1_024
STEPS = 150
N_FROZEN = 9  # 9/12 on base ~= the kernel's 18/24 on large
SEED = 42
RANDOM_LOSS = math.log(5)

ARMS = [
    ("closed book (no context)", 0),
    ("context 500 chars", 500),
    ("context 2000 chars", 2000),
    ("context 8000 chars (the kernel)", 8000),
]


def run_arm(label: str, chars: int, base_df, tok, device, lr: float) -> list[float]:
    torch.manual_seed(SEED)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32)
    assert_trainable_dtype(model)
    model = model.to(device)
    for p in model.deberta.embeddings.parameters():
        p.requires_grad = False
    for layer in model.deberta.encoder.layer[:N_FROZEN]:
        for p in layer.parameters():
            p.requires_grad = False

    df = base_df.copy()
    if chars == 0:
        context_col = None  # MultipleChoiceDataset passes context="" for closed book
    else:
        df["context"] = df["context"].str.slice(0, chars)
        context_col = "context"

    collator = DataCollatorForMultipleChoice(tok)
    loader = DataLoader(
        MultipleChoiceDataset(df, tok, max_length=MAX_LENGTH, context_col=context_col),
        batch_size=MICRO_BATCH, shuffle=True, collate_fn=collator,
        generator=torch.Generator().manual_seed(SEED),
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, eps=1e-6, weight_decay=0.01
    )
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * STEPS), STEPS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # How many context tokens does this arm actually show the model?
    probe = MultipleChoiceDataset(df.head(50), tok, max_length=MAX_LENGTH, context_col=context_col)
    kept = []
    for i in range(len(probe)):
        tt = np.asarray(probe[i]["token_type_ids"][0])
        kept.append(int((tt == 0).sum()) - 2)
    print(f"\n  {label}   (~{np.mean(kept):.0f} ctx tokens of {MAX_LENGTH})", flush=True)

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
            flag = "  <- below ln(5)" if m < RANDOM_LOSS - 0.02 else ""
            print(f"    step {step:3d}/{STEPS}  train_loss {m:.4f}{flag}  [{time.time()-t0:.0f}s]",
                  flush=True)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rows", type=int, default=N_ROWS)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)
    print(f"device {device} | {MODEL_NAME} | lr={args.lr:.0e} | freeze {N_FROZEN}/12 | "
          f"{args.rows} rows | ln(5)={RANDOM_LOSS:.4f}")
    print("pass criterion (declared up front): train_loss clearly below ln(5) and still falling")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_parquet(TRAIN_SRC2).head(args.rows).reset_index(drop=True)

    results = {label: run_arm(label, chars, df, tok, device, args.lr) for label, chars in ARMS}

    print("\n" + "=" * 74)
    print(f"{'arm':<36} {'first':>8} {'last':>8} {'delta':>8}  verdict")
    for label, losses in results.items():
        if not losses:
            continue
        v = "LEARNS" if losses[-1] < RANDOM_LOSS - 0.05 else "null (at ln5)"
        print(f"{label:<36} {losses[0]:8.4f} {losses[-1]:8.4f} {losses[-1]-losses[0]:+8.4f}  {v}")
    print("=" * 74)

    learns = {k: v[-1] for k, v in results.items() if v and v[-1] < RANDOM_LOSS - 0.05}
    closed = results.get("closed book (no context)") or [9.9]
    full = results.get("context 8000 chars (the kernel)") or [9.9]
    if not learns:
        print("ALL ARMS NULL -- the context is not the obstruction either. Remaining suspect")
        print("is DATA SCALE: 4,586 rows to bootstrap open-book reading from a fresh head,")
        print("against the ~60k the public checkpoints trained on. Say that plainly rather")
        print("than testing a fifth variant, and do NOT spend Kaggle quota on one.")
    elif closed[-1] < RANDOM_LOSS - 0.05 and full[-1] >= RANDOM_LOSS - 0.05:
        print("CLOSED BOOK LEARNS, FULL CONTEXT DOES NOT: the context obstructs learning.")
        print("Fix the curriculum -- shorter context, and/or drop rows with no answer")
        print("support (~38% of them at recall 0.62) -- not the learning rate.")
    else:
        print("arms that learn:", *learns, sep="\n  ")
        print("\nConfirm on deberta-v3-large before spending quota.")


if __name__ == "__main__":
    main()
