"""Fast local confirmation that the corrected LR actually makes the loss fall.

The two corrected Kaggle kernels cannot be inspected mid-run (`kaggle kernels
output` only serves files from finished runs), so if lr=2e-5 were also wrong
we would not learn that until ~75 minutes of the remaining window had already
been spent. This runs the same recipe locally at production-equivalent
effective batch (32, via gradient accumulation -- unlike
scripts/diagnose_overfit_sanity.py, which deliberately used raw batch=2) for a
few hundred optimizer steps, purely to watch the loss.

Success criterion, decided before running: mean train loss must fall clearly
and durably below 1.55, against the ln(5)=1.6094 uniform-prediction floor that
every previous run in this project sat at. No MAP@3 is computed here -- the
loss floor is the thing being tested, and evaluating would only cost time.

Prints every few steps and flushes, so a partial result survives if the
machine loses power mid-run (nothing here needs a checkpoint: it is a
diagnostic, not a training run whose weights we intend to keep).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from llmsci.gpu_guard import cap_memory_fraction
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, assert_trainable_dtype

DATA = Path("data")
MODEL_NAME = "microsoft/deberta-v3-base"
LR = 2e-5
MAX_LENGTH = 384
BATCH_SIZE = 2          # 8 GB local card; Kaggle's T4 run uses 4
GRAD_ACCUM_STEPS = 16   # effective batch 32, matching the Kaggle runs
N_ROWS = 6_000
MAX_OPTIM_STEPS = 200
REPORT_EVERY = 10
SEED = 42
RANDOM_LOSS = math.log(5)
SUCCESS_THRESHOLD = 1.55


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    assert_trainable_dtype(model)  # fp16 params silently cannot train; see mc.py

    df = pd.read_parquet(DATA / "train_pool_own_context_general_big.parquet")
    df = df.sample(n=min(N_ROWS, len(df)), random_state=SEED).reset_index(drop=True)
    df["context"] = df["context"].str.slice(0, 8_000)

    collator = DataCollatorForMultipleChoice(tokenizer)
    loader = DataLoader(
        MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col="context"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * MAX_OPTIM_STEPS), num_training_steps=MAX_OPTIM_STEPS
    )
    print(
        f"lr={LR}, effective batch {BATCH_SIZE * GRAD_ACCUM_STEPS}, {len(df)} rows, "
        f"target {MAX_OPTIM_STEPS} optim steps\n"
        f"random-guess loss (ln 5) = {RANDOM_LOSS:.4f}; success = mean loss durably < {SUCCESS_THRESHOLD}\n",
        flush=True,
    )

    model.train()
    optimizer.zero_grad()
    start = time.time()
    optim_step = 0
    window: list[float] = []
    trend: list[float] = []
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        (loss / GRAD_ACCUM_STEPS).backward()
        window.append(loss.item())
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_step += 1
            if optim_step % REPORT_EVERY == 0:
                mean_loss = sum(window) / len(window)
                trend.append(mean_loss)
                window = []
                delta = mean_loss - RANDOM_LOSS
                print(
                    f"optim_step {optim_step:4d}/{MAX_OPTIM_STEPS}  mean_loss {mean_loss:.4f}  "
                    f"(vs ln5 {delta:+.4f})  lr {scheduler.get_last_lr()[0]:.2e}  "
                    f"[{time.time() - start:.0f}s]",
                    flush=True,
                )
            if optim_step >= MAX_OPTIM_STEPS:
                break

    print()
    if trend:
        first, last = trend[0], trend[-1]
        best = min(trend)
        print(f"first reported window {first:.4f} -> last {last:.4f} (best {best:.4f})")
        if best < SUCCESS_THRESHOLD:
            print(f"VERDICT: LEARNING -- loss broke below {SUCCESS_THRESHOLD}. The LR fix is correct.")
        else:
            print(
                f"VERDICT: STILL STUCK near ln(5). lr={LR} is insufficient at this scale -- "
                f"escalate (3e-5), and re-check the head is not frozen, before spending more GPU time."
            )


if __name__ == "__main__":
    main()
