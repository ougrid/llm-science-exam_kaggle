"""Decisive diagnostic: can the reader memorize a tiny batch of rows?

Every training run in this project has held its loss at ~1.61 = ln(5), the
exact loss of uniform predictions over 5 options -- i.e. the reader has never
demonstrably learned anything, and the "peaks at optim_step 15 then collapses"
pattern documented in reports/ablation_table.md is most likely noise around a
model still near its random initialization. Supporting arithmetic: at
lr=5e-6 with 10% warmup over 465 optimizer steps, the effective LR at step 15
is 1.63e-6, which is far too small to train a freshly-initialized
AutoModelForMultipleChoice head (classifier.weight + pooler.dense.* are
randomly initialized, per every run's own LOAD REPORT).

This is the standard sanity check that should have run before any real
training: a correct training loop MUST be able to drive loss to ~0 on a
handful of rows. If it can, the loop and data are fine and the problem is
purely optimization hyperparameters. If it can't, there's a real defect
(label misalignment, truncation eating the answer, a frozen head).

Sweeps LR so the result also tells us which LR to use for the real run,
rather than needing a second diagnostic pass.

Run from the repo root: `python scripts/diagnose_overfit_sanity.py`
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from llmsci.gpu_guard import cap_memory_fraction
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset, assert_trainable_dtype
from transformers import AutoModelForMultipleChoice, AutoTokenizer

DATA = Path("data")
MODEL_NAME = "microsoft/deberta-v3-base"
N_ROWS = 64
MAX_LENGTH = 384
BATCH_SIZE = 2
STEPS = 60
LRS = [5e-6, 2e-5, 5e-5]
RANDOM_LOSS = math.log(5)


def run_one(lr: float, df: pd.DataFrame, device) -> list[float]:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    assert_trainable_dtype(model)  # fp16 params silently cannot train; see mc.py
    collator = DataCollatorForMultipleChoice(tokenizer)
    ds = MultipleChoiceDataset(df, tokenizer, max_length=MAX_LENGTH, context_col="context")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-6)

    model.train()
    losses: list[float] = []
    step = 0
    while step < STEPS:
        for batch in loader:
            if step >= STEPS:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
    del model
    torch.cuda.empty_cache()
    return losses


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap_memory_fraction(0.975)

    df = pd.read_parquet(DATA / "train_pool_own_context_general.parquet").head(N_ROWS).reset_index(drop=True)
    df["context"] = df["context"].str.slice(0, 8_000)
    print(f"overfit target: {len(df)} rows, {STEPS} steps/LR, random-guess loss = {RANDOM_LOSS:.4f}\n")

    for lr in LRS:
        losses = run_one(lr, df, device)
        first10 = sum(losses[:10]) / 10
        last10 = sum(losses[-10:]) / 10
        verdict = "LEARNING" if last10 < RANDOM_LOSS - 0.15 else "STUCK AT RANDOM"
        print(
            f"lr={lr:.0e}: first-10-step mean loss {first10:.4f} -> last-10-step mean {last10:.4f} "
            f"(min {min(losses):.4f})  => {verdict}"
        )

    print(
        "\nInterpretation: any LR whose loss stays ~1.61 cannot train this head. "
        "The lowest LR that clearly drives loss down is the floor for the real run."
    )


if __name__ == "__main__":
    main()
