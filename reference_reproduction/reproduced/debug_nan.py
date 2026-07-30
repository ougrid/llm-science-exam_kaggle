"""Diagnose the non-finite loss that appeared right after the first optimizer step.

Runs the real data through the real config and reports, per micro-step, the
loss and the pre-clip gradient norm, under bf16 autocast and under fp32, so
the failure can be attributed instead of guessed at.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common import (
    GRAD_ACCUM_STEPS,
    LR,
    MAX_INPUT,
    MICRO_BATCH,
    MODEL_NAME,
    WEIGHT_DECAY,
    MCCollator,
    OpenBookMCDataset,
    build_model,
)
from llmsci.gpu_guard import cap_memory_fraction

DATA = Path(__file__).resolve().parent.parent / "data"


def run(mode: str, n_micro: int = 40):
    print(f"\n{'='*60}\nmode = {mode}\n{'='*60}")
    torch.manual_seed(42)
    device = torch.device("cuda")
    train_df = pd.read_parquet(DATA / "train_1024.parquet")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    ds = OpenBookMCDataset(train_df, tokenizer, MAX_INPUT, "context")
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True,
                        collate_fn=MCCollator(tokenizer), drop_last=True)

    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        if i >= n_micro:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        seq_len = batch["input_ids"].shape[-1]
        if mode == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
        else:
            out = model(**batch)
        loss = out.loss
        (loss / GRAD_ACCUM_STEPS).backward()
        flag = "" if torch.isfinite(loss) else "   <-- NON-FINITE LOSS"
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            print(f"micro {i:3d} L={seq_len:4d} loss={loss.item():.4f} "
                  f"grad_norm={gn.item():.4e} -> OPTIM STEP{flag}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        elif i < 3 or i > GRAD_ACCUM_STEPS - 2:
            print(f"micro {i:3d} L={seq_len:4d} loss={loss.item():.4f}{flag}")
        if not torch.isfinite(loss):
            # Where did it break: weights, or this batch?
            bad_w = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
            print(f"  non-finite params: {len(bad_w)} (first few: {bad_w[:5]})")
            print(f"  logits finite: {torch.isfinite(out.logits).all().item()}, "
                  f"logits={out.logits.detach().float().cpu().numpy()}")
            break
    del model, optimizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    cap_memory_fraction(0.975)
    run("bf16")
    run("fp32")
