"""Isolate the NaN to the optimizer kernel.

Established by debug_nan.py: loss and pre-clip grad norm are finite and
healthy (1.54), fp32 and bf16 fail identically, and the first optimizer step
turns all ~103 trainable tensors non-finite. That rules out precision, data,
and gradient explosion, and points at AdamW's update itself on sm_120.

This runs one identical accumulation window and one optimizer step under
several optimizer configurations, reporting how many params go non-finite.
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
    MCCollator,
    OpenBookMCDataset,
    build_model,
)
from llmsci.gpu_guard import cap_memory_fraction

DATA = Path(__file__).resolve().parent.parent / "data"


def make_optimizer(kind, params):
    if kind == "adamw_default":
        return torch.optim.AdamW(params, lr=LR)
    if kind == "adamw_foreach_false":
        return torch.optim.AdamW(params, lr=LR, foreach=False)
    if kind == "adamw_fused":
        return torch.optim.AdamW(params, lr=LR, fused=True)
    if kind == "adamw_eps1e-6_foreach_false":
        return torch.optim.AdamW(params, lr=LR, eps=1e-6, foreach=False)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=LR)
    raise ValueError(kind)


def trial(kind: str) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    df = pd.read_parquet(DATA / "train_1024.parquet")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(kind, params)

    ds = OpenBookMCDataset(df, tokenizer, MAX_INPUT, "context")
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True,
                        collate_fn=MCCollator(tokenizer), drop_last=True)
    opt.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        if i >= GRAD_ACCUM_STEPS:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        (out.loss / GRAD_ACCUM_STEPS).backward()
    n_bad_grad = sum(1 for p in params if p.grad is not None and not torch.isfinite(p.grad).all())
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    n_bad = sum(1 for p in params if not torch.isfinite(p).all())
    print(f"{kind:32s} grad_norm={gn.item():.4e} bad_grads={n_bad_grad:3d} "
          f"bad_params_after_step={n_bad:3d}/{len(params)}  "
          f"{'FAIL' if n_bad else 'ok'}")
    del model, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    cap_memory_fraction(0.975)
    print(f"torch {torch.__version__}  capability {torch.cuda.get_device_capability()}\n")
    for k in ["adamw_default", "adamw_foreach_false", "adamw_fused",
              "adamw_eps1e-6_foreach_false", "sgd"]:
        try:
            trial(k)
        except Exception as e:
            print(f"{k:32s} raised {type(e).__name__}: {str(e)[:100]}")
