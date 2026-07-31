"""Does our training step actually move the trainable weights?

The third and last suspect for the day-4 symptom (train_loss pinned at
ln(5)=1.6094 for 1,000 optimizer steps over 4 epochs of 4,586 rows, with 77.2M
trainable parameters). If the data is sound and the input format is sound, then
either gradients are not reaching the trainable weights, or the update is being
applied but is somehow ineffective.

This reuses the EXACT code path from notebooks/kaggle/day4-src2-train/script.py
-- same freezing recipe, same optimizer, same collator, same input format -- and
instruments it. Three things get measured:

  1. GRAD NORMS BY GROUP. Frozen groups must be exactly None/zero; the six
     unfrozen encoder layers, the pooler, and the classifier must all be
     strictly positive. A zero here on a group we believe is trainable is the
     whole bug: a detached graph or a mis-specified requires_grad sweep looks
     like a healthy loop that never learns.
  2. WEIGHT DELTA AFTER optimizer.step(). Nonzero grads still buy nothing if
     the optimizer was constructed over a different parameter list than the one
     that has grads -- a real and silent failure mode, since the script builds
     AdamW over a filtered generator.
  3. OVERFIT A TINY SLICE. 16 rows, ~60 steps, no accumulation. Loss must go
     to ~0.

On (3), the honest limitation, learned the hard way earlier in this project:
reaching loss ~0 on 16 rows does NOT prove the labels are meaningful --
memorization succeeds on scrambled labels too. That is not what this test is
for. It is a gradient-flow test only: FAILING to overfit 16 rows is conclusive
evidence the optimization is broken, while succeeding merely clears the
optimizer and hands the question back to the data (which cells A/B of
scripts/diagnose_train_data_and_format.py answer).

CPU-capable but slow; uses CUDA when free. ~5 min either way, no Kaggle quota.

Run: python scripts/diagnose_optimizer_path.py [--device cpu]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from llmsci.gpu_guard import cap_memory_fraction
from llmsci.reader.mc import DataCollatorForMultipleChoice, MultipleChoiceDataset

DATA = Path("data")
TRAIN_SRC2 = DATA / "train_pool_own_context_src2.parquet"

# Copied verbatim from the day-4 kernel so this tests that recipe, not a variant.
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 384
MAX_CONTEXT_CHARS = 8_000
LR = 2e-5
FREEZE_EMBEDDINGS = True
N_FROZEN_LAYERS = 18
SEED = 42

OVERFIT_ROWS = 16
OVERFIT_STEPS = 60
# The kernel trains at micro-batch 2 x accum 8 (effective 16) on a 16 GB T4.
# Locally the 8 GB card cannot do micro-batch 2 with a backward pass at maxlen
# 384 -- measured peak is 4.20 GiB at bs 1 and ~7.5 GiB at bs 2 -- so we use
# micro-batch 1 x accum 16. Same effective batch, same accumulation loop (which
# is itself under test here), just more micro-steps.
#
# Worth knowing how that limit announces itself: on WSL2 the over-budget case
# raises `RuntimeError: CUDA driver error: device not ready`, NOT
# torch.OutOfMemoryError. Any `except torch.OutOfMemoryError` fallback -- and
# our submission scripts have one -- will not catch it.
MICRO_BATCH = 1
GRAD_ACCUM_STEPS = OVERFIT_ROWS // MICRO_BATCH
RANDOM_LOSS = math.log(5)


def group_of(name: str) -> str:
    if name.startswith("deberta.embeddings"):
        return "embeddings (frozen)"
    if name.startswith("deberta.encoder.layer."):
        i = int(name.split(".")[3])
        return f"encoder.layer.{i:02d} ({'frozen' if i < N_FROZEN_LAYERS else 'TRAINABLE'})"
    if name.startswith("deberta.encoder.rel_embeddings"):
        return "encoder.rel_embeddings"
    if name.startswith("pooler"):
        return "pooler (TRAINABLE head)"
    if name.startswith("classifier"):
        return "classifier (TRAINABLE head)"
    return f"other: {name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    # transformers 5.x defaults from_pretrained to the CHECKPOINT's dtype, and
    # microsoft/deberta-v3-large ships fp16 -- so the default load gives fp16
    # PARAMETERS, not just fp16 compute. Pure-fp16 training is silently broken:
    # an AdamW update of lr=2e-5 is ~1.3 ULP for a weight near 0.03 (fp16 ULP
    # there is 2^-16 = 1.5e-5), so updates round away and the model never leaves
    # the uniform-prediction fixed point, whose loss is exactly ln(5).
    # Under transformers 4.x the same call returned fp32, which is why this
    # never bit earlier in the project.
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="float16 reproduces the broken run; float32 is the fix")
    # fp32 doubles weight and activation memory, so the 8 GB local card cannot
    # run the kernel's 384 at fp32 -- measured fp16 peak is already 4.20 GiB.
    # 256 fits and proves the same point; the Kaggle T4 has 16 GB and runs 384.
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    max_length = args.max_length
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        # A previous process releasing the device can leave the driver briefly
        # unready on WSL2 ("CUDA driver error: device not ready"); a real kernel
        # launch here surfaces that before the model is on the GPU.
        cap_memory_fraction(0.975)
        torch.zeros(8, 8, device="cuda").sum().item()
        torch.cuda.synchronize()
    print(f"device {device}, random-guess loss ln(5)={RANDOM_LOSS:.4f}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    seen = {str(p.dtype) for p in model.parameters()}
    print(f"PARAM DTYPE: {seen}   (fp16 params => updates below ULP => ln(5) plateau)")
    ulp = float(torch.tensor(0.03, dtype=dtype).nextafter(torch.tensor(1.0, dtype=dtype)) - 0.03)
    print(f"  AdamW step-1 update ~= lr = {LR:.1e};  ULP near a weight of 0.03 = {ulp:.1e}"
          f"  -> update is {LR/ulp:.1f} ULP")

    if FREEZE_EMBEDDINGS:
        for p in model.deberta.embeddings.parameters():
            p.requires_grad = False
    for layer in model.deberta.encoder.layer[:N_FROZEN_LAYERS]:
        for p in layer.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M of {total/1e6:.1f}M ({100*trainable/total:.1f}%)")

    df = pd.read_parquet(TRAIN_SRC2).head(OVERFIT_ROWS).reset_index(drop=True)
    df["context"] = df["context"].str.slice(0, MAX_CONTEXT_CHARS)
    collator = DataCollatorForMultipleChoice(tok)
    ds = MultipleChoiceDataset(df, tok, max_length=max_length, context_col="context")
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=False, collate_fn=collator)
    batch = {k: v.to(device) for k, v in next(iter(loader)).items()}

    # ---- 1. grad norms by group -------------------------------------------
    print("\n[1] GRAD NORMS after one backward (frozen must be none; heads must be > 0)")
    model.train()
    loss = model(**batch).loss
    loss.backward()
    print(f"    first-batch loss {loss.item():.4f}")
    norms: dict[str, float] = {}
    nograd: dict[str, int] = {}
    for name, p in model.named_parameters():
        g = group_of(name)
        if p.grad is None:
            nograd[g] = nograd.get(g, 0) + 1
        else:
            norms[g] = norms.get(g, 0.0) + float(p.grad.detach().norm() ** 2)
    for g in sorted(set(list(norms) + list(nograd))):
        n = math.sqrt(norms.get(g, 0.0))
        tag = "" if g not in nograd else f"  [{nograd[g]} params with grad=None]"
        flag = ""
        if "TRAINABLE" in g and n == 0.0:
            flag = "   <-- ZERO GRAD ON A TRAINABLE GROUP: this is the bug"
        if "frozen" in g and n > 0.0:
            flag = "   <-- grad on a frozen group (unexpected)"
        print(f"    {g:<34} grad_norm {n:.3e}{tag}{flag}")

    # ---- 2. does optimizer.step() actually change the weights? ------------
    print("\n[2] WEIGHT DELTA after one optimizer.step()")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, eps=1e-6, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=100)
    watch = {
        "classifier.weight": model.classifier.weight,
        "pooler.dense.weight": model.pooler.dense.weight,
        "layer.23 attn q": model.deberta.encoder.layer[23].attention.self.query_proj.weight,
        "layer.00 attn q (frozen)": model.deberta.encoder.layer[0].attention.self.query_proj.weight,
    }
    before = {k: v.detach().clone() for k, v in watch.items()}
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    for k, v in watch.items():
        d = float((v.detach() - before[k]).abs().max())
        expect_move = "frozen" not in k
        flag = ""
        if expect_move and d == 0.0:
            flag = "   <-- DID NOT MOVE: optimizer is not updating this parameter"
        if not expect_move and d != 0.0:
            flag = "   <-- moved despite being frozen (unexpected)"
        print(f"    {k:<26} max|delta| {d:.3e}{flag}")
    print(f"    lr in optimizer: {optimizer.param_groups[0]['lr']:.3e}")

    # ---- 3. can the exact pipeline overfit 16 rows? -----------------------
    print(f"\n[3] OVERFIT {OVERFIT_ROWS} ROWS for {OVERFIT_STEPS} optim steps at micro-batch "
          f"{MICRO_BATCH} x accum {GRAD_ACCUM_STEPS}")
    print("    (gradient-flow test only; success does NOT prove the labels are")
    print("     meaningful, but failure is conclusive)")
    micro = [
        {k: v.to(device) for k, v in b.items()}
        for b in DataLoader(ds, batch_size=MICRO_BATCH, shuffle=False, collate_fn=collator)
    ]
    losses = []
    for step in range(OVERFIT_STEPS):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for mb in micro:  # mirrors the kernel's accumulation loop
            out = model(**mb)
            (out.loss / GRAD_ACCUM_STEPS).backward()
            total += out.loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(total / len(micro))
        if (step + 1) % 10 == 0:
            print(f"    step {step+1:3d}  loss {losses[-1]:.4f}  (ln5 {RANDOM_LOSS:.4f})", flush=True)
    print("\nREADING")
    if losses[-1] < 0.20:
        print(f"    loss {losses[0]:.4f} -> {losses[-1]:.4f}: gradients flow and the optimizer")
        print("    updates the weights. The optimizer path is CLEARED -- the ln(5) plateau")
        print("    on the real pool is not a broken training loop.")
    elif losses[-1] > RANDOM_LOSS - 0.05:
        print(f"    loss {losses[0]:.4f} -> {losses[-1]:.4f}, still at ln(5) on SIXTEEN rows.")
        print("    This is conclusive: the training loop cannot learn anything at all.")
        print("    Fix this before any further GPU run; it fully explains day 4.")
    else:
        print(f"    loss {losses[0]:.4f} -> {losses[-1]:.4f}: partial descent. Gradients flow but")
        print("    slowly -- suspect the LR/warmup schedule or the frozen-layer depth, and")
        print("    re-run with N_FROZEN_LAYERS lowered before spending GPU quota.")


if __name__ == "__main__":
    main()
