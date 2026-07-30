"""One-off benchmark: ms/step for the frozen deberta-v3-large MC config.

CLAUDE.md forbids guessing the `max_ms_per_step` threshold the WSL2 guard
uses ("don't guess the threshold; benchmark it once and hardcode the number
with a comment citing the measurement"). This script produces that number.

`llmsci.gpu_guard.probe_training_speed` can't be used to generate it: it
builds an unfrozen model with AdamW over all 435M params (~3.5 GB of
optimizer state alone), which does not fit alongside everything else in
8.55 GB and is not the shape the real run uses. So the probe here mirrors
the real config's freezing.

Run: python bench_step.py
"""

from __future__ import annotations

import time

import torch

from common import GRAD_ACCUM_STEPS, LR, MAX_INPUT, MICRO_BATCH, build_model
from llmsci.gpu_guard import cap_memory_fraction


def main() -> None:
    device = torch.device("cuda")
    cap_memory_fraction(0.975)
    model = build_model(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01
    )

    ids = torch.randint(0, 1000, (MICRO_BATCH, 5, MAX_INPUT), device=device)
    mask = torch.ones_like(ids)
    tt = torch.zeros_like(ids)
    labels = torch.randint(0, 5, (MICRO_BATCH,), device=device)

    def micro_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=mask, token_type_ids=tt, labels=labels)
        (out.loss / GRAD_ACCUM_STEPS).backward()

    for _ in range(5):
        micro_step()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    n = 40
    start = time.time()
    for _ in range(n):
        micro_step()
    torch.cuda.synchronize()
    ms = (time.time() - start) / n * 1000
    optimizer.step()
    optimizer.zero_grad()

    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nmicro-step (batch={MICRO_BATCH} choices=5 seq_len={MAX_INPUT}, bf16 autocast): "
          f"{ms:.1f} ms")
    print(f"peak allocated {peak:.2f} GB / {total:.2f} GB card")
    print(f"suggested guard threshold (2.5x): {ms * 2.5:.0f} ms")
    est = ms * 1024 * 2 / 1000 / 60
    print(f"estimated train wall-clock for 1024 rows x 2 epochs: {est:.1f} min")


if __name__ == "__main__":
    main()
