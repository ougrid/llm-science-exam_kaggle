"""GPU safety guards for local training under WSL2.

WSL2's CUDA driver silently backs allocations past the physical VRAM limit
with system RAM ("shared GPU memory"), which runs 15-25x slower with zero
error, warning, or unusual `nvidia-smi` reading -- this cost one run 5 hours
before being caught by hand (see DEVLOG.md's "WSL2 shared-memory trap"
entry). Two independent defenses, since either alone can miss a different
failure mode:

1. `cap_memory_fraction` blocks allocations that would overflow VRAM,
   converting the silent slowdown into an immediate, loud CUDA OOM.
2. `probe_training_speed` catches everything else that can silently degrade
   a job -- thermal throttling, CPU contention, a WSL/driver hiccup -- by
   benchmarking a few real steps at the production batch shape *before* the
   real run starts, and failing loudly if they're much slower than a
   pre-registered reference measured on this hardware.

Call both at the top of every GPU training/inference script, before loading
the real model.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForMultipleChoice


def cap_memory_fraction(fraction: float = 0.975) -> None:
    """Cap PyTorch's reservable GPU memory below the physical card size.

    Call once per process, right after selecting the CUDA device. No-op on
    CPU. `fraction` is relative to `torch.cuda.get_device_properties().total_memory`,
    the true card size -- not to whatever is currently free.
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def assert_step_speed(
    step_fn,
    max_ms_per_step: float,
    n_warmup: int = 2,
    n_measure: int = 5,
    label: str = "step",
) -> float:
    """Run `step_fn()` a few times and raise if it's much slower than expected.

    `max_ms_per_step` must come from a real measurement on this hardware for
    this config (CLAUDE.md: numbers come from measurement, not intuition) --
    set it to roughly 2.5x a clean benchmark run, enough margin for normal
    variance but tight enough to still catch an order-of-magnitude
    regression. Returns the measured ms/step so it can be logged.
    """
    for _ in range(n_warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_measure):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_step = (time.time() - start) / n_measure * 1000
    if ms_per_step > max_ms_per_step:
        raise RuntimeError(
            f"{label} took {ms_per_step:.0f} ms/step, expected <= {max_ms_per_step:.0f} ms/step. "
            "This is the signature of the WSL2 shared-GPU-memory fallback (or another "
            "silent slowdown -- thermal throttling, CPU contention, a driver hiccup) -- "
            "stopping now instead of running for hours at a fraction of expected speed. "
            "See DEVLOG.md's 'WSL2 shared-memory trap' entry."
        )
    print(f"[gpu_guard] {label}: {ms_per_step:.0f} ms/step (limit {max_ms_per_step:.0f}) -- OK")
    return ms_per_step


def probe_training_speed(
    model_name: str,
    batch_size: int,
    num_choices: int,
    seq_len: int,
    device: torch.device,
    max_ms_per_step: float,
    lr: float = 5e-6,
    eps: float = 1e-6,
    n_warmup: int = 2,
    n_measure: int = 5,
) -> float:
    """Benchmark a throwaway model/optimizer at the production batch shape.

    Uses a SEPARATE model+optimizer instance from the one used for real
    training, so the probe never touches production weights or optimizer
    state -- it exists purely to measure hardware throughput before
    committing to a run that might silently take 100x longer than expected.
    Frees its GPU memory before returning.
    """
    probe_model = AutoModelForMultipleChoice.from_pretrained(model_name, dtype=torch.float32).to(device)
    probe_model.train()
    probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=lr, eps=eps)

    input_ids = torch.randint(0, 1000, (batch_size, num_choices, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    labels = torch.randint(0, num_choices, (batch_size,), device=device)

    def step():
        out = probe_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels
        )
        out.loss.backward()
        probe_optimizer.step()
        probe_optimizer.zero_grad()

    try:
        return assert_step_speed(
            step,
            max_ms_per_step,
            n_warmup=n_warmup,
            n_measure=n_measure,
            label=f"speed probe (batch={batch_size} choices={num_choices} seq_len={seq_len})",
        )
    finally:
        del probe_model, probe_optimizer, input_ids, attention_mask, token_type_ids, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()
