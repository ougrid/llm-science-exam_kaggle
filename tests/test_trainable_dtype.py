"""Regression tests for the fp16-parameter trap that caused four failed runs.

The bug: `transformers` 5.x makes `from_pretrained` follow the checkpoint's
stored dtype, and `microsoft/deberta-v3-large` ships fp16. So the plain call
returns half-precision PARAMETERS, which AdamW then cannot move -- an lr=2e-5
update is ~1.3 ULP at fp16 near a weight of 0.03, so it rounds away and training
loss pins at ln(5) forever with healthy gradients and healthy data.

These tests do not need the real model; the failure is pure floating-point
arithmetic, so a two-parameter stub reproduces it exactly and runs in
milliseconds. That is the point -- the check that would have saved four runs was
always cheap.
"""

from __future__ import annotations

import math

import pytest
import torch

from llmsci.reader.mc import assert_trainable_dtype


class _Stub(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1, dtype=dtype)


def test_fp32_model_passes():
    assert_trainable_dtype(_Stub(torch.float32))  # must not raise


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_params_rejected(dtype):
    with pytest.raises(RuntimeError, match="cannot be trained"):
        assert_trainable_dtype(_Stub(dtype))


def test_error_message_names_the_fix_and_the_symptom():
    with pytest.raises(RuntimeError) as ei:
        assert_trainable_dtype(_Stub(torch.float16))
    msg = str(ei.value)
    assert "dtype=torch.float32" in msg, "must tell the caller how to fix it"
    assert "ln(5)" in msg, "must name the symptom so a future reader connects them"


def ulp(value: float, dtype) -> float:
    """Spacing between representable `dtype` values at `value`."""
    exponent = math.floor(math.log2(abs(value)))
    return 2.0**exponent * torch.finfo(dtype).eps


def test_the_arithmetic_that_makes_fp16_untrainable():
    """The actual mechanism, asserted rather than asserted-about.

    An AdamW step at step 1 has magnitude ~lr, because m/sqrt(v) = sign(g). So
    the only question is whether lr is representable as a change to the weight.
    """
    lr = 2e-5
    w = 0.03  # typical DeBERTa classifier weight magnitude
    assert ulp(w, torch.float16) == pytest.approx(1.526e-5, rel=1e-3)
    assert lr / ulp(w, torch.float16) < 2, "fp16: update is ~1 ULP, so it snaps or vanishes"
    assert lr / ulp(w, torch.float32) > 1000, "fp32: update resolves cleanly"


@pytest.mark.parametrize("w", [0.1, 0.3, 1.0])
def test_fp16_updates_vanish_entirely_for_ordinary_weight_magnitudes(w):
    """The severe case, and the reason the whole network stalls rather than crawls.

    ULP grows with magnitude, so the larger a weight is, the smaller lr looks
    relative to it. At w >= 0.1 an lr=2e-5 AdamW update is below half a ULP and
    rounds to nothing -- not "slower training", literally zero movement, forever.
    DeBERTa's LayerNorm weights sit near 1.0, so they are frozen solid.
    """
    lr = 2e-5
    assert lr / ulp(w, torch.float16) < 0.5, "precondition: update is sub-ULP here"
    moved = {}
    for dtype in (torch.float16, torch.float32):
        p = torch.nn.Parameter(torch.full((32,), w, dtype=dtype))
        before = p.detach().clone().float()
        opt = torch.optim.AdamW([p], lr=lr, eps=1e-6, weight_decay=0.0)
        for _ in range(20):
            p.grad = torch.full((32,), 0.1, dtype=dtype)  # constant sign: easiest possible case
            opt.step()
        moved[dtype] = float((p.detach().float() - before).abs().mean())

    assert moved[torch.float32] == pytest.approx(20 * lr, rel=0.3), moved
    assert moved[torch.float16] == 0.0, (
        f"expected fp16 to be frozen at w={w}, saw {moved[torch.float16]:.3e}. If this now "
        f"moves, the upstream behaviour changed -- re-measure before relaxing the guard."
    )


def test_fp16_updates_are_quantized_to_ulp_multiples_when_they_do_move():
    """At small magnitudes fp16 moves, but only in ULP steps -- never by lr."""
    lr, w = 2e-5, 0.03
    p = torch.nn.Parameter(torch.full((32,), w, dtype=torch.float16))
    before = p.detach().clone().float()
    opt = torch.optim.AdamW([p], lr=lr, eps=1e-6, weight_decay=0.0)
    for _ in range(5):
        p.grad = torch.full((32,), 0.1, dtype=torch.float16)
        opt.step()
    delta = float((p.detach().float() - before).abs().mean())
    n_ulp = delta / ulp(w, torch.float16)
    assert n_ulp == pytest.approx(round(n_ulp), abs=1e-6), (
        f"fp16 delta {delta:.3e} should be an exact ULP multiple, got {n_ulp:.4f} ULP"
    )
    assert delta != pytest.approx(5 * lr, abs=1e-7), "fp16 must not match the intended 5*lr"


def test_random_loss_is_the_plateau_we_saw():
    """ln(5) is not a coincidence -- it is the loss of predicting uniformly."""
    uniform = torch.full((1, 5), 0.0)
    loss = torch.nn.functional.cross_entropy(uniform, torch.tensor([2]))
    assert float(loss) == pytest.approx(math.log(5), abs=1e-6)
    assert float(loss) == pytest.approx(1.6094, abs=1e-4)
