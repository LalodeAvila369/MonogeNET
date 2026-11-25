"""Numerical stability of the features: degenerate inputs, gradients, precision."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from monogenet.features._ops import promote_half, safe_atan2, safe_sqrt
from monogenet.features.congruency import phase_congruency
from monogenet.features.local import (
    feature_type,
    local_amplitude,
    local_features,
    local_orientation,
    local_phase,
)
from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import phase_asymmetry, phase_symmetry
from monogenet.features.threshold import LearnableThreshold, noise_threshold
from monogenet.filters.learnable import LearnableLogGaborBank
from monogenet.filters.log_gabor import log_gabor_bank

FEATURES = [
    local_amplitude,
    local_phase,
    local_orientation,
    feature_type,
    phase_symmetry,
    phase_asymmetry,
    phase_congruency,
    lambda m: noise_threshold(m),
    lambda m: noise_threshold(m, "geometric"),
]
DTYPES = [torch.float32, torch.float64]


def _degenerate(kind: str, dtype: torch.dtype) -> torch.Tensor:
    """Monogenic tensors (S=2, 3, H, W) at the edge cases of the formulas."""
    shape = (2, 3, 8, 8)
    m = torch.zeros(shape, dtype=dtype)
    if kind == "zero":
        pass
    elif kind == "pure_even":  # odd components exactly zero
        m[:, 0] = 1.0
    elif kind == "pure_odd":  # even exactly zero
        m[:, 1] = 0.6
        m[:, 2] = -0.8
    elif kind == "parallel_scales":  # cross product exactly zero
        m[0] = torch.tensor([1.0, 0.4, 0.2], dtype=dtype).view(3, 1, 1)
        m[1] = 2.5 * m[0]
    elif kind == "tiny":
        m = torch.full(shape, 1e-30, dtype=dtype)
    return m


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["zero", "pure_even", "pure_odd", "parallel_scales", "tiny"])
@pytest.mark.parametrize("fn", FEATURES)
def test_finite_values_and_gradients_on_degenerate_inputs(fn, kind: str, dtype: torch.dtype) -> None:
    mono = _degenerate(kind, dtype).requires_grad_(True)
    out = fn(mono)
    outs = out if isinstance(out, tuple) else (out,)
    assert all(torch.isfinite(o).all() for o in outs)
    sum(o.sum() for o in outs).backward()
    assert torch.isfinite(mono.grad).all(), f"non-finite gradient for {kind}"


def test_soft_features_finite_on_degenerate_inputs() -> None:
    for kind in ("zero", "pure_even", "pure_odd", "parallel_scales"):
        for fn in (phase_symmetry, phase_asymmetry, phase_congruency):
            mono = _degenerate(kind, torch.float32).requires_grad_(True)
            out = fn(mono, sharpness=20.0)
            out.sum().backward()
            assert torch.isfinite(out).all() and torch.isfinite(mono.grad).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_blank_image_through_whole_pipeline(dtype: torch.dtype) -> None:
    # All-zero and constant images (e.g. black ultrasound background or padding).
    bank_module = LearnableLogGaborBank((6.0, 9.0), min_wavelength=3.0, max_wavelength=30.0).to(dtype)
    lt = LearnableThreshold(2).to(dtype)
    for value in (0.0, 3.0):
        img = torch.full((1, 1, 32, 32), value, dtype=dtype, requires_grad=True)
        mono = monogenic_signal(img, bank_module(32, 32))
        outs = [
            phase_symmetry(mono, lt(mono), sharpness=20.0),
            phase_asymmetry(mono, lt(mono)),
            phase_congruency(mono),
            *local_features(mono),
            feature_type(mono),
        ]
        assert all(torch.isfinite(o).all() for o in outs)
        sum(o.sum() for o in outs).backward()
        assert torch.isfinite(img.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in (*bank_module.parameters(), *lt.parameters()))
    # A blank image has no structure: no symmetric or asymmetric response.
    assert phase_symmetry(mono).abs().max() < 1e-6 and phase_asymmetry(mono).abs().max() < 1e-6


@pytest.mark.parametrize("scale", [1e-30, 1e-12, 1e-3, 1.0, 1e3, 1e12])
def test_finite_over_a_wide_dynamic_range_float32(scale: float) -> None:
    img = torch.rand(1, 1, 32, 32) * scale
    mono = monogenic_signal(img, log_gabor_bank(32, 32, 3, 4.0, 1.5))
    for out in (phase_symmetry(mono), phase_asymmetry(mono), phase_congruency(mono), feature_type(mono)):
        assert torch.isfinite(out).all()


def test_contrast_invariance_over_orders_of_magnitude() -> None:
    img = torch.rand(1, 1, 48, 48, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    bank = log_gabor_bank(48, 48, 3, 4.0, 1.5, dtype=torch.float64)
    ref = phase_symmetry(monogenic_signal(img, bank), eps=1e-14)
    for c in (1e-4, 1e-2, 1e2, 1e5):
        out = phase_symmetry(monogenic_signal(c * img, bank), eps=1e-14)
        assert torch.allclose(out, ref, atol=1e-6)


def test_safe_primitives() -> None:
    x = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
    safe_sqrt(x * x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[1].item() == pytest.approx(1.0)
    y = torch.tensor([0.0, 0.5, -2.0], dtype=torch.float64, requires_grad=True)
    xs = torch.tensor([0.0, 1.5, -1.0], dtype=torch.float64, requires_grad=True)
    out = safe_atan2(y, xs)
    assert out[0].item() == 0.0
    assert torch.allclose(out[1:], torch.atan2(y[1:], xs[1:]))
    out.sum().backward()
    assert torch.isfinite(y.grad).all() and torch.isfinite(xs.grad).all()
    assert y.grad[0].item() == 0.0 and xs.grad[0].item() == 0.0
    # Away from zero the gradients are those of torch.atan2.
    y2, x2 = y.detach()[1:].clone().requires_grad_(True), xs.detach()[1:].clone().requires_grad_(True)
    torch.atan2(y2, x2).sum().backward()
    assert torch.allclose(y.grad[1:], y2.grad) and torch.allclose(xs.grad[1:], x2.grad)


@pytest.mark.parametrize("half", [torch.float16, torch.bfloat16])
def test_half_precision_inputs_are_promoted(half: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    mono32 = torch.randn(2, 3, 8, 8, generator=g)
    mono = mono32.to(half)
    ref = mono.float()
    for fn in (phase_symmetry, phase_asymmetry, phase_congruency, feature_type, local_phase, local_amplitude):
        out = fn(mono)
        assert out.dtype == half and torch.isfinite(out.float()).all()
        assert torch.allclose(out.float(), fn(ref), atol=3e-2, rtol=3e-2)
    feats = local_features(mono)
    assert all(t.dtype == half for t in feats)
    # Values that would underflow in half precision (1e-4 squared = 1e-8) stay accurate.
    small = (mono32 * 1e-4).to(torch.bfloat16)
    assert torch.isfinite(local_amplitude(small).float()).all()
    assert torch.allclose(local_amplitude(small).float(), local_amplitude(small.float()), rtol=2e-2)


def test_promote_half_passes_other_dtypes_through() -> None:
    @promote_half
    def f(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    assert f(torch.ones(2, dtype=torch.float64)).dtype == torch.float64
    assert f(torch.ones(2, dtype=torch.float16)).dtype == torch.float16


@pytest.mark.parametrize(
    "fn",
    [
        local_amplitude,
        local_phase,
        local_orientation,
        feature_type,
        lambda m: phase_symmetry(m, sharpness=20.0),
        lambda m: phase_asymmetry(m, sharpness=20.0),
        lambda m: phase_congruency(m, sharpness=20.0),
    ],
)
def test_gradcheck_features_at_generic_points(fn) -> None:
    g = torch.Generator().manual_seed(4)
    mono = (torch.randn(2, 3, 3, 3, generator=g, dtype=torch.float64) * 0.7).requires_grad_(True)
    assert gradcheck(fn, (mono,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_phase_is_continuous_across_the_zero_vector() -> None:
    # Phase of a vanishing vector is defined as 0; nearby values are not NaN or huge.
    vals = []
    for r in (1e-3, 1e-8, 1e-11, 0.0):
        mono = torch.tensor([r, r, 0.0], dtype=torch.float64).view(1, 3, 1, 1)
        vals.append(local_phase(mono).item())
    assert all(math.isfinite(v) for v in vals) and vals[-1] == 0.0
