"""Tests for local amplitude, phase and orientation (synthetic known answers)."""

import math

import pytest
import torch

from monogenet.features.local import (
    local_amplitude,
    local_features,
    local_orientation,
    local_phase,
)
from monogenet.features.monogenic import monogenic_signal
from monogenet.filters.log_gabor import log_gabor_bank

D = torch.float64


def _mono(e: float, o1: float, o2: float) -> torch.Tensor:
    return torch.tensor([e, o1, o2], dtype=D).view(1, 3, 1, 1)


def test_hand_computed_values() -> None:
    m = _mono(3.0, 4.0, 12.0)  # |odd| = sqrt(160)
    assert local_amplitude(m).item() == pytest.approx(math.sqrt(9 + 16 + 144))
    assert local_phase(m).item() == pytest.approx(math.atan2(math.sqrt(160), 3.0))
    assert local_orientation(m).item() == pytest.approx(math.atan2(12.0, 4.0))


def test_phase_special_values() -> None:
    assert local_phase(_mono(1.0, 0.0, 0.0)).item() == pytest.approx(0.0)  # bright symmetric
    assert local_phase(_mono(-1.0, 0.0, 0.0)).item() == pytest.approx(math.pi)  # dark symmetric
    assert local_phase(_mono(0.0, 0.3, -0.4)).item() == pytest.approx(math.pi / 2)  # step edge


def test_orientation_is_modulo_pi() -> None:
    a = local_orientation(_mono(1.0, 0.3, 0.4)).item()
    b = local_orientation(_mono(1.0, -0.3, -0.4)).item()  # odd part flipped
    assert a == pytest.approx(b) and a == pytest.approx(math.atan2(0.4, 0.3))
    assert -math.pi / 2 < local_orientation(_mono(0.0, -0.1, 0.9)).item() <= math.pi / 2
    assert local_orientation(_mono(0.0, 0.0, 1.0)).item() == pytest.approx(math.pi / 2)


def test_oblique_cosine_amplitude_phase_orientation() -> None:
    n, a, b = 64, 3, 4
    y, x = torch.meshgrid(torch.arange(n, dtype=D), torch.arange(n, dtype=D), indexing="ij")
    arg = 2 * math.pi * (a * x + b * y) / n
    bank = log_gabor_bank(n, n, 1, n / 5.0, 1.0, 0.55, dtype=D)
    mono = monogenic_signal(torch.cos(arg), bank)
    amp, phase, orient = local_features(mono)
    assert torch.allclose(amp, torch.ones_like(amp), atol=1e-6)
    wrapped = torch.atan2(torch.sin(arg), torch.cos(arg)).abs()  # phase = |arg| wrapped to (-pi, pi]
    assert torch.allclose(phase[0], wrapped, atol=1e-5)
    mask = torch.sin(arg).abs() > 0.2  # orientation is ill-defined where the odd part vanishes
    assert torch.allclose(orient[0][mask], torch.full_like(orient[0][mask], math.atan2(b, a)), atol=1e-5)


def test_line_has_phase_zero_at_its_centre() -> None:
    n = 128
    bank = log_gabor_bank(n, n, 1, 8.0, 1.0, dtype=D)
    line = torch.zeros(n, n, dtype=D)
    line[:, n // 2] = 1.0
    phase = local_phase(monogenic_signal(line, bank))[0]
    assert phase[10, n // 2] < 0.05
    dark = local_phase(monogenic_signal(-line, bank))[0]
    assert dark[10, n // 2] > math.pi - 0.05


def test_step_edge_phase_is_symmetric_about_the_edge() -> None:
    # Edge between columns col-1 and col: even is antisymmetric, odd symmetric, so
    # phi(col-1) + phi(col) = pi exactly, and both are within the odd-dominated range.
    n = 128
    bank = log_gabor_bank(n, n, 1, 8.0, 1.0, dtype=D)
    step = torch.zeros(n, n, dtype=D)
    step[:, n // 4 : 3 * n // 4] = 1.0
    phase = local_phase(monogenic_signal(step, bank))[0]
    col = n // 4
    assert phase[10, col - 1] + phase[10, col] == pytest.approx(math.pi, abs=1e-8)
    assert 0.3 * math.pi < phase[10, col] < 0.7 * math.pi


def test_contrast_scaling() -> None:
    # Amplitude scales with contrast; phase and orientation do not.
    g = torch.Generator().manual_seed(3)
    img = torch.randn(32, 32, generator=g, dtype=D)
    bank = log_gabor_bank(32, 32, 2, 4.0, 2.0, dtype=D)
    f1 = local_features(monogenic_signal(img, bank))
    f2 = local_features(monogenic_signal(5.0 * img, bank))
    assert torch.allclose(f2.amplitude, 5.0 * f1.amplitude, atol=1e-9)
    assert torch.allclose(f2.phase, f1.phase, atol=1e-9)
    assert torch.allclose(f2.orientation, f1.orientation, atol=1e-9)
    # Negative contrast flips (even, odd) together: phase -> pi - phase.
    f3 = local_features(monogenic_signal(-img, bank))
    assert torch.allclose(f3.phase, math.pi - f1.phase, atol=1e-9)


def test_eps_makes_gradient_finite_at_zero() -> None:
    m = torch.zeros(1, 3, 2, 2, requires_grad=True)
    local_amplitude(m, eps=1e-8).sum().backward()
    assert torch.isfinite(m.grad).all()


def test_shapes_and_invalid() -> None:
    mono = monogenic_signal(torch.rand(2, 4, 16, 16), log_gabor_bank(16, 16, 3, 3.0, 1.6))
    f = local_features(mono)
    assert f.amplitude.shape == f.phase.shape == f.orientation.shape == (2, 4, 3, 16, 16)
    with pytest.raises(ValueError):
        local_phase(torch.rand(2, 16, 16))
