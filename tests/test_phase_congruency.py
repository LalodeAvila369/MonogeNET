"""Tests for monogenic phase congruency."""

import pytest
import torch

from monogenet.features.congruency import phase_congruency
from monogenet.features.monogenic import monogenic_signal
from monogenet.filters.log_gabor import log_gabor_bank

D = torch.float64


def _stack(*vectors: tuple[float, float, float]) -> torch.Tensor:
    """Monogenic tensor (S, 3, 1, 1) from per-scale (e, o1, o2) vectors."""
    return torch.tensor(vectors, dtype=D).view(len(vectors), 3, 1, 1)


def test_scales_in_phase_give_one() -> None:
    m = _stack((1.0, 0.5, 0.0), (2.0, 1.0, 0.0), (0.5, 0.25, 0.0))  # all parallel
    assert phase_congruency(m, threshold=0.0, eps=0.0).item() == pytest.approx(1.0, abs=1e-5)


def test_threshold_subtracts_from_energy() -> None:
    m = _stack((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))  # sum of amplitudes = 2, energy = 2
    assert phase_congruency(m, threshold=0.5, eps=0.0).item() == pytest.approx((2.0 - 0.5) / 2.0, abs=1e-5)
    assert phase_congruency(m, threshold=5.0, eps=0.0).item() == 0.0


def test_orthogonal_scales_give_zero() -> None:
    # v1 = x, v2 = y: mean direction at 45 deg, dot = cross = 1/sqrt(2) for each -> 0.
    m = _stack((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert phase_congruency(m, threshold=0.0, eps=0.0).item() == pytest.approx(0.0, abs=1e-5)


def test_hand_computed_two_scale_value() -> None:
    # v1 = (1, 0, 0), v2 = (cos a, sin a, 0): the mean direction bisects them at a/2.
    a = 0.6
    m = _stack((1.0, 0.0, 0.0), (float(torch.cos(torch.tensor(a))), float(torch.sin(torch.tensor(a))), 0.0))
    half = torch.tensor(a / 2)
    expected = float(torch.cos(half) - torch.sin(half))  # per scale, both amplitudes 1
    assert phase_congruency(m, threshold=0.0, eps=0.0).item() == pytest.approx(expected, abs=1e-5)


def test_line_and_edge_are_congruent_noise_is_not() -> None:
    n = 128
    bank = log_gabor_bank(n, n, 3, 4.0, 1.6, dtype=D)
    line = torch.zeros(n, n, dtype=D)
    line[:, n // 2] = 1.0
    pc_line = phase_congruency(monogenic_signal(line, bank))
    assert pc_line[10, n // 2] > 0.5 and pc_line[10, n // 2 + 40] < 1e-3
    step = torch.zeros(n, n, dtype=D)
    step[:, n // 4 : 3 * n // 4] = 1.0
    pc_step = phase_congruency(monogenic_signal(step, bank))
    assert pc_step[10, n // 4] > 0.4  # edges are congruent too (unlike PS)
    assert pc_step[10, n // 2] < 1e-3  # flat interior
    noise = torch.randn(n, n, generator=torch.Generator().manual_seed(0), dtype=D)
    assert phase_congruency(monogenic_signal(noise, bank)).mean() < pc_line[10, n // 2] / 3


def test_contrast_and_polarity_invariance() -> None:
    img = torch.randn(48, 48, generator=torch.Generator().manual_seed(1), dtype=D)
    bank = log_gabor_bank(48, 48, 3, 4.0, 1.5, dtype=D)
    a = phase_congruency(monogenic_signal(img, bank), eps=1e-12, smooth=1e-24)
    b = phase_congruency(monogenic_signal(7.0 * img, bank), eps=1e-12, smooth=1e-24)
    c = phase_congruency(monogenic_signal(-img, bank), eps=1e-12, smooth=1e-24)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, c, atol=1e-6)


def test_shape_range_and_batching() -> None:
    img = torch.rand(2, 3, 24, 24, dtype=D)
    bank = log_gabor_bank(24, 24, 3, 4.0, 1.5, dtype=D)
    pc = phase_congruency(monogenic_signal(img, bank))
    assert pc.shape == (2, 3, 24, 24)
    assert (pc >= 0).all() and (pc <= 1 + 1e-9).all()


def test_gradient_finite_even_when_scales_are_parallel() -> None:
    v = torch.tensor([1.0, 0.4, 0.2], dtype=D)
    mono = (v[None, :, None, None] * torch.tensor([1.0, 2.0, 0.5], dtype=D)[:, None, None, None])
    mono = mono.clone().requires_grad_(True)  # exactly parallel scales -> cross product 0
    phase_congruency(mono, threshold=0.0).sum().backward()
    assert torch.isfinite(mono.grad).all()


def test_gradients_to_image() -> None:
    img = torch.rand(1, 1, 24, 24, dtype=D, requires_grad=True)
    bank = log_gabor_bank(24, 24, 3, 4.0, 1.5, dtype=D)
    phase_congruency(monogenic_signal(img, bank)).sum().backward()
    assert torch.isfinite(img.grad).all() and img.grad.abs().sum() > 0


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        phase_congruency(torch.rand(2, 8, 8))
