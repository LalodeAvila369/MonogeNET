"""Tests for the monogenic signal on synthetic images with known answers."""

import math

import pytest
import torch

from monogenet.features.monogenic import monogenic_signal
from monogenet.filters.learnable import LearnableLogGaborBank
from monogenet.filters.log_gabor import log_gabor_bank
from monogenet.filters.riesz import riesz_kernels


def _cos_image(n: int, a: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    arg = 2 * math.pi * (a * x + b * y) / n
    return torch.cos(arg), arg


def test_shapes() -> None:
    bank = log_gabor_bank(32, 24, 3, 4.0, 1.5)
    assert monogenic_signal(torch.rand(32, 24), bank).shape == (3, 3, 32, 24)
    assert monogenic_signal(torch.rand(2, 5, 32, 24), bank).shape == (2, 5, 3, 3, 32, 24)


def test_unit_gain_bank_gives_image_and_riesz_components() -> None:
    # A bank with G = 1 (except DC) must return the zero-mean image and its Riesz transforms.
    n = 31
    g = torch.Generator().manual_seed(0)
    img = torch.randn(n, n, generator=g, dtype=torch.float64)
    bank = torch.ones(1, n, n, dtype=torch.float64)
    bank[:, 0, 0] = 0
    out = monogenic_signal(img, bank)[0]
    h1, h2 = riesz_kernels(n, n, dtype=torch.float64)
    spec = torch.fft.fft2(img - img.mean())
    assert torch.allclose(out[0], img - img.mean(), atol=1e-10)
    assert torch.allclose(out[1], torch.fft.ifft2(spec * h1).real, atol=1e-10)
    assert torch.allclose(out[2], torch.fft.ifft2(spec * h2).real, atol=1e-10)


def test_oblique_cosine_at_centre_frequency() -> None:
    # |k| = 5 cycles over 64 px -> wavelength 12.8 px; bank centred there with gain 1.
    n, a, b = 64, 3, 4
    img, arg = _cos_image(n, a, b)
    bank = log_gabor_bank(n, n, 1, n / 5.0, 1.0, 0.55, dtype=torch.float64)
    e, o1, o2 = monogenic_signal(img, bank)[0]
    assert torch.allclose(e, torch.cos(arg), atol=1e-6)
    assert torch.allclose(o1, -torch.sin(arg) * a / 5, atol=1e-6)
    assert torch.allclose(o2, -torch.sin(arg) * b / 5, atol=1e-6)
    # Local amplitude of a unit cosine is 1 everywhere.
    amp = torch.sqrt(e**2 + o1**2 + o2**2)
    assert torch.allclose(amp, torch.ones_like(amp), atol=1e-6)


def test_line_is_even_and_step_edge_is_odd() -> None:
    n = 128
    bank = log_gabor_bank(n, n, 1, 8.0, 1.0, dtype=torch.float64)
    line = torch.zeros(n, n, dtype=torch.float64)
    line[:, n // 2] = 1.0
    e, o1, o2 = monogenic_signal(line, bank)[0]
    c = n // 2
    assert e[10, c].abs() > 20 * o1[10, c].abs() and o2.abs().max() < 1e-9
    step = torch.zeros(n, n, dtype=torch.float64)
    step[:, n // 4 : 3 * n // 4] = 1.0  # edges at n/4 (rising) and 3n/4 (falling)
    e, o1, o2 = monogenic_signal(step, bank)[0]
    for col in (n // 4, 3 * n // 4):
        # The edge lies half a pixel between col-1 and col: even is antisymmetric
        # around it, odd is symmetric, and the odd peak dominates the even response.
        assert e[10, col - 1] == pytest.approx(-e[10, col].item(), abs=1e-9)
        assert o1[10, col - 1] == pytest.approx(o1[10, col].item(), abs=1e-9)
        assert o1[10, col].abs() > 1.5 * e[10, col].abs()
        assert o1[10, col].abs() > 0.9 * o1[10].abs().max()
    assert o2.abs().max() < 1e-9


def test_translation_equivariance() -> None:
    g = torch.Generator().manual_seed(1)
    img = torch.randn(32, 32, generator=g, dtype=torch.float64)
    bank = log_gabor_bank(32, 32, 2, 4.0, 2.0, dtype=torch.float64)
    shifted = torch.roll(img, shifts=(3, -5), dims=(0, 1))
    out = monogenic_signal(img, bank)
    assert torch.allclose(monogenic_signal(shifted, bank), torch.roll(out, shifts=(3, -5), dims=(-2, -1)), atol=1e-9)


def test_linear_in_image() -> None:
    g = torch.Generator().manual_seed(2)
    a, b = torch.randn(2, 16, 16, generator=g, dtype=torch.float64)
    bank = log_gabor_bank(16, 16, 2, 3.0, 1.6, dtype=torch.float64)
    assert torch.allclose(
        monogenic_signal(2 * a - 3 * b, bank),
        2 * monogenic_signal(a, bank) - 3 * monogenic_signal(b, bank),
        atol=1e-9,
    )


def test_gradients_to_image_and_learnable_bank() -> None:
    bank_module = LearnableLogGaborBank((6.0, 9.0), min_wavelength=3.0, max_wavelength=30.0)
    img = torch.rand(1, 1, 24, 24, requires_grad=True)
    monogenic_signal(img, bank_module(24, 24)).pow(2).sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all() and img.grad.abs().sum() > 0
    for p in bank_module.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_invalid_inputs() -> None:
    bank = log_gabor_bank(16, 16, 2, 3.0, 1.6)
    with pytest.raises(ValueError):
        monogenic_signal(torch.rand(16, 20), bank)
    with pytest.raises(ValueError):
        monogenic_signal(torch.rand(16, 16), bank[0])
    with pytest.raises(TypeError):
        monogenic_signal(torch.rand(16, 16, dtype=torch.complex64), bank)
