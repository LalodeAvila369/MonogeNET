"""Tests for the Riesz kernels on synthetic signals with known answers."""

import math

import pytest
import torch

from monogenet.filters.riesz import riesz_kernels


def _apply(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(img) * kernel).real


def test_unit_modulus_and_dc() -> None:
    h1, h2 = riesz_kernels(32, 48)
    assert h1.shape == h2.shape == (32, 48) and h1.is_complex()
    assert h1[0, 0] == 0 and h2[0, 0] == 0
    h1o, h2o = riesz_kernels(31, 33)  # odd sizes: no Nyquist, unit modulus except DC
    mag2 = (h1o.abs() ** 2 + h2o.abs() ** 2).flatten()[1:]
    assert torch.allclose(mag2, torch.ones_like(mag2), atol=1e-5)


def test_purely_imaginary_and_odd_symmetry() -> None:
    for size in ((16, 16), (15, 17), (16, 15)):
        for h in riesz_kernels(*size):
            assert h.real.abs().max() == 0
            # H(-w) = -H(w): the kernels are odd, so the spatial filters are real.
            flipped = torch.roll(torch.flip(h, dims=(0, 1)), shifts=(1, 1), dims=(0, 1))
            assert torch.allclose(h, -flipped, atol=1e-6)
            assert torch.fft.ifft2(h).imag.abs().max() < 1e-6


def test_cosine_along_x_maps_to_minus_sine() -> None:
    n = 64
    x = torch.arange(n, dtype=torch.float32)
    arg = 2 * math.pi * 4 * x / n
    img = torch.cos(arg).expand(n, n).clone()  # varies along columns (x)
    h1, h2 = riesz_kernels(n, n)
    assert torch.allclose(_apply(img, h1), (-torch.sin(arg)).expand(n, n), atol=1e-4)
    assert _apply(img, h2).abs().max() < 1e-4


def test_cosine_along_y_maps_to_minus_sine() -> None:
    n = 64
    y = torch.arange(n, dtype=torch.float32)[:, None]
    arg = 2 * math.pi * 4 * y / n
    img = torch.cos(arg).expand(n, n).clone()
    h1, h2 = riesz_kernels(n, n)
    assert torch.allclose(_apply(img, h2), (-torch.sin(arg)).expand(n, n), atol=1e-4)
    assert _apply(img, h1).abs().max() < 1e-4


def test_oblique_cosine_splits_by_direction() -> None:
    # Wave vector (a, b): odd components are -sin * (a, b)/|(a, b)|.
    n = 64
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    a, b = 3, 4  # cycles per image along x and y, |k| = 5
    arg = 2 * math.pi * (a * x + b * y) / n
    h1, h2 = riesz_kernels(n, n)
    img = torch.cos(arg)
    assert torch.allclose(_apply(img, h1), -torch.sin(arg) * a / 5, atol=1e-4)
    assert torch.allclose(_apply(img, h2), -torch.sin(arg) * b / 5, atol=1e-4)


def test_vertical_step_edge_is_odd_in_x_only() -> None:
    # Vertical edge (intensity changes along x): odd1 is non-zero, odd2 ~ 0.
    n = 64
    img = torch.zeros(n, n)
    img[:, n // 2 :] = 1.0
    img = img - img.mean()
    h1, h2 = riesz_kernels(n, n)
    o1, o2 = _apply(img, h1), _apply(img, h2)
    assert o1.abs().max() > 10 * o2.abs().max()
    assert o2.abs().max() < 1e-4


def test_energy_preserved_for_zero_mean_image() -> None:
    g = torch.Generator().manual_seed(0)
    img = torch.randn(31, 33, generator=g)  # odd sizes: no Nyquist content
    img = img - img.mean()
    h1, h2 = riesz_kernels(31, 33)
    energy = (_apply(img, h1) ** 2 + _apply(img, h2) ** 2).sum()
    assert energy == pytest.approx(float((img**2).sum()), rel=1e-4)
