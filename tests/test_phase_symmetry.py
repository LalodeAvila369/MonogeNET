"""Tests for phase symmetry, including agreement with the NumPy thesis reference."""

import math

import numpy as np
import pytest
import torch

from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import noise_threshold, phase_symmetry
from monogenet.filters.log_gabor import log_gabor_bank
from monogenet.reference import reference_pipeline

D = torch.float64


def _mono(e: float, o1: float, o2: float) -> torch.Tensor:
    return torch.tensor([e, o1, o2], dtype=D).view(1, 3, 1, 1)


def test_hand_computed_value() -> None:
    m = _mono(3.0, 0.0, 1.0)
    ps = phase_symmetry(m, threshold=0.5, eps=0.0).item()
    assert ps == pytest.approx((3.0 - 1.0 - 0.5) / math.sqrt(10.0))
    # Odd dominates -> floor at 0; works for dark features too (|even|).
    assert phase_symmetry(_mono(1.0, 2.0, 0.0), 0.0, 0.0).item() == 0.0
    assert phase_symmetry(_mono(-3.0, 0.0, 1.0), 0.5, 0.0).item() == pytest.approx(ps)


def test_matches_numpy_reference() -> None:
    rows, cols = 64, 80
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    img = np.zeros((rows, cols))
    img[16:32, 20:40] = 0.6
    img[:, 55] += 0.8
    img += 0.9 * np.exp(-(((x - 65) ** 2 + (y - 45) ** 2) / (2 * 3.0**2)))
    img += 0.02 * np.random.default_rng(0).standard_normal((rows, cols))
    wavelengths = (6.0, 8.0, 10.0)
    bank = torch.stack([log_gabor_bank(rows, cols, 1, wl, 1.0, 0.55, dtype=D)[0] for wl in wavelengths])
    ps = phase_symmetry(monogenic_signal(torch.from_numpy(img), bank), eps=1e-6).numpy()
    ref = reference_pipeline(img, wavelengths, 0.55, eps=1e-6)
    assert np.allclose(ps, ref.ps, atol=1e-9)
    assert ps.max() > 0.1


def test_contrast_invariance_with_default_threshold() -> None:
    g = torch.Generator().manual_seed(0)
    img = torch.randn(48, 48, generator=g, dtype=D)
    bank = log_gabor_bank(48, 48, 3, 4.0, 1.5, dtype=D)
    a = phase_symmetry(monogenic_signal(img, bank), eps=1e-12)
    b = phase_symmetry(monogenic_signal(9.0 * img, bank), eps=1e-12)
    assert torch.allclose(a, b, atol=1e-8)


def test_line_symmetric_edge_not() -> None:
    n = 128
    bank = log_gabor_bank(n, n, 1, 8.0, 1.0, dtype=D)
    line = torch.zeros(n, n, dtype=D)
    line[:, n // 2] = 1.0
    ps = phase_symmetry(monogenic_signal(line, bank), eps=1e-9)[0]
    assert ps[10, n // 2] > 0.5 and ps[10, n // 2 + 40] < 1e-3
    step = torch.zeros(n, n, dtype=D)
    step[:, n // 4 : 3 * n // 4] = 1.0
    ps_step = phase_symmetry(monogenic_signal(step, bank), threshold=0.0, eps=1e-9)[0]
    assert ps_step[10, n // 4] < 0.5 * ps[10, n // 2]  # odd response dominates at an edge


def test_dark_line_is_also_symmetric() -> None:
    n = 128
    bank = log_gabor_bank(n, n, 1, 8.0, 1.0, dtype=D)
    line = torch.zeros(n, n, dtype=D)
    line[:, n // 2] = 1.0
    a = phase_symmetry(monogenic_signal(line, bank), eps=1e-9)
    b = phase_symmetry(monogenic_signal(-line, bank), eps=1e-9)
    assert torch.allclose(a, b, atol=1e-9)


def test_noise_threshold_statistics() -> None:
    g = torch.Generator().manual_seed(1)
    mono = torch.randn(2, 3, 3, 16, 16, generator=g, dtype=D)  # (B, S, 3, H, W)
    amp = torch.sqrt((mono**2).sum(dim=-3))
    t_mean = noise_threshold(mono, "mean")
    assert t_mean.shape == (2, 3, 1, 1)
    assert torch.allclose(t_mean[..., 0, 0], amp.mean(dim=(-2, -1)))
    t_geo = noise_threshold(mono, "geometric", eps=0.0)
    assert torch.allclose(t_geo[..., 0, 0], torch.exp(torch.log(amp).mean(dim=(-2, -1))))
    assert (t_geo <= t_mean + 1e-12).all()  # AM-GM
    assert torch.allclose(noise_threshold(mono, gain=2.5), 2.5 * t_mean)
    with pytest.raises(ValueError):
        noise_threshold(mono, "median")


def test_threshold_is_per_image() -> None:
    # The threshold of one image must not depend on the rest of the batch.
    g = torch.Generator().manual_seed(2)
    mono = torch.randn(4, 2, 3, 8, 8, generator=g, dtype=D)
    assert torch.allclose(noise_threshold(mono)[1], noise_threshold(mono[1:2])[0])


def test_range_shape_and_gradients() -> None:
    g = torch.Generator().manual_seed(3)
    img = torch.rand(2, 1, 24, 24, generator=g, dtype=D, requires_grad=True)
    bank = log_gabor_bank(24, 24, 3, 4.0, 1.5, dtype=D)
    ps = phase_symmetry(monogenic_signal(img, bank))
    assert ps.shape == (2, 1, 3, 24, 24)
    assert (ps >= 0).all() and (ps <= 1 + 1e-9).all()
    ps.sum().backward()
    assert torch.isfinite(img.grad).all()


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        phase_symmetry(torch.rand(2, 8, 8))
