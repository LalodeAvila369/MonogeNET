"""Tests for low-pass and windowing utilities."""

import math

import pytest
import torch

from monogenet.filters.log_gabor import radial_log_gabor
from monogenet.filters.window import apply_edge_taper, butterworth_lowpass, tukey_window


def test_butterworth_known_values() -> None:
    b = butterworth_lowpass(64, 64, cutoff=0.25, order=2)
    assert b[0, 0] == 1.0
    # |w| = 0.25 (index 16 on u axis) -> 1 / (1 + 1) = 0.5 at the cutoff.
    assert b[0, 16] == pytest.approx(0.5, abs=1e-6)
    # |w| = 0.125 -> 1 / (1 + 0.5**4)
    assert b[0, 8] == pytest.approx(1 / (1 + 0.5**4), abs=1e-6)
    assert (b > 0).all() and (b <= 1).all()


def test_butterworth_removes_corner_response() -> None:
    n = 64
    g = radial_log_gabor(n, n, 8.0)
    corner = (n // 2, n // 2)  # |w| = 0.707, outside the Nyquist disc
    filtered = g * butterworth_lowpass(n, n, 0.45, 15)
    assert g[corner] > 1e-3  # the raw log-Gabor leaks into the corner
    assert filtered[corner] < 1e-6
    assert filtered[0, 8] == pytest.approx(g[0, 8].item(), rel=1e-3)  # passband untouched


def test_tukey_shape() -> None:
    w = tukey_window(101, 0.4)
    assert w[0] == pytest.approx(0.0, abs=1e-6) and w[-1] == pytest.approx(0.0, abs=1e-6)
    assert w[50] == 1.0
    assert torch.allclose(w, torch.flip(w, dims=(0,)), atol=1e-6)
    assert (w >= 0).all() and (w <= 1).all()
    assert (tukey_window(10, 0.0) == 1).all()
    assert torch.allclose(tukey_window(65, 1.0), torch.hann_window(65, periodic=False), atol=1e-5)


def test_taper_keeps_interior_and_mean() -> None:
    img = torch.rand(40, 50) + 2.0
    out = apply_edge_taper(img, 0.2)
    assert torch.allclose(out[15:25, 20:30], img[15:25, 20:30])
    # Borders go to the mean.
    assert torch.allclose(out[0], torch.full((50,), float(img.mean())), atol=1e-5)
    assert torch.allclose(out[:, 0], torch.full((40,), float(img.mean())), atol=1e-5)


def test_taper_reduces_wraparound_edge_response() -> None:
    # A horizontal ramp is smooth but jumps between left and right borders under
    # the FFT's periodic assumption: strong spurious band-pass response there.
    n = 128
    ramp = torch.linspace(0, 1, n).expand(n, n).clone()
    g = radial_log_gabor(n, n, 8.0)

    def border_energy(img: torch.Tensor) -> float:
        out = torch.fft.ifft2(torch.fft.fft2(img) * g).real
        return float((out[:, :8] ** 2).mean() + (out[:, -8:] ** 2).mean())

    assert border_energy(apply_edge_taper(ramp, 0.2)) < 0.1 * border_energy(ramp)


def test_taper_batched_and_invalid() -> None:
    x = torch.rand(2, 3, 32, 32)
    assert apply_edge_taper(x).shape == x.shape
    with pytest.raises(ValueError):
        tukey_window(10, 1.5)
    with pytest.raises(ValueError):
        butterworth_lowpass(8, 8, cutoff=0.6)
