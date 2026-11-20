"""Tests for the radial log-Gabor filter (known answers)."""

import math

import pytest
import torch

from monogenet.filters.grids import frequency_radius
from monogenet.filters.log_gabor import radial_log_gabor


def test_dc_is_zero_and_range() -> None:
    g = radial_log_gabor(64, 64, 16.0)
    assert g[0, 0] == 0.0
    assert g.shape == (64, 64)
    assert (g >= 0).all() and (g <= 1 + 1e-6).all()


def test_peak_at_f0() -> None:
    # 64 px grid, wavelength 16 px -> f0 = 4/64 lies exactly on the grid.
    g = radial_log_gabor(64, 64, 16.0)
    r = frequency_radius(64, 64)
    assert g.max() == pytest.approx(1.0)
    assert torch.allclose(r[g > 1 - 1e-6], torch.tensor(1 / 16), atol=1e-6)
    # Entries at the ring |w| = f0 equal 1; the 1-D axis frequency u = f0 is one of them.
    assert g[0, 4] == pytest.approx(1.0)


def test_log_symmetry_around_f0() -> None:
    # G(f0 * a) == G(f0 / a): evaluate along the u axis with a = 2 (u = 8/64 and 2/64).
    g = radial_log_gabor(64, 64, 16.0, 0.55)
    assert g[0, 8] == pytest.approx(g[0, 2].item(), rel=1e-5)
    expected = math.exp(-(math.log(2) ** 2) / (2 * math.log(0.55) ** 2))
    assert g[0, 8] == pytest.approx(expected, rel=1e-5)


def test_bandwidth_ordering() -> None:
    # Convention: smaller sigma_onf -> larger |log sigma| -> WIDER passband.
    wide = radial_log_gabor(64, 64, 16.0, 0.3)
    narrow = radial_log_gabor(64, 64, 16.0, 0.75)
    assert wide[0, 8] > narrow[0, 8]


def test_isotropic() -> None:
    g = radial_log_gabor(64, 64, 16.0)
    assert g[0, 5] == pytest.approx(g[5, 0].item())
    assert torch.allclose(g, g.T)


def test_no_gain_at_dc_for_image() -> None:
    # Filtering a constant image gives zero (zero DC gain).
    img = torch.full((32, 32), 3.0)
    out = torch.fft.ifft2(torch.fft.fft2(img) * radial_log_gabor(32, 32, 8.0)).real
    assert out.abs().max() < 1e-5


def test_bandpass_response_to_sinusoid() -> None:
    # A sinusoid at f0 passes with gain 1; one far from f0 is strongly attenuated.
    n = 64
    x = torch.arange(n, dtype=torch.float32)
    g = radial_log_gabor(n, n, 16.0)
    for wavelength in (16.0, 4.0):
        img = torch.cos(2 * math.pi * x / wavelength).expand(n, n)
        out = torch.fft.ifft2(torch.fft.fft2(img) * g).real
        gain = math.exp(-(math.log(16.0 / wavelength) ** 2) / (2 * math.log(0.55) ** 2))
        assert out.std() / img.std() == pytest.approx(gain, abs=1e-3)
    assert gain < 0.1


def test_differentiable_in_wavelength_and_sigma() -> None:
    wl = torch.tensor(16.0, requires_grad=True)
    sg = torch.tensor(0.55, requires_grad=True)
    radial_log_gabor(32, 32, wl, sg).sum().backward()
    assert wl.grad is not None and torch.isfinite(wl.grad)
    assert sg.grad is not None and torch.isfinite(sg.grad)


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        radial_log_gabor(32, 32, 0.0)
    with pytest.raises(ValueError):
        radial_log_gabor(32, 32, 8.0, 1.0)
