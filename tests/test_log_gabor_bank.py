"""Tests for the multi-scale log-Gabor bank."""

import pytest
import torch

from monogenet.filters.log_gabor import (
    log_gabor_bank,
    log_gabor_wavelengths,
    radial_log_gabor,
)


def test_wavelengths_geometric() -> None:
    wl = log_gabor_wavelengths(4, 3.0, 2.0)
    assert torch.allclose(wl, torch.tensor([3.0, 6.0, 12.0, 24.0]))


def test_wavelength_defaults_match_thesis() -> None:
    w = log_gabor_wavelengths(3, 36.0, 1.0833)
    assert torch.allclose(w, torch.tensor([36.0, 39.0, 42.25]), atol=0.05)


def test_bank_shape_and_matches_single_filters() -> None:
    bank = log_gabor_bank(64, 64, n_scales=3, min_wavelength=4.0, scale_factor=2.0, sigma_onf=0.5)
    assert bank.shape == (3, 64, 64)
    for s, wl in enumerate((4.0, 8.0, 16.0)):
        assert torch.allclose(bank[s], radial_log_gabor(64, 64, wl, 0.5), atol=1e-6)


def test_dc_zero_and_peaks_ordered() -> None:
    bank = log_gabor_bank(64, 64, 3, 4.0, 2.0)
    assert (bank[:, 0, 0] == 0).all()
    # Peak along the u axis moves to lower frequency as wavelength grows.
    peaks = bank[:, 0, : 32].argmax(dim=1)
    assert peaks.tolist() == [16, 8, 4]


def test_nonsquare() -> None:
    assert log_gabor_bank(32, 48, 2, 4.0, 2.0).shape == (2, 32, 48)


def test_differentiable() -> None:
    lam = torch.tensor(4.0, requires_grad=True)
    m = torch.tensor(2.0, requires_grad=True)
    s = torch.tensor(0.55, requires_grad=True)
    log_gabor_bank(32, 32, 3, lam, m, s).sum().backward()
    for p in (lam, m, s):
        assert p.grad is not None and torch.isfinite(p.grad)


def test_invalid_n_scales() -> None:
    with pytest.raises(ValueError):
        log_gabor_wavelengths(0, 4.0)
