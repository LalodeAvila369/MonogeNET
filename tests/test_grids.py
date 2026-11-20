"""Tests for the frequency-grid utilities (known answers on small grids)."""

import math

import pytest
import torch

from monogenet.filters.grids import frequency_angle, frequency_grid, frequency_radius


def test_grid_values_even_size() -> None:
    u, v = frequency_grid(4, 4)
    expected = torch.tensor([0.0, 0.25, -0.5, -0.25])
    assert torch.allclose(u[0], expected)  # u varies along columns
    assert torch.allclose(v[:, 0], expected)  # v varies along rows


def test_grid_odd_size_and_nonsquare() -> None:
    u, v = frequency_grid(3, 5)
    assert u.shape == v.shape == (3, 5)
    assert torch.allclose(u[0], torch.tensor([0.0, 0.2, 0.4, -0.4, -0.2]))
    assert torch.allclose(v[:, 0], torch.tensor([0.0, 1 / 3, -1 / 3]))


def test_grid_range_and_dc() -> None:
    u, v = frequency_grid(64, 48)
    assert u[0, 0] == 0 and v[0, 0] == 0
    assert u.min() == -0.5 and u.max() < 0.5
    assert v.min() == -0.5 and v.max() < 0.5


def test_radius_dc_and_values() -> None:
    r = frequency_radius(4, 4)
    assert r[0, 0] == 1.0
    assert frequency_radius(4, 4, dc_value=0.0)[0, 0] == 0.0
    assert r[0, 1] == pytest.approx(0.25)
    assert r[1, 1] == pytest.approx(0.25 * math.sqrt(2))
    assert r[2, 2] == pytest.approx(0.5 * math.sqrt(2))
    assert torch.isfinite(r).all() and (r > 0).all()


def test_radius_symmetric() -> None:
    r = frequency_radius(8, 8, dc_value=0.0)
    flipped = torch.roll(torch.flip(r, dims=(0, 1)), shifts=(1, 1), dims=(0, 1))
    assert torch.allclose(r, flipped)


def test_angle_axes_and_diagonals() -> None:
    th = frequency_angle(4, 4)
    assert th[0, 0] == 0.0
    assert th[0, 1] == pytest.approx(0.0)  # +u axis
    assert th[1, 0] == pytest.approx(math.pi / 2)  # +v axis
    assert th[1, 1] == pytest.approx(math.pi / 4)
    assert th[3, 3] == pytest.approx(-3 * math.pi / 4)
    assert th.abs().max() <= math.pi + 1e-6


def test_dtype_and_device() -> None:
    u, v = frequency_grid(4, 6, dtype=torch.float64, device="cpu")
    assert u.dtype == v.dtype == torch.float64 and u.device.type == "cpu"
    assert frequency_radius(4, 6, dtype=torch.float64).dtype == torch.float64


def test_invalid_size() -> None:
    with pytest.raises(ValueError):
        frequency_grid(0, 4)
