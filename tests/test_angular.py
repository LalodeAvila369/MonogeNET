"""Tests for the angular spread function."""

import math

import pytest
import torch

from monogenet.filters.angular import angular_spread
from monogenet.filters.grids import frequency_angle


def test_peak_on_orientation_axis() -> None:
    # 64 px grid: u axis is theta=0, v axis is theta=pi/2.
    s0 = angular_spread(64, 64, 0.0, 0.3)
    assert s0[0, 5] == pytest.approx(1.0)  # +u axis
    s90 = angular_spread(64, 64, math.pi / 2, 0.3)
    assert s90[5, 0] == pytest.approx(1.0)  # +v axis


def test_gaussian_value_and_wrapping() -> None:
    sigma = 0.4
    s = angular_spread(64, 64, 0.0, sigma)
    # 45 degrees: u = v, so angle = pi/4.
    assert s[5, 5] == pytest.approx(math.exp(-((math.pi / 4) ** 2) / (2 * sigma**2)), rel=1e-5)
    # Opposite direction (-u axis, theta = pi): distance pi, tiny but not wrapped to 0.
    assert s[0, -5] == pytest.approx(math.exp(-(math.pi**2) / (2 * sigma**2)), rel=1e-4)
    # Wrapping: orientation just below pi, angle just above -pi are close.
    s_pi = angular_spread(64, 64, math.pi - 0.01, sigma)
    assert s_pi[0, -5] > 0.99


def test_symmetric_responds_both_directions() -> None:
    s = angular_spread(64, 64, 0.0, 0.3, symmetric=True)
    assert s[0, 5] == pytest.approx(1.0)
    assert s[0, -5] == pytest.approx(1.0)  # theta = pi
    assert s[5, 0] == pytest.approx(math.exp(-((math.pi / 2) ** 2) / (2 * 0.3**2)), rel=1e-4)
    # Symmetric spread is point-symmetric in frequency -> real spatial kernel.
    k = torch.fft.ifft2(s)
    assert k.imag.abs().max() < 1e-6


def test_range_and_shape() -> None:
    s = angular_spread(32, 48, 1.0, 0.5)
    assert s.shape == (32, 48)
    assert (s > 0).all() and (s <= 1 + 1e-6).all()


def test_orientation_rotates_peak() -> None:
    th = frequency_angle(64, 64)
    s = angular_spread(64, 64, math.pi / 4, 0.2)
    idx = s[1:, 1:].argmax()  # skip DC row/col artefacts
    i, j = divmod(int(idx), 63)
    assert th[1 + i, 1 + j] == pytest.approx(math.pi / 4, abs=0.1)


def test_differentiable() -> None:
    o = torch.tensor(0.5, requires_grad=True)
    sg = torch.tensor(0.4, requires_grad=True)
    angular_spread(16, 16, o, sg).sum().backward()
    assert torch.isfinite(o.grad) and torch.isfinite(sg.grad)


def test_invalid_sigma() -> None:
    with pytest.raises(ValueError):
        angular_spread(16, 16, 0.0, 0.0)
