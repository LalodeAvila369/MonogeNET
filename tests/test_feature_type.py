"""Tests for the feature type atan2(even, |odd|)."""

import math

import pytest
import torch

from monogenet.features.local import feature_type, local_phase
from monogenet.features.monogenic import monogenic_signal
from monogenet.filters.log_gabor import log_gabor_bank

D = torch.float64


def _mono(e: float, o1: float, o2: float) -> torch.Tensor:
    return torch.tensor([e, o1, o2], dtype=D).view(1, 3, 1, 1)


def test_special_values() -> None:
    assert feature_type(_mono(1.0, 0.0, 0.0)).item() == pytest.approx(math.pi / 2)  # bright line
    assert feature_type(_mono(-1.0, 0.0, 0.0)).item() == pytest.approx(-math.pi / 2)  # dark line
    assert feature_type(_mono(0.0, 0.6, -0.8)).item() == pytest.approx(0.0)  # step edge
    assert feature_type(_mono(1.0, 3.0, 4.0)).item() == pytest.approx(math.atan2(1.0, 5.0))
    assert feature_type(_mono(2.0, 2.0, 0.0)).item() == pytest.approx(math.pi / 4)


def test_complement_of_local_phase() -> None:
    g = torch.Generator().manual_seed(0)
    mono = torch.randn(2, 3, 3, 12, 12, generator=g, dtype=D)
    assert torch.allclose(feature_type(mono), math.pi / 2 - local_phase(mono), atol=1e-12)


def test_range() -> None:
    g = torch.Generator().manual_seed(1)
    ft = feature_type(torch.randn(2, 3, 3, 12, 12, generator=g, dtype=D))
    assert (ft >= -math.pi / 2 - 1e-12).all() and (ft <= math.pi / 2 + 1e-12).all()


def test_bright_line_dark_line_and_edge_on_images() -> None:
    n = 128
    bank = log_gabor_bank(n, n, 1, 8.0, 1.0, dtype=D)
    line = torch.zeros(n, n, dtype=D)
    line[:, n // 2] = 1.0
    assert feature_type(monogenic_signal(line, bank))[0, 10, n // 2] > math.pi / 2 - 0.05
    assert feature_type(monogenic_signal(-line, bank))[0, 10, n // 2] < -math.pi / 2 + 0.05
    step = torch.zeros(n, n, dtype=D)
    step[:, n // 4 : 3 * n // 4] = 1.0
    ft = feature_type(monogenic_signal(step, bank))[0]
    col = n // 4  # the edge lies half a pixel between col-1 and col
    assert ft[10, col - 1] == pytest.approx(-ft[10, col].item(), abs=1e-9)
    assert abs(ft[10, col]) < math.pi / 3


def test_contrast_invariant_and_polarity_flips_sign() -> None:
    img = torch.randn(32, 32, generator=torch.Generator().manual_seed(2), dtype=D)
    bank = log_gabor_bank(32, 32, 2, 4.0, 1.8, dtype=D)
    a = feature_type(monogenic_signal(img, bank))
    assert torch.allclose(feature_type(monogenic_signal(6.0 * img, bank)), a, atol=1e-9)
    assert torch.allclose(feature_type(monogenic_signal(-img, bank)), -a, atol=1e-9)


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        feature_type(torch.rand(2, 8, 8))
