"""Tests for phase asymmetry."""

import math

import pytest
import torch

from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import phase_asymmetry, phase_symmetry
from monogenet.filters.log_gabor import log_gabor_bank

D = torch.float64


def _mono(e: float, o1: float, o2: float) -> torch.Tensor:
    return torch.tensor([e, o1, o2], dtype=D).view(1, 3, 1, 1)


def test_hand_computed_value() -> None:
    m = _mono(1.0, 3.0, 0.0)
    pa = phase_asymmetry(m, threshold=0.5, eps=0.0).item()
    assert pa == pytest.approx((3.0 - 1.0 - 0.5) / math.sqrt(10.0))
    assert phase_asymmetry(_mono(3.0, 1.0, 0.0), 0.0, 0.0).item() == 0.0  # even dominates
    # Odd magnitude uses both components.
    assert phase_asymmetry(_mono(0.0, 3.0, 4.0), 0.0, 0.0).item() == pytest.approx(1.0)


def test_ps_and_pa_are_mutually_exclusive() -> None:
    g = torch.Generator().manual_seed(0)
    mono = torch.randn(2, 3, 3, 16, 16, generator=g, dtype=D)
    ps, pa = phase_symmetry(mono), phase_asymmetry(mono)
    assert ((ps > 0) & (pa > 0)).sum() == 0
    assert ((ps > 0) | (pa > 0)).any()


def test_pa_is_ps_of_swapped_roles() -> None:
    # PS(e, o) with e and |o| exchanged equals PA: check on a configuration built by hand.
    ps = phase_symmetry(_mono(4.0, 3.0, 0.0), 0.0, 0.0).item()  # (4-3)/5
    pa = phase_asymmetry(_mono(3.0, 4.0, 0.0), 0.0, 0.0).item()  # (4-3)/5
    assert ps == pytest.approx(pa) == pytest.approx(0.2)


def test_step_edge_asymmetric_line_not() -> None:
    n = 128
    bank = log_gabor_bank(n, n, 1, 8.0, 1.0, dtype=D)
    step = torch.zeros(n, n, dtype=D)
    step[:, n // 4 : 3 * n // 4] = 1.0
    pa_step = phase_asymmetry(monogenic_signal(step, bank), eps=1e-9)[0]
    col = n // 4
    assert pa_step[10, col] > 0.2 and pa_step[10, col - 1] == pytest.approx(pa_step[10, col].item(), abs=1e-9)
    assert pa_step[10, n // 2] < 1e-3  # flat interior
    line = torch.zeros(n, n, dtype=D)
    line[:, n // 2] = 1.0
    pa_line = phase_asymmetry(monogenic_signal(line, bank), eps=1e-9)[0]
    assert pa_line[10, n // 2] < 1e-3  # centre of a line is symmetric


def test_contrast_invariance_and_polarity() -> None:
    g = torch.Generator().manual_seed(1)
    img = torch.randn(48, 48, generator=g, dtype=D)
    bank = log_gabor_bank(48, 48, 2, 4.0, 1.5, dtype=D)
    a = phase_asymmetry(monogenic_signal(img, bank), eps=1e-12)
    b = phase_asymmetry(monogenic_signal(9.0 * img, bank), eps=1e-12)
    c = phase_asymmetry(monogenic_signal(-img, bank), eps=1e-12)
    assert torch.allclose(a, b, atol=1e-8)
    assert torch.allclose(a, c, atol=1e-8)  # rising and falling edges are both asymmetric


def test_range_shape_and_gradients() -> None:
    img = torch.rand(2, 1, 24, 24, dtype=D, requires_grad=True)
    bank = log_gabor_bank(24, 24, 3, 4.0, 1.5, dtype=D)
    pa = phase_asymmetry(monogenic_signal(img, bank))
    assert pa.shape == (2, 1, 3, 24, 24)
    assert (pa >= 0).all() and (pa <= 1 + 1e-9).all()
    pa.sum().backward()
    assert torch.isfinite(img.grad).all()


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        phase_asymmetry(torch.rand(2, 8, 8))
