"""Tests for the learnable log-Gabor bank parameterisation."""

import pytest
import torch

from monogenet.filters.learnable import LearnableLogGaborBank
from monogenet.filters.log_gabor import radial_log_gabor


def test_init_reproduces_wavelengths_and_sigma() -> None:
    m = LearnableLogGaborBank((36.0, 39.0, 42.0), 0.55)
    assert torch.allclose(m.wavelengths, torch.tensor([36.0, 39.0, 42.0]), rtol=1e-4)
    assert torch.allclose(m.sigma_onf, torch.full((3,), 0.55), rtol=1e-5)


def test_forward_matches_fixed_filters_at_init() -> None:
    m = LearnableLogGaborBank((8.0, 12.0), (0.5, 0.6), min_wavelength=2.0, max_wavelength=64.0)
    out = m(48, 40)
    assert out.shape == (2, 48, 40)
    for s, (wl, sg) in enumerate(((8.0, 0.5), (12.0, 0.6))):
        assert torch.allclose(out[s], radial_log_gabor(48, 40, wl, sg), atol=1e-4)


def test_bounds_hold_for_extreme_parameters() -> None:
    m = LearnableLogGaborBank((8.0, 12.0), 0.5, min_wavelength=4.0, max_wavelength=32.0)
    for value in (-1e3, -50.0, 50.0, 1e3):
        with torch.no_grad():
            m.raw_f0.fill_(value)
            m.log_bandwidth.fill_(value)
        assert (m.wavelengths >= 4.0 - 1e-3).all() and (m.wavelengths <= 32.0 + 1e-3).all()
        assert (m.sigma_onf >= 0).all() and (m.sigma_onf <= 1).all()


def test_sigma_strictly_in_unit_interval_at_moderate_values() -> None:
    m = LearnableLogGaborBank()
    with torch.no_grad():
        m.log_bandwidth.copy_(torch.tensor([-5.0, 0.0, 3.0]))
    assert ((m.sigma_onf > 0) & (m.sigma_onf < 1)).all()


def test_parameter_count_and_freezing() -> None:
    m = LearnableLogGaborBank((8.0, 12.0, 16.0))
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 6
    frozen = LearnableLogGaborBank((8.0, 12.0), learn_f0=False, learn_sigma=False)
    assert sum(p.numel() for p in frozen.parameters() if p.requires_grad) == 0


def test_gradients_flow_to_both_parameters() -> None:
    m = LearnableLogGaborBank((8.0, 12.0))
    m(32, 32).pow(2).sum().backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_gradient_descent_recovers_target_filter() -> None:
    # Known answer: fit a bank initialised at (10 px, 0.55) to a target (14 px, 0.45).
    target = radial_log_gabor(64, 64, 14.0, 0.45)[None]
    m = LearnableLogGaborBank((10.0,), 0.55, min_wavelength=3.0, max_wavelength=40.0)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = (m(64, 64) - target).pow(2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-5
    assert m.wavelengths.item() == pytest.approx(14.0, rel=0.03)
    assert m.sigma_onf.item() == pytest.approx(0.45, abs=0.02)


def test_dtype_conversion_and_state_dict() -> None:
    m = LearnableLogGaborBank((8.0,)).double()
    assert m(16, 16).dtype == torch.float64
    m2 = LearnableLogGaborBank((20.0,))
    m2.load_state_dict(LearnableLogGaborBank((8.0,)).state_dict())
    assert m2.wavelengths.item() == pytest.approx(8.0, rel=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(wavelengths=(1.0,)),
        dict(wavelengths=(200.0,)),
        dict(sigma_onf=1.0),
        dict(wavelengths=()),
        dict(min_wavelength=10.0, max_wavelength=5.0),
    ],
)
def test_invalid_args(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LearnableLogGaborBank(**kwargs)
