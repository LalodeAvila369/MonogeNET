"""Tests for soft floors and the learnable noise threshold."""

import math

import pytest
import torch

from monogenet.features.congruency import phase_congruency
from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import phase_asymmetry, phase_symmetry
from monogenet.features.threshold import LearnableThreshold, noise_threshold, soft_floor
from monogenet.filters.log_gabor import log_gabor_bank

D = torch.float64


def _scene_mono(seed: int = 0, n: int = 64, scales: int = 3) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(n, dtype=D), torch.arange(n, dtype=D), indexing="ij")
    img = torch.zeros(n, n, dtype=D)
    img[n // 4 : n // 2, n // 4 : n // 2] = 0.6
    img[:, int(n * 0.7)] += 0.8
    img += 0.9 * torch.exp(-(((x - 0.8 * n) ** 2 + (y - 0.75 * n) ** 2) / (2 * 3.0**2)))
    img += 0.03 * torch.randn(n, n, generator=torch.Generator().manual_seed(seed), dtype=D)
    bank = log_gabor_bank(n, n, scales, 4.0, 1.5, dtype=D)
    return monogenic_signal(img, bank)


def test_soft_floor_known_values() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=D)
    assert torch.equal(soft_floor(x), torch.tensor([0.0, 0.0, 1.0], dtype=D))
    soft = soft_floor(x, sharpness=10.0)
    assert soft[1].item() == pytest.approx(math.log(2) / 10.0)  # maximal error at 0
    assert soft[2].item() == pytest.approx(math.log1p(math.exp(-10.0)) / 10.0 + 1.0)
    assert (soft > 0).all()
    assert (soft_floor(x, 1e4) - soft_floor(x)).abs().max() < math.log(2) / 1e4 + 1e-12
    with pytest.raises(ValueError):
        soft_floor(x, 0.0)


def test_soft_gradient_nonzero_where_hard_is_zero() -> None:
    x = torch.tensor([-0.5], dtype=D, requires_grad=True)
    soft_floor(x).sum().backward()
    assert x.grad.item() == 0.0
    x2 = torch.tensor([-0.5], dtype=D, requires_grad=True)
    soft_floor(x2, 5.0).sum().backward()
    assert x2.grad.item() > 0.0


@pytest.mark.parametrize("fn", [phase_symmetry, phase_asymmetry, phase_congruency])
def test_soft_features_converge_to_hard(fn) -> None:
    mono = _scene_mono()
    hard = fn(mono)
    soft = fn(mono, sharpness=1e5)
    assert (soft - hard).abs().max() < math.log(2) / 1e5 * 1.01
    assert (soft >= hard - 1e-12).all()  # softplus >= relu
    # A moderate sharpness smooths but keeps the structure.
    mid = fn(mono, sharpness=30.0)
    assert (mid - hard).abs().max() < math.log(2) / 30.0 * 1.01
    assert torch.corrcoef(torch.stack([mid.flatten(), hard.flatten()]))[0, 1] > 0.95


@pytest.mark.parametrize("fn", [phase_symmetry, phase_asymmetry, phase_congruency])
def test_soft_features_keep_contrast_invariance(fn) -> None:
    img = torch.randn(48, 48, generator=torch.Generator().manual_seed(1), dtype=D)
    bank = log_gabor_bank(48, 48, 3, 4.0, 1.5, dtype=D)
    a = fn(monogenic_signal(img, bank), sharpness=20.0, eps=1e-12)
    b = fn(monogenic_signal(8.0 * img, bank), sharpness=20.0, eps=1e-12)
    assert torch.allclose(a, b, atol=1e-6)


def test_soft_ps_gives_gradient_below_threshold() -> None:
    # A pixel below threshold has zero gradient with the hard floor but not with the soft one.
    mono = torch.tensor([0.1, 0.01, 0.0], dtype=D).view(1, 3, 1, 1).requires_grad_(True)
    phase_symmetry(mono, threshold=1.0).sum().backward()
    assert mono.grad.abs().sum() == 0
    mono2 = torch.tensor([0.1, 0.01, 0.0], dtype=D).view(1, 3, 1, 1).requires_grad_(True)
    phase_symmetry(mono2, threshold=1.0, sharpness=10.0).sum().backward()
    assert mono2.grad.abs().sum() > 0


def test_learnable_threshold_init_matches_thesis() -> None:
    mono = _scene_mono()
    lt = LearnableThreshold(3).double()
    assert torch.allclose(lt.gain, torch.ones(3, 1, 1, dtype=D))
    assert torch.allclose(lt(mono), noise_threshold(mono, "mean"), rtol=1e-6)
    assert torch.allclose(phase_symmetry(mono, lt(mono)), phase_symmetry(mono), atol=1e-6)
    lt2 = LearnableThreshold(3, init_gain=2.05, stat="geometric", shared=True).double()
    assert lt2.gain.item() == pytest.approx(2.05, rel=1e-6)
    assert torch.allclose(lt2(mono), 2.05 * noise_threshold(mono, "geometric"), rtol=1e-6)


def test_learnable_threshold_parameters_and_positivity() -> None:
    assert sum(p.numel() for p in LearnableThreshold(3).parameters()) == 3
    assert sum(p.numel() for p in LearnableThreshold(3, shared=True).parameters()) == 1
    lt = LearnableThreshold(2)
    with torch.no_grad():
        lt.raw_gain.fill_(-50.0)
    assert (lt.gain > 0).all()


def test_learnable_threshold_scales_with_contrast() -> None:
    mono = _scene_mono()
    lt = LearnableThreshold(3, init_gain=1.7).double()
    assert torch.allclose(lt(5.0 * mono), 5.0 * lt(mono), rtol=1e-9)


def test_gradient_descent_recovers_gain() -> None:
    # Known answer: a target PS map made with gain 2.0 is recovered from an initial gain of 1.0.
    mono = _scene_mono()
    sharp = 30.0
    target = phase_symmetry(mono, noise_threshold(mono, gain=2.0), sharpness=sharp)
    lt = LearnableThreshold(3, init_gain=1.0).double()
    opt = torch.optim.Adam(lt.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = (phase_symmetry(mono, lt(mono), sharpness=sharp) - target).pow(2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-8
    assert torch.allclose(lt.gain, torch.full((3, 1, 1), 2.0, dtype=D), rtol=0.03)


def test_learnable_threshold_validation() -> None:
    mono = _scene_mono(scales=3)
    with pytest.raises(ValueError):
        LearnableThreshold(2).double()(mono)
    with pytest.raises(ValueError):
        LearnableThreshold(0)
    with pytest.raises(ValueError):
        LearnableThreshold(2, init_gain=0.0)
    with pytest.raises(ValueError):
        LearnableThreshold(2, stat="median")
