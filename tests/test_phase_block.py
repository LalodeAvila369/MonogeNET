"""Tests for PhaseBlock."""

import pytest
import torch
from torch import nn

from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import phase_asymmetry, phase_symmetry
from monogenet.layers.phase_block import PhaseBlock

KW = dict(wavelengths=(4.0, 6.0), min_wavelength=2.0)


def _x(b: int = 2, c: int = 3, n: int = 32, seed: int = 0) -> torch.Tensor:
    return torch.randn(b, c, n, n, generator=torch.Generator().manual_seed(seed))


def test_shape_and_feature_channel_count() -> None:
    block = PhaseBlock(3, 8, **KW)
    assert block(_x()).shape == (2, 8, 32, 32)
    assert block.n_feature_channels == 3 * 2 * 2  # C * S * (ps, pa)
    assert block.phase_features(_x()).shape == (2, 12, 32, 32)
    full = PhaseBlock(3, 8, features=("ps", "pa", "energy", "phase", "pc"), **KW)
    assert full.n_feature_channels == 3 * (2 * 4 + 1)
    assert full.phase_features(_x()).shape == (2, 27, 32, 32) and full(_x()).shape == (2, 8, 32, 32)


def test_non_square_odd_sizes_and_batch_of_one() -> None:
    assert PhaseBlock(2, 4, **KW)(torch.rand(1, 2, 21, 34)).shape == (1, 4, 21, 34)


def test_parameter_count_formula() -> None:
    c, o, s = 5, 7, 2
    block = PhaseBlock(c, o, **KW)
    filtering = 3 * s * c  # f0, bandwidth, gain per scale per channel
    mixing = c * s * 2 * o + o  # 1x1 conv over C * S * F maps, with bias
    assert sum(p.numel() for p in block.parameters()) == filtering + mixing
    shared = PhaseBlock(c, o, per_channel=False, bias=False, **KW)
    assert sum(p.numel() for p in shared.parameters()) == 3 * s + c * s * 2 * o


def test_matches_manual_composition() -> None:
    block = PhaseBlock(2, 3, sharpness=None, **KW).double()
    x = _x(2, 2).double()
    mono = monogenic_signal(x, block.bank(32, 32))
    thr = block.threshold(mono)
    ps, pa = phase_symmetry(mono, thr, 1e-6, None), phase_asymmetry(mono, thr, 1e-6, None)
    expected_maps = torch.stack([ps, pa], dim=3).flatten(1, 3)  # (B, C*S*2, H, W): scale-major, then feature
    feats = block.phase_features(x)
    assert torch.allclose(feats, expected_maps, atol=1e-10)
    assert torch.allclose(block(x), nn.functional.conv2d(expected_maps, block.mix.weight, block.mix.bias), atol=1e-10)


def test_feature_order_follows_the_features_argument() -> None:
    a = PhaseBlock(1, 1, features=("ps", "pa"), sharpness=None, **KW).double()
    b = PhaseBlock(1, 1, features=("pa", "ps"), sharpness=None, **KW).double()
    x = _x(1, 1).double()
    fa, fb = a.phase_features(x), b.phase_features(x)
    assert torch.allclose(fa[:, 0], fb[:, 1]) and torch.allclose(fa[:, 1], fb[:, 0])  # swapped per scale


def test_filtering_is_contrast_invariant_with_phase_only_features() -> None:
    block = PhaseBlock(3, 4, features=("ps", "pa", "pc", "phase"), **KW).double()
    x = _x(2, 3).double()
    scale = torch.tensor([0.1, 3.0, 40.0], dtype=torch.float64).view(1, 3, 1, 1)  # per-channel contrast
    assert torch.allclose(block.phase_features(x), block.phase_features(x * scale), atol=1e-5)
    assert torch.allclose(block(x), block(x * scale), atol=1e-5)
    energy = PhaseBlock(3, 4, features=("energy",), **KW).double()
    assert not torch.allclose(energy.phase_features(x), energy.phase_features(x * scale), atol=1e-3)


def test_channels_are_filtered_independently_before_mixing() -> None:
    block = PhaseBlock(3, 4, **KW).double()
    x = _x(1, 3).double()
    ref = block.phase_features(x)
    x2 = x.clone()
    x2[:, 1] = torch.randn(32, 32, dtype=torch.float64)  # replace channel 1
    out = block.phase_features(x2)
    per = 2 * 2  # S * F maps per input channel
    assert torch.allclose(out[:, :per], ref[:, :per], atol=1e-10)
    assert torch.allclose(out[:, 2 * per :], ref[:, 2 * per :], atol=1e-10)
    assert not torch.allclose(out[:, per : 2 * per], ref[:, per : 2 * per], atol=1e-3)
    assert not torch.allclose(block(x2), block(x), atol=1e-3)  # but the 1x1 mixing couples them


@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("features", [("ps", "pa"), ("ps", "pa", "energy", "phase", "pc")])
def test_gradients_reach_every_parameter(per_channel: bool, features) -> None:
    block = PhaseBlock(3, 4, features=features, per_channel=per_channel, **KW)
    x = _x().requires_grad_(True)
    block(x).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for n, p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n


def test_stacked_blocks_with_nonlinearity_train() -> None:
    net = nn.Sequential(
        PhaseBlock(1, 6, **KW), nn.GroupNorm(2, 6), nn.ReLU(), PhaseBlock(6, 4, **KW), nn.ReLU()
    )
    x = _x(2, 1)
    net(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_blank_input_is_finite() -> None:
    block = PhaseBlock(2, 3, features=("ps", "pa", "pc", "phase"), **KW)
    x = torch.zeros(1, 2, 24, 24, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in block.parameters())


def test_batch_independence_taper_and_dtype() -> None:
    block = PhaseBlock(2, 3, taper=0.1, **KW)
    x = _x(3, 2)
    assert torch.allclose(block(x)[1:2], block(x[1:2]), atol=1e-6)
    assert block.double()(x.double()).dtype == torch.float64


def test_learns_a_contrast_robust_line_detector() -> None:
    # Task: segment thin vertical/horizontal lines whose contrast varies wildly between samples.
    torch.manual_seed(0)
    n, b = 32, 16

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(b, 1, n, n)
        mask = torch.zeros(b, 1, n, n)
        for i in range(b):
            pos = int(torch.randint(6, n - 6, (1,)))
            if i % 2 == 0:
                mask[i, 0, :, pos] = 1.0
            else:
                mask[i, 0, pos, :] = 1.0
            x[i] = mask[i] * 10 ** float(torch.empty(1).uniform_(-2, 1))  # contrast over 3 decades
        return x + 0.02 * x.amax(dim=(1, 2, 3), keepdim=True) * torch.randn_like(x), mask

    net = nn.Sequential(PhaseBlock(1, 4, wavelengths=(4.0, 6.0, 9.0)), nn.ReLU(), nn.Conv2d(4, 1, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0))
    losses = []
    for _ in range(80):
        x, y = batch()
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-10:]) / 10 < 0.5 * sum(losses[:10]) / 10


def test_validation() -> None:
    block = PhaseBlock(3, 4, **KW)
    with pytest.raises(ValueError):
        block(torch.rand(2, 2, 16, 16))  # wrong channel count
    with pytest.raises(ValueError):
        block(torch.rand(3, 16, 16))
    for bad in (("ps", "ps"), ("foo",), ()):
        with pytest.raises(ValueError):
            PhaseBlock(3, 4, features=bad, **KW)
    with pytest.raises(ValueError):
        PhaseBlock(0, 4, **KW)
