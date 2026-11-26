"""Tests for parameter and FLOP counting."""

import math

import pytest
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from monogenet.complexity import (
    Complexity,
    bank_flops,
    compare_with_conv,
    conv2d_flops,
    count_parameters,
    estimate_flops,
    fft_flops,
    front_end_flops,
    phase_block_flops,
)
from monogenet.layers.frontend import MonogenicFrontEnd
from monogenet.layers.phase_block import PhaseBlock


def test_count_parameters() -> None:
    conv = nn.Conv2d(3, 8, 3)
    assert count_parameters(conv) == 3 * 8 * 9 + 8
    conv.bias.requires_grad_(False)
    assert count_parameters(conv) == 3 * 8 * 9 and count_parameters(conv, trainable_only=False) == 3 * 8 * 9 + 8
    assert count_parameters(MonogenicFrontEnd()) == 9


def test_fft_flops_known_values() -> None:
    assert fft_flops(8, 8) == 5 * 64 * 6
    assert fft_flops(1, 1) == 0
    assert fft_flops(16, 32) == int(5 * 512 * 9)


def test_conv2d_flops_hand_values() -> None:
    assert conv2d_flops(3, 8, 3, 10, 10) == 2 * 3 * 9 * 8 * 100
    assert conv2d_flops(3, 8, 3, 10, 10, bias=True) == 2 * 3 * 9 * 8 * 100 + 8 * 100
    assert conv2d_flops(4, 4, 1, 5, 6) == 2 * 4 * 4 * 30
    assert conv2d_flops(8, 8, 3, 10, 10, groups=8) == 2 * 1 * 9 * 8 * 100  # depthwise
    assert conv2d_flops(3, 8, (3, 5), 10, 10) == 2 * 3 * 15 * 8 * 100
    assert conv2d_flops(3, 8, 3, 10, 10, stride=2) == 2 * 3 * 9 * 8 * 25


@pytest.mark.parametrize(
    "kwargs,shape",
    [
        (dict(in_channels=3, out_channels=8, kernel_size=3, padding=1), (1, 3, 16, 16)),
        (dict(in_channels=4, out_channels=4, kernel_size=1), (1, 4, 9, 11)),
        (dict(in_channels=8, out_channels=8, kernel_size=5, padding=2, groups=8), (1, 8, 12, 12)),
        (dict(in_channels=2, out_channels=6, kernel_size=3, padding=1, stride=2), (1, 2, 16, 16)),
    ],
)
def test_conv_flops_match_torch_flop_counter(kwargs: dict, shape: tuple) -> None:
    conv = nn.Conv2d(**kwargs, bias=False)
    with FlopCounterMode(display=False) as counter:
        conv(torch.rand(*shape))
    est = estimate_flops(conv, shape)
    assert est.flops == counter.get_total_flops()
    assert est.params == count_parameters(conv)


def test_phase_block_mixing_matches_torch_counter_and_total_adds_up() -> None:
    block = PhaseBlock(4, 6, wavelengths=(4.0, 6.0), bias=False)
    with FlopCounterMode(display=False) as counter:
        block(torch.rand(1, 4, 16, 16))
    mixing = conv2d_flops(block.n_feature_channels, 6, 1, 16, 16)
    assert counter.get_total_flops() == mixing  # torch counts only the 1x1 conv (no FFT model)
    est = phase_block_flops(block, 16, 16)
    assert est.flops > mixing and est.params == count_parameters(block)
    spectral = 4 * fft_flops(16, 16) + 3 * 4 * 2 * fft_flops(16, 16)
    assert est.flops > spectral + mixing  # FFTs + products + features + mixing


def test_phase_block_params_vs_conv_follow_the_formulas() -> None:
    c, o, s = 64, 64, 2
    block = PhaseBlock(c, o, wavelengths=(4.0, 8.0))
    out = compare_with_conv(block, 32, 32, kernel_size=3)
    assert out["conv2d"].params == c * o * 9 + o
    assert out["phase_block"].params == 3 * s * c + c * s * 2 * o + o
    assert out["params_ratio"] == pytest.approx(out["conv2d"].params / out["phase_block"].params)
    assert out["params_ratio"] > 1.0  # fewer parameters than a 3x3 conv (2 scales x 2 features)
    large_k = compare_with_conv(block, 32, 32, kernel_size=7)
    assert large_k["params_ratio"] > out["params_ratio"] * 5  # the gap widens with kernel size
    assert large_k["flops_ratio"] > out["flops_ratio"]


def test_scaling_with_resolution_and_channels() -> None:
    block = PhaseBlock(8, 8, wavelengths=(4.0, 6.0))
    small, large = phase_block_flops(block, 32, 32), phase_block_flops(block, 64, 64)
    assert 4.0 < large.flops / small.flops < 4.6  # N log N: a bit more than 4x for 4x pixels
    assert small.params == large.params  # parameters do not depend on resolution
    more_scales = PhaseBlock(8, 8, wavelengths=(4.0, 6.0, 9.0, 13.0))
    assert phase_block_flops(more_scales, 32, 32).flops > 1.5 * small.flops


def test_feature_selection_changes_cost() -> None:
    base = PhaseBlock(4, 4, features=("ps",), wavelengths=(4.0, 6.0))
    more = PhaseBlock(4, 4, features=("ps", "pa", "energy", "phase", "pc"), wavelengths=(4.0, 6.0))
    assert phase_block_flops(more, 32, 32).flops > phase_block_flops(base, 32, 32).flops
    assert phase_block_flops(more, 32, 32).params > phase_block_flops(base, 32, 32).params


def test_front_end_and_dispatch() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), in_channels=3, min_wavelength=3.0)
    est = estimate_flops(layer, (8, 3, 64, 64))  # batch size ignored
    assert est == front_end_flops(layer, 64, 64) and est.params == 6
    assert estimate_flops(layer, (3, 64, 64)) == est
    per = MonogenicFrontEnd((6.0, 9.0), in_channels=3, per_channel=True, min_wavelength=3.0)
    assert estimate_flops(per, (3, 64, 64)).params == 18 and estimate_flops(per, (3, 64, 64)).flops == est.flops
    with pytest.raises(TypeError):
        estimate_flops(nn.ReLU(), (1, 3, 8, 8))


def test_bank_flops_and_dataclass() -> None:
    assert bank_flops(3, 2, 8, 8) == 3 * 2 * 64 * 20
    c = Complexity(10, 20)
    assert (c.params, c.flops) == (10, 20)
    with pytest.raises(Exception):
        c.params = 5  # frozen
    assert math.isfinite(compare_with_conv(PhaseBlock(2, 2, wavelengths=(4.0,)), 16, 16)["flops_ratio"])
