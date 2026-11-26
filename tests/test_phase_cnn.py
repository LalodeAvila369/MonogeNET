"""Tests for the CNN with PhaseBlocks replacing convolutions."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from monogenet.complexity import count_parameters
from monogenet.layers import PhaseBlock
from monogenet.models import PhaseCNN, SmallCNN
from monogenet.train import TrainConfig, fit


def _filters(net: PhaseCNN) -> list[nn.Module]:
    return [stage[0] for stage in net.features]


def test_all_stages_phase_by_default_and_shape() -> None:
    net = PhaseCNN(1, 4, width=4)
    assert all(isinstance(f, PhaseBlock) for f in _filters(net)) and net.phase_stages == (0, 1, 2, 3)
    assert net(torch.rand(2, 1, 64, 64)).shape == (2, 4)


def test_selected_stages_mix_phase_and_conv() -> None:
    net = PhaseCNN(1, 2, width=4, phase_stages=(2, 3))
    kinds = [type(f) for f in _filters(net)]
    assert kinds == [nn.Conv2d, nn.Conv2d, PhaseBlock, PhaseBlock]
    assert net(torch.rand(1, 1, 64, 64)).shape == (1, 2)
    none = PhaseCNN(1, 2, width=4, phase_stages=())
    assert all(isinstance(f, nn.Conv2d) for f in _filters(none))


def test_no_phase_stages_equals_small_cnn_parameter_count() -> None:
    assert count_parameters(PhaseCNN(1, 2, width=8, phase_stages=())) == count_parameters(SmallCNN(1, 2, width=8))


def test_phase_stages_use_fewer_parameters_than_convs() -> None:
    conv = count_parameters(PhaseCNN(1, 2, width=16, phase_stages=()))
    phase = count_parameters(PhaseCNN(1, 2, width=16))
    assert phase < conv  # the deep 3x3 convs dominate the baseline


def test_gradients_reach_phase_parameters_and_kwargs_pass_through() -> None:
    net = PhaseCNN(1, 2, width=4, depth=3, features=("ps", "pa", "energy"), taper=0.1)
    net(torch.rand(4, 1, 32, 32)).sum().backward()
    for n, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert _filters(net)[0].features == ("ps", "pa", "energy")


def test_learns_contrast_varying_lines() -> None:
    torch.manual_seed(0)
    n = 48
    y = torch.arange(n) % 2
    x = torch.zeros(n, 1, 32, 32)
    for i in range(n):
        pos = 6 + (i * 7) % 20
        if y[i]:
            x[i, 0, :, pos] = 1.0
        else:
            x[i, 0, pos, :] = 1.0
        x[i] *= 10 ** (-2 + 3 * (i % 5) / 4)
    x = x + 0.01 * x.amax(dim=(1, 2, 3), keepdim=True) * torch.randn_like(x)
    loader = DataLoader(TensorDataset(x, y.view(-1, 1)), batch_size=16)
    hist = fit(PhaseCNN(1, 2, width=4, depth=3, phase_stages=(0,)), loader, loader, TrainConfig(epochs=30, lr=0.01))
    assert hist[-1]["val_acc"] >= 0.9


def test_validation() -> None:
    with pytest.raises(ValueError):
        PhaseCNN(depth=3, phase_stages=(3,))
