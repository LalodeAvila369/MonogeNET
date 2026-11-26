"""Tests for ResNet-18 with a MonogenicFrontEnd."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from monogenet.complexity import count_parameters
from monogenet.models import MonogenicResNet18, ResNet18
from monogenet.train import TrainConfig, fit

KW = dict(wavelengths=(4.0, 8.0), stem="small")


def test_shapes_and_channel_counts() -> None:
    net = MonogenicResNet18(1, 4, **KW)
    assert net(torch.rand(2, 1, 32, 32)).shape == (2, 4)
    assert net.backbone.stem[0].in_channels == 4 * 2  # [energy, phase, ps, pa] x 2 scales
    cat = MonogenicResNet18(1, 2, mode="concat", **KW)
    assert cat.backbone.stem[0].in_channels == 4 * 2 + 1 and cat(torch.rand(2, 1, 32, 32)).shape == (2, 2)
    rgb = MonogenicResNet18(3, 2, **KW)
    assert rgb.backbone.stem[0].in_channels == 3 * 4 * 2 and rgb(torch.rand(1, 3, 32, 32)).shape == (1, 2)


@pytest.mark.parametrize("size", [64, 128])
def test_imagenet_stem_sizes(size: int) -> None:
    assert MonogenicResNet18(1, 2, wavelengths=(4.0, 8.0))(torch.rand(1, 1, size, size)).shape == (1, 2)


def test_extra_parameters_are_the_front_end_and_input_widening() -> None:
    mono = MonogenicResNet18(1, 2, **KW)
    base = ResNet18(1, 2, "small")
    front = count_parameters(mono.frontend)
    assert front == 2 * 3  # f0, sigma, threshold gain per scale
    widening = 64 * 9 * (8 - 1)  # first conv sees 8 maps instead of 1
    assert count_parameters(mono) - count_parameters(base) == front + widening + 2 * 8  # + BatchNorm affine


def test_gradients_reach_front_end_parameters() -> None:
    net = MonogenicResNet18(1, 2, **KW)
    net(torch.rand(4, 1, 32, 32)).sum().backward()
    for n, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert all(p.grad.abs().sum() > 0 for p in net.frontend.parameters())


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
        x[i] *= 10 ** (-2 + 3 * (i % 5) / 4)  # contrast over 3 decades
    x = x + 0.01 * x.amax(dim=(1, 2, 3), keepdim=True) * torch.randn_like(x)
    loader = DataLoader(TensorDataset(x, y.view(-1, 1)), batch_size=16)
    hist = fit(MonogenicResNet18(1, 2, **KW), loader, loader, TrainConfig(epochs=8, lr=3e-3))
    assert hist[-1]["val_acc"] >= 0.9


def test_validation() -> None:
    with pytest.raises(ValueError):
        MonogenicResNet18(mode="foo")
