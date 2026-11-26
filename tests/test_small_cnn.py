"""Tests for the small CNN baseline."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from monogenet.complexity import count_parameters
from monogenet.models import SmallCNN
from monogenet.train import TrainConfig, fit


@pytest.mark.parametrize("size", [28, 64, 128, 224])
def test_output_shape_for_medmnist_sizes(size: int) -> None:
    assert SmallCNN(1, 4)(torch.rand(2, 1, size, size)).shape == (2, 4)


def test_parameter_count_formula() -> None:
    w = 16
    convs = sum(a * b * 9 for a, b in [(1, w), (w, 2 * w), (2 * w, 4 * w), (4 * w, 8 * w)])
    bns = 2 * sum([w, 2 * w, 4 * w, 8 * w])
    head = 8 * w * 2 + 2
    assert count_parameters(SmallCNN(1, 2, width=w)) == convs + bns + head


def test_gradients_reach_all_parameters() -> None:
    net = SmallCNN(1, 2)
    net(torch.rand(4, 1, 32, 32)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_learns_bright_vs_dark_blobs() -> None:
    g = torch.Generator().manual_seed(0)
    y = torch.arange(48) % 2
    x = torch.randn(48, 1, 32, 32, generator=g) * 0.2
    x[:, :, 8:24, 8:24] += (y.float() * 2 - 1).view(-1, 1, 1, 1)
    loader = DataLoader(TensorDataset(x, y.view(-1, 1)), batch_size=16)
    hist = fit(SmallCNN(1, 2, width=4), loader, loader, TrainConfig(epochs=10, lr=0.01))
    assert hist[-1]["val_acc"] > 0.95 and hist[-1]["train_loss"] < hist[0]["train_loss"]
