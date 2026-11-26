"""Tests for the ResNet-18 baseline."""

import pytest
import torch

from monogenet.complexity import count_parameters
from monogenet.models import BasicBlock, ResNet18


def test_matches_torchvision_parameter_count() -> None:
    # torchvision.models.resnet18(): 11,689,512 parameters (3 channels, 1000 classes).
    assert count_parameters(ResNet18(3, 1000, "imagenet")) == 11_689_512


def test_head_and_stem_change_only_their_parameters() -> None:
    base = count_parameters(ResNet18(1, 2, "imagenet"))
    assert count_parameters(ResNet18(3, 2, "imagenet")) - base == 2 * 64 * 49  # extra input channels
    assert count_parameters(ResNet18(1, 4, "imagenet")) - base == 2 * 512 + 2  # extra class rows


@pytest.mark.parametrize("stem,size", [("small", 28), ("small", 64), ("imagenet", 128), ("imagenet", 224)])
def test_output_shape(stem: str, size: int) -> None:
    assert ResNet18(1, 4, stem)(torch.rand(2, 1, size, size)).shape == (2, 4)


def test_downsampling_factors() -> None:
    x = torch.rand(1, 1, 128, 128)
    net = ResNet18(1, 2, "imagenet").eval()
    assert net.layers(net.stem(x)).shape == (1, 512, 4, 4)  # total stride 32
    small = ResNet18(1, 2, "small").eval()
    assert small.layers(small.stem(torch.rand(1, 1, 32, 32))).shape == (1, 512, 4, 4)  # stride 8


def test_block_shortcut_is_identity_only_without_reshaping() -> None:
    assert isinstance(BasicBlock(8, 8).shortcut, torch.nn.Identity)
    assert BasicBlock(8, 16, 2)(torch.rand(1, 8, 8, 8)).shape == (1, 16, 4, 4)
    x = torch.rand(1, 8, 6, 6)
    zero = BasicBlock(8, 8).eval()  # eval BN with fresh stats is the identity map
    torch.nn.init.zeros_(zero.conv2.weight)  # residual branch outputs 0 -> block = relu(shortcut(x))
    assert torch.allclose(zero(x), torch.relu(x), atol=1e-6)


def test_gradients_and_validation() -> None:
    net = ResNet18(1, 2, "small")
    net(torch.rand(2, 1, 16, 16)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    with pytest.raises(ValueError):
        ResNet18(stem="foo")
