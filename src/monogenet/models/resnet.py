"""ResNet-18 baseline (He et al., 2016), implemented here to avoid a torchvision dependency.

With ``stem="imagenet"``, 3 input channels and 1000 classes it has exactly the same
11,689,512 parameters as ``torchvision.models.resnet18``.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["BasicBlock", "ResNet18"]


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with an identity (or 1x1 projection) shortcut."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18: stem, four stages of two basic blocks (64-128-256-512), pooling, linear head.

    ``stem="imagenet"``: 7x7 stride-2 convolution + max-pool (for 128 and 224 pixel
    inputs). ``stem="small"``: a 3x3 stride-1 convolution and no pooling (for 28 or 64
    pixel inputs). ``in_channels`` lets a front end with more than one channel feed
    the network.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2, stem: str = "imagenet") -> None:
        super().__init__()
        if stem == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )
        elif stem == "small":
            self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        else:
            raise ValueError(f"stem must be 'imagenet' or 'small', got {stem!r}")
        layers: list[nn.Module] = []
        c = 64
        for i, width in enumerate((64, 128, 256, 512)):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(c, width, stride), BasicBlock(width, width)]
            c = width
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, n_classes)
        self.n_classes = n_classes
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.layers(self.stem(x))).flatten(1))
