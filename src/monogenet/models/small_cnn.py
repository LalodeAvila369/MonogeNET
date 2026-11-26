"""Small CNN baseline for MedMNIST-style classification."""

from __future__ import annotations

from torch import nn

__all__ = ["SmallCNN"]


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Four ``conv3x3 - BN - ReLU - maxpool`` stages, global average pooling, linear head.

    Works for any input size of at least ``2**depth`` pixels (28, 64, 128, 224, ...).
    The widths double per stage (``width, 2*width, 4*width, 8*width``).
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2, width: int = 16, depth: int = 4) -> None:
        super().__init__()
        widths = [in_channels] + [width * 2**i for i in range(depth)]
        self.features = nn.Sequential(*[_block(a, b) for a, b in zip(widths[:-1], widths[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], n_classes)
        self.n_classes = n_classes

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.head(self.pool(self.features(x)).flatten(1))
