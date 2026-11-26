"""ResNet-18 with a learnable monogenic front end (research idea, step 2)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from monogenet.layers.frontend import MonogenicFrontEnd
from monogenet.models.resnet import ResNet18

__all__ = ["MonogenicResNet18"]


class MonogenicResNet18(nn.Module):
    """``image -> MonogenicFrontEnd -> BatchNorm -> ResNet-18``.

    The front end turns each image into ``[energy, phase, PS, PA]`` per scale
    (``4 * S`` channels, per-image and contrast-normalised except for the energy).
    A ``BatchNorm2d`` puts these maps, whose ranges differ (energy is unbounded,
    phase is in ``[0, pi]``, PS and PA in ``[0, 1]``), on a common scale before the
    first convolution.

    ``mode="replace"`` feeds only the monogenic maps to the ResNet; ``mode="concat"``
    also keeps the (normalised) image as an extra channel. ``wavelengths`` are in
    pixels: the thesis values (36, 39, 42) suit its ultrasound scale, the default
    ``(4, 8, 16)`` suits 128-pixel MedMNIST images (use larger values for 224).
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        wavelengths: Sequence[float] = (4.0, 8.0, 16.0),
        stem: str = "imagenet",
        mode: str = "replace",
        **frontend_kwargs: object,
    ) -> None:
        super().__init__()
        if mode not in ("replace", "concat"):
            raise ValueError(f"mode must be 'replace' or 'concat', got {mode!r}")
        self.mode = mode
        self.frontend = MonogenicFrontEnd(wavelengths, in_channels=in_channels, **frontend_kwargs)  # type: ignore[arg-type]
        mono_channels = self.frontend.out_channels
        total = mono_channels + (in_channels if mode == "concat" else 0)
        self.norm = nn.BatchNorm2d(total)
        self.backbone = ResNet18(total, n_classes, stem)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.frontend(x)
        if self.mode == "concat":
            maps = torch.cat([x, maps], dim=1)
        return self.backbone(self.norm(maps))
