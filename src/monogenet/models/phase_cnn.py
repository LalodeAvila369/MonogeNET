"""CNN whose convolutions are replaced by PhaseBlocks (research idea, step 3)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from monogenet.layers.phase_block import PhaseBlock

__all__ = ["PhaseCNN"]


class PhaseCNN(nn.Module):
    """:class:`~monogenet.models.SmallCNN` with ``PhaseBlock`` in place of ``Conv2d``.

    Stage ``i`` is ``filter(c_in, c_out) - BatchNorm - ReLU - MaxPool(2)`` where the
    filter is a :class:`~monogenet.layers.PhaseBlock` if ``i`` is in ``phase_stages``
    and a 3x3 ``Conv2d`` otherwise. ``phase_stages=None`` replaces every stage; for
    example ``(0,)`` gives a phase first layer, and ``(2, 3)`` tests the open question
    of whether phase layers help in deeper layers. Widths double per stage.

    ``PhaseBlock`` has no stride, so downsampling is done by the max-pool. Its filters
    are periodic in the image (use ``taper`` to soften the borders), and the
    wavelengths are in pixels of the stage's own feature map; the default
    ``(4, 6, 9)`` stays below the map size down to 16 x 16 maps.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        width: int = 16,
        depth: int = 4,
        phase_stages: Iterable[int] | None = None,
        wavelengths: Sequence[float] = (4.0, 6.0, 9.0),
        features: Sequence[str] = ("ps", "pa"),
        **phase_kwargs: object,
    ) -> None:
        super().__init__()
        stages = set(range(depth)) if phase_stages is None else set(phase_stages)
        if not stages <= set(range(depth)):
            raise ValueError(f"phase_stages must lie in range({depth}), got {sorted(stages)}")
        self.phase_stages = tuple(sorted(stages))
        widths = [in_channels] + [width * 2**i for i in range(depth)]
        blocks: list[nn.Module] = []
        for i, (c_in, c_out) in enumerate(zip(widths[:-1], widths[1:])):
            if i in stages:
                filt: nn.Module = PhaseBlock(c_in, c_out, wavelengths, features=tuple(features), **phase_kwargs)  # type: ignore[arg-type]
            else:
                filt = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
            blocks.append(nn.Sequential(filt, nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.MaxPool2d(2)))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], n_classes)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)).flatten(1))
