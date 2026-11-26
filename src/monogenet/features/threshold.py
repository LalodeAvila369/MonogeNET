"""Noise thresholds and soft floors for the phase features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from monogenet.features._ops import promote_half, safe_sqrt

__all__ = ["noise_threshold", "soft_floor", "LearnableThreshold"]


def _amplitude(mono: torch.Tensor) -> torch.Tensor:
    if mono.ndim < 4 or mono.shape[-3] != 3:
        raise ValueError(f"expected monogenic tensor (..., S, 3, H, W), got {tuple(mono.shape)}")
    return safe_sqrt((mono * mono).sum(dim=-3))


@promote_half
def noise_threshold(
    mono: torch.Tensor,
    stat: str = "mean",
    gain: float | torch.Tensor = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Per-scale noise threshold :math:`T_s = k\,\mathrm{stat}(A_s)`, shape ``(..., S, 1, 1)``.

    The statistic is taken over the image (last two dims), separately for every
    image, channel and scale, so it does not depend on the batch and scales
    linearly with contrast (keeping the phase features contrast invariant).

    * ``"mean"``: :math:`\bar A_s`. This is the thesis value
      :math:`\exp(\log(\mathrm{mean}(A)))`, which equals the mean amplitude.
    * ``"geometric"``: :math:`\exp(\mathrm{mean}(\log A_s))`, the geometric
      mean (more robust to a few very strong responses).

    ``gain`` :math:`k` multiplies the statistic (1 reproduces the thesis).
    """
    amp = _amplitude(mono)
    if stat == "mean":
        t = amp.mean(dim=(-2, -1), keepdim=True)
    elif stat == "geometric":
        t = torch.exp(torch.log(amp + eps).mean(dim=(-2, -1), keepdim=True))
    else:
        raise ValueError(f"unknown stat {stat!r}; use 'mean' or 'geometric'")
    return gain * t


def soft_floor(x: torch.Tensor, sharpness: float | None = None) -> torch.Tensor:
    r"""Floor :math:`\lfloor x \rfloor = \max(x, 0)`, or its smooth version.

    With ``sharpness`` :math:`\beta` it is the softplus
    :math:`\tfrac{1}{\beta}\log(1 + e^{\beta x})`, which tends to :math:`\max(x, 0)`
    as :math:`\beta \to \infty`, is always positive and has a non-zero gradient
    everywhere (the hard floor has none below the threshold). The error is at
    most :math:`\log 2/\beta`, reached at :math:`x = 0`. ``None`` gives the hard
    floor. The features apply it to dimensionless ratios, so :math:`\beta` does
    not depend on image contrast.
    """
    if sharpness is None:
        return torch.clamp(x, min=0.0)
    if sharpness <= 0:
        raise ValueError("sharpness must be positive")
    return F.softplus(x, beta=sharpness)


class LearnableThreshold(nn.Module):
    r"""Noise threshold with a learnable gain: :math:`T_s = k_s\,\mathrm{stat}(A_s)`.

    The statistic (mean or geometric mean of the amplitude over the image, see
    :func:`noise_threshold`) is computed from the data, so :math:`T` scales with
    contrast and the phase features stay contrast invariant. Only the gain is
    learned, :math:`k_s = \mathrm{softplus}(\kappa_s) > 0`, initialised at
    ``init_gain`` (1 reproduces the thesis). That is ``n_scales`` parameters, or
    one if ``shared=True``. With ``channels=C`` every channel gets its own gains
    (``C * n_scales`` parameters) and ``mono`` must be ``(..., C, S, 3, H, W)``.

    A Rayleigh noise model, :math:`T = \mu + c\,\sigma` with
    :math:`\mu = \bar A` and :math:`\sigma = \bar A\sqrt{(4-\pi)/\pi}`, is also a
    constant multiple of the mean: :math:`k = 1 + c\sqrt{(4-\pi)/\pi} \approx 1 + 0.52\,c`
    (about 2.05 for :math:`c = 2`). So a gain initialised there covers it.

    ``forward(mono)`` takes ``(..., S, 3, H, W)`` and returns ``(..., S, 1, 1)``.
    For :func:`phase_congruency` pass ``forward(mono).sum(dim=-3)``.
    """

    def __init__(
        self,
        n_scales: int,
        init_gain: float = 1.0,
        stat: str = "mean",
        *,
        shared: bool = False,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        if n_scales < 1:
            raise ValueError("n_scales must be >= 1")
        if init_gain <= 0:
            raise ValueError("init_gain must be positive")
        if stat not in ("mean", "geometric"):
            raise ValueError(f"unknown stat {stat!r}; use 'mean' or 'geometric'")
        if channels is not None and channels < 1:
            raise ValueError("channels must be >= 1")
        self.n_scales = n_scales
        self.stat = stat
        self.shared = shared
        self.channels = channels
        # Inverse softplus: softplus(raw) = init_gain.
        raw = torch.log(torch.expm1(torch.tensor(float(init_gain))))
        if shared:
            size = (1, 1, 1)
        elif channels is None:
            size = (n_scales, 1, 1)
        else:
            size = (channels, n_scales, 1, 1)
        self.raw_gain = nn.Parameter(raw.expand(size).clone())

    @property
    def gain(self) -> torch.Tensor:
        """Positive gain :math:`k_s`: ``(S, 1, 1)``, ``(C, S, 1, 1)`` or ``(1, 1, 1)``."""
        return F.softplus(self.raw_gain)

    def forward(self, mono: torch.Tensor) -> torch.Tensor:
        if not self.shared and mono.shape[-4] != self.n_scales:
            raise ValueError(f"expected {self.n_scales} scales, got {mono.shape[-4]}")
        if self.channels is not None and not self.shared and mono.shape[-5] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {mono.shape[-5]}")
        return noise_threshold(mono, self.stat, self.gain.to(mono.dtype))

    def extra_repr(self) -> str:
        return (
            f"n_scales={self.n_scales}, channels={self.channels}, "
            f"stat={self.stat!r}, shared={self.shared}"
        )
