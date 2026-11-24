"""Phase symmetry and phase asymmetry from a monogenic signal."""

from __future__ import annotations

import torch

__all__ = ["noise_threshold", "phase_symmetry"]


def _split(mono: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mono.ndim < 4 or mono.shape[-3] != 3:
        raise ValueError(f"expected monogenic tensor (..., S, 3, H, W), got {tuple(mono.shape)}")
    return mono.unbind(dim=-3)


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
    e, o1, o2 = _split(mono)
    amp = torch.sqrt(e * e + o1 * o1 + o2 * o2)
    if stat == "mean":
        t = amp.mean(dim=(-2, -1), keepdim=True)
    elif stat == "geometric":
        t = torch.exp(torch.log(amp + eps).mean(dim=(-2, -1), keepdim=True))
    else:
        raise ValueError(f"unknown stat {stat!r}; use 'mean' or 'geometric'")
    return gain * t


def phase_symmetry(
    mono: torch.Tensor,
    threshold: torch.Tensor | float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Phase symmetry per scale, shape ``(..., S, H, W)``.

    .. math::
        \mathrm{PS}_s = \frac{\lfloor\, |e_s| - \sqrt{o_{1,s}^2 + o_{2,s}^2} - T_s \,\rfloor}
                             {\sqrt{e_s^2 + o_{1,s}^2 + o_{2,s}^2} + \varepsilon},
        \qquad \lfloor x \rfloor = \max(x, 0).

    High where the even response dominates the odd one: lines, blobs and ridges
    (bright or dark). ``threshold`` is :math:`T_s`, broadcastable to
    ``(..., S, H, W)``; by default the thesis value from
    :func:`noise_threshold` (mean amplitude per scale). ``mono`` is a monogenic
    tensor ``(..., S, 3, H, W)`` from :func:`monogenet.features.monogenic_signal`.
    """
    e, o1, o2 = _split(mono)
    if threshold is None:
        threshold = noise_threshold(mono)
    odd = torch.sqrt(o1 * o1 + o2 * o2)
    amp = torch.sqrt(e * e + o1 * o1 + o2 * o2)
    return torch.clamp(e.abs() - odd - threshold, min=0.0) / (amp + eps)
