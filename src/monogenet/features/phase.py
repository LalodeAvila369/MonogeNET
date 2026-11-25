"""Phase symmetry and phase asymmetry from a monogenic signal."""

from __future__ import annotations

import torch

from monogenet.features._ops import promote_half, safe_sqrt
from monogenet.features.threshold import noise_threshold, soft_floor

__all__ = ["phase_symmetry", "phase_asymmetry"]


def _split(mono: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mono.ndim < 4 or mono.shape[-3] != 3:
        raise ValueError(f"expected monogenic tensor (..., S, 3, H, W), got {tuple(mono.shape)}")
    return mono.unbind(dim=-3)


@promote_half
def phase_symmetry(
    mono: torch.Tensor,
    threshold: torch.Tensor | float | None = None,
    eps: float = 1e-6,
    sharpness: float | None = None,
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
    ``sharpness`` replaces the hard floor by a softplus (see
    :func:`monogenet.features.soft_floor`), applied to the dimensionless ratio so it
    is independent of contrast; ``None`` keeps the hard floor.
    """
    e, o1, o2 = _split(mono)
    if threshold is None:
        threshold = noise_threshold(mono)
    odd = safe_sqrt(o1 * o1 + o2 * o2)
    amp = safe_sqrt(e * e + o1 * o1 + o2 * o2)
    return soft_floor((e.abs() - odd - threshold) / (amp + eps), sharpness)


@promote_half
def phase_asymmetry(
    mono: torch.Tensor,
    threshold: torch.Tensor | float | None = None,
    eps: float = 1e-6,
    sharpness: float | None = None,
) -> torch.Tensor:
    r"""Phase asymmetry per scale, shape ``(..., S, H, W)``.

    .. math::
        \mathrm{PA}_s = \frac{\lfloor\, \sqrt{o_{1,s}^2 + o_{2,s}^2} - |e_s| - T_s \,\rfloor}
                             {\sqrt{e_s^2 + o_{1,s}^2 + o_{2,s}^2} + \varepsilon}.

    The mirror image of :func:`phase_symmetry`: high where the odd response
    dominates, i.e. step edges and other antisymmetric structures. For
    :math:`T_s \ge 0`, PS and PA cannot both be non-zero at the same pixel.
    """
    e, o1, o2 = _split(mono)
    if threshold is None:
        threshold = noise_threshold(mono)
    odd = safe_sqrt(o1 * o1 + o2 * o2)
    amp = safe_sqrt(e * e + o1 * o1 + o2 * o2)
    return soft_floor((odd - e.abs() - threshold) / (amp + eps), sharpness)
