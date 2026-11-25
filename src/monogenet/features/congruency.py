"""Monogenic phase congruency."""

from __future__ import annotations

import torch

from monogenet.features.threshold import noise_threshold, soft_floor

__all__ = ["phase_congruency"]


def phase_congruency(
    mono: torch.Tensor,
    threshold: torch.Tensor | float | None = None,
    eps: float = 1e-6,
    smooth: float = 1e-12,
    sharpness: float | None = None,
) -> torch.Tensor:
    r"""Monogenic phase congruency, shape ``(..., H, W)`` (scales are combined).

    Let :math:`\mathbf v_s = (e_s, o_{1,s}, o_{2,s})` be the monogenic vector at
    scale :math:`s`, :math:`A_s = \lVert\mathbf v_s\rVert`, and
    :math:`\hat{\mathbf m} = \sum_s \mathbf v_s / \lVert\sum_s \mathbf v_s\rVert`
    the unit direction of the summed response. Then

    .. math::
        \mathrm{PC} = \frac{\lfloor\, \sum_s \big(\mathbf v_s\cdot\hat{\mathbf m}
                        - \lVert \mathbf v_s \times \hat{\mathbf m} \rVert\big)
                        - T \,\rfloor}{\sum_s A_s + \varepsilon}.

    :math:`\mathbf v_s\cdot\hat{\mathbf m} = A_s\cos\Delta_s` and
    :math:`\lVert\mathbf v_s\times\hat{\mathbf m}\rVert = A_s\sin\Delta_s`, with
    :math:`\Delta_s` the angle between scale :math:`s` and the mean direction
    (the monogenic form of Kovesi's phase deviation, after Felsberg and Sommer).
    PC is 1 when all scales are in phase and the response is above the noise,
    and close to 0 when scales disagree. It is invariant to contrast and,
    unlike PS and PA, does not distinguish between lines and edges.

    Args:
        mono: monogenic tensor ``(..., S, 3, H, W)``.
        threshold: total noise threshold :math:`T`, broadcastable to
            ``(..., H, W)``. Default: sum over scales of the per-scale thesis
            thresholds from :func:`monogenet.features.noise_threshold`.
        eps: stabiliser of the denominator.
        smooth: added under the square root of the cross-product norm so the
            gradient stays finite when a scale is exactly parallel to the mean
            direction (a bias of order ``sqrt(smooth)``).
        sharpness: softplus sharpness replacing the hard floor (``None`` = hard);
            see :func:`monogenet.features.soft_floor`.
    """
    if mono.ndim < 4 or mono.shape[-3] != 3:
        raise ValueError(f"expected monogenic tensor (..., S, 3, H, W), got {tuple(mono.shape)}")
    if threshold is None:
        threshold = noise_threshold(mono).sum(dim=-3)  # (..., 1, 1)
    total = mono.sum(dim=-4, keepdim=True)  # (..., 1, 3, H, W)
    norm = torch.sqrt((total * total).sum(dim=-3, keepdim=True) + smooth * smooth)
    m = total / (norm + eps)  # unit direction of the summed response
    dot = (mono * m).sum(dim=-3)  # (..., S, H, W)
    cross = torch.linalg.cross(mono, m.expand_as(mono), dim=-3)
    cross_norm = torch.sqrt((cross * cross).sum(dim=-3) + smooth)
    amp = torch.sqrt((mono * mono).sum(dim=-3))
    energy = (dot - cross_norm).sum(dim=-3)
    return soft_floor((energy - threshold) / (amp.sum(dim=-3) + eps), sharpness)
