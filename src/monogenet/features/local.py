"""Local amplitude, local phase and local orientation of a monogenic signal."""

from __future__ import annotations

from typing import NamedTuple

import torch

__all__ = [
    "LocalFeatures",
    "local_amplitude",
    "local_phase",
    "local_orientation",
    "local_features",
    "feature_type",
]


class LocalFeatures(NamedTuple):
    """Per-scale local features, each of shape ``(..., S, H, W)``."""

    amplitude: torch.Tensor
    phase: torch.Tensor
    orientation: torch.Tensor


def _split(mono: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mono.ndim < 4 or mono.shape[-3] != 3:
        raise ValueError(f"expected monogenic tensor (..., S, 3, H, W), got {tuple(mono.shape)}")
    return mono.unbind(dim=-3)


def local_amplitude(mono: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    r"""Local amplitude :math:`A = \sqrt{e^2 + o_1^2 + o_2^2 + \varepsilon}`.

    ``mono`` has shape ``(..., S, 3, H, W)`` with channels ``[even, odd1, odd2]``.
    With ``eps = 0`` the gradient is undefined where the signal is exactly zero;
    use a small positive ``eps`` when training.
    """
    e, o1, o2 = _split(mono)
    return torch.sqrt(e * e + o1 * o1 + o2 * o2 + eps)


def local_phase(mono: torch.Tensor) -> torch.Tensor:
    r"""Local phase :math:`\varphi = \mathrm{atan2}\big(\sqrt{o_1^2 + o_2^2},\, e\big) \in [0, \pi]`.

    :math:`\varphi = 0` for a bright symmetric feature (line, blob),
    :math:`\varphi = \pi` for a dark one, and :math:`\varphi = \pi/2` for an
    ideal step edge (feature type, see ROADMAP step 17).
    """
    e, o1, o2 = _split(mono)
    return torch.atan2(torch.sqrt(o1 * o1 + o2 * o2), e)


def local_orientation(mono: torch.Tensor) -> torch.Tensor:
    r"""Local orientation :math:`\theta \in (-\pi/2, \pi/2]` of the dominant gradient direction.

    .. math::
        \theta = \mathrm{atan2}(s\,o_2,\; s\,o_1), \qquad s = \mathrm{sign}(o_1)\ (s = 1 \text{ if } o_1 = 0),

    i.e. :math:`\arctan(o_2 / o_1)`. The odd components :math:`(o_1, o_2)` point
    along (or against) the direction of fastest variation, so orientation is
    defined modulo :math:`\pi`; the sign normalisation picks the representative
    with :math:`o_1 \ge 0`. The angle is measured from the +x (column) axis
    towards +y (row), the same convention as ``frequency_angle``.
    """
    _, o1, o2 = _split(mono)
    s = torch.where(o1 < 0, -torch.ones_like(o1), torch.ones_like(o1))
    return torch.atan2(s * o2, s * o1)


def local_features(mono: torch.Tensor, eps: float = 0.0) -> LocalFeatures:
    """Amplitude, phase and orientation of a monogenic tensor ``(..., S, 3, H, W)``."""
    return LocalFeatures(
        amplitude=local_amplitude(mono, eps),
        phase=local_phase(mono),
        orientation=local_orientation(mono),
    )


def feature_type(mono: torch.Tensor) -> torch.Tensor:
    r"""Feature type :math:`\psi = \mathrm{atan2}\big(e,\; \sqrt{o_1^2 + o_2^2}\big) \in [-\pi/2, \pi/2]`.

    A signed, polarity-aware summary of the local structure:

    * :math:`\psi = +\pi/2`: bright symmetric feature (line, blob);
    * :math:`\psi = -\pi/2`: dark symmetric feature;
    * :math:`\psi = 0`: ideal step edge (odd response only);
    * in between: mixtures of even and odd structure.

    It is related to the local phase by :math:`\psi = \pi/2 - \varphi`.
    Shape ``(..., S, H, W)`` for ``mono`` of shape ``(..., S, 3, H, W)``.
    """
    e, o1, o2 = _split(mono)
    return torch.atan2(e, torch.sqrt(o1 * o1 + o2 * o2))
