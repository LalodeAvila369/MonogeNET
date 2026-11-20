"""Frequency-grid utilities for filters built in the Fourier domain."""

from __future__ import annotations

import torch

__all__ = ["frequency_grid", "frequency_radius", "frequency_angle"]


def frequency_grid(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Normalised frequency meshgrid in FFT order (DC at index ``[0, 0]``).

    .. math::
        u_k = \mathrm{fftfreq}(W)_k \in [-1/2, 1/2), \qquad
        v_l = \mathrm{fftfreq}(H)_l \in [-1/2, 1/2)

    Units are cycles per pixel. ``u`` varies along the width (columns, x) and
    ``v`` along the height (rows, y).

    Returns:
        ``(u, v)``, each of shape ``(height, width)``.
    """
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    fv = torch.fft.fftfreq(height, device=device, dtype=dtype)
    fu = torch.fft.fftfreq(width, device=device, dtype=dtype)
    v, u = torch.meshgrid(fv, fu, indexing="ij")
    return u, v


def frequency_radius(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    dc_value: float = 1.0,
) -> torch.Tensor:
    r"""Radial frequency :math:`|w| = \sqrt{u^2 + v^2}`, shape ``(height, width)``.

    The DC entry is set to ``dc_value`` (default 1) so that expressions such as
    :math:`\log|w|` or :math:`u/|w|` (Riesz kernels) stay finite. Callers must
    mask DC themselves where it matters (e.g. log-Gabor has zero DC gain).
    """
    u, v = frequency_grid(height, width, device=device, dtype=dtype)
    radius = torch.sqrt(u * u + v * v)
    radius[0, 0] = dc_value
    return radius


def frequency_angle(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Frequency orientation :math:`\theta = \mathrm{atan2}(v, u) \in (-\pi, \pi]`.

    Shape ``(height, width)``. At DC the angle is 0 by convention.
    """
    u, v = frequency_grid(height, width, device=device, dtype=dtype)
    return torch.atan2(v, u)
