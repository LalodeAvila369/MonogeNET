"""Riesz transform kernels in the frequency domain."""

from __future__ import annotations

import torch

from monogenet.filters.grids import frequency_grid, frequency_radius

__all__ = ["riesz_kernels"]


def riesz_kernels(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Riesz transform transfer functions :math:`H_1, H_2` (complex, FFT order).

    .. math::
        H_1(\omega) = \frac{i\,u}{|\omega|}, \qquad
        H_2(\omega) = \frac{i\,v}{|\omega|}, \qquad
        H_1(0) = H_2(0) = 0,

    with :math:`u` along the width (x) and :math:`v` along the height (y).
    :math:`|H_1|^2 + |H_2|^2 = 1` everywhere except DC (and the Nyquist
    row/column of even-sized grids, see below), so the transform preserves
    energy of zero-mean, Nyquist-free signals. For a band-passed image
    :math:`f`, the monogenic signal is
    :math:`(f,\; \mathcal F^{-1}[H_1 \hat f],\; \mathcal F^{-1}[H_2 \hat f])`.

    Example: :math:`\cos(2\pi f x) \mapsto -\sin(2\pi f x)` under :math:`H_1`
    and :math:`0` under :math:`H_2`.

    Returns:
        ``(H1, H2)``, complex tensors of shape ``(height, width)``.
    """
    u, v = frequency_grid(height, width, device=device, dtype=dtype)
    radius = frequency_radius(height, width, device=device, dtype=dtype, dc_value=1.0)
    h1 = torch.complex(torch.zeros_like(u), u / radius)
    h2 = torch.complex(torch.zeros_like(v), v / radius)
    # Zero DC gain (the 1/|w| singularity has no defined value there).
    h1[0, 0] = 0
    h2[0, 0] = 0
    # Even sizes: the Nyquist frequency is its own mirror image, so odd symmetry
    # H(-w) = -H(w) forces H = 0 there. Zeroing it keeps the output exactly real.
    if width % 2 == 0:
        h1[:, width // 2] = 0
    if height % 2 == 0:
        h2[height // 2, :] = 0
    return h1, h2
