"""Low-pass and spatial windowing to reduce FFT edge artifacts."""

from __future__ import annotations

import torch

from monogenet.filters.grids import frequency_radius

__all__ = ["butterworth_lowpass", "tukey_window", "apply_edge_taper"]


def butterworth_lowpass(
    height: int,
    width: int,
    cutoff: float = 0.45,
    order: int = 15,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Radial Butterworth low-pass transfer function in FFT order.

    .. math::
        B(\omega) = \frac{1}{1 + (|\omega| / f_c)^{2n}}

    ``cutoff`` :math:`f_c` is in cycles per pixel, in (0, 0.5]; ``order``
    :math:`n` sets the sharpness. Multiplying a log-Gabor by :math:`B` removes
    the unbounded high-frequency tail of the log-Gabor, which otherwise gives
    non-zero response at the corners of the spectrum. :math:`B(0) = 1`.
    """
    if not 0 < cutoff <= 0.5:
        raise ValueError("cutoff must be in (0, 0.5]")
    if order < 1:
        raise ValueError("order must be >= 1")
    radius = frequency_radius(height, width, device=device, dtype=dtype, dc_value=0.0)
    return 1.0 / (1.0 + (radius / cutoff) ** (2 * order))


def tukey_window(
    n: int,
    alpha: float = 0.25,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""1-D Tukey (tapered cosine) window of length ``n``.

    Flat (= 1) in the middle and cosine tapers over a fraction ``alpha`` of
    the length (half of it at each end). ``alpha = 0`` is rectangular and
    ``alpha = 1`` is a Hann window. Symmetric, with values in [0, 1].
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if n == 1 or alpha == 0:
        return torch.ones(n, device=device, dtype=dtype)
    x = torch.linspace(0, 1, n, device=device, dtype=dtype)
    w = torch.ones(n, device=device, dtype=dtype)
    edge = alpha / 2
    left = x < edge
    right = x > 1 - edge
    w = torch.where(left, 0.5 * (1 + torch.cos(torch.pi * (2 * x / alpha - 1))), w)
    w = torch.where(right, 0.5 * (1 + torch.cos(torch.pi * (2 * (1 - x) / alpha - 1))), w)
    return w


def apply_edge_taper(img: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    r"""Taper an image to its mean towards the borders.

    .. math::
        \tilde f = \bar f + w(y)\,w(x)\,(f - \bar f)

    with :math:`w` a Tukey window and :math:`\bar f` the per-image mean over
    the last two dims. The FFT treats images as periodic, so a jump between
    opposite borders acts like a strong artificial edge; tapering to the mean
    removes it while leaving the interior untouched. Works on ``(..., H, W)``.
    """
    h, w_ = img.shape[-2:]
    wy = tukey_window(h, alpha, device=img.device, dtype=img.dtype)
    wx = tukey_window(w_, alpha, device=img.device, dtype=img.dtype)
    mean = img.mean(dim=(-2, -1), keepdim=True)
    return mean + (img - mean) * (wy[:, None] * wx[None, :])
