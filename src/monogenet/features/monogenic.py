"""Monogenic signal: even and odd (Riesz) components per scale."""

from __future__ import annotations

import torch

from monogenet.filters.riesz import riesz_kernels

__all__ = ["monogenic_signal"]


def monogenic_signal(image: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    r"""Band-pass monogenic signal of ``image`` for every filter in ``bank``.

    For each scale :math:`s` with radial transfer function :math:`G_s`,

    .. math::
        e_s = \mathcal F^{-1}\!\left[G_s \hat f\right], \quad
        o_{1,s} = \mathcal F^{-1}\!\left[H_1 G_s \hat f\right], \quad
        o_{2,s} = \mathcal F^{-1}\!\left[H_2 G_s \hat f\right],

    with :math:`H_1 = i u/|\omega|`, :math:`H_2 = i v/|\omega|` the Riesz
    kernels (see :func:`monogenet.filters.riesz_kernels`). The triple
    :math:`(e_s, o_{1,s}, o_{2,s})` is the monogenic signal at scale
    :math:`s`: ``even`` is the band-passed image and ``odd1``, ``odd2`` its
    two Riesz components. Everything is computed in the frequency domain
    (one FFT of the image, three inverse FFTs per scale) and is differentiable
    with respect to both the image and the bank.

    Args:
        image: real tensor ``(..., H, W)``, e.g. ``(B, C, H, W)`` or ``(H, W)``.
        bank: real transfer functions ``(S, H, W)`` in FFT order, as returned by
            :func:`monogenet.filters.log_gabor_bank` or
            :class:`monogenet.filters.LearnableLogGaborBank`.

    Returns:
        Real tensor ``(..., S, 3, H, W)`` with channel order
        ``[even, odd1, odd2]``.
    """
    if image.is_complex():
        raise TypeError("image must be real")
    if bank.ndim != 3 or bank.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"bank must have shape (S, H, W) matching the image, got {tuple(bank.shape)} "
            f"for image {tuple(image.shape)}"
        )
    h, w = image.shape[-2:]
    h1, h2 = riesz_kernels(h, w, device=image.device, dtype=image.dtype)
    spec = torch.fft.fft2(image).unsqueeze(-3) * bank  # (..., S, H, W)
    even = torch.fft.ifft2(spec).real
    odd1 = torch.fft.ifft2(spec * h1).real
    odd2 = torch.fft.ifft2(spec * h2).real
    return torch.stack((even, odd1, odd2), dim=-3)
