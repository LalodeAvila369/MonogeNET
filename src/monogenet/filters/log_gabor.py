"""Radial log-Gabor filter in the frequency domain."""

from __future__ import annotations

import math

import torch

from monogenet.filters.grids import frequency_radius

__all__ = ["radial_log_gabor", "log_gabor_wavelengths", "log_gabor_bank"]


def radial_log_gabor(
    height: int,
    width: int,
    wavelength: float | torch.Tensor,
    sigma_onf: float | torch.Tensor = 0.55,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Radial log-Gabor transfer function (isotropic, zero DC gain).

    .. math::
        G(\omega) = \exp\!\left(-\frac{\left(\log(|\omega| / f_0)\right)^2}
                                      {2\,\left(\log \sigma_{o/f}\right)^2}\right),
        \qquad G(0) = 0,

    with centre frequency :math:`f_0 = 1/\lambda` (cycles per pixel) and
    :math:`\sigma_{o/f} = k/\omega_0` the ratio of the standard deviation of
    the Gaussian (in log-frequency) to the centre frequency. Note the
    convention: *smaller* values give a *wider* bandwidth, because the width in
    log-frequency is :math:`|\log\sigma_{o/f}|`. Thesis defaults: :math:`\lambda \in \{36, 39,
    42\}` px and :math:`k/\omega_0 = 0.55`. The peak value is 1 at
    :math:`|\omega| = f_0`, and :math:`G` is symmetric in :math:`\log|\omega|`
    around :math:`f_0`. Everything is differentiable with respect to
    ``wavelength`` and ``sigma_onf`` when they are tensors.

    Args:
        height, width: image size.
        wavelength: centre wavelength in pixels (must be > 0).
        sigma_onf: :math:`k/\omega_0`, in (0, 1) (log of it must be non-zero).

    ``wavelength`` and ``sigma_onf`` may also be tensors of shape ``(..., 1, 1)``; the
    result then has shape ``(..., height, width)`` (used to build whole banks at once).

    Returns:
        Real tensor of shape ``(height, width)`` in FFT order (DC at ``[0, 0]``).
    """
    wavelength_t = torch.as_tensor(wavelength, device=device, dtype=dtype)
    sigma_t = torch.as_tensor(sigma_onf, device=device, dtype=dtype)
    if (wavelength_t <= 0).any():
        raise ValueError("wavelength must be positive")
    if ((sigma_t <= 0) | (sigma_t >= 1)).any():
        raise ValueError("sigma_onf must be in (0, 1)")
    radius = frequency_radius(height, width, device=device, dtype=dtype, dc_value=1.0)
    f0 = 1.0 / wavelength_t
    g = torch.exp(-(torch.log(radius / f0) ** 2) / (2 * torch.log(sigma_t) ** 2))
    dc_mask = torch.zeros(height, width, dtype=torch.bool, device=radius.device)
    dc_mask[0, 0] = True
    return torch.where(dc_mask, torch.zeros_like(g), g)


def log_gabor_wavelengths(
    n_scales: int,
    min_wavelength: float | torch.Tensor,
    scale_factor: float | torch.Tensor = 2.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Geometric sequence of centre wavelengths :math:`\lambda_s = \lambda_{\min}\, m^{s}`.

    ``m`` is ``scale_factor`` (> 1 for increasing wavelength), ``s = 0..n_scales-1``.
    Returns a tensor of shape ``(n_scales,)``.
    """
    if n_scales < 1:
        raise ValueError("n_scales must be >= 1")
    lam = torch.as_tensor(min_wavelength, device=device, dtype=dtype)
    m = torch.as_tensor(scale_factor, device=device, dtype=dtype)
    exponents = torch.arange(n_scales, device=lam.device, dtype=dtype)
    return lam * m**exponents


def log_gabor_bank(
    height: int,
    width: int,
    n_scales: int = 3,
    min_wavelength: float | torch.Tensor = 36.0,
    scale_factor: float | torch.Tensor = 1.0833,
    sigma_onf: float | torch.Tensor = 0.55,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Multi-scale radial log-Gabor bank, shape ``(n_scales, height, width)``.

    Scale :math:`s` is :func:`radial_log_gabor` with
    :math:`\lambda_s = \lambda_{\min}\, m^{s}` and a shared
    :math:`\sigma_{o/f} = k/\omega_0`. The defaults (36 px, ``m`` = 1.0833,
    3 scales) give wavelengths of about 36, 39 and 42 px, the thesis values for
    hip ultrasound. Differentiable in ``min_wavelength``, ``scale_factor`` and
    ``sigma_onf`` when they are tensors.
    """
    wavelengths = log_gabor_wavelengths(
        n_scales, min_wavelength, scale_factor, device=device, dtype=dtype
    )
    return torch.stack(
        [radial_log_gabor(height, width, wl, sigma_onf, device=device, dtype=dtype) for wl in wavelengths]
    )
