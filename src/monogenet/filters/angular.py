"""Angular spread function for oriented log-Gabor filters."""

from __future__ import annotations

import math

import torch

from monogenet.filters.grids import frequency_angle

__all__ = ["angular_spread"]


def angular_spread(
    height: int,
    width: int,
    orientation: float | torch.Tensor,
    sigma_theta: float | torch.Tensor,
    *,
    symmetric: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Gaussian angular spread around ``orientation`` in the frequency plane.

    .. math::
        S(\theta) = \exp\!\left(-\frac{\Delta\theta^2}{2\sigma_\theta^2}\right),
        \qquad
        \Delta\theta = \left|\mathrm{atan2}\big(\sin(\theta-\theta_0),
                                              \cos(\theta-\theta_0)\big)\right|,

    where :math:`\theta = \mathrm{atan2}(v, u)` is the frequency angle and
    :math:`\Delta\theta \in [0, \pi]` is the wrapped angular distance to
    :math:`\theta_0`. With ``symmetric=True`` the distance is taken modulo
    :math:`\pi` (:math:`\Delta\theta \in [0, \pi/2]`), so that the filter also
    responds at :math:`\theta_0 + \pi` and has a real, even spatial kernel.

    A common choice for ``n`` orientations is
    :math:`\sigma_\theta = \pi / (1.5\,n)`. Multiply with a radial filter to
    get an oriented log-Gabor. The value at DC (angle 0 by convention) is
    irrelevant because the radial factor is zero there.

    Returns:
        Real tensor of shape ``(height, width)`` in FFT order, values in (0, 1].
    """
    theta0 = torch.as_tensor(orientation, device=device, dtype=dtype)
    sigma = torch.as_tensor(sigma_theta, device=device, dtype=dtype)
    if (sigma <= 0).any():
        raise ValueError("sigma_theta must be positive")
    theta = frequency_angle(height, width, device=device, dtype=dtype)
    diff = theta - theta0
    if symmetric:
        diff = 2 * diff  # period pi -> period 2 pi
    # Wrapped |delta|; sin/cos form is smooth and differentiable.
    dtheta = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))
    if symmetric:
        dtheta = dtheta / 2
    return torch.exp(-(dtheta**2) / (2 * sigma**2))
