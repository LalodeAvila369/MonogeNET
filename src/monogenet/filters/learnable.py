"""Learnable (bounded) parameterisation of the log-Gabor bank."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from monogenet.filters.log_gabor import radial_log_gabor

__all__ = ["LearnableLogGaborBank"]


class LearnableLogGaborBank(nn.Module):
    r"""Multi-scale radial log-Gabor bank with trainable, bounded parameters.

    Per scale :math:`s` there are two unconstrained real parameters
    :math:`(a_s, b_s)`:

    .. math::
        f_{0,s} = f_{\min} + (f_{\max} - f_{\min})\,\sigma(a_s), \qquad
        \sigma_{o/f,s} = \exp\!\big(-e^{\,b_s}\big),

    where :math:`\sigma(\cdot)` is the logistic sigmoid. The centre frequency
    :math:`f_0` therefore always lies in :math:`(f_{\min}, f_{\max}) =
    (1/\lambda_{\max},\, 1/\lambda_{\min})` and the bandwidth ratio
    :math:`\sigma_{o/f} \in (0, 1)`. The bandwidth enters through
    :math:`|\log\sigma_{o/f}| = e^{b_s}`, so :math:`b_s` is the log of the
    log-frequency width: scale-free and always positive.

    The initial values reproduce the given wavelengths and ``sigma_onf``
    exactly (thesis defaults: 36, 39, 42 px and 0.55).

    Args:
        wavelengths: initial centre wavelengths in pixels, one per scale.
        sigma_onf: initial :math:`k/\omega_0` (float, or one per scale), in (0, 1).
        min_wavelength, max_wavelength: bounds on the learnable wavelength
            (``min_wavelength >= 2`` is the Nyquist limit).
        learn_f0, learn_sigma: set to ``False`` to freeze that parameter.
        channels: if given, every one of ``channels`` input channels gets its own
            ``(S,)`` set of parameters (all initialised identically), and ``forward``
            returns ``(channels, S, H, W)``; otherwise one shared set and ``(S, H, W)``.

    Call ``forward(height, width)`` to get the bank of shape
    ``(n_scales, height, width)`` (or ``(channels, n_scales, height, width)``) in FFT order.
    """

    def __init__(
        self,
        wavelengths: Sequence[float] = (36.0, 39.0, 42.0),
        sigma_onf: float | Sequence[float] = 0.55,
        min_wavelength: float = 2.0,
        max_wavelength: float = 128.0,
        *,
        learn_f0: bool = True,
        learn_sigma: bool = True,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        if not 0 < min_wavelength < max_wavelength:
            raise ValueError("need 0 < min_wavelength < max_wavelength")
        wl = torch.as_tensor(list(wavelengths), dtype=torch.float32)
        if wl.ndim != 1 or wl.numel() < 1:
            raise ValueError("wavelengths must be a non-empty 1-D sequence")
        if (wl <= min_wavelength).any() or (wl >= max_wavelength).any():
            raise ValueError("initial wavelengths must lie strictly inside the bounds")
        sg = torch.as_tensor(sigma_onf, dtype=torch.float32).expand(wl.numel()).clone()
        if ((sg <= 0) | (sg >= 1)).any():
            raise ValueError("sigma_onf must be in (0, 1)")

        if channels is not None:
            if channels < 1:
                raise ValueError("channels must be >= 1")
            wl = wl.expand(channels, -1)
            sg = sg.expand(channels, -1)
        self.register_buffer("f_min", torch.tensor(1.0 / max_wavelength))
        self.register_buffer("f_max", torch.tensor(1.0 / min_wavelength))
        frac = (1.0 / wl - self.f_min) / (self.f_max - self.f_min)
        self.channels = channels
        self.raw_f0 = nn.Parameter(torch.logit(frac).clone(), requires_grad=learn_f0)
        self.log_bandwidth = nn.Parameter(torch.log(-torch.log(sg)).clone(), requires_grad=learn_sigma)

    @property
    def n_scales(self) -> int:
        return self.raw_f0.shape[-1]

    @property
    def f0(self) -> torch.Tensor:
        """Centre frequencies (cycles/pixel), bounded in ``(f_min, f_max)``."""
        return self.f_min + (self.f_max - self.f_min) * torch.sigmoid(self.raw_f0)

    @property
    def wavelengths(self) -> torch.Tensor:
        """Centre wavelengths in pixels, :math:`1/f_0`."""
        return 1.0 / self.f0

    @property
    def sigma_onf(self) -> torch.Tensor:
        r""":math:`k/\omega_0 = \exp(-e^{b})`, in (0, 1)."""
        return torch.exp(-torch.exp(self.log_bandwidth))

    def forward(self, height: int, width: int) -> torch.Tensor:
        wl, sg = self.wavelengths, self.sigma_onf  # (S,) or (C, S)
        return radial_log_gabor(
            height, width, wl[..., None, None], sg[..., None, None], device=wl.device, dtype=wl.dtype
        )

    def extra_repr(self) -> str:
        return f"n_scales={self.n_scales}, channels={self.channels}, f0 in ({self.f_min.item():.4g}, {self.f_max.item():.4g})"
