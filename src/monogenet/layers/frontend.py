"""MonogenicFrontEnd: image -> [energy, phase, PS, PA] per scale."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from monogenet.features.local import local_amplitude, local_phase
from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import phase_asymmetry, phase_symmetry
from monogenet.features.threshold import LearnableThreshold
from monogenet.filters.learnable import LearnableLogGaborBank
from monogenet.filters.window import apply_edge_taper

__all__ = ["MonogenicFrontEnd"]

FEATURES = ("energy", "phase", "ps", "pa")


class MonogenicFrontEnd(nn.Module):
    r"""Entry layer turning an image into learnable local-phase feature maps.

    Pipeline (all differentiable, all in the frequency domain)::

        image -> [taper] -> FFT -> learnable log-Gabor bank -> Riesz
              -> monogenic tensor (e, o1, o2) per scale
              -> energy A, phase phi, phase symmetry PS, phase asymmetry PA

    For each of the ``S`` scales the layer outputs four channels, in this order:

    * ``energy``: local amplitude :math:`A = \sqrt{e^2 + o_1^2 + o_2^2}`;
    * ``phase``: local phase :math:`\varphi = \mathrm{atan2}(|o|, e) \in [0, \pi]`;
    * ``ps``: phase symmetry (contrast invariant, in [0, 1] for the hard floor);
    * ``pa``: phase asymmetry (contrast invariant, in [0, 1] for the hard floor).

    With ``in_channels = C`` input channels (e.g. 3 for RGB) every channel is
    processed independently and the outputs are concatenated:
    ``out_channels = 4 * S * C`` and channel ``(c * S + s) * 4 + k`` is feature ``k``
    of scale ``s`` of input channel ``c``. Energy scales with image contrast, the
    other three do not.

    Learnable parameters: per scale a bounded centre frequency and a log-space
    bandwidth (:class:`LearnableLogGaborBank`) and a positive threshold gain
    (:class:`LearnableThreshold`), i.e. ``3 * S`` in total. By default they are
    shared by all input channels; with ``per_channel=True`` each channel has its own
    copy (``3 * S * C``, identical at initialisation). The defaults are the thesis
    values for hip ultrasound: wavelengths 36, 39, 42 px, k/w0 = 0.55 and threshold
    gain 1 (T = mean amplitude).

    Args:
        wavelengths: initial centre wavelengths in pixels, one per scale.
        sigma_onf: initial :math:`k/\omega_0` (float or one per scale).
        threshold_gain: initial gain ``k`` of the noise threshold ``T = k * mean(A)``.
        min_wavelength, max_wavelength: bounds of the learnable wavelength.
        sharpness: softplus sharpness of the PS/PA floor. ``None`` gives the hard
            floor of the thesis (no gradient below the threshold); the default 20
            keeps gradients flowing during training.
        taper: Tukey taper fraction applied to the input to reduce FFT wrap-around
            artefacts, or ``None`` (default) for none.
        eps: stabiliser of the PS/PA denominators.
        learn_f0, learn_sigma, learn_threshold: freeze individual parameter groups.
        in_channels: number of input channels ``C``.
        per_channel: give every input channel its own filter and threshold parameters.

    Input: ``(B, C, H, W)`` (or ``(B, H, W)`` when ``C = 1``). Output:
    ``(B, 4 * S * C, H, W)``.
    """

    def __init__(
        self,
        wavelengths: Sequence[float] = (36.0, 39.0, 42.0),
        sigma_onf: float | Sequence[float] = 0.55,
        threshold_gain: float = 1.0,
        min_wavelength: float = 2.0,
        max_wavelength: float = 128.0,
        *,
        sharpness: float | None = 20.0,
        taper: float | None = None,
        eps: float = 1e-6,
        learn_f0: bool = True,
        learn_sigma: bool = True,
        learn_threshold: bool = True,
        in_channels: int = 1,
        per_channel: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        self.in_channels = in_channels
        self.per_channel = per_channel
        channels = in_channels if per_channel else None
        self.bank = LearnableLogGaborBank(
            wavelengths,
            sigma_onf,
            min_wavelength,
            max_wavelength,
            learn_f0=learn_f0,
            learn_sigma=learn_sigma,
            channels=channels,
        )
        self.threshold = LearnableThreshold(self.bank.n_scales, threshold_gain, channels=channels)
        self.threshold.raw_gain.requires_grad_(learn_threshold)
        self.sharpness = sharpness
        self.taper = taper
        self.eps = eps

    @property
    def n_scales(self) -> int:
        return self.bank.n_scales

    @property
    def out_channels(self) -> int:
        return len(FEATURES) * self.n_scales * self.in_channels

    @property
    def feature_names(self) -> list[str]:
        """Channel names in output order.

        ``['energy_0', 'phase_0', 'ps_0', 'pa_0', 'energy_1', ...]`` for one input
        channel and ``['c0_energy_0', ..., 'c1_energy_0', ...]`` for several.
        """
        if self.in_channels == 1:
            return [f"{name}_{s}" for s in range(self.n_scales) for name in FEATURES]
        return [
            f"c{c}_{name}_{s}"
            for c in range(self.in_channels)
            for s in range(self.n_scales)
            for name in FEATURES
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and self.in_channels == 1:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            expected = f"(B, {self.in_channels}, H, W)"
            if self.in_channels == 1:
                expected += " or (B, H, W)"
            raise ValueError(f"expected input {expected}, got {tuple(x.shape)}")
        if not x.is_floating_point():
            x = x.to(self.bank.raw_f0.dtype)
        h, w = x.shape[-2:]
        if self.taper:
            x = apply_edge_taper(x, self.taper)
        mono = monogenic_signal(x, self.bank(h, w))  # (B, C, S, 3, H, W)
        threshold = self.threshold(mono)
        feats = torch.stack(
            (
                local_amplitude(mono),
                local_phase(mono),
                phase_symmetry(mono, threshold, self.eps, self.sharpness),
                phase_asymmetry(mono, threshold, self.eps, self.sharpness),
            ),
            dim=-3,
        )  # (B, C, S, 4, H, W)
        return feats.flatten(1, 3)  # (B, C * S * 4, H, W): channel = (c * S + s) * 4 + k

    def extra_repr(self) -> str:
        return (
            f"n_scales={self.n_scales}, in_channels={self.in_channels}, "
            f"per_channel={self.per_channel}, out_channels={self.out_channels}, "
            f"sharpness={self.sharpness}, taper={self.taper}"
        )
