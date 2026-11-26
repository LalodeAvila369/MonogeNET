"""PhaseBlock: depthwise learnable log-Gabor + Riesz -> PS/PA -> 1x1 convolution."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from monogenet.features.congruency import phase_congruency
from monogenet.features.local import local_amplitude, local_phase
from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import phase_asymmetry, phase_symmetry
from monogenet.features.threshold import LearnableThreshold
from monogenet.filters.learnable import LearnableLogGaborBank
from monogenet.filters.window import apply_edge_taper

__all__ = ["PhaseBlock"]

#: Features computed per input channel and per scale.
PER_SCALE_FEATURES = ("energy", "phase", "ps", "pa")
#: Features computed per input channel with all scales combined.
PER_CHANNEL_FEATURES = ("pc",)


class PhaseBlock(nn.Module):
    r"""Drop-in alternative to ``Conv2d`` built from local-phase features.

    ::

        x (B, C, H, W) -> depthwise learnable log-Gabor bank -> Riesz
          -> monogenic signal (e, o1, o2) per channel and scale
          -> local-phase features (PS, PA, ...) -> 1x1 conv -> (B, C_out, H, W)

    Every input channel is filtered separately (depthwise) by its own learnable
    log-Gabor bank with ``S`` scales, so all spatial filtering costs
    ``3 * S`` parameters per channel (centre frequency, bandwidth and threshold
    gain per scale) instead of ``k^2 * C_out`` weights. The ``C * S * F`` feature
    maps (``F`` features per scale) are then mixed across channels by a 1x1
    convolution. With the default features ``("ps", "pa")`` the filtering stage is
    invariant to the contrast of each input channel, which is what makes local
    phase attractive for noisy modalities such as ultrasound.

    The convolution is computed in the frequency domain, so the block sees the
    image as periodic; ``taper`` optionally tapers the borders to reduce
    wrap-around artefacts. The spatial size is preserved (``same`` output).

    Args:
        in_channels, out_channels: channel counts.
        wavelengths: initial centre wavelengths in pixels, one per scale. The
            default suits feature maps of roughly 16 to 64 pixels; choose them
            relative to the map size (they must be above the Nyquist limit of 2).
        sigma_onf: initial :math:`k/\omega_0` (float or one per scale).
        features: any of ``"ps"``, ``"pa"``, ``"energy"``, ``"phase"`` (per scale) and
            ``"pc"`` (phase congruency over all scales, one map per channel).
        threshold_gain: initial gain of the noise threshold ``T = k * mean(A)``.
        sharpness: softplus sharpness of the PS/PA/PC floor; ``None`` is the hard
            floor (see :class:`MonogenicFrontEnd`).
        per_channel: separate filter parameters for every input channel (default);
            ``False`` shares one bank across channels.
        bias: add a bias to the 1x1 convolution.
        min_wavelength, max_wavelength, taper, eps: as in :class:`MonogenicFrontEnd`.

    Input ``(B, C_in, H, W)``, output ``(B, C_out, H, W)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelengths: Sequence[float] = (4.0, 6.0, 9.0),
        sigma_onf: float | Sequence[float] = 0.55,
        features: Sequence[str] = ("ps", "pa"),
        threshold_gain: float = 1.0,
        *,
        sharpness: float | None = 20.0,
        per_channel: bool = True,
        bias: bool = True,
        min_wavelength: float = 2.0,
        max_wavelength: float = 128.0,
        taper: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be >= 1")
        features = tuple(features)
        allowed = PER_SCALE_FEATURES + PER_CHANNEL_FEATURES
        if not features or any(f not in allowed for f in features) or len(set(features)) != len(features):
            raise ValueError(f"features must be distinct names from {allowed}, got {features}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.per_channel = per_channel
        self.sharpness = sharpness
        self.taper = taper
        self.eps = eps
        channels = in_channels if per_channel else None
        self.bank = LearnableLogGaborBank(
            wavelengths, sigma_onf, min_wavelength, max_wavelength, channels=channels
        )
        self.threshold = LearnableThreshold(self.bank.n_scales, threshold_gain, channels=channels)
        self.mix = nn.Conv2d(self.n_feature_channels, out_channels, kernel_size=1, bias=bias)

    @property
    def n_scales(self) -> int:
        return self.bank.n_scales

    @property
    def n_feature_channels(self) -> int:
        """Channels entering the 1x1 convolution: ``C * (S * F_scale + F_channel)``."""
        per_scale = sum(f in PER_SCALE_FEATURES for f in self.features)
        per_channel = sum(f in PER_CHANNEL_FEATURES for f in self.features)
        return self.in_channels * (self.n_scales * per_scale + per_channel)

    def phase_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature maps before the 1x1 mixing, shape ``(B, n_feature_channels, H, W)``.

        Channel order: for every input channel ``c`` the per-scale features
        (scale-major, in the order given by ``features``), then the per-channel
        ones (``"pc"``).
        """
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected input (B, {self.in_channels}, H, W), got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if self.taper:
            x = apply_edge_taper(x, self.taper)
        mono = monogenic_signal(x, self.bank(h, w))  # (B, C, S, 3, H, W)
        threshold = self.threshold(mono)  # (B, C, S, 1, 1)
        computed = {
            "energy": lambda: local_amplitude(mono),
            "phase": lambda: local_phase(mono),
            "ps": lambda: phase_symmetry(mono, threshold, self.eps, self.sharpness),
            "pa": lambda: phase_asymmetry(mono, threshold, self.eps, self.sharpness),
        }
        scale_feats = [computed[f]() for f in self.features if f in PER_SCALE_FEATURES]
        maps = []
        if scale_feats:
            maps.append(torch.stack(scale_feats, dim=-3).flatten(2, 3))  # (B, C, S * F, H, W)
        if "pc" in self.features:
            total = threshold.sum(dim=-3)  # (B, C, 1, 1): sum of per-scale thresholds
            pc = phase_congruency(mono, total, self.eps, sharpness=self.sharpness)
            maps.append(pc.unsqueeze(2))  # (B, C, 1, H, W)
        return torch.cat(maps, dim=2).flatten(1, 2)  # (B, C * (S * F + F'), H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.phase_features(x))

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, n_scales={self.n_scales}, "
            f"features={self.features}, per_channel={self.per_channel}"
        )
