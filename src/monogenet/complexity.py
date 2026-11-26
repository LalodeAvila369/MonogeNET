"""Parameter and FLOP counting, to compare PhaseBlock with ``Conv2d``.

Parameters are counted exactly. FLOPs are an *analytic estimate* of one forward
pass per sample (batch size 1), in the usual convention of 2 FLOPs per
multiply-accumulate:

* ``Conv2d``: ``2 * (C_in / groups) * k_h * k_w * C_out * H_out * W_out`` (+ bias adds).
  This matches ``torch.utils.flop_counter.FlopCounterMode`` (which ignores the bias).
* complex FFT/IFFT of an ``H x W`` array: ``5 * N * log2(N)`` with ``N = H * W``
  (the classic radix-2 estimate). The code uses full complex ``fft2`` even for real
  inputs, so this is not halved.
* complex x real product: 2 FLOPs per element; complex x complex: 6 FLOPs per element.
* elementwise phase features: :data:`FEATURE_FLOPS_PER_PIXEL` per feature map and pixel
  (a coarse constant for the sqrt/abs/divide chains; phase congruency counts
  :data:`PC_FLOPS_PER_PIXEL_SCALE` per scale and pixel).

Building the filter bank itself (``C * S`` radial filters, a few tens of FLOPs per
frequency sample, independent of the batch size) is reported separately as
``bank_flops`` and is *not* included in the per-sample total.

FFT-based and convolution FLOPs do not run at the same hardware efficiency, so FLOP
ratios indicate scaling, not wall-clock speed. Measure time separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import nn

from monogenet.layers.frontend import FEATURES as FRONTEND_FEATURES
from monogenet.layers.frontend import MonogenicFrontEnd
from monogenet.layers.phase_block import PER_CHANNEL_FEATURES, PER_SCALE_FEATURES, PhaseBlock

__all__ = [
    "Complexity",
    "count_parameters",
    "fft_flops",
    "conv2d_flops",
    "phase_block_flops",
    "front_end_flops",
    "estimate_flops",
    "compare_with_conv",
    "FEATURE_FLOPS_PER_PIXEL",
    "PC_FLOPS_PER_PIXEL_SCALE",
    "BANK_FLOPS_PER_SAMPLE",
]

FEATURE_FLOPS_PER_PIXEL = 10
PC_FLOPS_PER_PIXEL_SCALE = 30
BANK_FLOPS_PER_SAMPLE = 20  # radial log-Gabor value per frequency sample (log, exp, ...)
_MONO_UNPACK = 5  # amplitude shared by the features of one scale: squares, sums, sqrt


@dataclass(frozen=True)
class Complexity:
    """Parameter count and estimated forward FLOPs (per sample)."""

    params: int
    flops: int


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters of ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def fft_flops(height: int, width: int) -> int:
    """Estimated FLOPs of one complex 2-D FFT: ``5 N log2 N`` with ``N = height * width``."""
    n = height * width
    return int(5 * n * math.log2(n)) if n > 1 else 0


def conv2d_flops(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    height: int,
    width: int,
    *,
    groups: int = 1,
    bias: bool = False,
    stride: int = 1,
) -> int:
    """FLOPs of a ``same``-padded ``Conv2d`` (2 per MAC, optional bias adds)."""
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    out_h, out_w = -(-height // stride), -(-width // stride)
    macs = (in_channels // groups) * kh * kw * out_channels * out_h * out_w
    return 2 * macs + (out_channels * out_h * out_w if bias else 0)


def _spectral_flops(channels: int, scales: int, n: int, height: int, width: int) -> int:
    """FFT, bank product, Riesz products and 3 inverse FFTs for ``channels`` channels."""
    forward = channels * fft_flops(height, width)
    bank_product = channels * scales * 2 * n
    riesz_products = channels * scales * 2 * 6 * n
    inverse = 3 * channels * scales * fft_flops(height, width)
    return forward + bank_product + riesz_products + inverse


def phase_block_flops(block: PhaseBlock, height: int, width: int) -> Complexity:
    """Complexity of ``PhaseBlock`` for an input of spatial size ``height x width``."""
    n = height * width
    c, s = block.in_channels, block.n_scales
    n_scale_feats = sum(f in PER_SCALE_FEATURES for f in block.features)
    has_pc = any(f in PER_CHANNEL_FEATURES for f in block.features)
    flops = _spectral_flops(c, s, n, height, width)
    flops += c * s * n * (_MONO_UNPACK + n_scale_feats * FEATURE_FLOPS_PER_PIXEL)
    if has_pc:
        flops += c * s * n * PC_FLOPS_PER_PIXEL_SCALE
    flops += conv2d_flops(block.n_feature_channels, block.out_channels, 1, height, width, bias=block.mix.bias is not None)
    return Complexity(count_parameters(block), flops)


def front_end_flops(layer: MonogenicFrontEnd, height: int, width: int) -> Complexity:
    """Complexity of ``MonogenicFrontEnd`` for an input of size ``height x width``."""
    n = height * width
    c, s = layer.in_channels, layer.n_scales
    flops = _spectral_flops(c, s, n, height, width)
    flops += c * s * n * (_MONO_UNPACK + len(FRONTEND_FEATURES) * FEATURE_FLOPS_PER_PIXEL)
    return Complexity(count_parameters(layer), flops)


def bank_flops(channels: int, scales: int, height: int, width: int) -> int:
    """FLOPs to build the filter bank once per forward pass (not per sample)."""
    return channels * scales * height * width * BANK_FLOPS_PER_SAMPLE


def estimate_flops(module: nn.Module, input_shape: tuple[int, ...]) -> Complexity:
    """Complexity of a ``Conv2d``, ``PhaseBlock`` or ``MonogenicFrontEnd``.

    ``input_shape`` is ``(B, C, H, W)`` or ``(C, H, W)``; the batch size is ignored
    (per-sample numbers).
    """
    height, width = input_shape[-2:]
    if isinstance(module, PhaseBlock):
        return phase_block_flops(module, height, width)
    if isinstance(module, MonogenicFrontEnd):
        return front_end_flops(module, height, width)
    if isinstance(module, nn.Conv2d):
        flops = conv2d_flops(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            height,
            width,
            groups=module.groups,
            bias=module.bias is not None,
            stride=module.stride[0],
        )
        return Complexity(count_parameters(module), flops)
    raise TypeError(f"no FLOP model for {type(module).__name__}")


def compare_with_conv(
    block: PhaseBlock, height: int, width: int, kernel_size: int = 3
) -> dict[str, float | Complexity]:
    """Compare ``block`` with ``Conv2d(C_in, C_out, kernel_size, padding='same')``.

    Returns the two :class:`Complexity` values and the ratios ``conv / phase_block``
    (a ratio above 1 means the PhaseBlock is cheaper).
    """
    conv = nn.Conv2d(block.in_channels, block.out_channels, kernel_size, padding="same", bias=block.mix.bias is not None)
    pb = phase_block_flops(block, height, width)
    cv = estimate_flops(conv, (block.in_channels, height, width))
    return {
        "phase_block": pb,
        "conv2d": cv,
        "params_ratio": cv.params / pb.params,
        "flops_ratio": cv.flops / pb.flops,
    }
