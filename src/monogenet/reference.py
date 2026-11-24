"""Plain NumPy reference of the thesis pipeline, used to validate the PyTorch code.

Deliberately independent of :mod:`torch` and of the rest of the package (own
frequency grid, own log-Gabor and Riesz code), so that agreement with the
PyTorch implementation is a real cross-check and not a shared bug.

Pipeline (E. de Avila-Armenta et al., CCIS 2348, 2025)::

    FFT -> radial log-Gabor bank -> Riesz (H1 = i u/|w|, H2 = i v/|w|)
        -> monogenic tensor [even, odd1, odd2] per scale
        -> PS = floor(|even| - |sqrt(odd1^2 + odd2^2)| - T)
                / (sqrt(even^2 + odd1^2 + odd2^2) + eps)

with ``floor(x) = max(x, 0)`` and ``T = exp(log(mean(amplitude)))`` computed
per scale from the local amplitude. Tuned values on hip ultrasound: centre
wavelengths 36, 39, 42 px and k/w0 = 0.55 (the defaults here).

Note: the thesis text writes the threshold as ``exp(log(mean(amplitude)))``,
which is the mean amplitude itself; it is implemented literally, per scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["ReferenceResult", "reference_pipeline"]


@dataclass
class ReferenceResult:
    """Outputs of :func:`reference_pipeline` (all float64, scales first)."""

    even: np.ndarray  # (S, H, W)
    odd1: np.ndarray  # (S, H, W)
    odd2: np.ndarray  # (S, H, W)
    amplitude: np.ndarray  # (S, H, W)
    threshold: np.ndarray  # (S,)
    ps: np.ndarray  # (S, H, W)


def reference_pipeline(
    image: np.ndarray,
    wavelengths: Sequence[float] = (36.0, 39.0, 42.0),
    k_over_w0: float = 0.55,
    eps: float = 1e-6,
) -> ReferenceResult:
    """Run the thesis pipeline on a 2-D ``image`` and return per-scale results."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2-D")
    rows, cols = img.shape

    # Frequency grid in cycles/pixel, FFT order, built independently of monogenet.filters.
    u, v = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows), indexing="xy")
    radius = np.hypot(u, v)
    radius[0, 0] = 1.0  # avoid 0/0 at DC; DC is zeroed below

    # Riesz transfer functions.
    h1 = 1j * u / radius
    h2 = 1j * v / radius

    spectrum = np.fft.fft2(img)
    evens, odds1, odds2 = [], [], []
    for wavelength in wavelengths:
        f0 = 1.0 / wavelength
        log_gabor = np.exp(-(np.log(radius / f0) ** 2) / (2.0 * np.log(k_over_w0) ** 2))
        log_gabor[0, 0] = 0.0
        band = spectrum * log_gabor
        evens.append(np.real(np.fft.ifft2(band)))
        odds1.append(np.real(np.fft.ifft2(band * h1)))
        odds2.append(np.real(np.fft.ifft2(band * h2)))

    even, odd1, odd2 = np.stack(evens), np.stack(odds1), np.stack(odds2)
    amplitude = np.sqrt(even**2 + odd1**2 + odd2**2)
    # T = exp(log(mean(amplitude))) per scale; guard log(0) for an all-zero response.
    mean_amp = amplitude.mean(axis=(1, 2))
    with np.errstate(divide="ignore"):
        threshold = np.where(mean_amp > 0, np.exp(np.log(np.where(mean_amp > 0, mean_amp, 1.0))), 0.0)
    odd_mag = np.sqrt(odd1**2 + odd2**2)
    numerator = np.maximum(np.abs(even) - np.abs(odd_mag) - threshold[:, None, None], 0.0)
    ps = numerator / (amplitude + eps)
    return ReferenceResult(even, odd1, odd2, amplitude, threshold, ps)
