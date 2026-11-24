"""Validate the PyTorch monogenic signal against the NumPy thesis reference."""

import numpy as np
import pytest
import torch

from monogenet.features.local import local_amplitude
from monogenet.features.monogenic import monogenic_signal
from monogenet.filters.log_gabor import log_gabor_bank
from monogenet.reference import reference_pipeline

WAVELENGTHS = (6.0, 8.0, 10.0)


def _scene(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Synthetic image: rectangle (step edges), thin line, blob and noise."""
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    img = np.zeros((rows, cols))
    img[rows // 4 : rows // 2, cols // 4 : cols // 2] = 0.6
    img[:, int(cols * 0.7)] += 0.8
    img += 0.9 * np.exp(-(((x - cols * 0.8) ** 2 + (y - rows * 0.75) ** 2) / (2 * 3.0**2)))
    return img + 0.02 * np.random.default_rng(seed).standard_normal((rows, cols))


def _torch_mono(img: np.ndarray) -> np.ndarray:
    rows, cols = img.shape
    # Explicit wavelengths (not geometric), one scale per bank entry.
    bank = torch.stack(
        [log_gabor_bank(rows, cols, 1, wl, 1.0, 0.55, dtype=torch.float64)[0] for wl in WAVELENGTHS]
    )
    return monogenic_signal(torch.from_numpy(img), bank).numpy()  # (S, 3, H, W)


@pytest.mark.parametrize("shape", [(63, 65), (64, 96), (64, 63)])
def test_monogenic_matches_reference(shape: tuple[int, int]) -> None:
    # Agreement to round-off, including even sizes: the reference takes the real part of
    # the inverse FFT, which removes exactly the Nyquist term that the PyTorch Riesz
    # kernel zeroes explicitly.
    atol = 1e-10
    img = _scene(*shape)
    ref = reference_pipeline(img, WAVELENGTHS, 0.55)
    mono = _torch_mono(img)
    assert np.allclose(mono[:, 0], ref.even, atol=1e-10)
    assert np.allclose(mono[:, 1], ref.odd1, atol=atol)
    assert np.allclose(mono[:, 2], ref.odd2, atol=atol)
    amp = local_amplitude(torch.from_numpy(mono)).numpy()
    assert np.allclose(amp, ref.amplitude, atol=atol)


def test_ps_range_and_threshold() -> None:
    ref = reference_pipeline(_scene(64, 64), WAVELENGTHS)
    assert ref.ps.shape == (3, 64, 64)
    assert (ref.ps >= 0).all() and (ref.ps <= 1 + 1e-9).all()
    assert np.allclose(ref.threshold, ref.amplitude.mean(axis=(1, 2)))
    assert ref.ps.max() > 0.1  # the scene has symmetric structures


def test_ps_is_contrast_invariant() -> None:
    img = _scene(64, 64)
    a = reference_pipeline(img, WAVELENGTHS, eps=1e-12)
    b = reference_pipeline(7.0 * img, WAVELENGTHS, eps=1e-12)
    assert np.allclose(a.ps, b.ps, atol=1e-8)
    assert np.allclose(b.amplitude, 7.0 * a.amplitude, rtol=1e-9)


def test_ps_highlights_line_not_flat_background() -> None:
    n = 128
    img = np.zeros((n, n))
    img[:, n // 2] = 1.0
    ref = reference_pipeline(img, (8.0,), eps=1e-9)
    ps = ref.ps[0]
    assert ps[10, n // 2] > 0.5
    assert ps[10, n // 2 + 40] == pytest.approx(0.0, abs=1e-3)
    assert ps[10, n // 2] > 20 * ps[10, n // 2 + 40]


def test_constant_image_gives_zero_without_nan() -> None:
    ref = reference_pipeline(np.full((32, 32), 5.0), WAVELENGTHS)
    assert np.isfinite(ref.ps).all() and np.allclose(ref.ps, 0.0)
    assert np.allclose(ref.threshold, 0.0, atol=1e-12)


def test_default_scales_are_the_thesis_values() -> None:
    ref = reference_pipeline(_scene(128, 128))
    assert ref.even.shape == (3, 128, 128)  # wavelengths 36, 39, 42
    explicit = reference_pipeline(_scene(128, 128), (36.0, 39.0, 42.0), 0.55)
    assert np.array_equal(ref.ps, explicit.ps)


def test_invalid_image() -> None:
    with pytest.raises(ValueError):
        reference_pipeline(np.zeros((4, 4, 3)))
