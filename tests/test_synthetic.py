"""Synthetic test suite: lines, step edges and blobs under contrast, noise and geometry changes.

Each scene has a known answer: where PS, PA, PC and the feature type should be high
or low, and how the maps must (not) change when the image is transformed.
"""

import math

import pytest
import torch

from monogenet.features.congruency import phase_congruency
from monogenet.features.local import feature_type, local_orientation, local_phase
from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import phase_asymmetry, phase_symmetry
from monogenet.filters.log_gabor import log_gabor_bank

D = torch.float64
N = 129  # odd: no Nyquist frequency, so rotations and flips are exact symmetries
C = N // 2  # centre pixel
BANK = log_gabor_bank(N, N, 3, 5.0, 1.5, dtype=D)  # wavelengths 5, 7.5, 11.25 px


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(torch.arange(N, dtype=D), torch.arange(N, dtype=D), indexing="ij")
    return x - C, y - C


def line_image(width: float = 1.0) -> torch.Tensor:
    x, _ = _grid()
    return (x.abs() <= width / 2).to(D)


def diagonal_line_image() -> torch.Tensor:
    x, y = _grid()
    return (((x - y).abs() / math.sqrt(2)) <= 0.8).to(D)


def edge_image() -> torch.Tensor:
    """Vertical step edge between columns C and C+1 (half a pixel right of the centre)."""
    x, _ = _grid()
    return (x > 0).to(D)


def blob_image(sigma: float = 3.0, amplitude: float = 1.0) -> torch.Tensor:
    x, y = _grid()
    return amplitude * torch.exp(-(x**2 + y**2) / (2 * sigma**2))


def maps(img: torch.Tensor, eps: float = 1e-9) -> dict[str, torch.Tensor]:
    mono = monogenic_signal(img, BANK)
    return {
        "ps": phase_symmetry(mono, eps=eps).mean(dim=-3),
        "pa": phase_asymmetry(mono, eps=eps).mean(dim=-3),
        "pc": phase_congruency(mono, eps=eps),
        "ft": feature_type(mono).mean(dim=-3),
        "mono": mono,
    }


# ------------------------------------------------------------------ known answers


def test_line_is_symmetric_everywhere_along_it() -> None:
    m = maps(line_image())
    for row in (10, C, N - 10):
        assert m["ps"][row, C] > 0.5
        assert m["pa"][row, C] < 0.05
        assert m["pc"][row, C] > 0.5
        assert m["ft"][row, C] > math.pi / 2 - 0.15
    assert m["ps"][C, C + 40] < 1e-3  # background


def test_dark_line_has_negative_feature_type_same_ps() -> None:
    bright, dark = maps(line_image()), maps(-line_image())
    assert dark["ft"][C, C] < -math.pi / 2 + 0.15
    assert torch.allclose(bright["ps"], dark["ps"], atol=1e-9)
    assert torch.allclose(bright["pc"], dark["pc"], atol=1e-9)
    assert torch.allclose(bright["ft"], -dark["ft"], atol=1e-9)


def test_step_edge_is_asymmetric() -> None:
    m = maps(edge_image())
    for row in (10, C, N - 10):
        assert m["pa"][row, C] > 0.2 and m["pa"][row, C + 1] > 0.2
        assert m["ps"][row, C] < m["pa"][row, C] and m["ps"][row, C + 1] < m["pa"][row, C + 1]
        assert m["pc"][row, C] > 0.3  # edges are congruent
    # The edge lies half a pixel between C and C+1: feature type is antisymmetric around it
    # (up to the far-away wrap-around edge of the periodic image).
    assert torch.allclose(m["ft"][:, C], -m["ft"][:, C + 1], atol=5e-3)
    assert m["pa"][C, C + 30] < 1e-3 and m["pa"][C, C - 30] < 1e-3  # flat regions


def test_blob_is_symmetric_at_centre_and_ring_is_asymmetric() -> None:
    m = maps(blob_image(sigma=3.0))
    assert m["ps"][C, C] > 0.3 and m["ps"][C, C] > 5 * m["pa"][C, C]
    assert m["ft"][C, C] > math.pi / 2 - 0.3
    assert m["pc"][C, C] > 0.3
    # Dark blob: same PS, opposite feature type.
    dark = maps(-blob_image(sigma=3.0))
    assert torch.allclose(dark["ps"], m["ps"], atol=1e-9)
    assert dark["ft"][C, C] < -(math.pi / 2 - 0.3)
    # PA peaks on the flanks of the blob, not at its centre.
    flank = m["pa"][C, C + 3 : C + 9].max()
    assert flank > m["pa"][C, C]


def test_blob_phase_is_even_at_centre_and_odd_on_the_flank() -> None:
    mono = maps(blob_image(sigma=3.0))["mono"]
    assert local_phase(mono)[1, C, C] < 0.2  # middle scale, centre: bright symmetric
    assert 0.6 < local_phase(mono)[1, C, C + 5] < math.pi - 0.6  # flank: mixed


def test_orientation_of_lines_and_edges() -> None:
    # Gradient direction is perpendicular to the line: vertical line -> 0, diagonal -> -pi/4.
    # Evaluate on the flank (2 px off the centre): at a line's centre the odd part is zero.
    mono_v = monogenic_signal(line_image(), BANK)
    mono_e = monogenic_signal(edge_image(), BANK)
    mono_d = monogenic_signal(diagonal_line_image(), BANK)
    assert abs(local_orientation(mono_v)[1, 20, C + 2]) < 0.05
    assert abs(local_orientation(mono_e)[1, 20, C]) < 0.05
    theta = local_orientation(mono_d)[1, C - 1, C + 1]  # one step perpendicular to the line
    assert abs(theta + math.pi / 4) < 0.1  # direction (1, -1), modulo pi


def test_blob_scale_selectivity() -> None:
    # Relative to the coarse blob, a fine blob puts more of its response in the finest scale.
    def amplitude_at_centre(sigma: float) -> torch.Tensor:
        return monogenic_signal(blob_image(sigma), BANK)[:, :, C, C].pow(2).sum(dim=-1).sqrt()

    fine, coarse = amplitude_at_centre(1.2), amplitude_at_centre(4.5)
    assert fine[0] / fine[2] > 1.2 * coarse[0] / coarse[2]
    assert fine[0] / fine[2] > 1.0 > coarse[0] / coarse[2]


# -------------------------------------------------------- contrast changes (invariance)


@pytest.mark.parametrize("factory", [line_image, edge_image, blob_image])
@pytest.mark.parametrize("contrast", [0.01, 0.2, 5.0, 100.0])
def test_scaling_contrast_leaves_phase_features_unchanged(factory, contrast: float) -> None:
    ref = maps(factory(), eps=1e-14)
    out = maps(contrast * factory(), eps=1e-14)
    for key in ("ps", "pa", "pc"):
        assert torch.allclose(out[key], ref[key], atol=1e-7), key
    amp = ref["mono"].pow(2).sum(dim=-3).sqrt().amax(dim=0)
    # Feature type is ill-conditioned where the response is tiny: the absolute 1e-24 guard
    # under the square roots perturbs the angle by about 1e-12 / amplitude.
    mask = amp > 1e-3 * amp.max()
    assert torch.allclose(out["ft"][mask], ref["ft"][mask], atol=1e-5)
    assert torch.allclose(out["mono"], contrast * ref["mono"], atol=1e-9 * contrast)


@pytest.mark.parametrize("factory", [line_image, edge_image, blob_image])
def test_dc_offset_is_ignored(factory) -> None:
    ref, out = maps(factory()), maps(factory() + 37.0)
    for key in ("ps", "pa", "pc"):
        assert torch.allclose(out[key], ref[key], atol=1e-8), key
    # Feature type is ill-conditioned where the response is tiny, so compare it where
    # the amplitude is non-negligible.
    amp = ref["mono"].pow(2).sum(dim=-3).sqrt().amax(dim=0)
    mask = amp > 1e-3 * amp.max()
    assert torch.allclose(out["ft"][mask], ref["ft"][mask], atol=1e-6)


def test_monotonic_contrast_change_keeps_feature_locations() -> None:
    # Gamma correction is a non-linear contrast change: values move, structure stays.
    base = blob_image(3.0) + line_image() * 0.0
    img = 0.8 * line_image() + 0.8 * blob_image(3.0).roll(shifts=40, dims=1)
    ref = maps(img)["pc"]
    gamma = maps(img.clamp(min=0) ** 0.5)["pc"]
    top_ref = ref.flatten().topk(30).indices
    top_gamma = gamma.flatten().topk(30).indices
    overlap = len(set(top_ref.tolist()) & set(top_gamma.tolist()))
    assert overlap >= 15
    assert base.shape == img.shape


def test_raw_responses_do_change_with_contrast() -> None:
    # Sanity check of the invariance tests: amplitudes scale, features do not.
    a = maps(blob_image())["mono"]
    b = maps(0.1 * blob_image())["mono"]
    assert torch.allclose(b, 0.1 * a, atol=1e-12) and not torch.allclose(a, b, atol=1e-3)


# ------------------------------------------------------------------------------ noise


def _noisy(img: torch.Tensor, sigma: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return img + sigma * torch.randn(N, N, generator=g, dtype=D)


def test_pc_survives_moderate_noise_and_degrades_gracefully() -> None:
    clean_line = maps(line_image())["pc"][C, C].item()
    levels = [0.0, 0.05, 0.2, 0.6]
    pcs = [maps(_noisy(line_image(), s))["pc"][20:-20, C].mean().item() for s in levels]
    assert pcs[1] > 0.5 * clean_line  # strong line, weak noise: still detected
    assert pcs[0] >= pcs[1] >= pcs[2] >= pcs[3] - 1e-6  # monotone degradation
    # Noise alone never produces a strong response.
    noise_only = maps(_noisy(torch.zeros(N, N, dtype=D), 0.3))["pc"]
    assert noise_only.mean() < 0.15 and noise_only.mean() < pcs[1]


def test_line_stands_out_from_noise_background() -> None:
    m = maps(_noisy(line_image(), 0.1))
    on_line = m["ps"][20:-20, C].mean()
    background = torch.cat([m["ps"][20:-20, : C - 15].flatten(), m["ps"][20:-20, C + 15 :].flatten()]).mean()
    assert on_line > 5 * background


def test_edge_stays_asymmetric_under_noise() -> None:
    m = maps(_noisy(edge_image(), 0.1))
    assert m["pa"][20:-20, C : C + 2].mean() > 2 * m["ps"][20:-20, C : C + 2].mean()


# -------------------------------------------------------------------------- geometry


def test_rotation_by_90_degrees_is_equivariant() -> None:
    g = torch.Generator().manual_seed(3)
    img = blob_image(3.0) + line_image().roll(25, 0) * 0.7 + 0.1 * torch.randn(N, N, generator=g, dtype=D)
    ref, rot = maps(img), maps(torch.rot90(img, 1, (0, 1)))
    for key in ("ps", "pa", "pc"):
        assert torch.allclose(rot[key], torch.rot90(ref[key], 1, (0, 1)), atol=1e-9), key


def test_flips_and_translations_are_equivariant() -> None:
    g = torch.Generator().manual_seed(4)
    img = blob_image(3.0) + 0.7 * edge_image().roll(20, 1) + 0.05 * torch.randn(N, N, generator=g, dtype=D)
    ref = maps(img)
    flipped = maps(torch.flip(img, dims=(1,)))
    shifted = maps(torch.roll(img, shifts=(7, -11), dims=(0, 1)))
    for key in ("ps", "pa", "pc"):
        assert torch.allclose(flipped[key], torch.flip(ref[key], dims=(1,)), atol=1e-9), key
        assert torch.allclose(shifted[key], torch.roll(ref[key], shifts=(7, -11), dims=(0, 1)), atol=1e-9), key


def test_feature_ranges() -> None:
    g = torch.Generator().manual_seed(5)
    m = maps(blob_image() + line_image() + _noisy(torch.zeros(N, N, dtype=D), 0.1, 5))
    assert (m["ps"] >= 0).all() and (m["ps"] <= 1 + 1e-9).all()
    assert (m["pa"] >= 0).all() and (m["pa"] <= 1 + 1e-9).all()
    assert (m["pc"] >= 0).all() and (m["pc"] <= 1 + 1e-9).all()
    assert (m["ft"].abs() <= math.pi / 2 + 1e-9).all()
    assert g is not None
