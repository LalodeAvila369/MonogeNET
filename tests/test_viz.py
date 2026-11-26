"""Tests for filter visualisation: numerical responses always, plots when matplotlib is present."""

import math

import pytest
import torch

from monogenet.filters.learnable import LearnableLogGaborBank
from monogenet.filters.log_gabor import log_gabor_bank
from monogenet.layers.frontend import MonogenicFrontEnd
from monogenet.layers.phase_block import PhaseBlock
from monogenet.viz import filter_responses

N = 65
C = N // 2


def test_shapes_and_keys() -> None:
    r = filter_responses(log_gabor_bank(N, N, 3, 6.0, 1.5))
    assert set(r) == {"frequency", "even", "odd1", "odd2", "magnitude"}
    assert all(v.shape == (3, N, N) for v in r.values())


def test_frequency_response_is_centred_with_zero_dc_and_ring_peak() -> None:
    wavelength = 13.0  # f0 = 5/65 lies on the grid
    r = filter_responses(log_gabor_bank(N, N, 1, wavelength, 1.0))
    g = r["frequency"][0]
    assert g[C, C] == 0.0  # DC at the centre after fftshift
    assert g.max() == pytest.approx(1.0)
    assert g[C, C + 5] == pytest.approx(1.0) and g[C - 5, C] == pytest.approx(1.0)  # ring at |w| = f0


def test_even_kernel_is_symmetric_zero_mean_and_peaked_at_centre() -> None:
    k = filter_responses(log_gabor_bank(N, N, 2, 6.0, 1.5))["even"]
    for s in range(2):
        assert torch.allclose(k[s], torch.flip(k[s], dims=(0, 1)), atol=1e-6)  # point symmetry
        assert k[s].sum().abs() < 1e-4  # G(0) = 0 -> no DC
        assert int(k[s].argmax()) == C * N + C  # central lobe is the global maximum


def test_odd_kernels_are_antisymmetric_along_their_axis() -> None:
    r = filter_responses(log_gabor_bank(N, N, 1, 8.0, 1.0))
    o1, o2 = r["odd1"][0], r["odd2"][0]
    assert torch.allclose(o1, -torch.flip(o1, dims=(1,)), atol=1e-6)  # odd in x
    assert torch.allclose(o1, torch.flip(o1, dims=(0,)), atol=1e-6)  # even in y
    assert torch.allclose(o2, -torch.flip(o2, dims=(0,)), atol=1e-6)  # odd in y
    assert torch.allclose(o2, torch.flip(o2, dims=(1,)), atol=1e-6)  # even in x
    assert o1[C, C].abs() < 1e-8 and o2[C, C].abs() < 1e-8  # zero at the centre
    assert torch.allclose(o1, o2.T, atol=1e-6)  # isotropic bank: x and y kernels are rotations
    mag = r["magnitude"][0]
    assert torch.allclose(mag, torch.sqrt(r["even"][0] ** 2 + o1**2 + o2**2), atol=1e-7)
    assert int(mag.argmax()) == C * N + C  # the kernel energy peaks at the centre


def test_sources_module_layers_and_tensor_agree() -> None:
    module = LearnableLogGaborBank((6.0, 9.0), 0.5, min_wavelength=3.0)
    expected = module(N, N).detach()
    for source in (module, expected, MonogenicFrontEnd((6.0, 9.0), 0.5, min_wavelength=3.0), PhaseBlock(2, 3, (6.0, 9.0), 0.5)):
        r = filter_responses(source, N)
        assert torch.allclose(r["frequency"], torch.fft.fftshift(expected, dim=(-2, -1)), atol=1e-5)
    assert filter_responses(module, 33, 49)["even"].shape == (2, 33, 49)  # grid size for modules
    assert not filter_responses(module, N)["even"].requires_grad  # detached


def test_per_channel_bank_selects_channel() -> None:
    block = PhaseBlock(3, 2, (6.0, 9.0), per_channel=True)
    with torch.no_grad():
        block.bank.raw_f0[2] += 1.0  # channel 2 gets different filters
    a, b = filter_responses(block, N, channel=0), filter_responses(block, N, channel=2)
    assert not torch.allclose(a["frequency"], b["frequency"], atol=1e-3)
    assert torch.allclose(a["frequency"], filter_responses(block, N, channel=1)["frequency"], atol=1e-6)
    with pytest.raises(ValueError):
        filter_responses(block, N, channel=3)


def test_learned_wavelength_moves_the_ring() -> None:
    module = LearnableLogGaborBank((10.0,), 0.55, min_wavelength=3.0, max_wavelength=40.0)

    def peak_radius(m: LearnableLogGaborBank) -> float:
        g = filter_responses(m, 129)["frequency"][0]
        u = torch.fft.fftshift(torch.fft.fftfreq(129))
        return float(u[g[64].argmax()].abs())  # along the u axis through the centre

    before = peak_radius(module)
    with torch.no_grad():
        module.raw_f0 += 0.8  # larger f0 -> shorter wavelength
    assert peak_radius(module) > before and before == pytest.approx(1 / 10, abs=1 / 129)


def test_invalid_sources() -> None:
    with pytest.raises(TypeError):
        filter_responses(object())
    with pytest.raises(ValueError):
        filter_responses(torch.rand(4, 5))


# ------------------------------------------------------------------------------ plots


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_plot_filter_bank(plt, tmp_path) -> None:
    from monogenet.viz import plot_filter_bank

    fig = plot_filter_bank(MonogenicFrontEnd((6.0, 9.0, 13.0), min_wavelength=3.0), 65)
    assert len(fig.axes) == 4 * 3
    assert "scale 0" in fig.axes[0].get_title() and "6.0" in fig.axes[0].get_title()
    out = tmp_path / "bank.png"
    fig.savefig(out)
    assert out.stat().st_size > 5_000
    # Bare tensor (no wavelength info), cropping, non-square grid.
    fig2 = plot_filter_bank(log_gabor_bank(48, 64, 2, 6.0, 1.5), crop=10)
    assert len(fig2.axes) == 8 and fig2.axes[0].get_title() == "scale 0"


def test_plot_radial_profiles(plt) -> None:
    from monogenet.viz import plot_radial_profiles

    before = LearnableLogGaborBank((6.0, 9.0), min_wavelength=3.0)
    after = LearnableLogGaborBank((7.0, 11.0), 0.45, min_wavelength=3.0)
    fig = plot_radial_profiles({"initial": before, "learned": after})
    ax = fig.axes[0]
    assert len(ax.lines) == 4 and ax.get_xscale() == "log"
    assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == ["initial", "learned"]
    single = plot_radial_profiles(before, log_x=False)
    assert len(single.axes[0].lines) == 2 and single.axes[0].get_xscale() == "linear"
    # The profile peaks at f0 = 1 / wavelength (6 px -> 0.1667 cycles/pixel).
    line = ax.lines[0]
    x, y = line.get_xdata(), line.get_ydata()
    assert float(x[int(y.argmax())]) == pytest.approx(1 / 6.0, abs=0.01) and math.isclose(float(y.max()), 1.0, abs_tol=0.05)
