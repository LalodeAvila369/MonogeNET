"""Tests for MonogenicFrontEnd."""

import numpy as np
import pytest
import torch

from monogenet.features.local import local_amplitude, local_phase
from monogenet.features.monogenic import monogenic_signal
from monogenet.features.phase import phase_asymmetry, phase_symmetry
from monogenet.layers.frontend import MonogenicFrontEnd
from monogenet.reference import reference_pipeline


def _images(b: int = 2, n: int = 48, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    base = ((x - n // 2).abs() <= 1).float() + 0.6 * (y > n // 2).float()
    return base.expand(b, 1, n, n) + 0.05 * torch.randn(b, 1, n, n, generator=g)


def test_shapes_names_and_defaults() -> None:
    layer = MonogenicFrontEnd()
    assert layer.n_scales == 3 and layer.out_channels == 12
    assert layer(_images(2, 40)).shape == (2, 12, 40, 40)
    assert layer(_images(2, 40)[:, 0]).shape == (2, 12, 40, 40)  # (B, H, W) accepted
    assert layer.feature_names[:5] == ["energy_0", "phase_0", "ps_0", "pa_0", "energy_1"]
    assert torch.allclose(layer.bank.wavelengths, torch.tensor([36.0, 39.0, 42.0]), rtol=1e-4)
    assert torch.allclose(layer.bank.sigma_onf, torch.full((3,), 0.55), rtol=1e-5)
    assert torch.allclose(layer.threshold.gain, torch.ones(3, 1, 1))


def test_non_square_and_odd_sizes() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0)
    assert layer(torch.rand(1, 1, 33, 50)).shape == (1, 8, 33, 50)


def test_channel_layout_matches_the_feature_functions() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0, sharpness=None).double()
    x = _images(2, 48).double()
    out = layer(x)
    mono = monogenic_signal(x, layer.bank(48, 48))  # (B, 1, S, 3, H, W)
    thr = layer.threshold(mono)
    expected = {
        0: local_amplitude(mono),
        1: local_phase(mono),
        2: phase_symmetry(mono, thr, 1e-6, None),
        3: phase_asymmetry(mono, thr, 1e-6, None),
    }
    for s in range(2):
        for k in range(4):
            assert torch.allclose(out[:, 4 * s + k], expected[k][:, 0, s], atol=1e-9), (s, k)


def test_matches_numpy_thesis_reference() -> None:
    # Thesis configuration with the hard floor: PS channels equal the NumPy reference.
    wavelengths = (6.0, 8.0, 10.0)
    layer = MonogenicFrontEnd(wavelengths, 0.55, 1.0, min_wavelength=3.0, sharpness=None).double()
    img = _images(1, 64)[0, 0].double()
    out = layer(img[None, None])[0]
    ref = reference_pipeline(img.numpy(), wavelengths, 0.55, eps=1e-6)
    for s in range(3):
        assert np.allclose(out[4 * s].detach().numpy(), ref.amplitude[s], atol=1e-6)
        assert np.allclose(out[4 * s + 2].detach().numpy(), ref.ps[s], atol=1e-6)


def test_contrast_behaviour() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0).double()
    x = _images(1, 48).double()
    a, b = layer(x)[0], layer(5.0 * x)[0]
    for s in range(2):
        assert torch.allclose(b[4 * s], 5.0 * a[4 * s], rtol=1e-6, atol=1e-9)  # energy scales
        for k in (1, 2, 3):  # phase, PS, PA do not
            assert torch.allclose(b[4 * s + k], a[4 * s + k], atol=1e-5), (s, k)


def test_ranges_with_hard_floor() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0, sharpness=None)
    out = layer(_images(2, 48))
    for s in range(2):
        assert (out[:, 4 * s] >= 0).all()
        assert (out[:, 4 * s + 1] >= 0).all() and (out[:, 4 * s + 1] <= np.pi + 1e-5).all()
        for k in (2, 3):
            assert (out[:, 4 * s + k] >= 0).all() and (out[:, 4 * s + k] <= 1 + 1e-5).all()


def test_batch_independence_and_eval_mode() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0)
    x = _images(3, 40)
    full = layer(x)
    assert torch.allclose(full[1:2], layer(x[1:2]), atol=1e-6)  # no cross-sample statistics
    layer.eval()
    with torch.no_grad():
        assert torch.allclose(layer(x), full, atol=1e-6)


def test_gradients_reach_input_and_every_parameter() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0)
    x = _images(2, 32).requires_grad_(True)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    names = {n for n, _ in layer.named_parameters()}
    assert names == {"bank.raw_f0", "bank.log_bandwidth", "threshold.raw_gain"}
    for n, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert p.grad.abs().sum() > 0, n


def test_parameter_count_and_freezing() -> None:
    layer = MonogenicFrontEnd()
    assert sum(p.numel() for p in layer.parameters()) == 9  # 3 scales x (f0, bandwidth, gain)
    frozen = MonogenicFrontEnd(learn_f0=False, learn_sigma=False, learn_threshold=False)
    assert sum(p.numel() for p in frozen.parameters() if p.requires_grad) == 0
    only_thr = MonogenicFrontEnd(learn_f0=False, learn_sigma=False)
    assert [n for n, p in only_thr.named_parameters() if p.requires_grad] == ["threshold.raw_gain"]


def test_taper_option_reduces_wraparound_response() -> None:
    n = 64
    ramp = torch.linspace(0, 1, n).expand(1, 1, n, n).clone()
    plain = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0)
    tapered = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0, taper=0.2)

    def border_energy(layer: MonogenicFrontEnd) -> float:
        e = layer(ramp)[:, 0].detach()  # energy of the finest scale
        return float((e[..., :6] ** 2).mean() + (e[..., -6:] ** 2).mean())

    assert border_energy(tapered) < 0.2 * border_energy(plain)


def test_training_step_changes_parameters_and_reduces_loss() -> None:
    # Fit the PS map of the first scale to a target made with different filter parameters.
    target_layer = MonogenicFrontEnd((9.0, 14.0), 0.45, 2.0, min_wavelength=3.0)
    layer = MonogenicFrontEnd((7.0, 11.0), 0.55, 1.0, min_wavelength=3.0)
    x = _images(2, 48)
    with torch.no_grad():
        target = target_layer(x)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    before = [p.detach().clone() for p in layer.parameters()]
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = (layer(x) - target).pow(2).mean()
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.6 * first
    assert any(not torch.equal(b, p) for b, p in zip(before, layer.parameters()))


def test_dtype_and_state_dict() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0).double()
    assert layer(_images(1, 32).double()).dtype == torch.float64
    other = MonogenicFrontEnd((20.0, 30.0), min_wavelength=3.0).double()
    other.load_state_dict(layer.state_dict())
    assert torch.allclose(other.bank.wavelengths, layer.bank.wavelengths)


def test_integer_input_is_converted() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0)
    out = layer(torch.randint(0, 255, (1, 1, 32, 32), dtype=torch.uint8))
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_blank_image_is_finite_with_finite_gradients() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), min_wavelength=3.0)
    x = torch.zeros(1, 1, 32, 32, requires_grad=True)
    layer(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


def test_invalid_inputs() -> None:
    layer = MonogenicFrontEnd()
    with pytest.raises(ValueError):
        layer(torch.rand(2, 3, 32, 32))  # layer was built for a single channel
    with pytest.raises(ValueError):
        layer(torch.rand(32, 32))
