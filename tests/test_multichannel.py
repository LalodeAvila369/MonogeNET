"""Multi-channel support: per-channel banks, thresholds and the front-end."""

import pytest
import torch

from monogenet.features.monogenic import monogenic_signal
from monogenet.features.threshold import LearnableThreshold
from monogenet.filters.learnable import LearnableLogGaborBank
from monogenet.filters.log_gabor import log_gabor_bank
from monogenet.layers.frontend import MonogenicFrontEnd

D = torch.float64
KW = dict(min_wavelength=3.0)


def _rgb(b: int = 2, c: int = 3, n: int = 40, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    structures = [((x - n // 2).abs() <= 1).float(), 0.7 * (y > n // 2).float(), ((x - y).abs() <= 1).float()]
    imgs = torch.stack([structures[i % 3] for i in range(c)])
    return imgs.expand(b, c, n, n) + 0.05 * torch.randn(b, c, n, n, generator=g)


# ------------------------------------------------------------------ monogenic_signal


def test_monogenic_signal_with_per_channel_bank_matches_loop() -> None:
    g = torch.Generator().manual_seed(1)
    img = torch.randn(2, 3, 32, 32, generator=g, dtype=D)
    banks = torch.stack([log_gabor_bank(32, 32, 2, 4.0 + c, 1.7, dtype=D) for c in range(3)])  # (C, S, H, W)
    out = monogenic_signal(img, banks)
    assert out.shape == (2, 3, 2, 3, 32, 32)
    for c in range(3):
        assert torch.allclose(out[:, c], monogenic_signal(img[:, c], banks[c]), atol=1e-12)


def test_shared_bank_equals_per_channel_bank_with_identical_entries() -> None:
    img = torch.randn(2, 3, 24, 24, generator=torch.Generator().manual_seed(2), dtype=D)
    bank = log_gabor_bank(24, 24, 2, 4.0, 1.7, dtype=D)
    assert torch.allclose(monogenic_signal(img, bank), monogenic_signal(img, bank.expand(3, -1, -1, -1)))


# ------------------------------------------------------------- learnable bank/threshold


def test_per_channel_bank_shapes_and_init() -> None:
    bank = LearnableLogGaborBank((8.0, 12.0), 0.5, channels=3, **KW)
    assert bank.raw_f0.shape == (3, 2) and bank.n_scales == 2
    out = bank(32, 40)
    assert out.shape == (3, 2, 32, 40)
    assert torch.allclose(out[0], out[2])  # identical at initialisation
    shared = LearnableLogGaborBank((8.0, 12.0), 0.5, **KW)
    assert torch.allclose(out[1], shared(32, 40), atol=1e-6)
    assert torch.allclose(bank.wavelengths, torch.tensor([[8.0, 12.0]] * 3), rtol=1e-4)


def test_vectorised_bank_equals_scale_by_scale_construction() -> None:
    bank = LearnableLogGaborBank((8.0, 12.0, 20.0), (0.4, 0.5, 0.6), **KW)
    out = bank(32, 32)
    from monogenet.filters.log_gabor import radial_log_gabor

    for s, (wl, sg) in enumerate(zip(bank.wavelengths, bank.sigma_onf)):
        assert torch.allclose(out[s], radial_log_gabor(32, 32, wl, sg), atol=1e-6)


def test_channels_have_independent_parameters_and_gradients() -> None:
    bank = LearnableLogGaborBank((8.0, 12.0), channels=3, **KW)
    bank(24, 24)[1].pow(2).sum().backward()  # loss on channel 1 only
    for p in (bank.raw_f0, bank.log_bandwidth):
        assert p.grad[1].abs().sum() > 0
        assert p.grad[0].abs().sum() == 0 and p.grad[2].abs().sum() == 0


def test_per_channel_threshold() -> None:
    mono = torch.randn(2, 3, 2, 3, 8, 8, generator=torch.Generator().manual_seed(3), dtype=D)  # (B,C,S,3,H,W)
    lt = LearnableThreshold(2, init_gain=1.5, channels=3).double()
    assert lt.gain.shape == (3, 2, 1, 1) and lt(mono).shape == (2, 3, 2, 1, 1)
    with torch.no_grad():
        lt.raw_gain[1] += 1.0  # change channel 1 only
    t = lt(mono)
    plain = LearnableThreshold(2, init_gain=1.5).double()(mono[:, 0])
    assert torch.allclose(t[:, 0], plain) and not torch.allclose(t[:, 1], LearnableThreshold(2, 1.5).double()(mono[:, 1]))
    with pytest.raises(ValueError):
        lt(mono[:, :2])  # wrong channel count
    with pytest.raises(ValueError):
        LearnableThreshold(2, channels=0)


def test_invalid_channels() -> None:
    with pytest.raises(ValueError):
        LearnableLogGaborBank((8.0,), channels=0, **KW)


# ---------------------------------------------------------------------------- front-end


def test_rgb_shapes_names_and_counts() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), in_channels=3, **KW)
    x = _rgb(2, 3)
    assert layer(x).shape == (2, 3 * 2 * 4, 40, 40) and layer.out_channels == 24
    assert layer.feature_names[:5] == ["c0_energy_0", "c0_phase_0", "c0_ps_0", "c0_pa_0", "c0_energy_1"]
    assert layer.feature_names[8] == "c1_energy_0" and len(layer.feature_names) == 24
    assert sum(p.numel() for p in layer.parameters()) == 6  # shared: 3 params x 2 scales
    per = MonogenicFrontEnd((6.0, 9.0), in_channels=3, per_channel=True, **KW)
    assert sum(p.numel() for p in per.parameters()) == 18  # 3 channels x 6


def test_shared_filters_process_each_channel_like_a_single_channel_layer() -> None:
    multi = MonogenicFrontEnd((6.0, 9.0), in_channels=3, **KW).double()
    single = MonogenicFrontEnd((6.0, 9.0), **KW).double()
    x = _rgb(2, 3).double()
    out = multi(x)
    per = 2 * 4  # features per channel
    for c in range(3):
        assert torch.allclose(out[:, c * per : (c + 1) * per], single(x[:, c : c + 1]), atol=1e-10), c


def test_per_channel_filters_equal_shared_at_init_and_diverge_after_training_step() -> None:
    shared = MonogenicFrontEnd((6.0, 9.0), in_channels=3, **KW)
    per = MonogenicFrontEnd((6.0, 9.0), in_channels=3, per_channel=True, **KW)
    x = _rgb(2, 3)
    assert torch.allclose(shared(x), per(x), atol=1e-5)
    # Make channel 1's filters different: only that channel's outputs change.
    before = per(x).detach()
    with torch.no_grad():
        per.bank.raw_f0[1] += 0.5
        per.threshold.raw_gain[1] += 0.5
    after = per(x).detach()
    per_ch = 2 * 4
    assert torch.allclose(after[:, :per_ch], before[:, :per_ch], atol=1e-6)
    assert torch.allclose(after[:, 2 * per_ch :], before[:, 2 * per_ch :], atol=1e-6)
    assert not torch.allclose(after[:, per_ch : 2 * per_ch], before[:, per_ch : 2 * per_ch], atol=1e-3)


def test_per_channel_filters_match_single_channel_layers_with_those_parameters() -> None:
    per = MonogenicFrontEnd((6.0, 9.0), in_channels=2, per_channel=True, **KW).double()
    with torch.no_grad():
        per.bank.raw_f0[1] += 0.7
        per.bank.log_bandwidth[1] -= 0.3
        per.threshold.raw_gain[1] += 0.4
    x = _rgb(1, 2).double()
    out = per(x)
    for c in range(2):
        single = MonogenicFrontEnd((6.0, 9.0), **KW).double()
        with torch.no_grad():
            single.bank.raw_f0.copy_(per.bank.raw_f0[c])
            single.bank.log_bandwidth.copy_(per.bank.log_bandwidth[c])
            single.threshold.raw_gain.copy_(per.threshold.raw_gain[c])
        assert torch.allclose(out[:, c * 8 : (c + 1) * 8], single(x[:, c : c + 1]), atol=1e-10), c


def test_channels_do_not_interact() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), in_channels=3, **KW).double()
    x = _rgb(1, 3).double()
    ref = layer(x)
    x2 = x.clone()
    x2[:, 1] = 5.0 * x2[:, 1] + 2.0  # change contrast and offset of channel 1 only
    out = layer(x2)
    per = 8
    assert torch.allclose(out[:, :per], ref[:, :per], atol=1e-9)
    assert torch.allclose(out[:, 2 * per :], ref[:, 2 * per :], atol=1e-9)
    # In channel 1 energy scales by 5 and PS/PA/phase are unchanged (contrast invariance).
    assert torch.allclose(out[:, per], 5.0 * ref[:, per], rtol=1e-6)
    assert torch.allclose(out[:, per + 2], ref[:, per + 2], atol=1e-6)


@pytest.mark.parametrize("per_channel", [False, True])
def test_gradients_for_all_parameters_in_multichannel_mode(per_channel: bool) -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), in_channels=3, per_channel=per_channel, **KW)
    x = _rgb(2, 3).requires_grad_(True)
    layer(x).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for n, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n


def test_batch_independence_and_taper_in_multichannel_mode() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), in_channels=3, per_channel=True, taper=0.1, **KW)
    x = _rgb(3, 3)
    full = layer(x)
    assert torch.allclose(full[1:2], layer(x[1:2]), atol=1e-6)


def test_grayscale_default_is_unchanged() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), **KW)
    assert layer.in_channels == 1 and layer(torch.rand(1, 1, 32, 32)).shape == (1, 8, 32, 32)
    assert layer(torch.rand(1, 32, 32)).shape == (1, 8, 32, 32)
    assert layer.feature_names[0] == "energy_0"


def test_invalid_inputs() -> None:
    layer = MonogenicFrontEnd((6.0, 9.0), in_channels=3, **KW)
    with pytest.raises(ValueError):
        layer(torch.rand(2, 1, 32, 32))  # wrong number of channels
    with pytest.raises(ValueError):
        layer(torch.rand(2, 32, 32))  # (B, H, W) only valid for a single channel
    with pytest.raises(ValueError):
        MonogenicFrontEnd(in_channels=0)
