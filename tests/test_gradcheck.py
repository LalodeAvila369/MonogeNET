"""Gradient checks (torch.autograd.gradcheck, float64) for the filter bank."""

import torch
from torch.autograd import gradcheck

from monogenet.filters.angular import angular_spread
from monogenet.filters.learnable import LearnableLogGaborBank
from monogenet.filters.log_gabor import log_gabor_bank, radial_log_gabor
from monogenet.filters.riesz import riesz_kernels
from monogenet.filters.window import butterworth_lowpass

D = torch.float64


def _p(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=D, requires_grad=True)


def test_gradcheck_radial_log_gabor() -> None:
    assert gradcheck(
        lambda wl, sg: radial_log_gabor(12, 10, wl, sg, dtype=D),
        (_p(5.3), _p(0.55)),
        eps=1e-6,
        atol=1e-5,
    )


def test_gradcheck_log_gabor_bank() -> None:
    assert gradcheck(
        lambda lam, m, sg: log_gabor_bank(12, 12, 3, lam, m, sg, dtype=D),
        (_p(4.1), _p(1.4), _p(0.6)),
        eps=1e-6,
        atol=1e-5,
    )


def test_gradcheck_angular_spread() -> None:
    # Offsets keep us away from the kink at a wrapped angular distance of exactly pi.
    assert gradcheck(
        lambda o, s: angular_spread(10, 12, o, s, dtype=D),
        (_p(0.37), _p(0.6)),
        eps=1e-6,
        atol=1e-5,
    )
    assert gradcheck(
        lambda o, s: angular_spread(10, 12, o, s, symmetric=True, dtype=D),
        (_p(0.37), _p(0.6)),
        eps=1e-6,
        atol=1e-5,
    )


def test_gradcheck_learnable_bank_parameters() -> None:
    m = LearnableLogGaborBank((6.0, 9.0), (0.5, 0.6), min_wavelength=3.0, max_wavelength=30.0).double()

    def f(raw_f0: torch.Tensor, log_bw: torch.Tensor) -> torch.Tensor:
        wl =1.0 / (m.f_min + (m.f_max - m.f_min) * torch.sigmoid(raw_f0))
        sg = torch.exp(-torch.exp(log_bw))
        return torch.stack([radial_log_gabor(12, 12, wl[s], sg[s], dtype=D) for s in range(2)])

    raw = m.raw_f0.detach().clone().requires_grad_(True)
    bw = m.log_bandwidth.detach().clone().requires_grad_(True)
    assert gradcheck(f, (raw, bw), eps=1e-6, atol=1e-5)
    # The module's own forward must agree with this explicit chain.
    assert torch.allclose(m(12, 12), f(raw, bw))


def test_gradcheck_filtering_pipeline_wrt_params_and_image() -> None:
    # image -> FFT -> bank * Riesz -> IFFT -> energy: gradients wrt filter parameters and the image.
    h1, h2 = riesz_kernels(12, 12, dtype=D)
    lp = butterworth_lowpass(12, 12, dtype=D)

    def pipeline(img: torch.Tensor, wl: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
        g = radial_log_gabor(12, 12, wl, sg, dtype=D) * lp
        spec = torch.fft.fft2(img)
        even = torch.fft.ifft2(spec * g).real
        odd1 = torch.fft.ifft2(spec * g * h1).real
        odd2 = torch.fft.ifft2(spec * g * h2).real
        return torch.sqrt(even**2 + odd1**2 + odd2**2 + 1e-6)

    gen = torch.Generator().manual_seed(0)
    img = torch.randn(12, 12, generator=gen, dtype=D).requires_grad_(True)
    assert gradcheck(pipeline, (img, _p(5.2), _p(0.55)), eps=1e-6, atol=1e-4, rtol=1e-3)
