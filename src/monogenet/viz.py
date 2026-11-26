"""Visualisation of log-Gabor / Riesz filters in the frequency and spatial domains.

The numerical part (:func:`filter_responses`) only needs torch. The plotting
functions need matplotlib (``pip install monogenet[viz]``) and return the
``matplotlib.figure.Figure`` so callers can save or show it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from monogenet.filters.grids import frequency_radius
from monogenet.filters.riesz import riesz_kernels

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

__all__ = ["filter_responses", "plot_filter_bank", "plot_radial_profiles"]


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("plotting needs matplotlib: pip install monogenet[viz]") from exc
    return plt


def _bank_tensor(source: Any, height: int, width: int, channel: int) -> torch.Tensor:
    """Resolve a bank ``(S, H, W)`` from a tensor, a bank module or a layer holding one."""
    bank = getattr(source, "bank", source)  # MonogenicFrontEnd / PhaseBlock -> their bank
    if callable(bank) and not isinstance(bank, torch.Tensor):
        with torch.no_grad():
            bank = bank(height, width)
    if not isinstance(bank, torch.Tensor):
        raise TypeError(f"cannot get a filter bank from {type(source).__name__}")
    bank = bank.detach()
    if bank.ndim == 4:  # per-channel bank (C, S, H, W)
        if not 0 <= channel < bank.shape[0]:
            raise ValueError(f"channel {channel} out of range for {bank.shape[0]} channels")
        bank = bank[channel]
    if bank.ndim != 3:
        raise ValueError(f"expected a bank of shape (S, H, W) or (C, S, H, W), got {tuple(bank.shape)}")
    return bank


def filter_responses(
    source: Any, height: int = 65, width: int | None = None, channel: int = 0
) -> dict[str, torch.Tensor]:
    r"""Frequency responses and spatial kernels of a bank, centred for display.

    ``source`` is a bank tensor ``(S, H, W)`` (or per-channel ``(C, S, H, W)``), a
    :class:`~monogenet.filters.LearnableLogGaborBank`, or any layer with a ``bank``
    attribute (:class:`~monogenet.layers.MonogenicFrontEnd`,
    :class:`~monogenet.layers.PhaseBlock`). For a per-channel bank, ``channel``
    selects which channel's filters to show. ``height`` and ``width`` give the
    grid size when the bank is generated from a module (a tensor keeps its own size).

    Returns tensors of shape ``(S, H, W)``, all ``fftshift``-ed so that DC / the
    kernel centre is at index ``(H // 2, W // 2)``:

    * ``"frequency"``: the transfer function :math:`G_s(u, v)`;
    * ``"even"``: :math:`\mathcal F^{-1}[G_s]`, the symmetric (even) kernel;
    * ``"odd1"``, ``"odd2"``: :math:`\mathcal F^{-1}[H_{1,2} G_s]`, the Riesz (odd)
      kernels along x and y;
    * ``"magnitude"``: :math:`\sqrt{\mathrm{even}^2 + \mathrm{odd1}^2 + \mathrm{odd2}^2}`.
    """
    width = height if width is None else width
    bank = _bank_tensor(source, height, width, channel)
    h, w = bank.shape[-2:]
    h1, h2 = riesz_kernels(h, w, device=bank.device, dtype=bank.dtype)
    even = torch.fft.ifft2(bank).real
    odd1 = torch.fft.ifft2(bank * h1).real
    odd2 = torch.fft.ifft2(bank * h2).real
    shift = lambda t: torch.fft.fftshift(t, dim=(-2, -1))  # noqa: E731
    return {
        "frequency": shift(bank),
        "even": shift(even),
        "odd1": shift(odd1),
        "odd2": shift(odd2),
        "magnitude": shift(torch.sqrt(even**2 + odd1**2 + odd2**2)),
    }


def plot_filter_bank(
    source: Any,
    height: int = 65,
    width: int | None = None,
    channel: int = 0,
    crop: int | None = None,
    freq_limit: float = 0.25,
) -> Figure:
    """Grid of the bank: one column per scale; frequency response, even and odd kernels.

    Rows: ``|G|`` in the frequency plane (axes in cycles/pixel, limited to
    ``+-freq_limit``), the even kernel, and the two Riesz odd kernels. ``crop`` is the
    half-size in pixels of the central window shown for the spatial kernels (default:
    the whole kernel).
    """
    plt = _plt()
    r = filter_responses(source, height, width, channel)
    n_scales, h, w = r["frequency"].shape
    ch, cw = h // 2, w // 2
    half = min(ch, cw) if crop is None else min(crop, ch, cw)
    window = (slice(ch - half, ch + half + 1), slice(cw - half, cw + half + 1))
    extent = (-0.5, 0.5, 0.5, -0.5)
    rows = [("frequency", "$G$ (frequency)"), ("even", "even kernel"), ("odd1", "odd kernel (x)"), ("odd2", "odd kernel (y)")]
    fig, axes = plt.subplots(len(rows), n_scales, figsize=(2.6 * n_scales, 2.5 * len(rows)), squeeze=False)
    wavelengths = _wavelengths(source, channel)
    for j in range(n_scales):
        for i, (key, label) in enumerate(rows):
            ax = axes[i, j]
            data = r[key][j]
            if key == "frequency":
                ax.imshow(data, cmap="magma", extent=extent, vmin=0, vmax=1)
                ax.set_xlim(-freq_limit, freq_limit)
                ax.set_ylim(freq_limit, -freq_limit)
            else:
                patch = data[window]
                m = float(patch.abs().max()) or 1.0
                ax.imshow(patch, cmap="RdBu_r", vmin=-m, vmax=m)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label)
            if i == 0 and wavelengths is not None:
                ax.set_title(f"scale {j}: $\\lambda_0$={wavelengths[j]:.1f}px", fontsize=10)
            elif i == 0:
                ax.set_title(f"scale {j}", fontsize=10)
    fig.tight_layout()
    return fig


def _wavelengths(source: Any, channel: int) -> list[float] | None:
    bank = getattr(source, "bank", source)
    wl = getattr(bank, "wavelengths", None)
    if wl is None or not isinstance(wl, torch.Tensor):
        return None
    wl = wl.detach()
    if wl.ndim == 2:
        wl = wl[channel]
    return [float(v) for v in wl]


def plot_radial_profiles(
    sources: Mapping[str, Any] | Any,
    height: int = 128,
    channel: int = 0,
    log_x: bool = True,
) -> Figure:
    r"""Overlay the radial profiles :math:`G_s(|\omega|)` of one or several banks.

    ``sources`` is a bank / layer or a mapping ``name -> bank / layer`` (e.g. before
    and after training). Each source gets one colour, the scales of a bank are drawn
    with the same colour. The profile is taken along the :math:`u` axis.
    """
    plt = _plt()
    if not isinstance(sources, Mapping):
        sources = {"bank": sources}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    radius = frequency_radius(height, height, dc_value=0.0)[0, : height // 2]
    for color_index, (name, source) in enumerate(sources.items()):
        bank = _bank_tensor(source, height, height, channel)
        profile = bank[:, 0, : height // 2]
        color = f"C{color_index}"
        for s in range(profile.shape[0]):
            ax.plot(radius[1:], profile[s, 1:], color=color, alpha=0.9, label=name if s == 0 else None)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("frequency |w| (cycles/pixel)")
    ax.set_ylabel("G(|w|)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
