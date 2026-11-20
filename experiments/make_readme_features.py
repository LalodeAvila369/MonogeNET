"""Generate the monogenic-feature figure and GIF for the README.

Illustrative only (features are computed inline here; the library
implementations arrive in ROADMAP steps 11-17). Synthetic image with a step
edge, a thin line and a blob, plus noise. Writes to docs/images/.

Per scale s (radial log-Gabor g_s, Riesz H1 = i u/|w|, H2 = i v/|w|):
    even = F^-1[G f], odd1 = F^-1[G H1 f], odd2 = F^-1[G H2 f]
    A = sqrt(even^2 + odd1^2 + odd2^2),  odd = sqrt(odd1^2 + odd2^2)
    PS = floor(|even| - odd - T) / (A + eps)
    PA = floor(odd - |even| - T) / (A + eps)
    PC = floor(sum_s [even.cos + odd.sin - |even.sin - odd.cos|] - T) / (sum_s A + eps)
with floor(x) = max(x, 0) and T = noise threshold from the mean amplitude.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from monogenet.filters.grids import frequency_grid, frequency_radius

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
N = 256
WAVELENGTHS = (8.0, 14.0, 24.0)
K_OVER_W0 = 0.55
EPS = 1e-6


def synthetic(contrast: float = 1.0, seed: int = 0) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(N, dtype=torch.float32), torch.arange(N, dtype=torch.float32), indexing="ij")
    img = torch.zeros(N, N)
    img[(x > 120) & (x < 230) & (y > 135) & (y < 225)] += 0.5  # rectangle: step edges
    img += 0.9 * torch.exp(-(((x - 70) ** 2 + (y - 70) ** 2) / (2 * 8**2)))  # blob
    d = ((x - 130) + (y - 30)) / 2**0.5  # thin diagonal line
    img += 0.8 * ((d.abs() <= 1.2) & (x > 125) & (x < 215) & (y < 120)).float()
    img = contrast * img
    # Taper to the mean at the borders to limit FFT wrap-around artifacts.
    t = torch.hann_window(N, periodic=False).clamp(max=1)
    w = torch.minimum(torch.ones(N), t * 8).unsqueeze(0) * torch.minimum(torch.ones(N), t * 8).unsqueeze(1)
    img = img * w
    g = torch.Generator().manual_seed(seed)
    return img + 0.03 * torch.randn(N, N, generator=g)


def monogenic_features(img: torch.Tensor) -> dict[str, torch.Tensor]:
    u, v = frequency_grid(N, N)
    r = frequency_radius(N, N)
    r0 = frequency_radius(N, N, dc_value=0.0)
    h1, h2 = 1j * u / r, 1j * v / r
    h1[0, 0] = h2[0, 0] = 0
    F = torch.fft.fft2(img)
    log_k = torch.log(torch.tensor(K_OVER_W0))
    evens, odd1s, odd2s = [], [], []
    for wl in WAVELENGTHS:
        g = torch.exp(-(torch.log(r / (1 / wl)) ** 2) / (2 * log_k**2))
        g[r0 == 0] = 0
        evens.append(torch.fft.ifft2(g * F).real)
        odd1s.append(torch.fft.ifft2(g * h1 * F).real)
        odd2s.append(torch.fft.ifft2(g * h2 * F).real)
    e, o1, o2 = (torch.stack(t) for t in (evens, odd1s, odd2s))
    odd = torch.sqrt(o1**2 + o2**2)
    amp = torch.sqrt(e**2 + odd**2)
    # Noise threshold per scale, from the mean amplitude of the finest scale.
    T = 0.8 * amp[0].mean() * torch.tensor([1.0, 0.8, 0.6]).view(-1, 1, 1)
    ps = ((e.abs() - odd - T).clamp(min=0) / (amp + EPS)).sum(0) / len(WAVELENGTHS)
    pa = ((odd - e.abs() - T).clamp(min=0) / (amp + EPS)).sum(0) / len(WAVELENGTHS)
    sum_e, sum_o1, sum_o2 = e.sum(0), o1.sum(0), o2.sum(0)
    xe = torch.sqrt(sum_e**2 + sum_o1**2 + sum_o2**2) + EPS
    me, mo1, mo2 = sum_e / xe, sum_o1 / xe, sum_o2 / xe
    dot = e * me + o1 * mo1 + o2 * mo2
    cross = torch.sqrt((o1 * me - e * mo1) ** 2 + (o2 * me - e * mo2) ** 2 + (o1 * mo2 - o2 * mo1) ** 2)
    pc = ((dot - cross).sum(0) - T.sum(0)).clamp(min=0) / (amp.sum(0) + EPS)
    return {
        "even": e[1], "odd": odd[1], "amp": amp[1], "phase": torch.atan2(odd[1], e[1]),
        "ps": ps, "pa": pa, "pc": pc,
    }


def draw(axes, img: torch.Tensor, feats: dict[str, torch.Tensor]):
    panels = [
        ("Input", img, "gray", None),
        (r"Even (symmetric) $\;e$", feats["even"], "RdBu_r", None),
        (r"Odd (antisymmetric) $\;|o|$", feats["odd"], "viridis", None),
        (r"Local amplitude $\;A$", feats["amp"], "magma", None),
        (r"Local phase $\;\varphi$", feats["phase"], "twilight", (0, np.pi)),
        ("Phase symmetry (PS)", feats["ps"], "gray", (0, 1)),
        ("Phase asymmetry (PA)", feats["pa"], "gray", (0, 1)),
        ("Phase congruency (PC)", feats["pc"], "gray", (0, 1)),
    ]
    ims = []
    for ax, (title, data, cmap, lim) in zip(axes.ravel(), panels):
        arr = data.numpy()
        if title.startswith("Even"):
            m = np.abs(arr).max()
            lim = (-m, m)
        im = ax.imshow(arr, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        ims.append(im)
    return ims


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = synthetic()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    draw(axes, img, monogenic_features(img))
    fig.suptitle(r"MonogeNet: monogenic features (middle scale, $\lambda_0$=14 px)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    fig.savefig(OUT / "monogenic_features.png", dpi=100)
    plt.close(fig)

    # GIF: contrast sweep. Raw image and even/odd change; PS/PA/PC barely do.
    contrasts = np.concatenate([np.linspace(1.0, 0.15, 10), np.linspace(0.15, 1.0, 10)])
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.4))
    ims = draw(axes, img, monogenic_features(img))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0)
    text = fig.suptitle("")

    def update(k: int):
        c = float(contrasts[k])
        im_t = synthetic(c)
        f = monogenic_features(im_t)
        keys = [None, "even", "odd", "amp", "phase", "ps", "pa", "pc"]
        for im, key in zip(ims, keys):
            data = (im_t if key is None else f[key]).numpy()
            im.set_data(data)
        ims[0].set_clim(0, 1.3)  # fixed display range so the dimming is visible
        ims[1].set_clim(-1.0, 1.0)
        ims[2].set_clim(0, 1.0)
        ims[3].set_clim(0, 1.0)
        text.set_text(f"Image contrast x{c:.2f}: raw responses vanish, PS / PA / PC mostly survive")
        return ims

    FuncAnimation(fig, update, frames=len(contrasts)).save(
        OUT / "contrast_invariance.gif", writer=PillowWriter(fps=6), dpi=55
    )
    print(OUT)


if __name__ == "__main__":
    main()
