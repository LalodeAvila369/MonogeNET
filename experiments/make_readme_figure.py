"""Generate docs/images/log_gabor_bank.png for the README.

Radial log-Gabor bank with the thesis defaults (wavelengths 36, 39, 42 px,
k/w0 = 0.55). Row 1: frequency response; row 2: spatial kernel (real part,
i.e. the "even" filter) and the Riesz odd1 component for the first scale.
This is illustrative only; the library implementation arrives in ROADMAP step 4.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from monogenet.filters.grids import frequency_radius

N = 256
WAVELENGTHS = (36, 39, 42)
K_OVER_W0 = 0.55

radius = frequency_radius(N, N, dc_value=1.0)
radius_nodc = frequency_radius(N, N, dc_value=0.0)
u = torch.fft.fftfreq(N)[None, :].expand(N, N)

fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
for i, wl in enumerate(WAVELENGTHS):
    f0 = 1.0 / wl
    g = torch.exp(-(torch.log(radius / f0) ** 2) / (2 * torch.log(torch.tensor(K_OVER_W0)) ** 2))
    g[radius_nodc == 0] = 0.0  # zero DC gain
    axes[0, i].imshow(torch.fft.fftshift(g), cmap="magma", extent=(-0.5, 0.5, 0.5, -0.5))
    axes[0, i].set_title(f"Frequency response, $\lambda_0$={wl}px")
    axes[0, i].set_xlim(-0.15, 0.15)
    axes[0, i].set_ylim(0.15, -0.15)
    kernel = torch.fft.fftshift(torch.fft.ifft2(g).real)
    c = N // 2
    axes[1, i].imshow(kernel[c - 48 : c + 48, c - 48 : c + 48], cmap="gray")
    axes[1, i].set_title("Spatial kernel (even)")
    if i == 0:
        riesz1 = torch.fft.fftshift(torch.fft.ifft2(g * (1j * u / radius)))
        axes[1, 3].imshow(riesz1.real[c - 48 : c + 48, c - 48 : c + 48], cmap="gray")
        axes[1, 3].set_title("Riesz odd1 (real part)")
axes[0, 3].axis("off")
axes[0, 3].text(0.0, 0.5, "MonogeNet\nlog-Gabor bank\n" r"$k/\omega_0$ = 0.55", fontsize=14, va="center")
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "images" / "log_gabor_bank.png"
fig.savefig(out, dpi=110)
print(out)
