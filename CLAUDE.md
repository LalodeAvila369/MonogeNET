# CLAUDE.md: MonogeNet

Context for Claude Code working in this repository.

## What this project is
MonogeNet is a PyTorch library of **learnable monogenic signal layers**: log-Gabor filter banks, the Riesz transform, and local phase features (phase symmetry PS, phase asymmetry PA, phase congruency PC, feature type) as trainable CNN building blocks for medical imaging.

It continues the author's master's thesis (collaboration with Dalhousie University):
E. de Avila-Armenta et al., *Femoral Head Surface Segmentation Using Ultrasound Images as Input Source: A Local Phase and Rigid Filtering Approach*, CCIS 2348, Springer 2025. https://doi.org/10.1007/978-3-031-83435-6_20

Thesis pipeline (the reference to reproduce in step 13): FFT → radial log-Gabor bank → Riesz (H1 = i·u/|w|, H2 = i·v/|w|) → monogenic tensor [even, odd1, odd2] → PS = ⌊|even| − |sqrt(odd1² + odd2²)| − T⌋ / (sqrt(even² + odd1² + odd2²) + ε), with T = exp(log(mean(amplitude))). Tuned values on hip ultrasound: center wavelengths 36, 39, 42 and k/w0 = 0.55. Use these as defaults and as initialisation for learnable versions.

## Research idea
1. Make PS/PA/PC **learnable** (bounded f0, sigma in log space, soft thresholds, learnable T).
2. `MonogenicFrontEnd`: entry layer, image → [energy, phase, PS, PA] per scale.
3. `PhaseBlock`: depthwise learnable log-Gabor + Riesz per channel → PS/PA → 1×1 conv. A parameter-efficient replacement for Conv2d inside the network (open question: does it help in deeper layers?).

Related work to compare against: GaborNet (Alekseev & Bobe 2019), M6 trainable monogenic layer (Moya-Sánchez et al., IEEE Access 2021), Mono2D (arXiv 2503.09050), MonoUNet (arXiv 2604.07780). Known gap: none use PS/PA as learnable layer outputs.

## Data
MedMNIST+ (`pip install medmnist`), at size 128 (main) and 224 (final):
- BreastMNIST: breast ultrasound, 780 images, binary, small-data regime.
- OCTMNIST: retinal OCT, ~109k images, 4 classes, large-data regime.
- MedMNIST-C for robustness to contrast/noise/blur.
Never commit datasets, checkpoints or outputs (see .gitignore). Cite MedMNIST v1 and v2.

## Workflow rules
- Follow `ROADMAP.md`: **one step per commit**. Do the next unchecked step, tick its box in the same commit.
- Every new function gets tests in `tests/`, preferably on synthetic images with a known answer (lines, step edges, blobs, contrast changes).
- Run `pytest` before committing; do not commit failing tests.
- Commit messages: imperative, short, e.g. `Add radial log-Gabor filter with tests`.
- Code style: type hints, docstrings with the math (LaTeX-ish), filters built in the frequency domain with `torch.fft`, everything differentiable.
- Keep CPU-friendly defaults; GPU optional.
