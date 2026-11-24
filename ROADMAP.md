# MonogeNet Roadmap

One step per weekday, one commit per step. Tick each box when done.

## Phase 0: Setup
- [x] 1. Project structure, README, roadmap, packaging config
- [x] 2. Test setup (pytest), CI workflow (GitHub Actions, CPU)
- [x] 3. Frequency-grid utilities (`fftfreq` meshgrid, radius, angle) with tests

## Phase 1: Filters (`monogenet.filters`)
- [x] 4. Radial log-Gabor filter (fixed parameters), DC = 0, peak at f0
- [x] 5. Multi-scale log-Gabor bank (min wavelength, scale factor, sigma/f0)
- [x] 6. Angular spread function for oriented banks
- [x] 7. Riesz transform kernels (H1, H2) and tests on synthetic edges
- [x] 8. Low-pass / window handling to reduce FFT edge artifacts
- [x] 9. Learnable parameterisation (bounded f0 via sigmoid, sigma in log-space)
- [ ] 10. Gradient checks (`torch.autograd.gradcheck`) for the filter bank

## Phase 2: Features (`monogenet.features`)
- [ ] 11. Monogenic signal: even, odd1, odd2 per scale
- [ ] 12. Local amplitude, local phase, local orientation
- [ ] 13. NumPy reference implementation of the thesis pipeline (for validation)
- [ ] 14. Phase symmetry (fixed threshold) matching the reference
- [ ] 15. Phase asymmetry
- [ ] 16. Phase congruency (monogenic version)
- [ ] 17. Feature type: atan2(even, |odd|)
- [ ] 18. Soft thresholds (softplus) and learnable noise threshold T
- [ ] 19. Numerical stability (epsilon, safe atan2 gradients) with tests
- [ ] 20. Synthetic test suite: lines, step edges, blobs under contrast changes

## Phase 3: Layers (`monogenet.layers`)
- [ ] 21. `MonogenicFrontEnd`: image -> [energy, phase, PS, PA] per scale
- [ ] 22. Multi-channel support (per-channel filters)
- [ ] 23. `PhaseBlock`: depthwise learnable log-Gabor + Riesz -> PS/PA -> 1x1 conv
- [ ] 24. Parameter and FLOP counting utilities vs. Conv2d
- [ ] 25. Filter visualisation (spatial and frequency domain)

## Phase 4: Data and baselines (`monogenet.data`, `experiments/`)
- [ ] 26. MedMNIST+ loaders (BreastMNIST, OCTMNIST; 128 and 224)
- [ ] 27. Training loop, metrics (ACC, AUC), config files, seeding
- [ ] 28. Baseline: small CNN
- [ ] 29. Baseline: ResNet-18
- [ ] 30. ResNet-18 + MonogenicFrontEnd
- [ ] 31. CNN with PhaseBlocks replacing convolutions

## Phase 5: Experiments
- [ ] 32. BreastMNIST training-set-size curves (10%, 25%, 50%, 100%)
- [ ] 33. OCTMNIST full-data comparison
- [ ] 34. MedMNIST-C robustness evaluation (contrast, noise, blur)
- [ ] 35. Ablation: PS vs PA vs PC vs all features
- [ ] 36. Learned vs hand-tuned log-Gabor parameters

## Phase 6: Write-up
- [ ] 37. Results tables and figures
- [ ] 38. Documentation and usage examples
- [ ] 39. Package release (PyPI `monogenet`)
- [ ] 40. Short paper draft in `docs/`
