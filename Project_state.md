# Project State: Temporal SETI

> **Last updated:** 2026-09-03
> **Current phase:** build (initial release)
> **Overall health:** green

---

## 1. Goal
Search for extraterrestrial technosignatures encoded in the temporal dimension of X-ray photon arrival times — a search space invisible to traditional frequency-domain SETI.

## 2. Current Status
### Done
- [x] Core types (SignalType, TimeSeries, DetectionResult, AnalysisConfig, InstrumentConfig)
- [x] Configuration system with instrument catalog (13 X-ray instruments)
- [x] Six signal simulators (time dilation, multi-timescale, temporal beat, anti-glitch, time-reversed, variable dilation)
- [x] Six detectors with Monte Carlo permutation testing (Poisson null model)
- [x] GPU acceleration backend (cupy/torch auto-detection, numpy fallback)
- [x] Pipeline orchestrator with report generation
- [x] CLI interface (simulate, instruments, signals commands)
- [x] 54 tests passing (core, signals, detectors, pipeline, instruments)
- [x] Signal vs noise validation (4/6 signals detected, noise not detected)

### In Progress
- [ ] README with Phosphorus personality

### Not Started
- [ ] Real data ingestion (FITS/evt file readers for RXTE, NICER, Chandra archives)
- [ ] Advanced detectors (wavelet decomposition, algorithmic complexity)
- [ ] Web dashboard for interactive analysis
- [ ] Push to GitHub

## 3. Architecture & Key Decisions
| Decision | Rationale | Date |
|---|---|---|
| Pure numpy (no scipy dependency) | Jetson compatibility, minimal deps | 2026-09-03 |
| Poisson resampling null model (not shuffling) | Correct null for interval-based metrics | 2026-09-03 |
| FFT-based autocorrelation (not np.correlate) | O(n log n) vs O(n²) for 10K+ photons | 2026-09-03 |
| cupy/torch auto-detect with numpy fallback | GPU when available, CPU everywhere | 2026-09-03 |
| Batched null generation (2D array) | 10-50x faster than per-permutation loop | 2026-09-03 |

## 4. Blockers & Risks
- **Risk:** MultiTimescaleDetector has higher false-positive rate on noise due to resampling artifacts → needs improved null model or threshold calibration
- **Risk:** No real data ingestion yet — all testing on simulated signals
- **Risk:** CUDA driver too old on this Jetson for torch.cuda — GPU backend falls back to CPU (expected)

## 5. Next Step
Write README and push to GitHub.

## 6. Environment & Tooling Notes
- Runtime: Python 3.11
- Dependencies: numpy only (scipy removed, torch optional for GPU)
- Tests: pytest, 54 tests, ~7s runtime
- GPU: cupy/torch auto-detect, numpy fallback active on this Jetson
- Hermes skill: software-development-workflows, project-state-management

## 7. Recent Session Log
- 2026-09-03: Built full project — core types, 6 signals, 6 detectors, pipeline, CLI, GPU backend, 54 tests passing

## 8. References
- Corbet (1997): SETI at X-ray Energies, arXiv:1609.00330
- Sheikh et al. (2025): Earth Detecting Earth, arXiv:2502.02614
- SETI Institute: https://www.seti.org/research/seti/
- Gajjar et al. (2026): Plasma broadening of narrowband signals, DOI:10.3847/1538-4357/ae3d33