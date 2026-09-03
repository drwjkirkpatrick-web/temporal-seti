# 🔥 Temporal SETI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 54 passing](https://img.shields.io/badge/Tests-54%20passing-brightgreen.svg)](https://github.com/drwjkirkpatrick-web/temporal-seti/actions)
[![Code Quality: numpy only](https://img.shields.io/badge/Dependencies-numpy%20only-orange.svg)](#dependencies)
[![PDF: 13-page proofs](https://img.shields.io/badge/Proofs-13%20pages-red.svg)](docs/temporal_seti_proofs.pdf)
[![GPU: cupy/torch/auto](https://img.shields.io/badge/GPU-cupy%2Ftorch%2Fauto-purple.svg)](#gpu-acceleration)
[![Signal Types: 6](https://img.shields.io/badge/Signal%20Types-6-9cf.svg)](#what-it-does)
[![Instruments: 13](https://img.shields.io/badge/Instruments-13-lightgrey.svg)](#instrument-catalog)

**Searching for technosignatures encoded in time, not frequency.**

Most SETI searches look for narrowband radio spikes — sharp signals in frequency space. But a civilization that can manipulate gravitational fields near neutron stars could encode information in the *temporal* dimension of X-ray photon arrival times. That signal would be invisible to every Fourier-based search ever conducted. It would hide in the timing noise, glitches, and quasi-periodic oscillations we already observe but don't fully understand.

This project builds the tools to search that hidden dimension.

---

## What It Does

Temporal SETI implements six signal types that an advanced civilization could encode in X-ray emission from compact objects (neutron stars, X-ray binaries, magnetars):

| # | Signal Type | How It Works | What It Looks Like in Existing Data |
|---|---|---|---|
| 1 | **Time-dilation encoding** | Transmitter at varying gravitational depths modulates pulse spacing by the time dilation factor | Non-random timing residuals after pulsar spin-down subtraction |
| 2 | **Multi-timescale nested** | Same pattern encoded at nanosecond, millisecond, and second scales simultaneously | Structure only visible in wavelet/multi-resolution analysis, not any single periodogram |
| 3 | **Temporal beat patterns** | Phase modulation of quasi-periodic oscillations (QPOs) to carry binary data | Non-random QPO phase residuals (phase is never analyzed in standard pipelines) |
| 4 | **Anti-glitch encoding** | Controlled glitch/anti-glitch sequences (spin-up=1, spin-down=0) encode binary messages | Glitch patterns across magnetars that have never been tested for non-random structure |
| 5 | **Time-reversed structures** | Patterns that are incoherent forward but coherent when the time series is reversed | Asymmetric compressibility — impossible for natural time-asymmetric processes |
| 6 | **Variable time-dilation** | Slowly varying gravitational field produces a "drifting clock" with algorithmic structure | Smooth timing residuals that compress well (unlike stochastic red noise) |

Each signal type has a **simulator** (generates synthetic signals embedded in Poisson background) and a **detector** (runs Monte Carlo permutation tests against a Poisson null hypothesis to measure statistical significance).

---

## Why X-Rays?

Three properties make X-rays the natural domain for temporal SETI:

- **No dispersion** — Unlike radio, X-ray pulses cross interstellar space with temporal structure intact. No dispersion smearing.
- **Gravitational time dilation engines** — Neutron stars and X-ray binaries are the strongest gravitational wells in the galaxy. A civilization near one gets time dilation for free.
- **Instant photon escape** — X-rays from accretion escape immediately, unlike stellar atmospheres where photons random-walk for thousands of years.

---

## Quick Start

```bash
# Simulate all six signal types and run the full detection pipeline
python -m temporal_seti.cli simulate all

# Simulate a specific signal type
python -m temporal_seti.cli simulate time_dilation --exposure 500

# List known X-ray instruments
python -m temporal_seti.cli instruments

# List signal types
python -m temporal_seti.cli signals
```

### Python API

```python
from temporal_seti.core.config import Config
from temporal_seti.signals import TimeDilationSignal
from temporal_seti.pipeline.runner import Pipeline

# Generate a test signal
sim = TimeDilationSignal(background_rate=10.0, exposure=1000.0, seed=42)
series = sim.generate()

# Run all 6 detectors
pipe = Pipeline(Config.default())
results = pipe.run(series)

# Get a formatted report
print(pipe.report(results))
```

---

## Instrument Catalog

13 X-ray instruments are catalogued with time resolution, collecting area, and bandpass:

| Instrument | Time Resolution | Collecting Area | All-Sky? |
|---|---|---|---|
| RXTE PCA | 1 μs | 7000 cm² | No |
| NICER | 100 ns | 1900 cm² | No |
| XMM-Newton PN | 30 ns | 4660 cm² | No |
| Chandra HRC | 16 μs | 27 cm² | No |
| NuSTAR | 2 μs | 800 cm² | No |
| Swift BAT | 64 μs | 5200 cm² | Yes |
| MAXI | 1 s | 300 cm² | Yes |
| ...and 6 more | | | |

---

## GPU Acceleration

The Monte Carlo permutation tests are the computational bottleneck. The `temporal_seti.core.gpu` module auto-detects available GPU backends:

- **cupy** (preferred for CUDA GPUs)
- **torch** (fallback if cupy unavailable)
- **numpy** (CPU fallback, always available)

Null distributions are generated in batch as 2D arrays and sorted in one operation, giving 10-50x speedup over per-permutation generation.

```python
from temporal_seti.core.gpu import gpu_available, get_backend_name
print(f"Backend: {get_backend_name()}")  # 'cupy', 'torch', or 'cpu'
```

---

## Architecture

```
temporal_seti/
├── core/           # Types, config, GPU backend
│   ├── types.py        # SignalType, TimeSeries, DetectionResult, AnalysisConfig
│   ├── config.py       # Config with instrument catalog
│   └── gpu.py          # cupy/torch/numpy auto-detection
├── signals/        # Six signal simulators
│   ├── base.py         # SignalSimulator base class
│   ├── time_dilation.py
│   ├── multi_timescale.py
│   ├── temporal_beat.py
│   ├── anti_glitch.py
│   ├── time_reversed.py
│   └── variable_dilation.py
├── detectors/      # Six detectors + base
│   ├── base.py         # BaseDetector with Monte Carlo permutation
│   ├── time_dilation.py
│   ├── multi_timescale.py
│   ├── temporal_beat.py
│   ├── anti_glitch.py
│   ├── time_reversed.py
│   └── variable_dilation.py
├── instruments/    # X-ray instrument catalog
│   └── catalog.py      # 13 instruments with specs
├── pipeline/        # Analysis orchestrator
│   └── runner.py       # Pipeline class: run + report
└── cli.py           # Command-line interface
```

---

## Testing

```bash
cd temporal-seti
python -m pytest tests/ -v
```

54 tests covering core types, signal generation, detector behavior (signal vs noise), pipeline orchestration, and instrument catalog.

---

## Key Insight

Current SETI searches for patterns in *frequency* (narrowband radio spikes) or *energy* (infrared excess, atmospheric spectra). A civilization that can manipulate time would encode in the *temporal* dimension — photon arrival times, timing residuals, glitch sequences, temporal beats. These signals would be:

- **Already in our data** — 60+ years of X-ray timing archives exist
- **Invisible to our pipelines** — which fold at known periods and discard residuals
- **Self-selecting** — only a civilization that understands time manipulation could decode them
- **Naturally camouflaged** — timing noise, glitches, and QPOs already exist as unexplained phenomena

The search requires temporal decoding algorithms (entropy analysis, cross-scale correlation, phase structure, jump sequence analysis, reversal asymmetry, compressibility) rather than Fourier-based periodicity searches.

---

## Dependencies

- **numpy** (required)
- **pytest** (for testing)
- **cupy** (optional, GPU acceleration)
- **torch** (optional, GPU fallback)

No scipy dependency — all statistical functions implemented in pure numpy for Jetson compatibility.

---

## References

- Corbet, R.H.D. (1997). "SETI at X-ray Energies." *JBIS*, 50, 253-257. arXiv:1609.00330
- Sheikh, S.Z. et al. (2025). "Earth Detecting Earth." arXiv:2502.02614
- Gajjar, V. et al. (2026). "Plasma broadening of narrowband SETI signals." DOI:10.3847/1538-4357/ae3d33
- NASA Technosignatures Workshop (2018)
- SETI Institute: https://www.seti.org/research/seti/

---

## License

MIT

---

## Author

Walker Kirkpatrick