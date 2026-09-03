"""
Temporal SETI — searching for technosignatures encoded in time, not frequency.

This package implements a framework for detecting signals from advanced
civilizations that use time as an encoding dimension rather than frequency
or energy. The core insight: X-ray astronomy data (photon arrival times from
compact objects) has been collected for 60+ years, but standard analysis
pipelines fold at known periods and discard timing residuals. A civilization
that can manipulate gravitational fields or accretion near neutron stars
could encode information in temporal structures invisible to Fourier-based
searches.

Six signal types are modeled:
  1. Time-dilation-encoded pulse trains (gravitational well keying)
  2. Multi-timescale nested encoding (temporal fractals)
  3. Temporal beat patterns (QPO phase modulation)
  4. Anti-glitch encoded signals (glitch/anti-glitch binary sequences)
  5. Time-reversed temporal structures (entropy-asymmetric signals)
  6. Variable time-dilation signatures (drifting clocks)

Five detector classes analyze data for each signal type, and a pipeline
orchestrator ties them together with existing instrument catalogs.
"""

__version__ = "0.1.0"
__author__ = "Walker Kirkpatrick"
__license__ = "MIT"

from temporal_seti.core.types import (
    SignalType,
    DetectionResult,
    TimeSeries,
    PhotonEvent,
    InstrumentConfig,
    AnalysisConfig,
)
from temporal_seti.core.config import Config

__all__ = [
    "SignalType",
    "DetectionResult",
    "TimeSeries",
    "PhotonEvent",
    "InstrumentConfig",
    "AnalysisConfig",
    "Config",
    "__version__",
]