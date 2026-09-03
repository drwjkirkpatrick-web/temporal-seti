"""
Core type definitions for temporal SETI analysis.

WHY: Every module in this package depends on these types. By defining them
first, we establish the interface contract before any implementation, which
prevents cross-module API drift during parallel development.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


class SignalType(enum.Enum):
    """The six categories of time-encoded technosignatures.

    WHY: Each enum value maps to a detector class. Adding a new signal type
    means adding an enum member and a corresponding detector — no other code
    needs to change. This is the open-closed principle applied to SETI.
    """
    TIME_DILATION_ENCODED = "time_dilation_encoded"
    MULTI_TIMESCALE_NESTED = "multi_timescale_nested"
    TEMPORAL_BEAT = "temporal_beat"
    ANTI_GLITCH = "anti_glitch"
    TIME_REVERSED = "time_reversed"
    VARIABLE_TIME_DILATION = "variable_time_dilation"


@dataclass(frozen=True)
class PhotonEvent:
    """A single X-ray photon detection event.

    NOTE: X-ray instruments record photon arrival times at microsecond
    resolution. Unlike optical or radio data, each photon is an individual
    event — there is no continuous waveform. The information content lives
    in the pattern of arrival times, not in amplitude.
    """
    arrival_time: float        # seconds since observation epoch (MJD or TDB)
    energy: float = 0.0        # keV (0 if energy resolution unavailable)
    channel: int = 0            # detector channel number


@dataclass
class TimeSeries:
    """A sequence of photon arrival times from a single observation.

    WHY: This is the fundamental data structure of X-ray timing. All
    detectors operate on TimeSeries. The metadata allows a detector to
    know the source type (pulsar, magnetar, X-ray binary) and apply
    source-appropriate filtering.
    """
    arrival_times: np.ndarray       # shape (N,), sorted ascending, seconds
    energies: Optional[np.ndarray] = None   # shape (N,), keV
    source_name: str = ""
    source_type: str = ""            # "pulsar", "magnetar", "xrb", "isolated_ns"
    instrument: str = ""             # e.g. "RXTE_PCA", "NICER", "Chandra"
    obs_id: str = ""
    exposure: float = 0.0           # total observation time, seconds
    mjd_start: float = 0.0          # Modified Julian Date of observation start

    def __post_init__(self) -> None:
        """Validate that arrival times are sorted and non-negative."""
        if len(self.arrival_times) > 1:
            diffs = np.diff(self.arrival_times)
            if not np.all(diffs >= 0):
                raise ValueError("arrival_times must be sorted ascending")
        if np.any(self.arrival_times < 0):
            raise ValueError("arrival_times must be non-negative")

    @property
    def count(self) -> int:
        """Number of photons in the series."""
        return len(self.arrival_times)

    @property
    def rate(self) -> float:
        """Average count rate (counts per second)."""
        if self.exposure <= 0 or self.count == 0:
            return 0.0
        return self.count / self.exposure

    def to_bins(self, bin_size: float) -> np.ndarray:
        """Histogram arrival times into uniform bins.

        WHY: Many detectors need binned data (Fourier analysis, periodograms).
        This is the standard time-to-frequency-domain bridge.
        """
        if self.exposure <= 0:
            return np.array([])
        n_bins = int(np.ceil(self.exposure / bin_size))
        counts, _ = np.histogram(self.arrival_times, bins=n_bins,
                                   range=(0, self.exposure))
        return counts.astype(float)


@dataclass
class DetectionResult:
    """Result of running a detector on a TimeSeries.

    A result with confidence > 0.0 means the detector found structure
    consistent with its signal type. Confidence is not a probability of
    ETI — it's a measure of how far the observed structure deviates from
    the natural-noise null hypothesis for that specific detector.
    """
    signal_type: SignalType
    detected: bool
    confidence: float                # 0.0 to 1.0
    metric_name: str = ""             # e.g. "entropy_ratio", "glitch_pattern"
    metric_value: float = 0.0
    details: dict = field(default_factory=dict)
    source_name: str = ""
    instrument: str = ""


@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration for an X-ray astronomy instrument.

    NOTE: These values are drawn from published instrument specifications.
    Time resolution determines which signal types are detectable — a
    instrument with 1ms resolution cannot see nanosecond-scale encoding.
    """
    name: str                        # e.g. "RXTE_PCA"
    time_resolution: float           # microseconds (best available)
    energy_resolution: float         # E/dE at 6.4 keV
    collecting_area: float           # cm^2
    bandpass_min: float              # keV
    bandpass_max: float              # keV
    mission_dates: str = ""          # e.g. "1995-2012"
    all_sky_monitor: bool = False    # True if it surveys the whole sky


@dataclass
class AnalysisConfig:
    """Configuration for a temporal SETI analysis run.

    WHY: This lets the pipeline apply different search strategies without
    modifying detector code. The significance threshold controls how
    aggressive the search is — lower thresholds catch more candidates but
    increase false positives.
    """
    significance_threshold: float = 5.0    # sigma (SNR)
    min_counts: int = 100                    # minimum photons to analyze
    time_resolution: float = 1.0             # microseconds, analysis bin size
    seed: int = 42                           # for reproducible randomization tests
    n_permutations: int = 1000               # Monte Carlo permutations for p-value
    verbose: bool = False