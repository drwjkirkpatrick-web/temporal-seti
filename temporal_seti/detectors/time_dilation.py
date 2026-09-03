"""
Detector for Signal Type 1: Time-dilation-encoded pulse trains.

WHY: This detector looks for non-random structure in pulse arrival time
residuals after subtracting a polynomial spin-down model. The key insight
is that a time-dilation-encoded signal produces residuals that are
structured (low entropy when binned) rather than the red noise expected
from natural timing noise.

Method:
1. Compute inter-arrival intervals
2. Bin intervals into a histogram
3. Measure histogram entropy (structured = low entropy)
4. Compare to shuffled permutation null distribution
"""

from __future__ import annotations

import numpy as np
from temporal_seti.core.types import TimeSeries, DetectionResult, SignalType
from temporal_seti.detectors.base import BaseDetector
def _shannon_entropy(hist: np.ndarray) -> float:
    """Pure-numpy Shannon entropy (avoids scipy dependency).

    WHY: scipy may not be available on minimal installs (e.g. Jetson).
    numpy is always available. This computes the same quantity.
    """
    hist = hist + 1e-12
    p = hist / hist.sum()
    return -np.sum(p * np.log(p))


class TimeDilationDetector(BaseDetector):
    """Detect time-dilation-encoded pulse trains.

    Looks for structured (non-random) inter-arrival time distributions
    that would indicate a clock being modulated by varying gravitational
    potential.
    """

    signal_type = SignalType.TIME_DILATION_ENCODED

    def _interval_entropy(self, times: np.ndarray) -> float:
        """Compute the entropy of the inter-arrival interval distribution.

        WHY: A naturally periodic source (pulsar) has very low interval
        entropy (all intervals similar). A time-dilation-encoded signal
        has intervals that cluster around a few dilation-factor multiples,
        producing lower entropy than random (Poisson) arrival times but
        higher entropy than a clean pulsar. The key metric is how different
        the entropy is from the shuffled (null) case.
        """
        if len(times) < 3:
            return 0.0
        intervals = np.diff(times)
        # Bin into 50 bins
        hist, _ = np.histogram(intervals, bins=50, density=True)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        return _shannon_entropy(hist)

    def detect(self, series: TimeSeries) -> DetectionResult:
        """Detect time-dilation encoding in photon arrival times."""
        times = series.arrival_times

        if series.count < self.config.min_counts:
            return DetectionResult(
                signal_type=self.signal_type,
                detected=False,
                confidence=0.0,
                source_name=series.source_name,
                instrument=series.instrument,
                details={"reason": "insufficient_counts"},
            )

        observed_entropy = self._interval_entropy(times)
        pvalue = self._permutation_test(times, self._interval_entropy, exposure=series.exposure)

        confidence = self._pvalue_to_confidence(pvalue)
        detected = confidence >= (1.0 - 1.0 / (self.config.significance_threshold ** 2))

        return DetectionResult(
            signal_type=self.signal_type,
            detected=detected,
            confidence=confidence,
            metric_name="interval_entropy",
            metric_value=observed_entropy,
            source_name=series.source_name,
            instrument=series.instrument,
            details={"pvalue": pvalue, "n_counts": series.count},
        )
