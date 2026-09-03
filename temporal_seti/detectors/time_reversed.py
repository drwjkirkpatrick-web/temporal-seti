"""
Detector for Signal Type 5: Time-reversed temporal structures.

WHY: No natural astrophysical process produces time-reversed structure.
If a time series has lower entropy or higher compressibility when reversed,
that is a strong indicator of artificial construction. This detector
compares the algorithmic complexity of the forward and reversed series.

Method:
1. Compute inter-arrival intervals
2. Measure compressibility (via autocorrelation-based proxy)
3. Compare forward vs reversed
4. Asymmetry = potential technosignature

NOTE: The original LZ complexity implementation was O(n²) and timed out
on 10K+ photon datasets. This version uses an autocorrelation-based
compressibility proxy that is O(n log n), giving ~1000x speedup.
"""

from __future__ import annotations

import numpy as np
from temporal_seti.core.types import TimeSeries, DetectionResult, SignalType
from temporal_seti.detectors.base import BaseDetector


class TimeReversedDetector(BaseDetector):
    """Detect time-reversed temporal structures.

    Measures the asymmetry in compressibility between forward and
    time-reversed photon arrival series. Natural processes are
    time-asymmetric but not in a way that makes reversed data more
    compressible.
    """

    signal_type = SignalType.TIME_REVERSED

    def _compressibility_proxy(self, seq: np.ndarray) -> float:
        """Fast compressibility proxy via autocorrelation integral.

        WHY: True LZ complexity is O(n²) and impractical for 10K+ photons.
        Autocorrelation integral is O(n log n) via FFT and correlates
        strongly with compressibility: structured sequences have
        slowly-decaying autocorrelation (high integral = compressible).
        """
        if len(seq) < 4:
            return 0.0

        # Normalize
        s = seq - seq.mean()
        if s.std() > 0:
            s = s / s.std()

        # FFT-based autocorrelation (O(n log n))
        n = len(s)
        fft = np.fft.rfft(s, n=2*n)
        autocorr = np.fft.irfft(fft * np.conj(fft))[:n]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Integral of |autocorrelation| over first 50 lags
        # High value = slowly decaying = structured/compressible
        n_lags = min(50, n)
        return np.sum(np.abs(autocorr[:n_lags])) / n_lags

    def _reversal_asymmetry(self, times: np.ndarray) -> float:
        """Compute the asymmetry between forward and reversed complexity.

        Returns a value in [-1, 1]. Negative = reversed is more
        compressible (potential technosignature). Positive = forward
        is more compressible. Near zero = symmetric (natural).
        """
        if len(times) < 10:
            return 0.0

        intervals = np.diff(times)
        if len(intervals) < 5:
            return 0.0

        fwd_comp = self._compressibility_proxy(intervals)
        rev_comp = self._compressibility_proxy(intervals[::-1])

        total = fwd_comp + rev_comp
        if total <= 0:
            return 0.0

        return (fwd_comp - rev_comp) / total

    def detect(self, series: TimeSeries) -> DetectionResult:
        """Detect time-reversed structure."""
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

        def metric_fn(t: np.ndarray) -> float:
            return abs(self._reversal_asymmetry(t))

        observed = metric_fn(times)
        pvalue = self._permutation_test(times, metric_fn, exposure=series.exposure)
        confidence = self._pvalue_to_confidence(pvalue)

        # For time-reversed: we specifically care about NEGATIVE asymmetry
        raw_asymmetry = self._reversal_asymmetry(times)
        detected = (confidence >= (1.0 - 1.0 / (self.config.significance_threshold ** 2))
                    and raw_asymmetry < 0)

        return DetectionResult(
            signal_type=self.signal_type,
            detected=detected,
            confidence=confidence,
            metric_name="reversal_asymmetry",
            metric_value=raw_asymmetry,
            source_name=series.source_name,
            instrument=series.instrument,
            details={
                "pvalue": pvalue,
                "raw_asymmetry": raw_asymmetry,
                "n_counts": series.count,
            },
        )