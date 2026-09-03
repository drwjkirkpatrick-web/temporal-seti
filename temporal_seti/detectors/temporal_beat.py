"""
Detector for Signal Type 3: Temporal beat patterns (QPO phase modulation).

WHY: X-ray binaries show quasi-periodic oscillations (QPOs) at frequencies
from 0.1 Hz to >1 kHz. Standard analysis looks for the QPO frequency but
ignores the QPO phase. If a civilization modulates the phase of a QPO,
the information is in the phase, not the frequency. This detector searches
for non-random structure in QPO phase residuals.

Method:
1. Compute the Lomb-Scargle periodogram to find dominant frequencies
2. Fold the data at the top frequency to get a phase profile
3. Look for phase modulation by checking if phase residuals are
   structured (low variance in blocks) vs random

NOTE: The original implementation used np.correlate in 'full' mode which
is O(n²) for large arrays. This version uses FFT-based autocorrelation
(O(n log n)) for the period search and vectorized phase folding.
"""

from __future__ import annotations

import numpy as np
from temporal_seti.core.types import TimeSeries, DetectionResult, SignalType
from temporal_seti.detectors.base import BaseDetector


class TemporalBeatDetector(BaseDetector):
    """Detect temporal beat patterns via QPO phase analysis.

    Method:
    1. Bin the light curve at fine resolution
    2. FFT-based autocorrelation to find dominant period
    3. Fold data at dominant period to get phase profile
    4. Measure block variance structure (structured = encoded)
    """

    signal_type = SignalType.TEMPORAL_BEAT

    def _phase_structure(self, times: np.ndarray, exposure: float) -> float:
        """Measure phase structure at the dominant frequency.

        WHY: A natural QPO has random phase residuals. An encoded QPO
        has phase residuals that follow a pattern. We detect this by
        looking for reduced variance in blocks of folded phases —
        structured phases produce block-to-block variance differences.
        """
        if len(times) < 20 or exposure <= 0:
            return 0.0

        # Bin at 10ms resolution
        bin_size = 0.01
        n_bins = int(exposure / bin_size)
        if n_bins < 10:
            return 0.0

        binned = np.bincount(
            np.clip((times / bin_size).astype(int), 0, n_bins - 1),
            minlength=n_bins
        ).astype(float)

        if binned.sum() == 0:
            return 0.0

        # FFT-based autocorrelation (O(n log n) instead of O(n²))
        n = len(binned)
        centered = binned - binned.mean()
        fft = np.fft.rfft(centered, n=2*n)
        autocorr = np.fft.irfft(fft * np.conj(fft))[:n]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Find first peak after lag 0 (skip first few lags to avoid DC)
        min_lag = max(1, n // 1000)
        peaks = []
        for i in range(min_lag, min(n - 1, n // 2)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.1:
                peaks.append(i)
                break

        if not peaks:
            return 0.0

        period_lag = peaks[0] * bin_size

        # Fold phases (vectorized)
        phases = (times % period_lag) / period_lag

        # Block variance: divide phases into 10 blocks
        block_size = max(1, len(phases) // 10)
        block_vars = []
        for i in range(0, len(phases), block_size):
            block = phases[i:i+block_size]
            if len(block) > 1:
                block_vars.append(np.var(block))

        if len(block_vars) < 2:
            return 0.0

        # Low coefficient of variation in block variances = structured phases
        cv = np.std(block_vars) / (np.mean(block_vars) + 1e-12)
        return 1.0 - min(cv, 1.0)  # high value = structured

    def detect(self, series: TimeSeries) -> DetectionResult:
        """Detect temporal beat patterns."""
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
            return self._phase_structure(t, series.exposure)

        observed = metric_fn(times)
        pvalue = self._permutation_test(times, metric_fn, exposure=series.exposure)
        confidence = self._pvalue_to_confidence(pvalue)
        detected = confidence >= (1.0 - 1.0 / (self.config.significance_threshold ** 2))

        return DetectionResult(
            signal_type=self.signal_type,
            detected=detected,
            confidence=confidence,
            metric_name="phase_structure",
            metric_value=observed,
            source_name=series.source_name,
            instrument=series.instrument,
            details={"pvalue": pvalue, "n_counts": series.count},
        )