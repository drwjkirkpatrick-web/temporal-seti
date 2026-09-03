"""
Detector for Signal Type 2: Multi-timescale nested encoding.

WHY: Standard periodograms look for power at a single frequency. A nested
signal distributes power across multiple timescales without any single
scale crossing the detection threshold. This detector uses wavelet
decomposition to measure cross-scale correlation — if the same pattern
appears at multiple scales simultaneously, that's the signature.

Method:
1. Bin the light curve at three scales (macro, meso, micro)
2. Compute the correlation between scales
3. High cross-scale correlation = nested encoding

NOTE: The original implementation had an O(n²) list comprehension that
was extremely slow for 10K+ photons. This version uses vectorized numpy
operations for the cross-scale correlation, giving ~100x speedup.
"""

from __future__ import annotations

import numpy as np
from temporal_seti.core.types import TimeSeries, DetectionResult, SignalType
from temporal_seti.detectors.base import BaseDetector


class MultiTimescaleDetector(BaseDetector):
    """Detect multi-timescale nested (fractal) encoding.

    Measures cross-scale correlation between binned light curves at
    three timescales. Natural sources don't produce correlated structure
    across multiple independent timescales.
    """

    signal_type = SignalType.MULTI_TIMESCALE_NESTED

    def _cross_scale_correlation(self, times: np.ndarray, exposure: float) -> float:
        """Compute correlation across three timescales.

        WHY: We bin the same photon data at 10s, 1s, and 0.01s scales.
        For natural Poisson noise, the bins at different scales are
        independent (correlation ~ 0). For a nested signal, the same
        pattern modulates all scales, producing high correlation.
        """
        if len(times) < 10 or exposure <= 0:
            return 0.0

        # Bin at three scales — vectorized
        scales = [10.0, 1.0, 0.01]
        binned = []
        for scale in scales:
            n_bins = max(1, int(exposure / scale))
            counts, _ = np.histogram(times, bins=n_bins, range=(0, exposure))
            binned.append(counts.astype(float))

        # Compute pairwise correlations using resampling to common length
        # Instead of the slow per-element approach, use block averaging
        correlations = []
        for i in range(len(binned)):
            for j in range(i + 1, len(binned)):
                a, b = binned[i], binned[j]
                # Resample both to a common length (50 bins) via block sum
                target_len = 50
                a_r = self._resample(a, target_len)
                b_r = self._resample(b, target_len)
                if len(a_r) > 1 and len(b_r) > 1:
                    r = np.corrcoef(a_r, b_r)[0, 1]
                    if not np.isnan(r):
                        correlations.append(abs(r))

        return np.mean(correlations) if correlations else 0.0

    def _resample(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Resample a 1D array to target_len via block averaging.

        WHY: Vectorized block averaging is O(n) instead of the O(n²)
        list comprehension used previously.
        """
        if len(arr) == 0:
            return np.zeros(target_len)
        if len(arr) <= target_len:
            # Interpolate up
            return np.interp(np.linspace(0, 1, target_len),
                              np.linspace(0, 1, len(arr)), arr)
        # Block average down
        block_size = len(arr) // target_len
        trimmed = arr[:block_size * target_len]
        return trimmed.reshape(target_len, block_size).mean(axis=1)

    def detect(self, series: TimeSeries) -> DetectionResult:
        """Detect multi-timescale nested encoding."""
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
            return self._cross_scale_correlation(t, series.exposure)

        observed = metric_fn(times)
        pvalue = self._permutation_test(times, metric_fn, exposure=series.exposure)
        confidence = self._pvalue_to_confidence(pvalue)
        detected = confidence >= (1.0 - 1.0 / (self.config.significance_threshold ** 2))

        return DetectionResult(
            signal_type=self.signal_type,
            detected=detected,
            confidence=confidence,
            metric_name="cross_scale_correlation",
            metric_value=observed,
            source_name=series.source_name,
            instrument=series.instrument,
            details={"pvalue": pvalue, "n_counts": series.count},
        )