"""
Detector for Signal Type 6: Variable time-dilation signatures (drifting clocks).

WHY: Natural timing noise is stochastic (red noise power spectrum). An
encoded drift has algorithmic structure. This detector measures the
algorithmic information content of timing residuals and compares it to
the expectation from red noise.

Method:
1. Fit a polynomial spin-down model to arrival times
2. Compute residuals (observed - model)
3. Measure the compressibility of the residuals
4. Compressible residuals = algorithmic structure = potential signal
"""

from __future__ import annotations

import numpy as np
from temporal_seti.core.types import TimeSeries, DetectionResult, SignalType
from temporal_seti.detectors.base import BaseDetector


class VariableDilationDetector(BaseDetector):
    """Detect variable time-dilation (drifting clock) signals.

    Measures the compressibility of timing residuals after polynomial
    detrending. Natural red noise is incompressible; an encoded drift
    pattern compresses well.
    """

    signal_type = SignalType.VARIABLE_TIME_DILATION

    def _residual_compressibility(self, times: np.ndarray) -> float:
        """Measure compressibility of timing residuals.

        WHY: We fit a low-order polynomial to the cumulative arrival times
        (which should be linear for constant rate). The residuals from
        this fit represent timing noise. If the residuals are compressible
        (low Kolmogorov complexity), they contain algorithmic structure.
        """
        if len(times) < 20:
            return 0.0

        n = len(times)
        t_index = np.arange(n)

        # Fit polynomial (order 3) to cumulative times
        coeffs = np.polyfit(t_index, times, 3)
        model = np.polyval(coeffs, t_index)
        residuals = times - model

        # Normalize residuals
        if residuals.std() > 0:
            residuals = residuals / residuals.std()

        # Measure compressibility via autocorrelation decay
        # Compressible sequences have slowly-decaying autocorrelation
        if len(residuals) < 10:
            return 0.0

        autocorr = np.correlate(residuals, residuals, mode="full")
        autocorr = autocorr[len(autocorr)//2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Slow decay = structured = compressible
        # Measure the integral of autocorrelation (higher = more structured)
        decay_integral = np.sum(np.abs(autocorr[:min(50, len(autocorr))])) / min(50, len(autocorr))

        return decay_integral

    def detect(self, series: TimeSeries) -> DetectionResult:
        """Detect variable time-dilation signatures."""
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
            return self._residual_compressibility(t)

        observed = metric_fn(times)
        pvalue = self._permutation_test(times, metric_fn, exposure=series.exposure)
        confidence = self._pvalue_to_confidence(pvalue)
        detected = confidence >= (1.0 - 1.0 / (self.config.significance_threshold ** 2))

        return DetectionResult(
            signal_type=self.signal_type,
            detected=detected,
            confidence=confidence,
            metric_name="residual_compressibility",
            metric_value=observed,
            source_name=series.source_name,
            instrument=series.instrument,
            details={"pvalue": pvalue, "n_counts": series.count},
        )
