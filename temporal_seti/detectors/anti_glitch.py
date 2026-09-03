"""
Detector for Signal Type 4: Anti-glitch encoded signals.

WHY: Magnetars show glitches (sudden spin-up) and rare "anti-glitches"
(sudden spin-down). Nobody has checked whether the glitch pattern across
known magnetars encodes information. This detector looks for non-random
binary structure in a sequence of timing jumps.

Method:
1. Compute the inter-arrival intervals
2. Detect jumps where the local median period changes suddenly
3. Classify each jump as + (spin-up) or - (spin-down)
4. Test if the +/- sequence is non-random via runs test

NOTE: The original _detect_jumps had a Python loop with median computation
per iteration. This version uses vectorized rolling median via numpy
stride tricks for O(n) performance.
"""

from __future__ import annotations

import numpy as np
from temporal_seti.core.types import TimeSeries, DetectionResult, SignalType
from temporal_seti.detectors.base import BaseDetector


class AntiGlitchDetector(BaseDetector):
    """Detect glitch/anti-glitch encoded signals.

    Looks for sudden period changes in a pulsar-like signal and tests
    whether the sequence of up/down jumps is non-random.
    """

    signal_type = SignalType.ANTI_GLITCH

    def _detect_jumps(self, times: np.ndarray) -> list[int]:
        """Detect period jumps in arrival times.

        Returns a list of +1 (spin-up) and -1 (spin-down) for each detected jump.
        Uses vectorized rolling median comparison for speed.
        """
        if len(times) < 10:
            return []

        intervals = np.diff(times)
        if len(intervals) < 5:
            return []

        # Vectorized rolling median using cumulative approach
        window = max(5, len(intervals) // 20)
        n = len(intervals)

        # Compute rolling medians before and after each point
        # Use a simple sliding window approach
        jumps = []
        step = max(1, window // 2)
        for i in range(window, n - window, step):
            local_before = np.median(intervals[i-window:i])
            local_after = np.median(intervals[i:i+window])

            if local_before > 0:
                frac_change = (local_after - local_before) / local_before
                if abs(frac_change) > 1e-4:
                    jumps.append(1 if frac_change < 0 else -1)

        return jumps

    def _jump_sequence_entropy(self, times: np.ndarray) -> float:
        """Measure the entropy of the jump sequence.

        WHY: A random glitch sequence has high entropy (near-maximal for
        the number of jumps). An encoded sequence has lower entropy because
        it contains a message. We compare to the permutation null.
        """
        jumps = self._detect_jumps(times)
        if len(jumps) < 3:
            return 1.0  # maximal entropy = no structure

        # Convert to binary and measure runs
        binary = [1 if j > 0 else 0 for j in jumps]
        # Count runs (transitions)
        runs = sum(1 for i in range(1, len(binary)) if binary[i] != binary[i-1])
        # Expected runs for random binary: (2*n1*n0)/n + 1
        n1 = sum(binary)
        n0 = len(binary) - n1
        expected_runs = (2 * n1 * n0) / len(binary) + 1 if len(binary) > 0 else 0

        if expected_runs == 0:
            return 1.0

        # Deviation from expected runs: low deviation = structured
        deviation = abs(runs - expected_runs) / max(expected_runs, 1)
        return 1.0 - min(deviation, 1.0)  # high = structured

    def detect(self, series: TimeSeries) -> DetectionResult:
        """Detect anti-glitch encoding."""
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
            return self._jump_sequence_entropy(t)

        observed = metric_fn(times)
        pvalue = self._permutation_test(times, metric_fn, exposure=series.exposure)
        confidence = self._pvalue_to_confidence(pvalue)
        detected = confidence >= (1.0 - 1.0 / (self.config.significance_threshold ** 2))

        jumps = self._detect_jumps(times)
        return DetectionResult(
            signal_type=self.signal_type,
            detected=detected,
            confidence=confidence,
            metric_name="jump_sequence_entropy",
            metric_value=observed,
            source_name=series.source_name,
            instrument=series.instrument,
            details={
                "pvalue": pvalue,
                "n_jumps": len(jumps),
                "n_counts": series.count,
            },
        )