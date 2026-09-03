"""
Signal Type 5: Time-reversed temporal structures.

WHY: A signal designed to be read backwards through time. No natural
astrophysical process produces time-reversed structure because physical
processes are time-asymmetric (entropy increases). A pattern that is
mathematically incoherent forward but coherent backward would be the most
unambiguous technosignature possible.

The signal encodes a message that reads naturally only when the time
series is reversed. Forward, it looks like structured noise. Backward,
it forms a clean periodic pattern or recognizable mathematical constant.
"""

from __future__ import annotations
import numpy as np
from temporal_seti.signals.base import SignalSimulator
from temporal_seti.core.types import TimeSeries


class TimeReversedSignal(SignalSimulator):
    """Simulate a time-reversed signal.

    The signal contains a clear pattern (e.g. prime numbers, digits of pi)
    that is only visible when the photon arrival time series is reversed.
    In the forward direction, the pattern is scrambled by an asymmetric
    transformation that preserves entropy but destroys coherence.
    """

    def __init__(self, background_rate: float = 10.0, exposure: float = 1000.0,
                 seed: int = 42, pattern_period: float = 0.05,
                 n_pulses: int = 200):
        """Initialize time-reversed signal.

        Args:
            background_rate: Poisson background (counts/sec)
            exposure: observation duration (seconds)
            seed: random seed
            pattern_period: period of the hidden pattern (seconds)
            n_pulses: number of signal pulses to generate
        """
        super().__init__(background_rate, exposure, seed)
        self.pattern_period = pattern_period
        self.n_pulses = n_pulses

    def generate(self, source_name: str = "sim_ns_reversed",
                 source_type: str = "isolated_ns",
                 instrument: str = "Chandra_HRC") -> TimeSeries:
        """Generate a time-reversed signal.

        Step 1: Create a clean periodic pattern (the message).
        Step 2: Apply an asymmetric time-warping that stretches earlier
                intervals and compresses later ones (entropy-increasing).
        Step 3: The result looks like non-periodic noise forward.
        Step 4: Reversing the series and applying the inverse warp
                recovers the clean pattern.
        """
        # Clean pattern: regular pulses
        clean_times = np.arange(self.n_pulses) * self.pattern_period

        # Asymmetric warp: cumulative stretching that increases with time
        # This makes intervals grow non-linearly, destroying periodicity
        warp_factors = 1.0 + 0.5 * (clean_times / clean_times[-1]) ** 2
        warped_intervals = np.diff(clean_times) * warp_factors[:-1]
        warped_times = np.zeros(len(clean_times))
        warped_times[1:] = np.cumsum(warped_intervals)

        # Scale to fit within exposure
        if warped_times[-1] > 0:
            warped_times = warped_times / warped_times[-1] * (self.exposure * 0.9)

        return self._merge_with_background(
            warped_times,
            source_name=source_name,
            source_type=source_type,
            instrument=instrument,
        )
