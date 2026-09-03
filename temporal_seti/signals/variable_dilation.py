"""
Signal Type 6: Variable time-dilation signatures (drifting clocks).

WHY: If a civilization can alter the gravitational field near a source
(e.g. by moving compact masses), the time dilation factor itself varies.
The signal shows a "drifting clock" — pulses that gradually speed up or
slow down in a way that doesn\'t match any known orbital dynamics, spin-down,
or accretion model. The drift pattern encodes information.

Natural timing noise is stochastic (red noise power spectrum). An encoded
drift would have algorithmic structure — it would compress well or show
algorithmic information content above random expectation.
"""

from __future__ import annotations
import numpy as np
from temporal_seti.signals.base import SignalSimulator
from temporal_seti.core.types import TimeSeries


class VariableDilationSignal(SignalSimulator):
    """Simulate a variable time-dilation (drifting clock) signal.

    A transmitter at a fixed gravitational depth emits regular pulses.
    The civilization modulates the local gravitational field slowly,
    causing the observed pulse spacing to drift. The drift encodes
    a smooth function (e.g. digits of a constant) rather than random noise.
    """

    def __init__(self, background_rate: float = 10.0, exposure: float = 1000.0,
                 seed: int = 42, base_period: float = 0.1,
                 drift_amplitude: float = 0.01,
                 drift_pattern: str = "sinusoidal"):
        """Initialize variable time-dilation signal.

        Args:
            background_rate: Poisson background (counts/sec)
            exposure: observation duration (seconds)
            seed: random seed
            base_period: transmitter pulse period (seconds)
            drift_amplitude: fractional period change due to drift
            drift_pattern: 'sinusoidal', 'linear', or 'exponential'
        """
        super().__init__(background_rate, exposure, seed)
        self.base_period = base_period
        self.drift_amplitude = drift_amplitude
        self.drift_pattern = drift_pattern

    def _drift_factor(self, t: float) -> float:
        """Compute the drift factor at time t.

        WHY: The drift factor modulates the observed period. Different
        patterns produce different observable signatures. A sinusoidal
        drift might mimic orbital modulation; an exponential or
        algorithmic drift would be clearly artificial.
        """
        t_norm = t / self.exposure
        if self.drift_pattern == "sinusoidal":
            return 1.0 + self.drift_amplitude * np.sin(2 * np.pi * t_norm)
        elif self.drift_pattern == "linear":
            return 1.0 + self.drift_amplitude * t_norm
        elif self.drift_pattern == "exponential":
            return 1.0 + self.drift_amplitude * np.exp(t_norm) - self.drift_amplitude
        else:
            return 1.0

    def generate(self, source_name: str = "sim_pulsar_drift",
                 source_type: str = "pulsar",
                 instrument: str = "NICER") -> TimeSeries:
        """Generate a variable time-dilation signal.

        Each pulse is emitted at the base period, but the observed
        interval is modulated by the drift factor. The cumulative effect
        produces a smooth deviation from strict periodicity.
        """
        signal_times = []
        t = 0.0

        while t < self.exposure:
            signal_times.append(t)
            period = self.base_period * self._drift_factor(t)
            t += period

        signal_arr = np.array(signal_times)
        return self._merge_with_background(
            signal_arr,
            source_name=source_name,
            source_type=source_type,
            instrument=instrument,
        )
