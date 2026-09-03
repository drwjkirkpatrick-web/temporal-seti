"""
Signal Type 1: Time-dilation-encoded pulse trains.

WHY: A transmitter near a neutron star at different gravitational potentials
sends pulses. The interval between pulses is compressed or stretched by the
local time dilation factor. To a receiver who doesn\'t know the time
transformation, the pulses look like timing noise. To one who applies the
correct de-warping, the pulses resolve into a clean signal.

The Schwarzschild time dilation factor for a clock at radius r from a mass M:

    t_inf / t_local = 1 / sqrt(1 - r_s / r)

where r_s = 2GM/c^2 is the Schwarzschild radius. For a neutron star,
r_s ~ 3 km and the surface is at r ~ 10 km, giving a dilation factor of
~1.26. A transmitter that moves between r=10km and r=20km modulates
the clock rate by ~14%.
"""

from __future__ import annotations
import numpy as np
from temporal_seti.signals.base import SignalSimulator
from temporal_seti.core.types import TimeSeries


class TimeDilationSignal(SignalSimulator):
    """Simulate a time-dilation-encoded pulse train.

    The signal consists of regularly-spaced pulses in the transmitter\'s
    local frame. When observed from infinity, each pulse is shifted by
    a time dilation factor that varies according to a key sequence.
    Without the key, the arrival times appear as non-periodic noise.
    """

    def __init__(self, background_rate: float = 10.0, exposure: float = 1000.0,
                 seed: int = 42, base_period: float = 0.1,
                 dilation_key: list[float] | None = None):
        """Initialize the time-dilation signal simulator.

        Args:
            background_rate: Poisson background counts/sec
            exposure: observation duration in seconds
            seed: random seed
            base_period: pulse period in the local frame (seconds)
            dilation_key: sequence of time dilation factors (e.g. [1.0, 1.1, 1.25])
        """
        super().__init__(background_rate, exposure, seed)
        self.base_period = base_period
        # Default key: three gravitational well depths
        self.dilation_key = dilation_key or [1.0, 1.1, 1.26]

    def generate(self, source_name: str = "sim_pulsar_td",
                 source_type: str = "pulsar",
                 instrument: str = "RXTE_PCA") -> TimeSeries:
        """Generate a time-dilation-encoded signal.

        The transmitter emits pulses at regular intervals in its local
        frame. We apply a cycling sequence of dilation factors (the key)
        to each pulse. An observer without the key sees irregular spacing.
        """
        signal_times = []
        t_local = 0.0

        while t_local < self.exposure:
            # Cycle through the dilation key
            key_idx = len(signal_times) % len(self.dilation_key)
            gamma = self.dilation_key[key_idx]

            # Observed time = local time * dilation factor
            t_observed = t_local * gamma

            if t_observed < self.exposure:
                signal_times.append(t_observed)

            t_local += self.base_period

        signal_arr = np.array(signal_times)
        return self._merge_with_background(
            signal_arr,
            source_name=source_name,
            source_type=source_type,
            instrument=instrument,
        )
