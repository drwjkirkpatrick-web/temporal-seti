"""
Signal Type 2: Multi-timescale nested encoding (temporal fractals).

WHY: Information encoded simultaneously at nanosecond, millisecond, and
second scales. Current X-ray analysis folds data at a single characteristic
period. Nested encoding is invisible to any single-scale Fourier analysis
but visible in a wavelet or multi-resolution decomposition.

The signal is a temporal fractal: the same pattern repeats at different
timescales. At each scale, the pattern contributes a small modulation
that is individually sub-threshold but collectively structured.
"""

from __future__ import annotations
import numpy as np
from temporal_seti.signals.base import SignalSimulator
from temporal_seti.core.types import TimeSeries


class MultiTimescaleSignal(SignalSimulator):
    """Simulate a multi-timescale nested (fractal) signal.

    The signal encodes the same binary pattern at three timescales:
    - Macro: 10-second blocks (coarse structure)
    - Meso: 1-second blocks (medium structure)
    - Micro: 0.01-second blocks (fine structure)

    Each scale adds a small rate modulation. No single scale is
    detectable above noise, but the cross-scale correlation is.
    """

    def __init__(self, background_rate: float = 10.0, exposure: float = 1000.0,
                 seed: int = 42, pattern: list[int] | None = None,
                 modulation_depth: float = 0.3):
        """Initialize multi-timescale signal.

        Args:
            background_rate: base Poisson rate (counts/sec)
            exposure: observation duration (seconds)
            seed: random seed
            pattern: binary pattern to encode (e.g. [1,0,1,1,0])
            modulation_depth: fractional rate change per scale (0-1)
        """
        super().__init__(background_rate, exposure, seed)
        self.pattern = pattern or [1, 0, 1, 1, 0, 0, 1, 0]
        self.modulation_depth = modulation_depth

    def generate(self, source_name: str = "sim_xrb_nested",
                 source_type: str = "xrb",
                 instrument: str = "NICER") -> TimeSeries:
        """Generate a multi-timescale nested signal.

        The rate at any time is the product of modulations at three scales.
        A photon\'s probability of arriving in a given interval depends on
        all three scales simultaneously.
        """
        # Generate non-uniform Poisson process via thinning
        # Base rate * macro_modulation * meso_modulation * micro_modulation
        dt = 0.001  # 1ms time step
        n_steps = int(self.exposure / dt)
        times = np.arange(n_steps) * dt

        # Macro scale: 10s blocks
        macro_period = 10.0
        macro_idx = ((times / macro_period).astype(int) % len(self.pattern))
        macro_mod = 1.0 + self.modulation_depth * (2 * np.array(self.pattern)[macro_idx] - 1)

        # Meso scale: 1s blocks
        meso_period = 1.0
        meso_idx = ((times / meso_period).astype(int) % len(self.pattern))
        meso_mod = 1.0 + self.modulation_depth * (2 * np.array(self.pattern)[meso_idx] - 1)

        # Micro scale: 0.01s blocks
        micro_period = 0.01
        micro_idx = ((times / micro_period).astype(int) % len(self.pattern))
        micro_mod = 1.0 + self.modulation_depth * (2 * np.array(self.pattern)[micro_idx] - 1)

        # Combined rate
        rate = self.background_rate * macro_mod * meso_mod * micro_mod

        # Thinning: generate at max rate, accept by probability
        max_rate = rate.max()
        n_candidates = self.rng.poisson(max_rate * dt, n_steps)
        candidates = np.repeat(times, n_candidates)
        accept_prob = rate / max_rate
        accepts = self.rng.uniform(0, 1, len(candidates)) < accept_prob[np.searchsorted(times, candidates)]
        signal_times = candidates[accepts]

        return TimeSeries(
            arrival_times=np.sort(signal_times),
            source_name=source_name,
            source_type=source_type,
            instrument=instrument,
            exposure=self.exposure,
        )
