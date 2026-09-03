"""
Signal Type 3: Temporal beat patterns (QPO phase modulation).

WHY: Two natural X-ray pulsation rates (e.g. spin frequency and orbital
frequency in a binary) create a beat frequency. If a civilization can
slightly perturb one rate at controlled times, the beat pattern itself
becomes the carrier. The beat is not a new frequency — it\'s a temporal
interference pattern. It would look like quasi-periodic oscillations
(QPOs), which are already observed but not fully explained.
"""

from __future__ import annotations
import numpy as np
from temporal_seti.signals.base import SignalSimulator
from temporal_seti.core.types import TimeSeries


class TemporalBeatSignal(SignalSimulator):
    """Simulate a temporal beat pattern with phase modulation.

    The signal has two carrier frequencies (f1, f2) whose beat (f1-f2)
    is phase-modulated to carry information. The phase modulation is
    subtle — it looks like noise in the beat frequency but encodes
    binary data in the phase.
    """

    def __init__(self, background_rate: float = 10.0, exposure: float = 1000.0,
                 seed: int = 42, f1: float = 100.0, f2: float = 97.0,
                 phase_key: list[float] | None = None):
        """Initialize temporal beat signal.

        Args:
            background_rate: Poisson background counts/sec
            exposure: observation duration (seconds)
            seed: random seed
            f1: primary carrier frequency (Hz)
            f2: secondary carrier frequency (Hz)
            phase_key: sequence of phase offsets (radians) to encode
        """
        super().__init__(background_rate, exposure, seed)
        self.f1 = f1
        self.f2 = f2
        self.phase_key = phase_key or [0.0, np.pi/4, np.pi/2, 3*np.pi/4]

    def generate(self, source_name: str = "sim_xrb_beat",
                 source_type: str = "xrb",
                 instrument: str = "RXTE_PCA") -> TimeSeries:
        """Generate a temporal beat signal with phase encoding.

        The intensity modulation is:
            I(t) = I0 * [1 + A*cos(2*pi*f1*t + phi(t))]
        where phi(t) cycles through the phase key at the beat frequency.
        """
        dt = 0.0001  # 100 microsecond step
        n_steps = int(self.exposure / dt)
        times = np.arange(n_steps) * dt

        beat_freq = abs(self.f1 - self.f2)
        # Phase modulation: cycle through key at beat period
        beat_period = 1.0 / beat_freq if beat_freq > 0 else self.exposure
        key_idx = ((times / beat_period).astype(int) % len(self.phase_key))
        phase = np.array(self.phase_key)[key_idx]

        # Intensity modulation (fractional)
        modulation = 0.3 * np.cos(2 * np.pi * self.f1 * times + phase)

        # Poisson rate = background * (1 + modulation)
        rate = np.maximum(self.background_rate * (1 + modulation), 0.1)

        # Thinning
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
