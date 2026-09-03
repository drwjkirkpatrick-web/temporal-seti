"""
Signal Type 4: Anti-glitch encoded signals.

WHY: Magnetars show "glitches" (sudden spin-up) and rare "anti-glitches"
(sudden spin-down). The 2013 anti-glitch in 1E 2259+586 was physically
puzzling. If a civilization can manipulate accretion or magnetic field
geometry near a neutron star, a glitch/anti-glitch sequence is an ideal
information carrier: spin-up = 1, spin-down = 0. Nobody has checked
whether the glitch pattern across all known magnetars encodes information.
"""

from __future__ import annotations
import numpy as np
from temporal_seti.signals.base import SignalSimulator
from temporal_seti.core.types import TimeSeries


class AntiGlitchSignal(SignalSimulator):
    """Simulate a glitch/anti-glitch encoded signal.

    A magnetar-like source has a steady spin period. At controlled times,
    the spin rate jumps up (glitch = binary 1) or down (anti-glitch = 0).
    The jumps are small enough to look like natural timing noise but
    encode a binary message in their sequence.
    """

    def __init__(self, background_rate: float = 10.0, exposure: float = 10000.0,
                 seed: int = 42, spin_period: float = 7.0,
                 message: list[int] | None = None,
                 glitch_interval: float = 500.0,
                 glitch_magnitude: float = 1e-6):
        """Initialize anti-glitch signal.

        Args:
            background_rate: Poisson rate (counts/sec)
            exposure: total observation time (seconds)
            seed: random seed
            spin_period: neutron star spin period (seconds)
            message: binary message to encode (e.g. [1,0,1,1,0,1])
            glitch_interval: time between glitches (seconds)
            glitch_magnitude: fractional spin rate change per glitch
        """
        super().__init__(background_rate, exposure, seed)
        self.spin_period = spin_period
        self.message = message or [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        self.glitch_interval = glitch_interval
        self.glitch_magnitude = glitch_magnitude

    def generate(self, source_name: str = "sim_magnetar_glitch",
                 source_type: str = "magnetar",
                 instrument: str = "Swift_BAT") -> TimeSeries:
        """Generate a glitch-encoded signal.

        The source emits pulses at a steady spin period. At each glitch
        time, the spin rate changes by +/- glitch_magnitude depending
        on the message bit. The pulses are X-ray peaks from accretion.
        """
        signal_times = []
        t = 0.0
        current_period = self.spin_period
        msg_idx = 0

        while t < self.exposure:
            # Check if we need to apply a glitch
            glitch_times = np.arange(len(self.message)) * self.glitch_interval
            for gi, gt in enumerate(glitch_times):
                if gi < len(self.message) and abs(t - gt) < current_period / 2:
                    # Apply glitch: bit=1 speeds up (shorter period), bit=0 slows down
                    if self.message[gi] == 1:
                        current_period *= (1 - self.glitch_magnitude)
                    else:
                        current_period *= (1 + self.glitch_magnitude)
                    break

            signal_times.append(t)
            t += current_period

        signal_arr = np.array(signal_times)
        return self._merge_with_background(
            signal_arr,
            source_name=source_name,
            source_type=source_type,
            instrument=instrument,
        )
