"""Base class for temporal SETI signal simulators.

Provides the common infrastructure shared by all signal types in the
``temporal_seti.signals`` subpackage: a Poisson background generator, a
helper to merge injected signal photons with the background, and the
abstract :meth:`generate` interface that every concrete simulator must
implement.
"""

import numpy as np

from temporal_seti.core.types import TimeSeries


class SignalSimulator:
    """Abstract base class for simulating temporal SETI signals.

    Subclasses implement :meth:`generate` to produce an injected signal,
    typically merging it with a stochastic Poisson background via
    :meth:`_merge_with_background`.

    Parameters
    ----------
    background_rate : float, default 10.0
        Mean number of background events per unit time (e.g. counts/s).
    exposure : float, default 1000.0
        Total observation window length in the same time units used for
        ``background_rate``.
    seed : int, default 42
        Seed for the reproducible random number generator used by this
        simulator instance.
    """

    def __init__(self, background_rate=10.0, exposure=1000.0, seed=42):
        self.background_rate = float(background_rate)
        self.exposure = float(exposure)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def _generate_background(self) -> np.ndarray:
        """Generate Poisson-distributed background arrival times.

        Draws the total background count from a Poisson distribution with
        expectation ``background_rate * exposure`` and scatters the arrival
        times uniformly across the observation window ``[0, exposure]``.

        Returns
        -------
        np.ndarray
            Sorted 1-D array of background event arrival times.
        """
        expected_count = self.background_rate * self.exposure
        n_background = self.rng.poisson(expected_count)
        times = self.rng.uniform(0.0, self.exposure, size=n_background)
        times.sort()
        return times

    def _merge_with_background(
        self,
        signal_times: np.ndarray,
        source_name: str,
        source_type: str,
        instrument: str,
    ) -> TimeSeries:
        """Merge injected signal arrival times with the Poisson background.

        Parameters
        ----------
        signal_times : np.ndarray
            1-D array of injected signal arrival times.
        source_name : str
            Identifier for the simulated source.
        source_type : str
            Classification label for the source (e.g. ``"temporal_seti"``).
        instrument : str
            Name of the simulated observing instrument.

        Returns
        -------
        TimeSeries
            Combined, sorted arrival times (signal + background) wrapped in
            a :class:`~temporal_seti.core.types.TimeSeries` with the supplied
            metadata and the configured exposure.
        """
        signal_times = np.asarray(signal_times, dtype=float)
        background_times = self._generate_background()
        all_times = np.concatenate([signal_times, background_times])
        all_times.sort()
        return TimeSeries(
            arrival_times=all_times,
            source_name=source_name,
            source_type=source_type,
            instrument=instrument,
            exposure=self.exposure,
        )

    def generate(self, **kwargs) -> TimeSeries:
        """Generate a signal merged with the background.

        Concrete subclasses must override this method to produce the injected
        signal arrival times and call :meth:`_merge_with_background`.

        Raises
        ------
        NotImplementedError
            Always, when called on the base class or a subclass that fails to
            override it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement generate()"
        )