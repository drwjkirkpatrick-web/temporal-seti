"""
Tests for core types and configuration.
"""

import numpy as np
import pytest
from temporal_seti.core.types import (
    SignalType, TimeSeries, DetectionResult, AnalysisConfig, InstrumentConfig,
)
from temporal_seti.core.config import Config


class TestSignalType:
    def test_six_types(self):
        assert len(list(SignalType)) == 6

    def test_values(self):
        for st in SignalType:
            assert isinstance(st.value, str)
            assert st.value  # non-empty


class TestTimeSeries:
    def test_creation(self):
        times = np.sort(np.random.uniform(0, 100, 1000))
        ts = TimeSeries(arrival_times=times, exposure=100.0)
        assert ts.count == 1000
        assert ts.exposure == 100.0

    def test_sorted_validation(self):
        unsorted = np.array([3.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="sorted"):
            TimeSeries(arrival_times=unsorted, exposure=10.0)

    def test_non_negative_validation(self):
        negative = np.array([-1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            TimeSeries(arrival_times=negative, exposure=10.0)

    def test_rate(self):
        times = np.sort(np.random.uniform(0, 100, 500))
        ts = TimeSeries(arrival_times=times, exposure=100.0)
        assert ts.rate == pytest.approx(5.0, abs=0.5)

    def test_to_bins(self):
        times = np.sort(np.random.uniform(0, 100, 1000))
        ts = TimeSeries(arrival_times=times, exposure=100.0)
        binned = ts.to_bins(1.0)
        assert len(binned) == 100
        assert binned.sum() == 1000

    def test_empty(self):
        ts = TimeSeries(arrival_times=np.array([]), exposure=10.0)
        assert ts.count == 0
        assert ts.rate == 0.0


class TestAnalysisConfig:
    def test_defaults(self):
        config = AnalysisConfig()
        assert config.significance_threshold == 5.0
        assert config.min_counts == 100
        assert config.n_permutations == 1000

    def test_custom(self):
        config = AnalysisConfig(n_permutations=50, significance_threshold=3.0)
        assert config.n_permutations == 50
        assert config.significance_threshold == 3.0


class TestInstrumentConfig:
    def test_creation(self):
        inst = InstrumentConfig(
            name="TEST", time_resolution=1.0, energy_resolution=10.0,
            collecting_area=100.0, bandpass_min=2.0, bandpass_max=10.0,
        )
        assert inst.name == "TEST"
        assert inst.time_resolution == 1.0


class TestConfig:
    def test_default(self):
        config = Config.default()
        assert len(config.instruments) > 0
        assert "RXTE_PCA" in config.instruments

    def test_analysis_defaults(self):
        config = Config.default()
        assert config.analysis.n_permutations == 1000
