"""
Tests for signal simulators.
"""

import numpy as np
import pytest
from temporal_seti.core.types import TimeSeries
from temporal_seti.signals import (
    TimeDilationSignal, MultiTimescaleSignal, TemporalBeatSignal,
    AntiGlitchSignal, TimeReversedSignal, VariableDilationSignal,
)


class TestTimeDilationSignal:
    def test_generate(self):
        sim = TimeDilationSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        assert isinstance(series, TimeSeries)
        assert series.count > 0
        assert series.exposure == 200.0

    def test_reproducible(self):
        sim1 = TimeDilationSignal(seed=99)
        sim2 = TimeDilationSignal(seed=99)
        s1 = sim1.generate()
        s2 = sim2.generate()
        np.testing.assert_array_equal(s1.arrival_times, s2.arrival_times)

    def test_custom_key(self):
        sim = TimeDilationSignal(dilation_key=[1.0, 1.5, 2.0])
        assert sim.dilation_key == [1.0, 1.5, 2.0]


class TestMultiTimescaleSignal:
    def test_generate(self):
        sim = MultiTimescaleSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        assert isinstance(series, TimeSeries)
        assert series.count > 0


class TestTemporalBeatSignal:
    def test_generate(self):
        sim = TemporalBeatSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        assert isinstance(series, TimeSeries)
        assert series.count > 0

    def test_beat_frequency(self):
        sim = TemporalBeatSignal(f1=100.0, f2=97.0)
        beat = abs(sim.f1 - sim.f2)
        assert beat == 3.0


class TestAntiGlitchSignal:
    def test_generate(self):
        sim = AntiGlitchSignal(background_rate=5.0, exposure=2000.0, seed=42)
        series = sim.generate()
        assert isinstance(series, TimeSeries)
        assert series.count > 0

    def test_message(self):
        msg = [1, 0, 1, 1, 0]
        sim = AntiGlitchSignal(message=msg)
        assert sim.message == msg


class TestTimeReversedSignal:
    def test_generate(self):
        sim = TimeReversedSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        assert isinstance(series, TimeSeries)
        assert series.count > 0


class TestVariableDilationSignal:
    def test_generate(self):
        sim = VariableDilationSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        assert isinstance(series, TimeSeries)
        assert series.count > 0

    def test_drift_patterns(self):
        for pattern in ["sinusoidal", "linear", "exponential"]:
            sim = VariableDilationSignal(drift_pattern=pattern)
            series = sim.generate()
            assert series.count > 0
