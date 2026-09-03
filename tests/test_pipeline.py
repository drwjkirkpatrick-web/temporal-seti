"""
Tests for the analysis pipeline.
"""

import numpy as np
import pytest
from temporal_seti.core.config import Config
from temporal_seti.core.types import TimeSeries
from temporal_seti.pipeline.runner import Pipeline
from temporal_seti.signals import TimeDilationSignal


@pytest.fixture
def pipe():
    p = Pipeline(Config.default())
    p.config.analysis.n_permutations = 20
    return p


class TestPipeline:
    def test_run_returns_results(self, pipe):
        sim = TimeDilationSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        results = pipe.run(series)
        assert len(results) == 6  # one per detector

    def test_results_sorted_by_confidence(self, pipe):
        sim = TimeDilationSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        results = pipe.run(series)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_report_is_string(self, pipe):
        sim = TimeDilationSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        results = pipe.run(series)
        report = pipe.report(results)
        assert isinstance(report, str)
        assert "TEMPORAL SETI" in report

    def test_run_and_report(self, pipe):
        sim = TimeDilationSignal(background_rate=5.0, exposure=200.0, seed=42)
        series = sim.generate()
        report = pipe.run_and_report(series)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_detector_error_handling(self, pipe):
        """Pipeline should not crash if a detector raises an exception."""
        # Create a series that might cause edge cases
        series = TimeSeries(
            arrival_times=np.array([0.0, 0.001, 0.002]),
            exposure=1.0,
        )
        results = pipe.run(series)
        assert len(results) == 6
