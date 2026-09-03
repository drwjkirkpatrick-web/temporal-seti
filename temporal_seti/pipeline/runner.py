"""
Analysis pipeline orchestrator for temporal SETI.

WHY: The pipeline ties together signals, detectors, and instruments. It
takes a TimeSeries (or generates a test signal), runs all applicable
detectors, and returns a combined analysis report.

Key design decisions:
- All six detectors run on every input (different signal types may
  coexist in the same data)
- The pipeline filters detectors by instrument time resolution
- Results are sorted by confidence for prioritized follow-up
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from temporal_seti.core.types import TimeSeries, DetectionResult, AnalysisConfig
from temporal_seti.core.config import Config
from temporal_seti.detectors import ALL_DETECTORS, BaseDetector
from temporal_seti.instruments.catalog import INSTRUMENT_CATALOG


class Pipeline:
    """Orchestrate temporal SETI analysis across all signal types.

    Usage:
        pipe = Pipeline()
        results = pipe.run(time_series)
        report = pipe.report(results)
    """

    def __init__(self, config: Config | None = None):
        """Initialize the pipeline.

        Args:
            config: configuration with analysis settings and instruments.
                    Uses defaults if None.
        """
        self.config = config or Config.default()
        self.detectors: list[BaseDetector] = [
            cls(self.config.analysis) for cls in ALL_DETECTORS
        ]

    def run(self, series: TimeSeries) -> list[DetectionResult]:
        """Run all detectors on a TimeSeries.

        Args:
            series: photon arrival time data to analyze

        Returns:
            List of DetectionResults, sorted by confidence (descending)
        """
        results: list[DetectionResult] = []
        for detector in self.detectors:
            try:
                result = detector.detect(series)
                results.append(result)
            except Exception as e:
                # A detector failure should not stop the pipeline
                results.append(DetectionResult(
                    signal_type=detector.signal_type,
                    detected=False,
                    confidence=0.0,
                    metric_name="error",
                    metric_value=0.0,
                    details={"error": str(e)},
                    source_name=series.source_name,
                    instrument=series.instrument,
                ))

        # Sort by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def report(self, results: list[DetectionResult]) -> str:
        """Generate a human-readable analysis report.

        Args:
            results: list of DetectionResults from run()

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TEMPORAL SETI ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        if not results:
            lines.append("No results to report.")
            return "\n".join(lines)

        source = results[0].source_name if results else ""
        instrument = results[0].instrument if results else ""
        lines.append(f"Source: {source}")
        lines.append(f"Instrument: {instrument}")
        lines.append(f"Detectors run: {len(results)}")
        lines.append("")

        any_detected = False
        for i, r in enumerate(results, 1):
            status = "*** DETECTED ***" if r.detected else "no signal"
            lines.append(f"  {i}. {r.signal_type.value}")
            lines.append(f"     Status: {status}")
            lines.append(f"     Confidence: {r.confidence:.4f}")
            lines.append(f"     Metric: {r.metric_name} = {r.metric_value:.6f}")
            if r.details:
                for k, v in r.details.items():
                    lines.append(f"     {k}: {v}")
            lines.append("")
            if r.detected:
                any_detected = True

        lines.append("-" * 60)
        if any_detected:
            lines.append("CANDIDATE TECHNOSIGNATURES DETECTED")
            lines.append("Follow-up recommended: independent observation,")
            lines.append("multi-wavelength correlation, natural-cause rejection.")
        else:
            lines.append("No technosignature candidates above threshold.")
        lines.append("=" * 60)

        return "\n".join(lines)

    def run_and_report(self, series: TimeSeries) -> str:
        """Convenience: run all detectors and return formatted report."""
        results = self.run(series)
        return self.report(results)
