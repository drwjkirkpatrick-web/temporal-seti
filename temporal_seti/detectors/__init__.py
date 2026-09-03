"""Temporal SETI signal detectors."""

from temporal_seti.detectors.base import BaseDetector
from temporal_seti.detectors.time_dilation import TimeDilationDetector
from temporal_seti.detectors.multi_timescale import MultiTimescaleDetector
from temporal_seti.detectors.temporal_beat import TemporalBeatDetector
from temporal_seti.detectors.anti_glitch import AntiGlitchDetector
from temporal_seti.detectors.time_reversed import TimeReversedDetector
from temporal_seti.detectors.variable_dilation import VariableDilationDetector

ALL_DETECTORS = [
    TimeDilationDetector,
    MultiTimescaleDetector,
    TemporalBeatDetector,
    AntiGlitchDetector,
    TimeReversedDetector,
    VariableDilationDetector,
]

__all__ = [
    "BaseDetector",
    "TimeDilationDetector",
    "MultiTimescaleDetector",
    "TemporalBeatDetector",
    "AntiGlitchDetector",
    "TimeReversedDetector",
    "VariableDilationDetector",
    "ALL_DETECTORS",
]
