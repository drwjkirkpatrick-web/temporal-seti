"""
Configuration for temporal SETI analysis.

WHY: Centralized config prevents hard-coded values from scattering across
detectors. The significance threshold, Monte Carlo parameters, and instrument
selection all live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from temporal_seti.core.types import AnalysisConfig, InstrumentConfig


@dataclass
class Config:
    """Top-level configuration for the temporal SETI pipeline.

    NOTE: The instruments dict maps instrument name to its config. The
    pipeline uses this to decide which detectors can run on which data
    based on time resolution requirements.
    """
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    instruments: dict[str, InstrumentConfig] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Config":
        """Create a config with default analysis settings and known instruments."""
        from temporal_seti.instruments.catalog import INSTRUMENT_CATALOG
        return cls(
            analysis=AnalysisConfig(),
            instruments=dict(INSTRUMENT_CATALOG),
        )