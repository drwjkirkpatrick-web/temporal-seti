"""
Command-line interface for temporal SETI analysis.

WHY: The CLI lets users generate test signals, run detectors on them,
and see results without writing Python. This makes the tool accessible
for quick testing and demonstrations.
"""

from __future__ import annotations

import argparse
import json
import sys

from temporal_seti.core.config import Config
from temporal_seti.core.types import TimeSeries
from temporal_seti.pipeline.runner import Pipeline


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: command-line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 = success)
    """
    parser = argparse.ArgumentParser(
        prog="temporal-seti",
        description="Temporal SETI: search for time-encoded technosignatures",
    )
    sub = parser.add_subparsers(dest="command")

    # Simulate command
    sim = sub.add_parser("simulate", help="Generate a test signal and analyze it")
    sim.add_argument("signal_type", choices=[
        "time_dilation", "multi_timescale", "temporal_beat",
        "anti_glitch", "time_reversed", "variable_dilation",
        "all",
    ], help="Type of signal to simulate")
    sim.add_argument("--background", type=float, default=10.0,
                     help="Background count rate (counts/sec)")
    sim.add_argument("--exposure", type=float, default=1000.0,
                     help="Observation exposure (seconds)")
    sim.add_argument("--seed", type=int, default=42, help="Random seed")
    sim.add_argument("--verbose", action="store_true", help="Verbose output")

    # List instruments
    sub.add_parser("instruments", help="List known X-ray instruments")

    # List signal types
    sub.add_parser("signals", help="List signal types")

    args = parser.parse_args(argv)

    if args.command == "simulate":
        return _cmd_simulate(args)
    elif args.command == "instruments":
        return _cmd_instruments()
    elif args.command == "signals":
        return _cmd_signals()
    else:
        parser.print_help()
        return 0


def _cmd_simulate(args) -> int:
    """Generate a signal and run the full pipeline."""
    from temporal_seti.signals import (
        TimeDilationSignal, MultiTimescaleSignal, TemporalBeatSignal,
        AntiGlitchSignal, TimeReversedSignal, VariableDilationSignal,
    )

    signal_map = {
        "time_dilation": TimeDilationSignal,
        "multi_timescale": MultiTimescaleSignal,
        "temporal_beat": TemporalBeatSignal,
        "anti_glitch": AntiGlitchSignal,
        "time_reversed": TimeReversedSignal,
        "variable_dilation": VariableDilationSignal,
    }

    pipe = Pipeline(Config.default())

    if args.signal_type == "all":
        for name, cls in signal_map.items():
            print(f"\n{'='*60}")
            print(f"Simulating: {name}")
            print(f"{'='*60}")
            sim = cls(background_rate=args.background, exposure=args.exposure, seed=args.seed)
            series = sim.generate()
            print(f"Generated {series.count} photons over {series.exposure}s")
            report = pipe.run_and_report(series)
            print(report)
    else:
        cls = signal_map[args.signal_type]
        sim = cls(background_rate=args.background, exposure=args.exposure, seed=args.seed)
        series = sim.generate()
        print(f"Generated {series.count} photons over {series.exposure}s")
        report = pipe.run_and_report(series)
        print(report)

    return 0


def _cmd_instruments() -> int:
    """List all known instruments."""
    from temporal_seti.instruments.catalog import INSTRUMENT_CATALOG
    print(f"{'Name':<20} {'Resolution (us)':<18} {'Area (cm^2)':<14} {'Bandpass':<15} {'All-Sky'}")
    print("-" * 80)
    for inst in INSTRUMENT_CATALOG.values():
        bp = f"{inst.bandpass_min}-{inst.bandpass_max} keV"
        asm = "Yes" if inst.all_sky_monitor else "No"
        print(f"{inst.name:<20} {inst.time_resolution:<18.1f} {inst.collecting_area:<14.0f} {bp:<15} {asm}")
    return 0


def _cmd_signals() -> int:
    """List all signal types."""
    from temporal_seti.core.types import SignalType
    print("Temporal SETI Signal Types:")
    print("-" * 60)
    for st in SignalType:
        print(f"  {st.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
