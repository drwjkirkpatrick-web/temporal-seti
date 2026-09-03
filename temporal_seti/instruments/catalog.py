"""
Catalog of X-ray astronomy instruments relevant to temporal SETI.

WHY: The detector needs to know what instruments exist, their time
resolution, and whether they're all-sky monitors or pointed. This
determines which signal types are searchable in existing archives.

Data sources:
- Corbet (1997) Table 1 & 2 (arXiv:1609.00330)
- NASA HEASARC mission summaries
- NICER, NuSTAR, Chandra, XMM-Newton instrument pages

NOTE: Time resolution is the best AVAILABLE resolution, not the typical
observation mode. For example, RXTE PCA can do 1 microsecond but most
observations used 2^-22 second bins (~0.25 ms).
"""

from __future__ import annotations

from temporal_seti.core.types import InstrumentConfig

# --- All-sky monitors (good for transient/glitch detection) ---

INSTRUMENT_CATALOG: dict[str, InstrumentConfig] = {
    "RXTE_ASM": InstrumentConfig(
        name="RXTE_ASM",
        time_resolution=3.0,           # ~3s per scan
        energy_resolution=1.0,          # low res, broad bands
        collecting_area=4500.0,        # cm^2
        bandpass_min=2.0,
        bandpass_max=10.0,
        mission_dates="1995-2012",
        all_sky_monitor=True,
    ),
    "MAXI": InstrumentConfig(
        name="MAXI",
        time_resolution=1.0,           # ~1s for all-sky
        energy_resolution=2.0,
        collecting_area=300.0,
        bandpass_min=2.0,
        bandpass_max=30.0,
        mission_dates="2009-present",
        all_sky_monitor=True,
    ),
    "Swift_BAT": InstrumentConfig(
        name="Swift_BAT",
        time_resolution=64.0,          # 64 microseconds for GRBs
        energy_resolution=3.0,
        collecting_area=5200.0,
        bandpass_min=15.0,
        bandpass_max=150.0,
        mission_dates="2004-present",
        all_sky_monitor=True,
    ),

    # --- Pointed instruments (high time resolution, targeted) ---

    "RXTE_PCA": InstrumentConfig(
        name="RXTE_PCA",
        time_resolution=1.0,           # 1 microsecond (best mode)
        energy_resolution=6.0,          # E/dE at 6.4 keV
        collecting_area=7000.0,
        bandpass_min=2.0,
        bandpass_max=60.0,
        mission_dates="1995-2012",
        all_sky_monitor=False,
    ),
    "NICER": InstrumentConfig(
        name="NICER",
        time_resolution=100.0,         # 100 nanoseconds (0.1 microsecond)
        energy_resolution=50.0,        # moderate spectral res
        collecting_area=1900.0,        # cm^2 (56 aligned pairs)
        bandpass_min=0.2,
        bandpass_max=12.0,
        mission_dates="2017-present",
        all_sky_monitor=False,
    ),
    "NuSTAR": InstrumentConfig(
        name="NuSTAR",
        time_resolution=2.0,          # 2 microseconds event timing
        energy_resolution=50.0,        # good for hard X-rays
        collecting_area=800.0,
        bandpass_min=3.0,
        bandpass_max=79.0,
        mission_dates="2012-present",
        all_sky_monitor=False,
    ),
    "Chandra_HRC": InstrumentConfig(
        name="Chandra_HRC",
        time_resolution=16.0,         # 16 microseconds
        energy_resolution=1.0,         # imaging, low spectral
        collecting_area=27.0,         # cm^2 (small but high angular res)
        bandpass_min=0.1,
        bandpass_max=10.0,
        mission_dates="2000-present",
        all_sky_monitor=False,
    ),
    "XMM_PN": InstrumentConfig(
        name="XMM_PN",
        time_resolution=30.0,         # 30 nanoseconds in timing mode
        energy_resolution=50.0,
        collecting_area=4660.0,       # cm^2 (largest X-ray telescope)
        bandpass_min=0.1,
        bandpass_max=15.0,
        mission_dates="2000-present",
        all_sky_monitor=False,
    ),
    "AstroSAT_SXT": InstrumentConfig(
        name="AstroSAT_SXT",
        time_resolution=2000.0,       # 2 ms
        energy_resolution=20.0,
        collecting_area=900.0,
        bandpass_min=0.3,
        bandpass_max=8.0,
        mission_dates="2015-present",
        all_sky_monitor=False,
    ),
    "Hitomi_SXS": InstrumentConfig(
        name="Hitomi_SXS",
        time_resolution=20.0,         # 20 microseconds (microcalorimeter)
        energy_resolution=670.0,      # extremely high spectral resolution
        collecting_area=58.0,
        bandpass_min=0.4,
        bandpass_max=12.0,
        mission_dates="2016 (brief)",
        all_sky_monitor=False,
    ),
    "XRISM_Resolve": InstrumentConfig(
        name="XRISM_Resolve",
        time_resolution=20.0,         # 20 microseconds
        energy_resolution=670.0,      # microcalorimeter, like Hitomi
        collecting_area=58.0,
        bandpass_min=0.4,
        bandpass_max=12.0,
        mission_dates="2023-present",
        all_sky_monitor=False,
    ),

    # --- Future instruments ---

    "STROBE_X": InstrumentConfig(
        name="STROBE_X",
        time_resolution=1.0,           # design goal: 1 microsecond
        energy_resolution=50.0,
        collecting_area=50000.0,      # very large area
        bandpass_min=0.2,
        bandpass_max=30.0,
        mission_dates="proposed",
        all_sky_monitor=False,
    ),
    "THESEUS_IST": InstrumentConfig(
        name="THESEUS_IST",
        time_resolution=10.0,
        energy_resolution=20.0,
        collecting_area=1000.0,
        bandpass_min=0.3,
        bandpass_max=10.0,
        mission_dates="proposed 2030s",
        all_sky_monitor=True,
    ),
}


def get_instruments_by_resolution(max_resolution_us: float) -> list[InstrumentConfig]:
    """Return instruments with time resolution better than the given threshold.

    WHY: A detector that needs nanosecond timing should only be run on data
    from instruments that provide it. This function filters the catalog.
    """
    return [
        inst for inst in INSTRUMENT_CATALOG.values()
        if inst.time_resolution <= max_resolution_us
    ]


def get_all_sky_monitors() -> list[InstrumentConfig]:
    """Return only all-sky monitoring instruments.

    WHY: All-sky monitors are essential for glitch and transient detection
    because they survey the whole sky rather than pointed observations.
    """
    return [inst for inst in INSTRUMENT_CATALOG.values() if inst.all_sky_monitor]