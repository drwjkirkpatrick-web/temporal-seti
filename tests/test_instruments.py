"""
Tests for instrument catalog.
"""

from temporal_seti.instruments.catalog import (
    INSTRUMENT_CATALOG, get_instruments_by_resolution, get_all_sky_monitors,
)


class TestInstrumentCatalog:
    def test_catalog_not_empty(self):
        assert len(INSTRUMENT_CATALOG) > 0

    def test_known_instruments(self):
        assert "RXTE_PCA" in INSTRUMENT_CATALOG
        assert "NICER" in INSTRUMENT_CATALOG
        assert "Chandra_HRC" in INSTRUMENT_CATALOG

    def test_all_have_required_fields(self):
        for name, inst in INSTRUMENT_CATALOG.items():
            assert inst.name == name
            assert inst.time_resolution > 0
            assert inst.collecting_area > 0
            assert inst.bandpass_min < inst.bandpass_max


class TestResolutionFilter:
    def test_filters_by_resolution(self):
        fast = get_instruments_by_resolution(1.0)  # 1 microsecond
        for inst in fast:
            assert inst.time_resolution <= 1.0

    def test_all_returned_for_large_threshold(self):
        all_inst = get_instruments_by_resolution(10000.0)
        assert len(all_inst) == len(INSTRUMENT_CATALOG)


class TestAllSkyMonitors:
    def test_returns_monitors(self):
        monitors = get_all_sky_monitors()
        for inst in monitors:
            assert inst.all_sky_monitor is True

    def test_at_least_one(self):
        monitors = get_all_sky_monitors()
        assert len(monitors) >= 1
