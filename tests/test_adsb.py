"""Tests for concam.adsb: ADS-B flight loader."""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import pytest

from concam.adsb import (
    Flight,
    Ping,
    _choose_altitude,
    _haversine_km,
    _upsample_pings,
    load_flights,
)
from concam.config import AdsbConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FIXTURE = FIXTURES_DIR / "adsb_april8_sample.json"

SITE_LAT = 42.360444
SITE_LON = -71.089238


# ---------------------------------------------------------------------------
# Unit tests: haversine
# ---------------------------------------------------------------------------


def test_haversine_zero():
    assert _haversine_km(42.0, -71.0, 42.0, -71.0) == pytest.approx(0.0)


def test_haversine_known():
    # Boston to Cambridge: ~5 km
    d = _haversine_km(42.3601, -71.0589, 42.3736, -71.1097)
    assert 3.0 < d < 8.0


# ---------------------------------------------------------------------------
# Unit tests: upsampling
# ---------------------------------------------------------------------------

_UTC = datetime.timezone.utc


def _make_ping(offset_s: int, lat: float = 42.0, lon: float = -71.0, alt: float = 10000.0) -> Ping:
    base = datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=_UTC)
    return Ping(
        time=base + datetime.timedelta(seconds=offset_s),
        lat=lat,
        lon=lon,
        alt_m=alt,
    )


def test_upsample_single_ping():
    pings = [_make_ping(0)]
    result = _upsample_pings(pings)
    assert len(result) == 1


def test_upsample_already_1s():
    pings = [_make_ping(0), _make_ping(1), _make_ping(2)]
    result = _upsample_pings(pings)
    assert len(result) == 3


def test_upsample_fills_gap():
    # 5-second gap → should produce 4 interior points (t=1,2,3,4) + 2 endpoints = 6
    pings = [_make_ping(0, lat=42.0), _make_ping(5, lat=42.5)]
    result = _upsample_pings(pings)
    assert len(result) == 6
    # Check monotone times
    times = [p.time for p in result]
    assert times == sorted(times)
    # Check linear interpolation at t=1s: lat should be 42.0 + 0.1 = 42.1
    assert result[1].lat == pytest.approx(42.0 + (42.5 - 42.0) * (1 / 5), rel=1e-4)


def test_upsample_skips_large_gap():
    # Gap >= 300s: no interpolation, just two endpoints
    pings = [_make_ping(0), _make_ping(300)]
    result = _upsample_pings(pings)
    assert len(result) == 2


def test_upsample_preserves_order():
    pings = [_make_ping(0, lat=42.0), _make_ping(10, lat=43.0)]
    result = _upsample_pings(pings)
    lats = [p.lat for p in result]
    assert lats == sorted(lats)


# ---------------------------------------------------------------------------
# Unit tests: altitude source policy
# ---------------------------------------------------------------------------


def test_choose_altitude_auto_uses_gnss_when_consistent():
    cfg = AdsbConfig(altitude_source="auto", altitude_discrepancy_threshold_m=122.0)
    # baro within 50m of gnss -> prefer gnss
    alt, src = _choose_altitude(alt_baro_m=9950.0, alt_gnss_m=10000.0, config=cfg)
    assert src == "gnss"
    assert alt == 10000.0


def test_choose_altitude_auto_falls_back_to_baro_on_major_discrepancy():
    cfg = AdsbConfig(altitude_source="auto", altitude_discrepancy_threshold_m=122.0)
    # 500m gap -> GNSS is suspect, trust baro
    alt, src = _choose_altitude(alt_baro_m=9500.0, alt_gnss_m=10000.0, config=cfg)
    assert src == "barometric"
    assert alt == 9500.0


def test_choose_altitude_auto_handles_missing_gnss():
    cfg = AdsbConfig(altitude_source="auto")
    alt, src = _choose_altitude(alt_baro_m=9500.0, alt_gnss_m=None, config=cfg)
    assert src == "barometric"
    assert alt == 9500.0


def test_choose_altitude_auto_handles_missing_baro():
    cfg = AdsbConfig(altitude_source="auto")
    alt, src = _choose_altitude(alt_baro_m=None, alt_gnss_m=10000.0, config=cfg)
    assert src == "gnss"
    assert alt == 10000.0


def test_choose_altitude_auto_handles_both_missing():
    cfg = AdsbConfig(altitude_source="auto")
    alt, src = _choose_altitude(alt_baro_m=None, alt_gnss_m=None, config=cfg)
    assert alt is None
    assert src == ""


def test_choose_altitude_policy_gnss_ignores_discrepancy():
    cfg = AdsbConfig(altitude_source="gnss", altitude_discrepancy_threshold_m=122.0)
    alt, src = _choose_altitude(alt_baro_m=9000.0, alt_gnss_m=10000.0, config=cfg)
    assert src == "gnss"
    assert alt == 10000.0


def test_choose_altitude_policy_barometric_always_baro():
    cfg = AdsbConfig(altitude_source="barometric")
    alt, src = _choose_altitude(alt_baro_m=9500.0, alt_gnss_m=10000.0, config=cfg)
    assert src == "barometric"
    assert alt == 9500.0


def test_choose_altitude_invalid_policy_raises():
    cfg = AdsbConfig(altitude_source="invalid")
    with pytest.raises(ValueError):
        _choose_altitude(alt_baro_m=9000.0, alt_gnss_m=10000.0, config=cfg)


def test_upsample_preserves_both_altitudes_when_present():
    base = datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=_UTC)

    def _ping(s, gnss, baro):
        return Ping(
            time=base + datetime.timedelta(seconds=s),
            lat=42.0,
            lon=-71.0,
            alt_m=gnss,
            alt_gnss_m=gnss,
            alt_baro_m=baro,
            alt_source="gnss",
        )

    pings = [_ping(0, 10000.0, 9950.0), _ping(5, 10100.0, 10050.0)]
    result = _upsample_pings(pings)
    assert len(result) == 6
    midpoint = result[1]
    assert midpoint.alt_gnss_m == pytest.approx(10020.0)
    assert midpoint.alt_baro_m == pytest.approx(9970.0)
    assert midpoint.alt_source == "gnss"


# ---------------------------------------------------------------------------
# Integration test: fixture-based (no live feder needed)
# ---------------------------------------------------------------------------


def _load_fixture() -> list[Flight]:
    """Parse the on-disk fixture into Flight objects."""
    with open(SAMPLE_FIXTURE) as f:
        data = json.load(f)
    flights = []
    for entry in data:
        pings = [
            Ping(
                time=datetime.datetime.fromisoformat(p["time"]),
                lat=p["lat"],
                lon=p["lon"],
                alt_m=p["alt_m"],
            )
            for p in entry["pings"]
        ]
        flights.append(
            Flight(
                callsign=entry["callsign"],
                transponder_id=entry["transponder_id"],
                aircraft_type=entry["aircraft_type"],
                orig=entry["orig"],
                dest=entry["dest"],
                pings=pings,
            )
        )
    return flights


def test_fixture_has_expected_flights():
    flights = _load_fixture()
    callsigns = {f.callsign for f in flights}
    assert "AAL141" in callsigns


def test_fixture_pings_count():
    flights = _load_fixture()
    for f in flights:
        assert len(f.pings) == 30


def test_fixture_pings_1s_resolution():
    """All consecutive pings in fixture are exactly 1 second apart."""
    flights = _load_fixture()
    for f in flights:
        for a, b in zip(f.pings, f.pings[1:]):
            gap = (b.time - a.time).total_seconds()
            assert gap == pytest.approx(1.0), f"{f.callsign}: gap={gap}s"


def test_fixture_altitude_above_threshold():
    cfg = AdsbConfig()
    flights = _load_fixture()
    for f in flights:
        for p in f.pings:
            assert p.alt_m >= cfg.min_altitude_m, (
                f"{f.callsign} ping at {p.alt_m:.0f}m below threshold {cfg.min_altitude_m}m"
            )


def test_fixture_within_radius():
    cfg = AdsbConfig()
    flights = _load_fixture()
    for f in flights:
        for p in f.pings:
            dist = _haversine_km(SITE_LAT, SITE_LON, p.lat, p.lon)
            assert dist <= cfg.max_radius_km, (
                f"{f.callsign} ping at {dist:.1f}km exceeds {cfg.max_radius_km}km"
            )


# ---------------------------------------------------------------------------
# Live integration test: requires FEDER_DATA_DIR to be accessible
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("/home/mcast/data/feder").exists(),
    reason="feder data store not available",
)
def test_load_flights_april8_nonempty():
    cfg = AdsbConfig()
    flights = load_flights(datetime.date(2026, 4, 8), cfg)
    assert len(flights) > 0, "Expected at least one flight on April 8, 2026"


@pytest.mark.skipif(
    not Path("/home/mcast/data/feder").exists(),
    reason="feder data store not available",
)
def test_load_flights_all_pings_pass_filters():
    cfg = AdsbConfig()
    flights = load_flights(datetime.date(2026, 4, 8), cfg)
    for f in flights:
        for p in f.pings:
            assert p.alt_m >= cfg.min_altitude_m
            dist = _haversine_km(SITE_LAT, SITE_LON, p.lat, p.lon)
            assert dist <= cfg.max_radius_km


@pytest.mark.skipif(
    not Path("/home/mcast/data/feder").exists(),
    reason="feder data store not available",
)
def test_load_flights_1s_resolution():
    """Upsampled pings are 1s apart, except across large gaps (>=300s) that are intentionally preserved."""
    cfg = AdsbConfig()
    flights = load_flights(datetime.date(2026, 4, 8), cfg)
    for f in flights:
        for a, b in zip(f.pings, f.pings[1:]):
            gap = (b.time - a.time).total_seconds()
            assert gap == pytest.approx(1.0) or gap >= 300, (
                f"{f.callsign}: consecutive pings {gap}s apart (expected 1s or >=300s)"
            )
