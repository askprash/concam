"""Tests for concam.adsb: ADS-B flight loader."""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import pytest

from concam.adsb import (
    FederFlightSource,
    Flight,
    Ping,
    RawPoint,
    RawTrajectory,
    RecordedFlightSource,
    _choose_altitude,
    _day_window_utc,
    _haversine_km,
    _upsample_pings,
    load_flights,
)
from concam.config import AdsbConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FIXTURE = FIXTURES_DIR / "adsb_april8_sample.json"
RAW_TRACE_FIXTURE = FIXTURES_DIR / "adsb_raw_trace.json"

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
# Unit tests: _day_window_utc (pure — no feder, no I/O)
# ---------------------------------------------------------------------------


def test_day_window_utc_none_is_exact_utc_day():
    """timezone=None → window is exactly UTC midnight to UTC midnight, 24 h."""
    date = datetime.date(2026, 6, 15)
    t_start, t_end = _day_window_utc(date, None)
    assert t_start == datetime.datetime(2026, 6, 15, 0, 0, 0, tzinfo=_UTC)
    assert t_end == datetime.datetime(2026, 6, 16, 0, 0, 0, tzinfo=_UTC)
    assert (t_end - t_start) == datetime.timedelta(hours=24)


def test_day_window_utc_edt_regular_day():
    """America/New_York regular summer day (EDT = UTC-4): window starts 04:00 UTC."""
    date = datetime.date(2026, 6, 15)
    t_start, t_end = _day_window_utc(date, "America/New_York")
    assert t_start == datetime.datetime(2026, 6, 15, 4, 0, 0, tzinfo=_UTC)
    assert t_end == datetime.datetime(2026, 6, 16, 4, 0, 0, tzinfo=_UTC)
    assert (t_end - t_start) == datetime.timedelta(hours=24)


def test_day_window_utc_dst_spring_forward_23h():
    """Spring-forward 2026-03-08 (2am→3am): clocks skip an hour → 23 h UTC window."""
    date = datetime.date(2026, 3, 8)
    t_start, t_end = _day_window_utc(date, "America/New_York")
    # Before DST: EST = UTC-5, so local midnight → 05:00 UTC
    assert t_start == datetime.datetime(2026, 3, 8, 5, 0, 0, tzinfo=_UTC)
    # After DST: EDT = UTC-4, so next local midnight → 04:00 UTC
    assert t_end == datetime.datetime(2026, 3, 9, 4, 0, 0, tzinfo=_UTC)
    assert (t_end - t_start) == datetime.timedelta(hours=23)


def test_day_window_utc_dst_fall_back_25h():
    """Fall-back 2026-11-01 (2am→1am): clocks repeat an hour → 25 h UTC window."""
    date = datetime.date(2026, 11, 1)
    t_start, t_end = _day_window_utc(date, "America/New_York")
    # Before DST ends: EDT = UTC-4, so local midnight → 04:00 UTC
    assert t_start == datetime.datetime(2026, 11, 1, 4, 0, 0, tzinfo=_UTC)
    # After DST ends: EST = UTC-5, so next local midnight → 05:00 UTC
    assert t_end == datetime.datetime(2026, 11, 2, 5, 0, 0, tzinfo=_UTC)
    assert (t_end - t_start) == datetime.timedelta(hours=25)


# ---------------------------------------------------------------------------
# FlightSource seam: convert/filter/upsample driven by a fake source
# (no feder, no skip — exercises the previously feder-gated logic everywhere)
# ---------------------------------------------------------------------------

_FT_TO_M = 0.3048


def _raw_point(t_s, lat, lon, alt_ft, alt_gnss_ft):
    base = datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=_UTC)
    return RawPoint(
        time=base + datetime.timedelta(seconds=t_s),
        lat=lat,
        lon=lon,
        alt_ft=alt_ft,
        alt_gnss_ft=alt_gnss_ft,
    )


def test_feder_source_construction_is_inert():
    """Constructing FederFlightSource must not import feder or touch the
    filesystem; it just records the data dir and satisfies the fetch port."""
    src = FederFlightSource("/nonexistent/data/dir")
    assert src._data_dir == "/nonexistent/data/dir"
    assert callable(src.fetch)


def test_load_flights_recorded_source_basic():
    """RecordedFlightSource drives convert/filter/upsample with no feder."""
    cfg = AdsbConfig()
    source = RecordedFlightSource.from_json(RAW_TRACE_FIXTURE)
    flights = load_flights(datetime.date(2026, 4, 8), cfg, source=source)

    # FAKE002 is entirely below threshold -> dropped; only FAKE001 survives.
    assert [f.callsign for f in flights] == ["FAKE001"]
    f = flights[0]
    assert f.transponder_id == "ABC123"
    assert f.aircraft_type == "A320"
    assert f.orig == "KBOS"
    assert f.dest == "KJFK"


def test_load_flights_recorded_altitude_filter_drops_low_point():
    """The 5000 ft point and the whole low trajectory are filtered out;
    every surviving ping is above the configured altitude threshold."""
    cfg = AdsbConfig()
    source = RecordedFlightSource.from_json(RAW_TRACE_FIXTURE)
    flights = load_flights(datetime.date(2026, 4, 8), cfg, source=source)
    for f in flights:
        for p in f.pings:
            assert p.alt_m >= cfg.min_altitude_m


def test_load_flights_recorded_radius_filter_drops_far_point():
    """The point ~127 km away is dropped; survivors are within max_radius_km."""
    cfg = AdsbConfig()
    source = RecordedFlightSource.from_json(RAW_TRACE_FIXTURE)
    flights = load_flights(datetime.date(2026, 4, 8), cfg, source=source)
    for f in flights:
        for p in f.pings:
            dist = _haversine_km(SITE_LAT, SITE_LON, p.lat, p.lon)
            assert dist <= cfg.max_radius_km


def test_load_flights_recorded_upsamples_to_1s():
    """Two surviving points 5 s apart upsample to 6 pings at 1 s spacing."""
    cfg = AdsbConfig()
    source = RecordedFlightSource.from_json(RAW_TRACE_FIXTURE)
    flights = load_flights(datetime.date(2026, 4, 8), cfg, source=source)
    pings = flights[0].pings
    # Original 2 surviving points 5 s apart -> 6 pings after upsampling.
    assert len(pings) == 6
    for a, b in zip(pings, pings[1:]):
        assert (b.time - a.time).total_seconds() == pytest.approx(1.0)


def test_load_flights_recorded_altitude_conversion():
    """Effective altitude reflects feet->metre conversion and policy choice.

    With altitude_source='auto' and GNSS (35100 ft) vs baro+geoid
    (35000 ft * 0.3048 - 28 m) agreeing within threshold, GNSS is used."""
    cfg = AdsbConfig()
    source = RecordedFlightSource.from_json(RAW_TRACE_FIXTURE)
    flights = load_flights(datetime.date(2026, 4, 8), cfg, source=source)
    first = flights[0].pings[0]
    assert first.alt_source == "gnss"
    assert first.alt_gnss_m == pytest.approx(35100.0 * _FT_TO_M)
    assert first.alt_baro_m == pytest.approx(35000.0 * _FT_TO_M + cfg.site_geoid_offset_m)
    assert first.alt_m == pytest.approx(first.alt_gnss_m)


def test_load_flights_recorded_from_in_memory_list():
    """RecordedFlightSource also accepts an in-memory RawTrajectory list."""
    cfg = AdsbConfig()
    traj = RawTrajectory(
        callsign="MEM001",
        transponder_id="XYZ",
        aircraft_type=None,
        orig=None,
        dest=None,
        points=[
            _raw_point(0, 42.40, -71.10, 35000.0, 35100.0),
            _raw_point(1, 42.41, -71.11, 35000.0, 35100.0),
        ],
    )
    flights = load_flights(
        datetime.date(2026, 4, 8), cfg, source=RecordedFlightSource([traj])
    )
    assert [f.callsign for f in flights] == ["MEM001"]
    assert flights[0].aircraft_type is None
    assert len(flights[0].pings) == 2


def test_load_flights_invalid_altitude_policy_raises_without_source():
    """The policy validation fires before any source is touched."""
    cfg = AdsbConfig(altitude_source="invalid")
    with pytest.raises(ValueError):
        load_flights(
            datetime.date(2026, 4, 8), cfg, source=RecordedFlightSource([])
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
