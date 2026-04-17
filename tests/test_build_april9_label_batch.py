"""Tests for scripts/build_april9_label_batch.py (PRD item 28)."""

from __future__ import annotations

import datetime
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

_spec = importlib.util.spec_from_file_location(
    "build_april9_label_batch",
    SCRIPTS_DIR / "build_april9_label_batch.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["build_april9_label_batch"] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)

FlightPeak = _module.FlightPeak
stratify = _module.stratify
time_bucketed_select = _module.time_bucketed_select
haversine_km = _module.haversine_km
SITE_LAT = _module.SITE_LAT
SITE_LON = _module.SITE_LON


def _make_peak(
    *,
    tid: str,
    callsign: str = "TEST",
    t: datetime.datetime,
    alt_baro_m: float,
    ground_distance_km: float,
    sun_alt: float,
    pixel_x: float = 1920.0,
    pixel_y: float = 1080.0,
) -> FlightPeak:
    proj = {
        "wall_time_utc": t.isoformat(),
        "callsign": callsign,
        "transponder_id": tid,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "path_dx": 1.0,
        "path_dy": 0.0,
        "roi": {"x": 0, "y": 0, "w": 100, "h": 100},
        "lat": 42.36,
        "lon": -71.08,
        "alt_m": alt_baro_m,
        "alt_baro_m": alt_baro_m,
    }
    return FlightPeak(
        transponder_id=tid,
        callsign=callsign,
        proj=proj,
        time=t,
        alt_baro_m=alt_baro_m,
        ground_distance_km=ground_distance_km,
        sun_alt=sun_alt,
    )


# ---------- haversine ----------


def test_haversine_zero_distance() -> None:
    assert haversine_km(SITE_LAT, SITE_LON, SITE_LAT, SITE_LON) == pytest.approx(0.0, abs=1e-6)


def test_haversine_reference_value() -> None:
    # Roughly 111 km per degree of latitude at 42°N.
    d = haversine_km(SITE_LAT, SITE_LON, SITE_LAT + 1.0, SITE_LON)
    assert 110 < d < 112


# ---------- time_bucketed_select ----------


def _peaks_across_day(
    n: int, *, alt: float, dist: float, sun: float, spacing_minutes: int = 45
) -> list[FlightPeak]:
    """Space flights at ``spacing_minutes`` apart so each gets its own bucket."""
    start = datetime.datetime(2026, 4, 9, 4, 0, tzinfo=datetime.timezone.utc)
    return [
        _make_peak(
            tid=f"TID{i:03d}",
            callsign=f"CS{i}",
            t=start + datetime.timedelta(minutes=spacing_minutes * i),
            alt_baro_m=alt,
            ground_distance_km=dist,
            sun_alt=sun,
        )
        for i in range(n)
    ]


def test_time_bucketed_select_returns_all_when_pool_small() -> None:
    peaks = _peaks_across_day(3, alt=11_000, dist=20, sun=40)
    out = time_bucketed_select(peaks, 10)
    assert len(out) == 3


def test_time_bucketed_select_downsamples_to_quota() -> None:
    peaks = _peaks_across_day(40, alt=11_000, dist=20, sun=40)
    out = time_bucketed_select(peaks, 10)
    assert len(out) == 10
    # Picks should be unique.
    assert len({p.transponder_id for p in out}) == 10


def test_time_bucketed_select_zero_quota() -> None:
    peaks = _peaks_across_day(5, alt=11_000, dist=20, sun=40)
    assert time_bucketed_select(peaks, 0) == []


# ---------- stratify ----------


def _mixed_pool() -> list[FlightPeak]:
    t0 = datetime.datetime(2026, 4, 9, 12, 0, tzinfo=datetime.timezone.utc)
    peaks: list[FlightPeak] = []
    # 20 high-cirrus
    for i in range(20):
        peaks.append(
            _make_peak(
                tid=f"CIR{i}",
                t=t0 + datetime.timedelta(minutes=45 * i),
                alt_baro_m=11_000,
                ground_distance_km=20,
                sun_alt=40,
            )
        )
    # 20 mid-cruise
    for i in range(20):
        peaks.append(
            _make_peak(
                tid=f"MID{i}",
                t=t0 + datetime.timedelta(minutes=45 * i),
                alt_baro_m=9_000,
                ground_distance_km=20,
                sun_alt=40,
            )
        )
    # 15 wide-radius (mid-altitude, far distance)
    for i in range(15):
        peaks.append(
            _make_peak(
                tid=f"WID{i}",
                t=t0 + datetime.timedelta(minutes=45 * i),
                alt_baro_m=9_500,
                ground_distance_km=90,
                sun_alt=40,
            )
        )
    # 10 marginal (low sun)
    for i in range(10):
        peaks.append(
            _make_peak(
                tid=f"MAR{i}",
                t=t0 + datetime.timedelta(minutes=45 * i),
                alt_baro_m=9_500,
                ground_distance_km=20,
                sun_alt=5,
            )
        )
    return peaks


def test_stratify_fills_all_quotas_disjointly() -> None:
    peaks = _mixed_pool()
    result = stratify(peaks, (10, 10, 10, 5))

    assert len(result["high_cirrus"]) == 10
    assert len(result["mid_cruise"]) == 10
    assert len(result["wide_radius"]) == 10
    assert len(result["marginal"]) == 5

    # Every selected TID is unique across all four strata.
    all_tids = [
        p.transponder_id
        for stratum in result.values()
        for p in stratum
    ]
    assert len(all_tids) == len(set(all_tids)) == 35


def test_stratify_honours_filter_conditions() -> None:
    peaks = _mixed_pool()
    result = stratify(peaks, (10, 10, 10, 5))

    for p in result["high_cirrus"]:
        assert 10_400 <= p.alt_baro_m <= 12_200 and p.sun_alt > 15
    for p in result["mid_cruise"]:
        assert 8_500 <= p.alt_baro_m < 10_400 and p.sun_alt > 15
    for p in result["wide_radius"]:
        assert 60 <= p.ground_distance_km <= 150 and p.sun_alt > 15
    for p in result["marginal"]:
        assert p.sun_alt < 15


def test_stratify_warns_and_returns_short_when_pool_underfilled(caplog) -> None:
    # Only 3 marginal peaks available but we ask for 5.
    pool = _mixed_pool()
    trimmed = [p for p in pool if not p.transponder_id.startswith("MAR")]
    t0 = datetime.datetime(2026, 4, 9, 2, 0, tzinfo=datetime.timezone.utc)
    for i in range(3):
        trimmed.append(
            _make_peak(
                tid=f"MAR_SMALL{i}",
                t=t0 + datetime.timedelta(hours=i),
                alt_baro_m=9_500,
                ground_distance_km=20,
                sun_alt=5,
            )
        )
    with caplog.at_level("WARNING"):
        result = stratify(trimmed, (10, 10, 10, 5))
    assert len(result["marginal"]) == 3
    assert any("marginal" in rec.message for rec in caplog.records)


def test_stratify_prefers_earlier_strata_for_overlapping_tids() -> None:
    # One flight could fit BOTH mid_cruise and wide_radius; high_cirrus and
    # marginal exclude it.  Because mid_cruise is applied before wide_radius,
    # mid_cruise should take it and wide_radius should see an empty pool.
    t0 = datetime.datetime(2026, 4, 9, 12, 0, tzinfo=datetime.timezone.utc)
    p_dual = _make_peak(
        tid="DUAL",
        t=t0,
        alt_baro_m=9_500,          # mid-cruise band
        ground_distance_km=90,     # wide-radius band
        sun_alt=40,
    )
    result = stratify([p_dual], (10, 10, 10, 5))
    assert [p.transponder_id for p in result["mid_cruise"]] == ["DUAL"]
    assert result["wide_radius"] == []
