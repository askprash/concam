"""ADS-B flight data loader using the feder package."""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import zoneinfo
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

from concam.config import AdsbConfig

_log = logging.getLogger(__name__)

_FT_TO_M = 0.3048

_VALID_ALTITUDE_SOURCES = ("auto", "gnss", "barometric")


# Temporary workaround for a feder 1.0.0 bug: ``DB.__init__`` opens each per-day
# SQLite file with ``mode=ro`` only. SQLite still attempts journal cleanup on
# first access, which fails with "attempt to write a readonly database" for any
# reader who is not ``mcast`` (i.e. all of us) on the Hex data files. Adding
# ``immutable=1`` to the URI tells SQLite the file cannot change so it skips the
# write attempt. Reaches into ``feder.common.db`` (private), so the version
# assertion below makes us fail loudly the moment we drift off the version we
# tested against. Awaiting upstream fix from collaborator (a one-line
# `&immutable=1` addition in ``DB.__init__``).
#
# Removal: when feder 1.0.1+ ships with the fix, delete ``_patch_feder_readonly_open``
# and the call site in ``load_flights``, then bump the pin in pyproject.toml.
_FEDER_PATCHED_VERSION = "1.0.0"


def _patch_feder_readonly_open() -> None:
    import sqlite3 as _real

    import feder

    if feder.__version__ != _FEDER_PATCHED_VERSION:
        raise RuntimeError(
            f"feder {feder.__version__} differs from the version this monkey-patch "
            f"was tested against ({_FEDER_PATCHED_VERSION}). Confirm the bug is "
            f"fixed upstream and remove _patch_feder_readonly_open, or update "
            f"_FEDER_PATCHED_VERSION after re-verifying."
        )

    import feder.common.db as feder_db

    if getattr(feder_db, "_concam_immutable_patched", False):
        return

    class _SqliteCompat:
        def __getattr__(self, name: str):
            return getattr(_real, name)

        @staticmethod
        def connect(database, *args, **kwargs):
            if (
                isinstance(database, str)
                and database.startswith("file:")
                and "mode=ro" in database
                and "immutable=" not in database
            ):
                database = database + "&immutable=1"
            return _real.connect(database, *args, **kwargs)

    feder_db.sqlite3 = _SqliteCompat()
    feder_db._concam_immutable_patched = True


# ---------------------------------------------------------------------------
# FlightSource port + raw types
# ---------------------------------------------------------------------------
#
# The ADS-B loader is split into two halves across a port/adapter seam:
#
#   * A ``FlightSource`` (the port) yields raw, pre-conversion trajectories in
#     exactly the units feder provides (altitudes in *feet*). This is the only
#     surface that touches the feder package.
#   * ``load_flights`` (the consumer) owns the units conversion, altitude-policy
#     selection, radius/altitude filtering, and 1-second upsampling. It is
#     feder-agnostic and so can be exercised with any ``FlightSource``.
#
# Production uses ``FederFlightSource``; tests can use ``RecordedFlightSource``
# to drive the convert/filter/upsample path without the live feder data store.


@dataclass(slots=True)
class RawPoint:
    """One raw position fix straight off the source, in feder's native units.

    Altitudes are in *feet* (``alt_ft`` is barometric, ``alt_gnss_ft`` is GNSS),
    matching feder's ``Point.alt`` / ``Point.alt_gnss`` exactly. ``load_flights``
    is responsible for the feet→metre conversion and the geoid offset.
    """

    time: datetime.datetime  # UTC, timezone-aware
    lat: float
    lon: float
    alt_ft: float | None  # barometric altitude (feet); feder Point.alt
    alt_gnss_ft: float | None  # GNSS altitude (feet); feder Point.alt_gnss


@dataclass(slots=True)
class RawTrajectory:
    """One raw flight trajectory off the source, before conversion/filtering."""

    callsign: str
    transponder_id: str | None
    aircraft_type: str | None
    orig: str | None
    dest: str | None
    points: list[RawPoint]


class FlightSource(Protocol):
    """Port: yields raw trajectories for a time window and spatial pre-filter.

    The parameters mirror exactly what the feder query pre-filters on today: a
    half-open UTC time window, a lat/lon bounding box, and a minimum barometric
    altitude in feet. Implementations may apply the bbox / altitude pre-filter
    (as feder does, for efficiency) or yield everything and let ``load_flights``
    do the strict per-ping filtering — both are correct, because the consumer
    re-applies the exact radius/altitude thresholds after conversion.
    """

    def fetch(
        self,
        t_start: datetime.datetime,
        t_end: datetime.datetime,
        bbox: tuple[float, float, float, float],
        min_altitude_ft: float,
    ) -> Iterator[RawTrajectory]:
        """Yield ``RawTrajectory`` objects for the window/box. ``bbox`` is
        ``(min_lat, max_lat, min_lon, max_lon)``."""
        ...


class FederFlightSource:
    """Production adapter: queries the feder per-day SQLite store.

    Owns the feder coupling end-to-end: the read-only monkeypatch, the
    ``FEDER_DATA_DIR`` environment variable, the ``FlightQuery`` chain, and the
    mapping of feder ``Trajectory``/``Point`` objects onto ``RawTrajectory`` /
    ``RawPoint``. Nothing outside this class imports feder.
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir

    def fetch(
        self,
        t_start: datetime.datetime,
        t_end: datetime.datetime,
        bbox: tuple[float, float, float, float],
        min_altitude_ft: float,
    ) -> Iterator[RawTrajectory]:
        import feder

        _patch_feder_readonly_open()

        os.environ["FEDER_DATA_DIR"] = self._data_dir

        min_lat, max_lat, min_lon, max_lon = bbox
        trajectories = (
            feder.FlightQuery(t_start, t_end)
            .with_bounds(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                min_alt=min_altitude_ft,
            )
            .spatially_crosses()
            .filter_waypoints()
            .run()
        )

        for traj in trajectories:
            yield RawTrajectory(
                callsign=traj.callsign,
                transponder_id=traj.transponder_id,
                aircraft_type=traj.aircraft_type,
                orig=traj.orig,
                dest=traj.dest,
                points=[
                    RawPoint(
                        time=pt.time,
                        lat=pt.lat,
                        lon=pt.lon,
                        alt_ft=pt.alt,
                        alt_gnss_ft=pt.alt_gnss,
                    )
                    for pt in traj.points
                ],
            )


class RecordedFlightSource:
    """Fake adapter: replays an in-memory list of ``RawTrajectory``.

    The query window / bbox / altitude pre-filter are ignored — the recorded
    trace is yielded verbatim and ``load_flights`` applies its strict per-ping
    filters. Use ``from_json`` to load a committed raw trace for tests.
    """

    def __init__(self, trajectories: list[RawTrajectory]) -> None:
        self._trajectories = list(trajectories)

    @classmethod
    def from_json(cls, path) -> "RecordedFlightSource":
        """Build from a JSON file of raw trajectories.

        Schema: a list of objects with ``callsign``, ``transponder_id``,
        ``aircraft_type``, ``orig``, ``dest`` and a ``points`` list, each point
        having ``time`` (ISO 8601), ``lat``, ``lon``, ``alt_ft``,
        ``alt_gnss_ft`` (the latter two may be null).
        """
        with open(path) as f:
            data = json.load(f)
        trajectories = [
            RawTrajectory(
                callsign=entry["callsign"],
                transponder_id=entry.get("transponder_id"),
                aircraft_type=entry.get("aircraft_type"),
                orig=entry.get("orig"),
                dest=entry.get("dest"),
                points=[
                    RawPoint(
                        time=datetime.datetime.fromisoformat(p["time"]),
                        lat=p["lat"],
                        lon=p["lon"],
                        alt_ft=p.get("alt_ft"),
                        alt_gnss_ft=p.get("alt_gnss_ft"),
                    )
                    for p in entry["points"]
                ],
            )
            for entry in data
        ]
        return cls(trajectories)

    def fetch(
        self,
        t_start: datetime.datetime,
        t_end: datetime.datetime,
        bbox: tuple[float, float, float, float],
        min_altitude_ft: float,
    ) -> Iterator[RawTrajectory]:
        yield from self._trajectories


@dataclass
class Ping:
    """One position fix for a flight, after filtering and upsampling.

    ``alt_m`` is the effective altitude used by downstream projection. It is
    WGS-84 ellipsoidal height (HAE), selected per ``AdsbConfig.altitude_source``
    from the GNSS and barometric fields.
    """

    time: datetime.datetime  # UTC, timezone-aware
    lat: float
    lon: float
    alt_m: float  # effective altitude in meters (WGS-84 HAE)
    alt_gnss_m: float | None = None  # GNSS altitude, WGS-84 HAE
    alt_baro_m: float | None = None  # barometric (ISA) + geoid offset, in HAE frame
    alt_source: str = "gnss"  # "gnss" or "barometric"


@dataclass
class Flight:
    """A single flight's trajectory, filtered to the site window."""

    callsign: str
    transponder_id: str
    aircraft_type: str | None
    orig: str | None
    dest: str | None
    pings: list[Ping]  # 1-second resolution after upsampling


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _bbox_for_radius(
    site_lat: float, site_lon: float, radius_km: float
) -> tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) bounding box for a radius in km."""
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * math.cos(math.radians(site_lat)))
    return (
        site_lat - delta_lat,
        site_lat + delta_lat,
        site_lon - delta_lon,
        site_lon + delta_lon,
    )


def _choose_altitude(
    alt_baro_m: float | None,
    alt_gnss_m: float | None,
    config: AdsbConfig,
) -> tuple[float | None, str]:
    """Pick the effective altitude for projection per config.altitude_source.

    Returns (alt_m, source) where source is "gnss" or "barometric". Returns
    (None, "") if no usable altitude is available.
    """
    policy = config.altitude_source
    if policy not in _VALID_ALTITUDE_SOURCES:
        raise ValueError(
            f"altitude_source must be one of {_VALID_ALTITUDE_SOURCES}, got {policy!r}"
        )

    if policy == "gnss":
        if alt_gnss_m is not None:
            return alt_gnss_m, "gnss"
        return (alt_baro_m, "barometric") if alt_baro_m is not None else (None, "")

    if policy == "barometric":
        if alt_baro_m is not None:
            return alt_baro_m, "barometric"
        return (alt_gnss_m, "gnss") if alt_gnss_m is not None else (None, "")

    # "auto": prefer GNSS unless missing, or discrepant beyond threshold
    if alt_gnss_m is None and alt_baro_m is None:
        return None, ""
    if alt_gnss_m is None:
        return alt_baro_m, "barometric"
    if alt_baro_m is None:
        return alt_gnss_m, "gnss"
    if abs(alt_gnss_m - alt_baro_m) > config.altitude_discrepancy_threshold_m:
        # GNSS and baro disagree materially — trust baro per surveillance
        # literature (GNSS multipath / RAIM drop more common than static-system
        # leaks at cruise).
        return alt_baro_m, "barometric"
    return alt_gnss_m, "gnss"


def _upsample_pings(pings: list[Ping]) -> list[Ping]:
    """
    Linearly interpolate a list of Pings to 1-second resolution.

    Only interpolates gaps shorter than 5 minutes; larger gaps are left as-is.
    All altitude fields (alt_m, alt_gnss_m, alt_baro_m) are interpolated when
    present on both endpoints; alt_source is inherited from the earlier ping.
    """
    if len(pings) <= 1:
        return list(pings)

    # Deduplicate: feder sometimes returns >1 ping per second.
    # Keep last value for each truncated-to-second timestamp.
    seen: dict[datetime.datetime, Ping] = {}
    for p in pings:
        key = p.time.replace(microsecond=0)
        seen[key] = Ping(
            time=key,
            lat=p.lat,
            lon=p.lon,
            alt_m=p.alt_m,
            alt_gnss_m=p.alt_gnss_m,
            alt_baro_m=p.alt_baro_m,
            alt_source=p.alt_source,
        )
    pings = sorted(seen.values(), key=lambda x: x.time)

    if len(pings) <= 1:
        return list(pings)

    def _interp_opt(a: float | None, b: float | None, frac: float) -> float | None:
        if a is None or b is None:
            return None
        return a + frac * (b - a)

    result: list[Ping] = []
    for i, p in enumerate(pings):
        result.append(p)
        if i >= len(pings) - 1:
            break
        q = pings[i + 1]
        gap_s = (q.time - p.time).total_seconds()
        if gap_s <= 1 or gap_s >= 300:
            continue
        n = int(gap_s)
        fracs = np.arange(1, n) / n
        lats = p.lat + fracs * (q.lat - p.lat)
        lons = p.lon + fracs * (q.lon - p.lon)
        alts = p.alt_m + fracs * (q.alt_m - p.alt_m)
        for j, frac in enumerate(fracs):
            t = p.time + datetime.timedelta(seconds=round(frac * gap_s))
            result.append(
                Ping(
                    time=t,
                    lat=float(lats[j]),
                    lon=float(lons[j]),
                    alt_m=float(alts[j]),
                    alt_gnss_m=_interp_opt(p.alt_gnss_m, q.alt_gnss_m, float(frac)),
                    alt_baro_m=_interp_opt(p.alt_baro_m, q.alt_baro_m, float(frac)),
                    alt_source=p.alt_source,
                )
            )

    result.sort(key=lambda x: x.time)
    return result


def _day_window_utc(
    date: datetime.date,
    timezone: str | None,
) -> tuple[datetime.datetime, datetime.datetime]:
    """Return (t_start, t_end) as timezone-aware UTC datetimes spanning *date*.

    When *timezone* is None the window is exactly UTC 00:00:00 to the next
    day's UTC 00:00:00 (always 24 h).  When a tz string is given the window
    runs from local midnight to local midnight, converted to UTC — so on DST
    spring-forward days the window is 23 h and on fall-back days it is 25 h.
    """
    utc = datetime.timezone.utc
    if timezone is not None:
        tz = zoneinfo.ZoneInfo(timezone)
        t_start = datetime.datetime.combine(
            date, datetime.time.min, tzinfo=tz
        ).astimezone(utc)
        t_end = datetime.datetime.combine(
            date + datetime.timedelta(days=1), datetime.time.min, tzinfo=tz
        ).astimezone(utc)
    else:
        t_start = datetime.datetime(date.year, date.month, date.day, 0, 0, 0,
                                    tzinfo=utc)
        t_end = t_start + datetime.timedelta(days=1)
    return t_start, t_end


def load_flights(
    date: datetime.date,
    config: AdsbConfig,
    timezone: str | None = None,
    source: FlightSource | None = None,
) -> list[Flight]:
    """
    Load and filter ADS-B flights for a calendar day.

    The day is interpreted in ``timezone`` (e.g. ``"America/New_York"``) when
    given, otherwise UTC. Local-time framing matters because the daily timelapse
    video covers a local-time day, so a UTC-day query truncates the last
    ``UTC-offset`` hours of overlay coverage. DST transitions are handled
    correctly via ``zoneinfo`` — the resulting UTC window is 23 h or 25 h on
    transition days.

    Pipeline:
      1. Converts feder Points to Pings, storing both GNSS (HAE) and barometric
         (ISA-MSL → HAE via site geoid offset) altitudes.
      2. Selects the effective altitude per ``config.altitude_source``.
      3. Discards pings below ``config.min_altitude_m`` (applied to the effective
         altitude) or beyond ``config.max_radius_km``.
      4. Discards trajectories with no pings surviving the filters.
      5. Upsamples each surviving trajectory to 1-second resolution.

    Emits a single summary log line with the count of pings where barometric
    and GNSS altitudes disagreed by more than the configured threshold.

    ``source`` is the :class:`FlightSource` to pull raw trajectories from. When
    ``None`` (the default) a :class:`FederFlightSource` is constructed for
    ``config.data_dir`` — i.e. the production path is unchanged.
    """
    if config.altitude_source not in _VALID_ALTITUDE_SOURCES:
        raise ValueError(
            f"altitude_source must be one of {_VALID_ALTITUDE_SOURCES}, "
            f"got {config.altitude_source!r}"
        )

    if source is None:
        source = FederFlightSource(config.data_dir)

    t_start, t_end = _day_window_utc(date, timezone)

    bbox = _bbox_for_radius(
        config.site_lat, config.site_lon, config.max_radius_km
    )
    # feder's altitude pre-filter operates on barometric feet; use the
    # barometric-equivalent threshold so we don't drop legitimate pings whose
    # GNSS altitude is above threshold but whose baro is slightly below.
    # Pre-filter generously; the per-ping filter applies the strict threshold.
    min_alt_ft_prefilter = max(0.0, (config.min_altitude_m - 500.0) / _FT_TO_M)

    trajectories = source.fetch(t_start, t_end, bbox, min_alt_ft_prefilter)

    flights: list[Flight] = []
    discrepant = 0
    total_considered = 0
    for traj in trajectories:
        pings: list[Ping] = []
        for pt in traj.points:
            alt_gnss_m = (
                pt.alt_gnss_ft * _FT_TO_M if pt.alt_gnss_ft is not None else None
            )
            alt_baro_hae_m = (
                pt.alt_ft * _FT_TO_M + config.site_geoid_offset_m
                if pt.alt_ft is not None
                else None
            )

            if alt_gnss_m is not None and alt_baro_hae_m is not None:
                total_considered += 1
                if abs(alt_gnss_m - alt_baro_hae_m) > config.altitude_discrepancy_threshold_m:
                    discrepant += 1

            alt_m, alt_source = _choose_altitude(alt_baro_hae_m, alt_gnss_m, config)
            if alt_m is None:
                continue
            if alt_m < config.min_altitude_m:
                continue
            if config.max_altitude_m is not None and alt_m > config.max_altitude_m:
                continue
            dist_km = _haversine_km(config.site_lat, config.site_lon, pt.lat, pt.lon)
            if dist_km > config.max_radius_km:
                continue
            pings.append(
                Ping(
                    time=pt.time,
                    lat=pt.lat,
                    lon=pt.lon,
                    alt_m=alt_m,
                    alt_gnss_m=alt_gnss_m,
                    alt_baro_m=alt_baro_hae_m,
                    alt_source=alt_source,
                )
            )

        if not pings:
            continue

        pings.sort(key=lambda x: x.time)
        upsampled = _upsample_pings(pings)

        flights.append(
            Flight(
                callsign=traj.callsign,
                transponder_id=traj.transponder_id or "",
                aircraft_type=traj.aircraft_type,
                orig=traj.orig,
                dest=traj.dest,
                pings=upsampled,
            )
        )

    if total_considered:
        frac = discrepant / total_considered
        _log.info(
            "ADS-B altitude: %d/%d pings (%.1f%%) had |GNSS-baro| > %.0f m "
            "(policy=%s, geoid_offset=%.1f m)",
            discrepant,
            total_considered,
            100.0 * frac,
            config.altitude_discrepancy_threshold_m,
            config.altitude_source,
            config.site_geoid_offset_m,
        )

    return flights
