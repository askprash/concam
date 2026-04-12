"""ADS-B flight data loader using the feder package."""

from __future__ import annotations

import datetime
import math
import os
from dataclasses import dataclass

import numpy as np

from concam.config import AdsbConfig

# feet → meters conversion constant
_FT_TO_M = 0.3048


@dataclass
class Ping:
    """One position fix for a flight, after filtering and upsampling."""

    time: datetime.datetime  # UTC, timezone-aware
    lat: float
    lon: float
    alt_m: float  # GNSS altitude in meters (WGS-84)


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


def _upsample_pings(pings: list[Ping]) -> list[Ping]:
    """
    Linearly interpolate a list of Pings to 1-second resolution.

    Only interpolates gaps shorter than 5 minutes; larger gaps are left as-is.
    """
    if len(pings) <= 1:
        return list(pings)

    # Deduplicate: feder sometimes returns >1 ping per second.
    # Keep last value for each truncated-to-second timestamp.
    seen: dict[datetime.datetime, Ping] = {}
    for p in pings:
        key = p.time.replace(microsecond=0)
        seen[key] = Ping(time=key, lat=p.lat, lon=p.lon, alt_m=p.alt_m)
    pings = sorted(seen.values(), key=lambda x: x.time)

    if len(pings) <= 1:
        return list(pings)

    result: list[Ping] = []
    for i, p in enumerate(pings):
        result.append(p)
        if i >= len(pings) - 1:
            break
        q = pings[i + 1]
        gap_s = (q.time - p.time).total_seconds()
        if gap_s <= 1 or gap_s >= 300:
            # Nothing to interpolate (already 1s, or too large a gap)
            continue
        n = int(gap_s)
        # Fractional positions for the n-1 interior points (t=1..n-1 out of n)
        fracs = np.arange(1, n) / n
        lats = p.lat + fracs * (q.lat - p.lat)
        lons = p.lon + fracs * (q.lon - p.lon)
        alts = p.alt_m + fracs * (q.alt_m - p.alt_m)
        for j, frac in enumerate(fracs):
            t = p.time + datetime.timedelta(seconds=round(frac * gap_s))
            result.append(Ping(time=t, lat=float(lats[j]), lon=float(lons[j]), alt_m=float(alts[j])))

    # Sort by time (upsampled points may be out of order if inserted mid-loop)
    result.sort(key=lambda x: x.time)
    return result


def load_flights(date: datetime.date, config: AdsbConfig) -> list[Flight]:
    """
    Load and filter ADS-B flights for a UTC calendar day.

    Queries feder for all flights with any pings during the UTC day, then:
      1. Converts feder Points to Pings (GNSS alt in meters).
      2. Discards pings below config.min_altitude_m or beyond config.max_radius_km.
      3. Discards trajectories with no pings surviving the filters.
      4. Upsamples each surviving trajectory to 1-second resolution.

    Args:
        date: UTC calendar date to load.
        config: AdsbConfig carrying data_dir and filter parameters.

    Returns:
        List of Flight objects, each with at least one Ping.
    """
    import feder

    os.environ["FEDER_DATA_DIR"] = config.data_dir

    # Full-day UTC window
    t_start = datetime.datetime(date.year, date.month, date.day, 0, 0, 0,
                                tzinfo=datetime.timezone.utc)
    t_end = t_start + datetime.timedelta(days=1)

    # Approximate bounding box for radius pre-filter (feder bbox is in degrees)
    min_lat, max_lat, min_lon, max_lon = _bbox_for_radius(
        config.site_lat, config.site_lon, config.max_radius_km
    )
    # Altitude pre-filter: feder uses feet; convert our metres threshold
    min_alt_ft = config.min_altitude_m / _FT_TO_M

    trajectories = (
        feder.FlightQuery(t_start, t_end)
        .with_bounds(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            min_alt=min_alt_ft,
        )
        .spatially_crosses()
        .filter_waypoints()
        .run()
    )

    flights: list[Flight] = []
    for traj in trajectories:
        pings: list[Ping] = []
        for pt in traj.points:
            # Use GNSS altitude; fall back to pressure alt if missing
            alt_gnss_ft = pt.alt_gnss if pt.alt_gnss is not None else pt.alt
            if alt_gnss_ft is None:
                continue
            alt_m = alt_gnss_ft * _FT_TO_M
            if alt_m < config.min_altitude_m:
                continue
            dist_km = _haversine_km(config.site_lat, config.site_lon, pt.lat, pt.lon)
            if dist_km > config.max_radius_km:
                continue
            pings.append(Ping(time=pt.time, lat=pt.lat, lon=pt.lon, alt_m=alt_m))

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

    return flights
