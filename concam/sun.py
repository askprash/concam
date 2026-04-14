"""Solar position helper for the concam pipeline.

Single public function: ``sun_alt_deg(dt, lat_deg, lon_deg) -> float``

Algorithm: NOAA simplified solar position equations.
Accuracy: ±1° for most dates and locations — sufficient for a day/night gate
(sun_alt > 0 = daylight) and for NRBR-aware detection gating.

Reference: NOAA Solar Calculator equations
https://www.esrl.noaa.gov/gmd/grad/solcalc/solareqns.PDF
"""

from __future__ import annotations

import datetime
import math


def sun_alt_deg(
    dt: datetime.datetime,
    lat_deg: float,
    lon_deg: float,
) -> float:
    """Return the solar elevation angle in degrees for a given UTC datetime.

    Args:
        dt: Observation datetime.  If timezone-aware, it is converted to UTC
            first.  If naive, it is assumed to already be UTC.
        lat_deg: Observer latitude in degrees (positive = north).
        lon_deg: Observer longitude in degrees (positive = east).

    Returns:
        Solar elevation angle in degrees.  Positive = sun above the horizon;
        negative = sun below the horizon (night / civil twilight).
    """
    # Normalise to a naive UTC datetime.
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    # Day of year (1 = Jan 1).
    day_of_year = dt.timetuple().tm_yday

    # Fractional UTC hour.
    hour_utc = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    # NOAA fractional year in radians.
    gamma = (2.0 * math.pi / 365.0) * (day_of_year - 1.0 + (hour_utc - 12.0) / 24.0)

    # Equation of time in minutes (accounts for Earth's orbital eccentricity
    # and axial tilt in the ecliptic plane).
    eqtime = (
        229.18
        * (
            0.000075
            + 0.001868 * math.cos(gamma)
            - 0.032077 * math.sin(gamma)
            - 0.014615 * math.cos(2.0 * gamma)
            - 0.040890 * math.sin(2.0 * gamma)
        )
    )

    # Solar declination in radians.
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.001480 * math.sin(3.0 * gamma)
    )

    # True solar time in minutes from midnight UTC, corrected for the observer's
    # longitude.  (lon_deg * 4 converts degrees to minutes at 4 min/degree.)
    tst = hour_utc * 60.0 + eqtime + 4.0 * lon_deg

    # Solar hour angle in radians.  0 at solar noon; negative in the morning.
    ha = math.radians(tst / 4.0 - 180.0)

    lat_rad = math.radians(lat_deg)

    # Cosine of solar zenith angle.
    cos_zenith = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    )
    # Clamp to [-1, 1] against floating-point rounding.
    cos_zenith = max(-1.0, min(1.0, cos_zenith))

    zenith_deg = math.degrees(math.acos(cos_zenith))
    return 90.0 - zenith_deg
