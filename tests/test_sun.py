"""Tests for concam.sun.sun_alt_deg."""

from __future__ import annotations

import datetime

import pytest

from concam.sun import sun_alt_deg

# MIT Green Building camera site
_LAT = 42.360444
_LON = -71.089238

# A known daytime moment: 2026-04-08 16:00 UTC ≈ noon in Boston
_NOON_UTC = datetime.datetime(2026, 4, 8, 16, 0, 0)

# A known night-time moment: 2026-04-08 06:00 UTC ≈ 2 AM EDT (well after sunset)
_NIGHT_UTC = datetime.datetime(2026, 4, 8, 6, 0, 0)


class TestSunAltDeg:
    def test_daytime_positive(self):
        """Solar altitude is positive around local noon."""
        alt = sun_alt_deg(_NOON_UTC, _LAT, _LON)
        assert alt > 40.0, f"Expected sun well above horizon at {_NOON_UTC} UTC, got {alt:.1f}°"

    def test_night_negative(self):
        """Solar altitude is negative well after sunset."""
        alt = sun_alt_deg(_NIGHT_UTC, _LAT, _LON)
        assert alt < 0.0, f"Expected sun below horizon at {_NIGHT_UTC} UTC, got {alt:.1f}°"

    def test_equinox_noon_at_equator(self):
        """At the March equinox, solar noon at the equator/prime-meridian gives ~90°."""
        # 2026-03-20 is close to the spring equinox.  At lon=0, UTC noon, lat=0:
        # the sun should be nearly overhead (altitude ≈ 90°, ≥80° is a safe check).
        dt = datetime.datetime(2026, 3, 20, 12, 0, 0)
        alt = sun_alt_deg(dt, lat_deg=0.0, lon_deg=0.0)
        assert alt > 80.0, f"Sun should be near zenith at equator on equinox noon, got {alt:.1f}°"

    def test_midnight_below_horizon(self):
        """UTC midnight at Boston in April is always night."""
        dt = datetime.datetime(2026, 4, 8, 0, 0, 0)
        alt = sun_alt_deg(dt, _LAT, _LON)
        assert alt < 0.0

    def test_timezone_aware_input(self):
        """Timezone-aware input is converted to UTC correctly."""
        # EDT is UTC-4.  16:00 UTC = 12:00 EDT.
        dt_aware = datetime.datetime(
            2026, 4, 8, 12, 0, 0,
            tzinfo=datetime.timezone(datetime.timedelta(hours=-4)),
        )
        alt_aware = sun_alt_deg(dt_aware, _LAT, _LON)
        alt_utc = sun_alt_deg(_NOON_UTC, _LAT, _LON)
        assert abs(alt_aware - alt_utc) < 0.01

    def test_return_type_is_float(self):
        dt = datetime.datetime(2026, 4, 8, 16, 0, 0)
        result = sun_alt_deg(dt, _LAT, _LON)
        assert isinstance(result, float)

    def test_altitude_range(self):
        """Altitude is always in [-90, 90]."""
        for hour in range(0, 24, 4):
            dt = datetime.datetime(2026, 4, 8, hour, 0, 0)
            alt = sun_alt_deg(dt, _LAT, _LON)
            assert -90.0 <= alt <= 90.0, f"Out-of-range altitude {alt:.1f}° at hour {hour}"

    def test_night_gate_logic(self):
        """Gate condition `sun_alt > 0` is True exactly during daylight."""
        # On April 8 near Boston, sunrise is ~10:00 UTC, sunset ~23:15 UTC.
        daytime_hours = [12, 14, 16, 18, 20]
        nighttime_hours = [0, 2, 4, 6, 8]
        for h in daytime_hours:
            dt = datetime.datetime(2026, 4, 8, h, 0, 0)
            assert sun_alt_deg(dt, _LAT, _LON) > 0, f"Expected daylight at UTC hour {h}"
        for h in nighttime_hours:
            dt = datetime.datetime(2026, 4, 8, h, 0, 0)
            assert sun_alt_deg(dt, _LAT, _LON) < 0, f"Expected night at UTC hour {h}"
