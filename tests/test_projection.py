"""Tests for the GPS-to-pixel projection module.

Uses known POI landmarks from the calibration data and real ADS-B flight
positions to verify that the projection pipeline produces sensible pixel
coordinates.
"""

from __future__ import annotations

import datetime
import math
import os

import numpy as np
import pytest

from concam.adsb import Ping
from concam.config import CalibrationConfig, DetectionConfig
from concam.projection import (
    Calibration,
    PixelPoint,
    Rect,
    flight_path_vector,
    load_calibration,
    oriented_roi,
    project_ping,
    project_pixel_to_meters,
    project_pings,
)

NPZ_PATH = "/home/prash/contrails/LAE_skycam/calibration/pointpicker_calibration_estimate.npz"


def _have_calibration() -> bool:
    return os.path.isfile(NPZ_PATH)


skip_no_calib = pytest.mark.skipif(
    not _have_calibration(), reason="Calibration npz not found"
)


def _make_ping(lat: float, lon: float, alt_m: float) -> Ping:
    return Ping(
        time=datetime.datetime(2026, 4, 9, 12, 0, 0, tzinfo=datetime.timezone.utc),
        lat=lat,
        lon=lon,
        alt_m=alt_m,
    )


@pytest.fixture
def calib():
    config = CalibrationConfig(npz_path=NPZ_PATH, calibration_resolution=(3840, 2160))
    return load_calibration(config)


# --- POI-based projection tests ---


@skip_no_calib
class TestPOIProjection:
    """Verify projection against known POI landmarks from MITSC_POIs.csv.

    These landmarks were manually picked in the camera image, so their
    pixel coordinates are known ground truth.
    """

    # (name, expected_x, expected_y, lat, lon, alt)
    POIS = [
        ("ProtoAptLeftCorner", 3561, 1614, 42.3630891, -71.0879636, 75.487),
        ("KochNearLeftCorner", 2833, 2026, 42.3628113, -71.0891338, 42.669),
        ("StataTopLeftCorner", 340, 2025, 42.3614880, -71.0913638, 44.629),
        ("BroadLeftCorner", 3013, 1693, 42.3634039, -71.0888331, 73.464),
    ]

    def test_pois_land_near_expected_pixel(self, calib: Calibration):
        """Each POI should project within ~80px of its picked position.

        The calibration is not perfect (RMS reprojection error exists),
        so we allow generous tolerance, but it should be in the right ballpark.
        """
        for name, exp_x, exp_y, lat, lon, alt in self.POIS:
            ping = _make_ping(lat, lon, alt)
            pt = project_ping(ping, calib)
            assert pt is not None, f"{name} projected to None (behind camera?)"
            dist = math.hypot(pt.x - exp_x, pt.y - exp_y)
            assert dist < 80, (
                f"{name}: projected to ({pt.x:.0f}, {pt.y:.0f}), "
                f"expected ~({exp_x}, {exp_y}), error={dist:.0f}px"
            )

    def test_poi_in_correct_quadrant(self, calib: Calibration):
        """POIs on the left of the image should have x < 1920 and vice versa."""
        # StataTopLeftCorner is at x=340 (left side)
        ping = _make_ping(42.3614880, -71.0913638, 44.629)
        pt = project_ping(ping, calib)
        assert pt is not None
        assert pt.x < 1920, f"Stata should be in left half, got x={pt.x:.0f}"

        # ProtoAptLeftCorner is at x=3561 (right side)
        ping = _make_ping(42.3630891, -71.0879636, 75.487)
        pt = project_ping(ping, calib)
        assert pt is not None
        assert pt.x > 1920, f"Proto should be in right half, got x={pt.x:.0f}"


# --- Behind-camera and out-of-bounds tests ---


@skip_no_calib
class TestEdgeCases:
    def test_point_behind_camera_returns_none(self, calib: Calibration):
        """A point far south of the camera (behind it) should return None."""
        ping = _make_ping(42.0, -71.0, 100.0)
        assert project_ping(ping, calib) is None

    def test_high_altitude_in_fov_in_bounds(self, calib: Calibration):
        """A flight at cruising altitude north of the camera (in its FOV) should be in-bounds."""
        # Camera faces roughly north; a point ~15km north at 10km alt is in frame
        ping = _make_ping(42.50, -71.09, 10000.0)
        pt = project_ping(ping, calib)
        assert pt is not None
        assert 0 <= pt.x < 3840 and 0 <= pt.y < 2160

    def test_empty_pings_returns_empty(self, calib: Calibration):
        assert project_pings([], calib) == []


# --- Batch projection test ---


@skip_no_calib
class TestBatchProjection:
    def test_batch_matches_individual(self, calib: Calibration):
        """project_pings should give same results as individual project_ping calls."""
        pings = [
            _make_ping(42.3630891, -71.0879636, 75.487),
            _make_ping(42.3628113, -71.0891338, 42.669),
            _make_ping(42.0, -71.0, 100.0),  # behind camera
        ]
        batch = project_pings(pings, calib)
        for i, ping in enumerate(pings):
            single = project_ping(ping, calib)
            if single is None:
                assert batch[i] is None
            else:
                assert batch[i] is not None
                assert abs(batch[i].x - single.x) < 0.01
                assert abs(batch[i].y - single.y) < 0.01


# --- Flight path vector tests ---


@skip_no_calib
class TestFlightPathVector:
    def test_vector_is_unit_length(self, calib: Calibration):
        a = _make_ping(42.50, -71.10, 10000.0)
        b = _make_ping(42.55, -71.08, 10000.0)
        vx, vy = flight_path_vector(a, b, calib)
        mag = math.hypot(vx, vy)
        assert abs(mag - 1.0) < 1e-6

    def test_raises_when_ping_behind_camera(self, calib: Calibration):
        a = _make_ping(42.0, -71.0, 100.0)  # behind camera
        b = _make_ping(42.36, -71.09, 10000.0)
        with pytest.raises(ValueError, match="valid pixel"):
            flight_path_vector(a, b, calib)


# --- Oriented ROI tests ---


class TestOrientedROI:
    def test_horizontal_path_gives_wide_rect(self):
        center = PixelPoint(x=1000.0, y=1000.0)
        config = DetectionConfig(roi_padding=20)
        roi = oriented_roi(center, (1.0, 0.0), config)
        # Along-track = 3*20=60 each side = 120 wide, cross-track = 20 each side = 40 tall
        assert roi.w > roi.h

    def test_vertical_path_gives_tall_rect(self):
        center = PixelPoint(x=1000.0, y=1000.0)
        config = DetectionConfig(roi_padding=20)
        roi = oriented_roi(center, (0.0, 1.0), config)
        assert roi.h > roi.w

    def test_roi_clipped_to_image(self):
        center = PixelPoint(x=5.0, y=5.0)
        config = DetectionConfig(roi_padding=20)
        roi = oriented_roi(center, (1.0, 0.0), config)
        assert roi.x >= 0
        assert roi.y >= 0

    def test_roi_center_inside(self):
        center = PixelPoint(x=500.0, y=500.0)
        config = DetectionConfig(roi_padding=30)
        roi = oriented_roi(center, (0.7071, 0.7071), config)
        assert roi.x <= 500 <= roi.x + roi.w
        assert roi.y <= 500 <= roi.y + roi.h


# --- project_pixel_to_meters tests ---


@skip_no_calib
class TestProjectPixelToMeters:
    """Pinhole-ray length conversion sanity checks."""

    def test_overhead_aircraft_at_10km_200px_approx_2km(self, calib: Calibration):
        """A 200-px contrail directly overhead at 10 km should be ~2 km.

        The focal length of the MIT Green Building camera is ~1700-2000 px
        (depending on exact calibration).  At 10 km slant range:
          length_m ≈ slant_range × length_px / focal_px
                   ≈ 10 000 × 200 / 1800 ≈ 1 110 m

        We allow a wide range (500 m – 4 000 m) to be robust to calibration
        uncertainty; the key assertion is order-of-magnitude correctness.
        """
        # Aircraft directly above the camera position at 10 km altitude.
        # Camera is at approx 42.360 N, 71.089 W, 84 m.
        ping = _make_ping(42.360444, -71.089238, 10_000.0 + 84.23)
        length_m = project_pixel_to_meters(200.0, ping, calib)
        assert 500 < length_m < 4_000, (
            f"Expected 500–4000 m for 200 px at 10 km, got {length_m:.0f} m"
        )

    def test_zero_pixel_length_returns_zero(self, calib: Calibration):
        ping = _make_ping(42.360444, -71.089238, 10_000.0)
        assert project_pixel_to_meters(0.0, ping, calib) == 0.0

    def test_longer_pixel_length_gives_longer_metres(self, calib: Calibration):
        ping = _make_ping(42.360444, -71.089238, 10_000.0 + 84.23)
        m100 = project_pixel_to_meters(100.0, ping, calib)
        m200 = project_pixel_to_meters(200.0, ping, calib)
        assert m200 > m100, "Twice the pixel length must give more metres"

    def test_higher_altitude_gives_longer_metres(self, calib: Calibration):
        """Higher aircraft → longer slant range → more metres per pixel."""
        ping_low = _make_ping(42.360444, -71.089238, 5_000.0)
        ping_high = _make_ping(42.360444, -71.089238, 12_000.0)
        m_low = project_pixel_to_meters(100.0, ping_low, calib)
        m_high = project_pixel_to_meters(100.0, ping_high, calib)
        assert m_high > m_low, "Higher altitude must produce more metres per pixel"
