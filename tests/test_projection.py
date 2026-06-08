"""Tests for the GPS-to-pixel projection module.

Uses known POI landmarks from the calibration data and real ADS-B flight
positions to verify that the projection pipeline produces sensible pixel
coordinates.

Synthetic-calibration tests (TestGPStoENU, TestProjectPingsSynthetic,
TestProjectPixelToMetersSynthetic, TestFlightPathVectorSynthetic,
TestBatchProjectionSynthetic) use ``synthetic_calibration()`` and run on
every machine — no .npz file required.  The real-calibration tests
(TestPOIProjection, TestEdgeCases, TestBatchProjection,
TestFlightPathVector, TestProjectPixelToMeters) are still gated by
``skip_no_calib`` because they assert geometry specific to the MIT Green
Building camera.
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
    _gps_to_enu,
    flight_path_vector,
    load_calibration,
    oriented_roi,
    project_ping,
    project_pixel_to_meters,
    project_pings,
    synthetic_calibration,
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
        # roi_along_px=180, roi_cross_px=40 (production defaults) → wide rect
        config = DetectionConfig(roi_padding=20)
        roi = oriented_roi(center, (1.0, 0.0), config)
        # roi_along_px (180) > roi_cross_px (40) → width > height for horizontal path
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


# ---------------------------------------------------------------------------
# Synthetic-calibration tests — run on every machine, no .npz required.
#
# ``synthetic_calibration()`` places a pinhole camera at the MIT Green
# Building GPS with identity rotation, so the camera frame = ENU frame:
#   +X → East,  +Y → North,  +Z → Up (the camera looks straight up).
#
# This makes expected values derivable from first-principles ENU geometry and
# the pinhole projection formula   pixel = K @ [enu_x/enu_z, enu_y/enu_z, 1].
# ---------------------------------------------------------------------------


@pytest.fixture
def syn_calib() -> Calibration:
    """Synthetic calibration used across all synthetic tests."""
    return synthetic_calibration()


def _syn_ping(lat: float, lon: float, alt_m: float) -> Ping:
    return Ping(
        time=datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
        lat=lat,
        lon=lon,
        alt_m=alt_m,
    )


class TestGPStoENU:
    """Unit tests for the _gps_to_enu coordinate transform.

    All expectations are derived from the ENU definition and pyproj's
    ECEF→ENU rotation; they do not depend on opaque magic numbers.
    """

    def test_self_ping_is_origin(self, syn_calib: Calibration):
        """A ping at the camera's own GPS must yield ENU ≈ (0, 0, 0).

        The cam_ecef inside the Calibration is derived from camera_gps via the
        same transformer, so the ECEF subtraction cancels exactly.
        """
        lat, lon, alt = syn_calib.camera_gps
        enu = _gps_to_enu(
            np.array([lat]), np.array([lon]), np.array([alt]), syn_calib
        )
        assert enu.shape == (1, 3)
        np.testing.assert_allclose(enu[0], [0.0, 0.0, 0.0], atol=1e-2)

    def test_north_offset_has_positive_y(self, syn_calib: Calibration):
        """A ping ~1 km due north should have ENU_y ≈ +1000 m, |ENU_x| ≈ 0.

        We derive the latitude step from the rule 1° lat ≈ 111 000 m,
        which gives the expected great-circle distance to within ~0.1%.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        north_m = 1_000.0
        # 1 degree of latitude ≈ 111 000 m; invert to get the angular step.
        delta_lat = north_m / 111_000.0
        enu = _gps_to_enu(
            np.array([lat0 + delta_lat]),
            np.array([lon0]),
            np.array([alt0]),
            syn_calib,
        )
        # North component should be close to the requested distance.
        assert enu[0, 1] > 0, "north ping must have positive ENU y"
        np.testing.assert_allclose(enu[0, 1], north_m, rtol=1e-2)
        # East component should be negligible (< 1 m for a pure-north offset).
        assert abs(enu[0, 0]) < 1.0, f"east bleed unexpected: {enu[0, 0]:.3f} m"

    def test_east_offset_has_positive_x(self, syn_calib: Calibration):
        """A ping ~1 km due east should have ENU_x ≈ +1000 m, |ENU_y| ≈ 0.

        The longitude step is scaled by cos(lat) to preserve metric distance.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        east_m = 1_000.0
        # 1 degree of longitude ≈ cos(lat) × 111 000 m at latitude lat0.
        deg_per_m = 1.0 / (math.cos(math.radians(lat0)) * 111_000.0)
        delta_lon = east_m * deg_per_m
        enu = _gps_to_enu(
            np.array([lat0]),
            np.array([lon0 + delta_lon]),
            np.array([alt0]),
            syn_calib,
        )
        assert enu[0, 0] > 0, "east ping must have positive ENU x"
        np.testing.assert_allclose(enu[0, 0], east_m, rtol=1e-2)
        assert abs(enu[0, 1]) < 1.0, f"north bleed unexpected: {enu[0, 1]:.3f} m"

    def test_altitude_offset_has_positive_z(self, syn_calib: Calibration):
        """A ping directly overhead (same lat/lon, higher alt) must have ENU_z > 0.

        The vertical distance is exact because the WGS-84 ellipsoid height
        and ECEF Z map linearly for small Δalt at a fixed lat/lon.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        up_m = 1_000.0
        enu = _gps_to_enu(
            np.array([lat0]),
            np.array([lon0]),
            np.array([alt0 + up_m]),
            syn_calib,
        )
        assert enu[0, 2] > 0, "up ping must have positive ENU z"
        np.testing.assert_allclose(enu[0, 2], up_m, rtol=1e-6)
        # Horizontal components should be < 1 mm for a pure-up offset.
        assert abs(enu[0, 0]) < 1e-4
        assert abs(enu[0, 1]) < 1e-4

    def test_batch_shape(self, syn_calib: Calibration):
        """_gps_to_enu must return (N, 3) for an N-point batch."""
        lat0, lon0, alt0 = syn_calib.camera_gps
        lats = np.array([lat0, lat0 + 0.01, lat0 + 0.02])
        lons = np.full(3, lon0)
        alts = np.full(3, alt0 + 500.0)
        enu = _gps_to_enu(lats, lons, alts, syn_calib)
        assert enu.shape == (3, 3)


class TestProjectPingsSynthetic:
    """Sanity tests for project_pings / project_ping using synthetic_calibration.

    With identity rotation the camera looks straight up (+Z ≡ Up), so:
      * A ping overhead (Δalt > 0, same lat/lon) projects near (cx, cy).
      * A ping displaced east (ENU_x > 0) maps to pixel_x > cx.
      * A ping displaced west maps to pixel_x < cx.
      * A ping displaced north (ENU_y > 0) maps to pixel_y > cy
        (pixel Y grows downward; north is towards the bottom of this
        upward-looking camera's image).
      * A ping below the camera (ENU_z ≤ 0) is behind the camera → None.
    """

    def test_overhead_projects_near_principal_point(self, syn_calib: Calibration):
        """A ping directly overhead (same lat/lon, +1000 m alt) must project
        within 1 pixel of the principal point (cx, cy).

        Derivation: ENU = (0, 0, 1000).  With R=I, t=0, the pinhole formula
        gives  pixel = (fx×0/1000 + cx,  fy×0/1000 + cy) = (cx, cy).
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        cx = syn_calib.camera_matrix[0, 2]
        cy = syn_calib.camera_matrix[1, 2]
        pt = project_ping(_syn_ping(lat0, lon0, alt0 + 1000.0), syn_calib)
        assert pt is not None, "overhead ping projected to None"
        assert abs(pt.x - cx) < 1.0, f"x={pt.x:.3f} not near cx={cx}"
        assert abs(pt.y - cy) < 1.0, f"y={pt.y:.3f} not near cy={cy}"

    def test_east_offset_projects_right_of_center(self, syn_calib: Calibration):
        """A ping ~500 m east and 1000 m up must project to pixel_x > cx,
        and the pixel must be consistent with the pinhole formula applied to
        the *actual* ENU coordinates (not the spherical approximation).

        Derivation: with R=I and t=0, the pinhole formula gives
            pixel_x = fx × ENU_x / ENU_z + cx.
        We compute ENU directly via _gps_to_enu and verify the projection
        matches — this tests the full stack without relying on the flat-Earth
        approximation (1° lon ≈ cos(lat) × 111 000 m) being exact.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        cx = syn_calib.camera_matrix[0, 2]
        fx = syn_calib.camera_matrix[0, 0]
        deg_per_m = 1.0 / (math.cos(math.radians(lat0)) * 111_000.0)
        delta_lon = 500.0 * deg_per_m  # ≈ 500 m east (WGS-84 may give ~502 m)
        ping = _syn_ping(lat0, lon0 + delta_lon, alt0 + 1000.0)
        pt = project_ping(ping, syn_calib)
        assert pt is not None
        assert pt.x > cx
        # Derive expected pixel from the actual ENU so the test is independent
        # of the flat-Earth approximation error (~0.5%).
        enu = _gps_to_enu(
            np.array([lat0]), np.array([lon0 + delta_lon]), np.array([alt0 + 1000.0]),
            syn_calib,
        )
        expected_x = fx * enu[0, 0] / enu[0, 2] + cx
        assert abs(pt.x - expected_x) < 0.5, (
            f"projected x={pt.x:.3f} inconsistent with pinhole formula {expected_x:.3f}"
        )

    def test_west_offset_projects_left_of_center(self, syn_calib: Calibration):
        """500 m west + 1000 m up must project to pixel_x < cx."""
        lat0, lon0, alt0 = syn_calib.camera_gps
        cx = syn_calib.camera_matrix[0, 2]
        deg_per_m = 1.0 / (math.cos(math.radians(lat0)) * 111_000.0)
        delta_lon = 500.0 * deg_per_m
        pt = project_ping(_syn_ping(lat0, lon0 - delta_lon, alt0 + 1000.0), syn_calib)
        assert pt is not None
        assert pt.x < cx

    def test_east_west_symmetric_about_cx(self, syn_calib: Calibration):
        """East and west offsets of equal magnitude must project symmetrically
        about cx (within floating-point noise).
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        cx = syn_calib.camera_matrix[0, 2]
        deg_per_m = 1.0 / (math.cos(math.radians(lat0)) * 111_000.0)
        delta_lon = 300.0 * deg_per_m
        pt_e = project_ping(_syn_ping(lat0, lon0 + delta_lon, alt0 + 1000.0), syn_calib)
        pt_w = project_ping(_syn_ping(lat0, lon0 - delta_lon, alt0 + 1000.0), syn_calib)
        assert pt_e is not None and pt_w is not None
        assert abs((pt_e.x - cx) + (pt_w.x - cx)) < 0.1, (
            "east/west projections not symmetric about cx"
        )

    def test_below_camera_returns_none(self, syn_calib: Calibration):
        """A ping 100 m below the camera (alt < cam alt) must return None.

        With R=I, the camera-frame Z coordinate equals ENU_z.  A ping below
        the camera has ENU_z < 0, i.e. z_cam ≤ 0, so project_pings rejects it.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        pt = project_ping(_syn_ping(lat0, lon0, alt0 - 100.0), syn_calib)
        assert pt is None, "below-camera ping must return None"

    def test_empty_list_returns_empty(self, syn_calib: Calibration):
        assert project_pings([], syn_calib) == []


class TestBatchProjectionSynthetic:
    """project_pings results must match individual project_ping calls."""

    def test_batch_matches_individual(self, syn_calib: Calibration):
        lat0, lon0, alt0 = syn_calib.camera_gps
        deg_per_m = 1.0 / (math.cos(math.radians(lat0)) * 111_000.0)
        pings = [
            _syn_ping(lat0, lon0, alt0 + 1000.0),          # overhead — valid
            _syn_ping(lat0, lon0 + 300.0 * deg_per_m, alt0 + 1000.0),  # east — valid
            _syn_ping(lat0, lon0, alt0 - 50.0),             # below — invalid
        ]
        batch = project_pings(pings, syn_calib)
        for i, ping in enumerate(pings):
            single = project_ping(ping, syn_calib)
            if single is None:
                assert batch[i] is None, f"batch[{i}] should be None"
            else:
                assert batch[i] is not None, f"batch[{i}] should not be None"
                assert abs(batch[i].x - single.x) < 0.01
                assert abs(batch[i].y - single.y) < 0.01


class TestFlightPathVectorSynthetic:
    """flight_path_vector tests that don't need the real .npz.

    Both pings must be in-frame (above camera, small horizontal offset, all
    within the 3840×2160 bounds).  We use 200 m offsets at 1000 m altitude
    where the pinhole formula keeps pixels well inside the frame.
    """

    def test_vector_is_unit_length(self, syn_calib: Calibration):
        """The returned vector must have magnitude 1 regardless of direction."""
        lat0, lon0, alt0 = syn_calib.camera_gps
        deg_per_m = 1.0 / (math.cos(math.radians(lat0)) * 111_000.0)
        a = _syn_ping(lat0, lon0 + 100.0 * deg_per_m, alt0 + 1000.0)
        b = _syn_ping(lat0, lon0 + 300.0 * deg_per_m, alt0 + 1000.0)
        vx, vy = flight_path_vector(a, b, syn_calib)
        assert abs(math.hypot(vx, vy) - 1.0) < 1e-6

    def test_eastward_flight_has_positive_vx(self, syn_calib: Calibration):
        """Pings moving east → increasing pixel_x → vx > 0.

        With R=I: ENU_x maps to pixel_x = fx×ENU_x/ENU_z + cx, so
        a flight that increases longitude increases ENU_x and therefore
        pixel_x, giving a positive x-component of the direction vector.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        deg_per_m = 1.0 / (math.cos(math.radians(lat0)) * 111_000.0)
        a = _syn_ping(lat0, lon0 + 100.0 * deg_per_m, alt0 + 1000.0)
        b = _syn_ping(lat0, lon0 + 300.0 * deg_per_m, alt0 + 1000.0)
        vx, vy = flight_path_vector(a, b, syn_calib)
        assert vx > 0.99, f"eastward flight must dominate x, got vx={vx:.4f}"
        assert abs(vy) < 0.01, f"y-component should be near zero, got vy={vy:.4f}"

    def test_northward_flight_has_positive_vy(self, syn_calib: Calibration):
        """Pings moving north → increasing ENU_y → increasing pixel_y → vy > 0.

        In an upward-looking camera (R=I), the image's +Y axis is aligned
        with ENU north, so a northward flight produces positive vy.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        a = _syn_ping(lat0 + 100.0 / 111_000.0, lon0, alt0 + 1000.0)
        b = _syn_ping(lat0 + 300.0 / 111_000.0, lon0, alt0 + 1000.0)
        vx, vy = flight_path_vector(a, b, syn_calib)
        assert vy > 0.99, f"northward flight must dominate y, got vy={vy:.4f}"
        assert abs(vx) < 0.01, f"x-component should be near zero, got vx={vx:.4f}"

    def test_raises_when_ping_behind_camera(self, syn_calib: Calibration):
        """A ping below the camera is behind; flight_path_vector must raise."""
        lat0, lon0, alt0 = syn_calib.camera_gps
        a = _syn_ping(lat0, lon0, alt0 - 50.0)   # below camera → None
        b = _syn_ping(lat0, lon0, alt0 + 1000.0)  # overhead — valid
        with pytest.raises(ValueError, match="valid pixel"):
            flight_path_vector(a, b, syn_calib)


class TestProjectPixelToMetersSynthetic:
    """project_pixel_to_meters formula verification using synthetic_calibration.

    The function implements:
        length_m = slant_range_m × length_px / focal_px
    where focal_px = sqrt(fx × fy).

    With synthetic_calibration defaults: fx = fy = 2000, focal_px = 2000.
    A ping directly overhead at +1000 m has slant_range = 1000 m (ENU_z = 1000,
    ENU_x = ENU_y ≈ 0 → ‖ENU‖ = 1000).  So:
        length_m = 1000 × length_px / 2000 = 0.5 × length_px.
    We derive all expected values from this formula rather than hard-coding.
    """

    def test_formula_at_known_slant_range(self, syn_calib: Calibration):
        """100 px at 1000 m overhead with focal_px=2000 must give exactly 50 m.

        This pins the formula to: slant_range × length_px / focal_px.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        ping = _syn_ping(lat0, lon0, alt0 + 1000.0)
        fx = syn_calib.camera_matrix[0, 0]
        fy = syn_calib.camera_matrix[1, 1]
        focal_px = math.sqrt(fx * fy)
        slant_range_m = 1000.0
        expected_m = slant_range_m * 100.0 / focal_px  # = 50.0
        actual_m = project_pixel_to_meters(100.0, ping, syn_calib)
        np.testing.assert_allclose(actual_m, expected_m, rtol=1e-6)

    def test_proportional_to_pixel_length(self, syn_calib: Calibration):
        """Doubling pixel length must exactly double the metre output."""
        lat0, lon0, alt0 = syn_calib.camera_gps
        ping = _syn_ping(lat0, lon0, alt0 + 1000.0)
        m100 = project_pixel_to_meters(100.0, ping, syn_calib)
        m200 = project_pixel_to_meters(200.0, ping, syn_calib)
        np.testing.assert_allclose(m200, 2.0 * m100, rtol=1e-6)

    def test_proportional_to_slant_range(self, syn_calib: Calibration):
        """A ping at 5× the altitude (5× slant range) must give 5× the metres.

        Both pings are directly overhead so slant_range = Δalt exactly.
        """
        lat0, lon0, alt0 = syn_calib.camera_gps
        ping_1km = _syn_ping(lat0, lon0, alt0 + 1_000.0)
        ping_5km = _syn_ping(lat0, lon0, alt0 + 5_000.0)
        m_1km = project_pixel_to_meters(100.0, ping_1km, syn_calib)
        m_5km = project_pixel_to_meters(100.0, ping_5km, syn_calib)
        np.testing.assert_allclose(m_5km, 5.0 * m_1km, rtol=1e-5)

    def test_zero_pixel_length_returns_zero(self, syn_calib: Calibration):
        lat0, lon0, alt0 = syn_calib.camera_gps
        ping = _syn_ping(lat0, lon0, alt0 + 1000.0)
        assert project_pixel_to_meters(0.0, ping, syn_calib) == 0.0

    def test_higher_altitude_gives_longer_metres(self, syn_calib: Calibration):
        """Higher aircraft → longer slant range → more metres per pixel."""
        lat0, lon0, alt0 = syn_calib.camera_gps
        ping_low = _syn_ping(lat0, lon0, alt0 + 1_000.0)
        ping_high = _syn_ping(lat0, lon0, alt0 + 10_000.0)
        m_low = project_pixel_to_meters(100.0, ping_low, syn_calib)
        m_high = project_pixel_to_meters(100.0, ping_high, syn_calib)
        assert m_high > m_low
