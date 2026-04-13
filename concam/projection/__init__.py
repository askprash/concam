"""GPS-to-pixel projection using camera calibration.

Converts GPS (lat, lon, alt) coordinates to pixel (x, y) via:
  GPS -> ECEF (pyproj) -> ENU (rotation) -> pixel (cv2.projectPoints)

The calibration data comes from LAE_skycam/calibration/pointpicker_calibration_estimate.npz,
which contains camera intrinsics, distortion coefficients, rotation, translation, and the
camera's GPS/ECEF position. The projection pipeline follows the reference implementation
in LAE_skycam/calibration/camera_utils.py (gps_to_camxy_vasha_fixed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from pyproj import Transformer

from concam.adsb import Ping
from concam.config import CalibrationConfig, DetectionConfig


@dataclass
class PixelPoint:
    """A projected point in pixel coordinates."""

    x: float
    y: float


@dataclass
class Rect:
    """An axis-aligned bounding rectangle in pixel space."""

    x: int
    y: int
    w: int
    h: int


class Calibration:
    """Loaded camera calibration parameters."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        cam_ecef: np.ndarray,
        camera_gps: np.ndarray,
        calibration_resolution: tuple[int, int],
    ):
        self.camera_matrix = camera_matrix  # (3, 3)
        self.distortion_coefficients = distortion_coefficients  # (N, 1)
        self.rotation = rotation  # (3, 3) rotation matrix
        self.translation = translation  # (3, 1)
        self.cam_ecef = cam_ecef  # (3,)
        self.camera_gps = camera_gps  # (lat, lon, alt)
        self.calibration_resolution = calibration_resolution  # (width, height)

        # Pre-compute the rvec for cv2.projectPoints
        self._rvec, _ = cv2.Rodrigues(self.rotation)
        self._tvec = self.translation.astype(np.float64)

        # Pre-compute ECEF-to-ENU rotation matrix
        lat0 = np.radians(self.camera_gps[0])
        lon0 = np.radians(self.camera_gps[1])
        self._ecef_to_enu = np.array(
            [
                [-np.sin(lon0), np.cos(lon0), 0],
                [
                    -np.sin(lat0) * np.cos(lon0),
                    -np.sin(lat0) * np.sin(lon0),
                    np.cos(lat0),
                ],
                [
                    np.cos(lat0) * np.cos(lon0),
                    np.cos(lat0) * np.sin(lon0),
                    np.sin(lat0),
                ],
            ]
        )

        # Pre-compute the geodetic-to-ECEF transformer (lon, lat, alt -> X, Y, Z)
        self._transformer = Transformer.from_crs(
            "epsg:4979", "epsg:4978", always_xy=True
        )


def load_calibration(config: CalibrationConfig) -> Calibration:
    """Load calibration parameters from an .npz file."""
    data = np.load(config.npz_path, allow_pickle=True)
    return Calibration(
        camera_matrix=data["camera_matrix"],
        distortion_coefficients=data["distortion_coefficients"],
        rotation=data["rotation"],
        translation=data["translation"],
        cam_ecef=data["cam_ecef"],
        camera_gps=data["camera_gps"],
        calibration_resolution=tuple(config.calibration_resolution),
    )


def _gps_to_enu(
    lats: np.ndarray,
    lons: np.ndarray,
    alts: np.ndarray,
    calib: Calibration,
) -> np.ndarray:
    """Convert GPS arrays to ENU coordinates relative to the camera. Returns (N, 3)."""
    ex, ey, ez = calib._transformer.transform(lons, lats, alts)
    ecef_points = np.column_stack((ex, ey, ez))
    delta = ecef_points - calib.cam_ecef[None, :]
    return (calib._ecef_to_enu @ delta.T).T


def project_pings(
    pings: list[Ping],
    calib: Calibration,
) -> list[PixelPoint | None]:
    """Project a batch of GPS pings to pixel coordinates.

    Returns a list parallel to pings, with None for points behind the camera
    or outside the image bounds.
    """
    if not pings:
        return []

    lats = np.array([p.lat for p in pings], dtype=np.float64)
    lons = np.array([p.lon for p in pings], dtype=np.float64)
    alts = np.array([p.alt_m for p in pings], dtype=np.float64)

    enu_points = _gps_to_enu(lats, lons, alts, calib)

    # Check which points are behind the camera (Z <= 0 in camera space)
    points_cam = (calib.rotation @ enu_points.T + calib.translation).T
    behind = points_cam[:, 2] <= 0

    w, h = calib.calibration_resolution
    results: list[PixelPoint | None] = [None] * len(pings)

    valid_mask = ~behind
    if not np.any(valid_mask):
        return results

    valid_enu = enu_points[valid_mask].astype(np.float64)
    image_pts, _ = cv2.projectPoints(
        valid_enu, calib._rvec, calib._tvec,
        calib.camera_matrix, calib.distortion_coefficients,
    )
    image_pts = image_pts.reshape(-1, 2)

    valid_indices = np.where(valid_mask)[0]
    for i, idx in enumerate(valid_indices):
        px, py = float(image_pts[i, 0]), float(image_pts[i, 1])
        if 0 <= px < w and 0 <= py < h:
            results[idx] = PixelPoint(x=px, y=py)

    return results


def project_ping(
    ping: Ping,
    calib: Calibration,
) -> PixelPoint | None:
    """Project a single GPS ping to pixel coordinates.

    Returns None if the point is behind the camera or outside the image.
    """
    return project_pings([ping], calib)[0]


def flight_path_vector(
    ping_a: Ping,
    ping_b: Ping,
    calib: Calibration,
) -> tuple[float, float]:
    """Return the unit vector of the flight path in pixel space (from a to b).

    Both pings must project to valid pixel coordinates; raises ValueError otherwise.
    """
    pts = project_pings([ping_a, ping_b], calib)
    if pts[0] is None or pts[1] is None:
        raise ValueError(
            "Both pings must project to valid pixel coordinates to compute a path vector"
        )
    dx = pts[1].x - pts[0].x
    dy = pts[1].y - pts[0].y
    mag = math.hypot(dx, dy)
    if mag < 1e-9:
        raise ValueError("Pings project to the same pixel; cannot compute direction")
    return (dx / mag, dy / mag)


def _roi_dimensions(config: DetectionConfig) -> tuple[float, float]:
    """Return ``(along_half, cross_half)`` for the rotated ROI in pixels.

    ``roi_along_px`` / ``roi_cross_px`` take precedence when set; otherwise
    fall back to the legacy ``roi_padding``-based derivation (along=3*pad,
    cross=pad) so older configs keep working.
    """
    along_full = getattr(config, "roi_along_px", None)
    cross_full = getattr(config, "roi_cross_px", None)
    if along_full and cross_full:
        return float(along_full) / 2.0, float(cross_full) / 2.0
    pad = config.roi_padding
    return float(pad * 3), float(pad)


def rotated_polygon(
    center: PixelPoint,
    path_vector: tuple[float, float],
    config: DetectionConfig,
) -> np.ndarray:
    """Return the four corners of the oriented detection rectangle.

    The rectangle is centered on ``center``, elongated along ``path_vector``,
    with along-track half-length and cross-track half-width derived from
    ``config``. Corners are returned CCW as a ``(4, 2) float32`` array in
    full-frame pixel coords. Caller is responsible for rounding / clipping
    if an integer polygon is required.
    """
    along, cross = _roi_dimensions(config)
    vx, vy = path_vector
    nx, ny = -vy, vx
    corners = [
        (center.x - along * vx - cross * nx, center.y - along * vy - cross * ny),
        (center.x + along * vx - cross * nx, center.y + along * vy - cross * ny),
        (center.x + along * vx + cross * nx, center.y + along * vy + cross * ny),
        (center.x - along * vx + cross * nx, center.y - along * vy + cross * ny),
    ]
    return np.asarray(corners, dtype=np.float32)


def oriented_roi(
    center: PixelPoint,
    path_vector: tuple[float, float],
    config: DetectionConfig,
    image_size: tuple[int, int] = (3840, 2160),
) -> Rect:
    """Compute an axis-aligned bounding box around the oriented detection region.

    The oriented region is the rotated rectangle returned by
    ``rotated_polygon()``; this helper just returns its AABB clipped to the
    image, which the detection stage uses as a cheap bounding crop before
    applying the rotated mask.
    """
    poly = rotated_polygon(center, path_vector, config)
    w, h = image_size
    xs = poly[:, 0]
    ys = poly[:, 1]
    x_min = max(0, int(math.floor(float(xs.min()))))
    y_min = max(0, int(math.floor(float(ys.min()))))
    x_max = min(w - 1, int(math.ceil(float(xs.max()))))
    y_max = min(h - 1, int(math.ceil(float(ys.max()))))
    return Rect(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min)
