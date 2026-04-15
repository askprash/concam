"""Configuration loading and dataclass definitions for the concam pipeline."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VideoConfig:
    root: str
    timelapse_glob: str = "{date:%Y_%m_%d}_0000_2359.mp4"
    raw_segment_glob: str = "raw_segments_clean/{date:%Y-%m-%d}_*.mp4"


@dataclass
class OcrConfig:
    # Top-right timestamp region (confirmed from probing real frames)
    timestamp_position: str = "top_right"
    # (height, width) of the timestamp crop box in pixels at 3840x2160
    timestamp_region: tuple[int, int] = (80, 875)
    # Confidence below which EasyOCR fallback is triggered
    fallback_confidence_threshold: float = 0.6


@dataclass
class AdsbConfig:
    # Path to the feder data store
    data_dir: str = "/home/mcast/data/feder"
    # Filter parameters
    min_altitude_m: float = 8000.0
    max_radius_km: float = 50.0
    # Camera site center (lat, lon) used for radius filter
    site_lat: float = 42.360444
    site_lon: float = -71.089238
    # Altitude source policy:
    #   "auto" (default): GNSS when available and consistent with barometric; fall
    #     back to barometric+geoid-offset when GNSS is missing or disagrees by more
    #     than altitude_discrepancy_threshold_m
    #   "gnss": always use GNSS (WGS-84 HAE)
    #   "barometric": always use barometric (ISA pressure altitude) + geoid offset
    altitude_source: str = "auto"
    # Red-flag threshold for baro/GNSS disagreement at cruise. ~400 ft (122 m)
    # is the level above which the delta is more likely to indicate GNSS trouble
    # (multipath, RAIM dropout, jamming/spoofing) or a static-system leak than
    # normal ISA deviation. Under typical conditions the gap is 150-300 ft.
    # Sourced from a non-primary aviation/surveillance literature sweep (FAA
    # ATC Altitude Assignment; EUROCONTROL Mode S monitoring guidance summarised
    # on avionicswest.com; ION/NAVIGATION 71(2), 2024 on baro-augmented GNSS
    # integrity monitoring). Not independently verified against RTCA DO-260B;
    # tune if a primary source indicates a different level.
    altitude_discrepancy_threshold_m: float = 122.0
    # Geoid-to-ellipsoid offset used when converting barometric (≈ MSL / ISA) to
    # WGS-84 HAE for projection. Positive = ellipsoid above geoid.
    #   h_ellipsoid = h_msl + site_geoid_offset_m
    # EGM96 at Boston MA is approximately -28 m. Leave as a site-level constant;
    # a per-point geoid query is overkill inside a 50 km radius.
    site_geoid_offset_m: float = -28.0


@dataclass
class CalibrationConfig:
    # Absolute path to the calibration .npz file
    npz_path: str = "/home/prash/contrails/LAE_skycam/calibration/pointpicker_calibration_estimate.npz"
    # Resolution at which calibration was performed
    calibration_resolution: tuple[int, int] = (3840, 2160)


@dataclass
class DetectionConfig:
    # Minimum Hough score (0-1) to count as a detection
    score_threshold: float = 0.3
    # Fallback Canny thresholds when use_adaptive_canny=False
    canny_low: int = 50
    canny_high: int = 150
    # Hough parameters
    hough_threshold: int = 30
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 10
    # Legacy ROI knob: along=3*pad, cross=pad. Kept for back-compat when
    # roi_along_px/roi_cross_px are unset.
    roi_padding: int = 20
    # Explicit along-track length and cross-track width of the rotated
    # detection rectangle (pixels). When both are set they override
    # ``roi_padding``-based sizing. Defaults mirror the legacy 120x40 box.
    roi_along_px: int = 120
    roi_cross_px: int = 40
    # Adaptive percentile-based Canny thresholds on the masked ROI pixel
    # distribution. Mirrors camera-flight-overlay/contrail_labeler prod detector.
    use_adaptive_canny: bool = True
    canny_percentile_low: float = 96.0  # pixel floor (THRESH_TOZERO)
    canny_percentile_high: float = 99.5  # -> Canny high threshold
    canny_low_ratio: float = 0.25  # Canny low = ratio * Canny high
    canny_min_high: int = 60  # clamp floor for Canny high threshold
    # Angle constraint: keep only Hough lines within ±tol degrees of the
    # flight-path vector. Disabled when path_vec is None.
    angle_tolerance_deg: float = 8.0
    # Minimum Hough-line length (pixels) to count toward the score.
    long_line_min_px: float = 40.0
    # Score normalisation: count of aligned long lines that maps to score=1.0.
    score_norm_count: int = 6
    # When True, the detector masks the rotated polygon before Canny/Hough.
    # False falls back to the AABB-only legacy path.
    use_rotated_mask: bool = True
    # Preprocessing: Gaussian blur kernel (set to 0 to disable).
    blur_kernel: int = 3
    # Pixels in this full-frame region are zeroed before Canny so that burned-in
    # overlay text (e.g. the clock/date OSD) cannot produce false Hough lines.
    # Format: [y0, y1, x0, x1] in full-frame pixel coordinates (same convention
    # as numpy array slicing: frame[y0:y1, x0:x1]).  None = disabled.
    # For the MIT Green Building camera the timestamp occupies the top-right
    # corner; set this in the site YAML to mask it with a safety pad.
    timestamp_exclusion_region: list | None = None
    # Pre-processing applied to the (possibly temporal-diffed) grayscale frame
    # before Canny edge detection.  Options:
    #   "none"            – raw grayscale (default, current behaviour)
    #   "local_contrast"  – subtract large Gaussian (sigma=local_contrast_sigma)
    #                       to remove slow cloud background while keeping the
    #                       narrow contrail streak; re-centres at 128.
    #   "cross_grad"      – gradient magnitude along the direction perpendicular
    #                       to the flight-path vector (emphasises cross-track
    #                       edges = contrail profile; requires path_vec kwarg).
    preprocessing: str = "none"
    # Gaussian sigma for the "local_contrast" preprocessing mode.
    local_contrast_sigma: float = 25.0
    # Adaptive contrail length measurement via iterative ROI growth.
    # After a detection scores above zero the along-track axis is grown by
    # growth_step_px (each direction) until the aligned-long-line count stops
    # increasing or the total along-track dimension reaches roi_max_along_px.
    growth_step_px: int = 20
    roi_max_along_px: int = 600


@dataclass
class AggregationConfig:
    # Minimum score to include a frame in an episode
    detection_threshold: float = 0.3
    # Maximum gap (seconds) between detections to merge into one episode
    max_gap_seconds: float = 30.0


@dataclass
class SiteConfig:
    name: str
    timezone: str
    camera_lat: float
    camera_lon: float
    camera_alt_m: float
    video: VideoConfig
    ocr: OcrConfig
    adsb: AdsbConfig
    calibration: CalibrationConfig
    detection: DetectionConfig
    aggregation: AggregationConfig


def load_config(path: str | Path) -> SiteConfig:
    """Load a site YAML config file and return a SiteConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    def _get(section: str, cls, defaults=None):
        data = raw.get(section, {}) or {}
        if defaults:
            data = {**defaults, **data}
        # Filter to known fields only
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    video = VideoConfig(**{k: v for k, v in (raw.get("video") or {}).items()
                           if k in ("root", "timelapse_glob", "raw_segment_glob")}) \
        if raw.get("video") else VideoConfig(root="/net/d16/data/contrail-camera")

    return SiteConfig(
        name=raw.get("name", "unknown"),
        timezone=raw.get("timezone", "America/New_York"),
        camera_lat=raw.get("camera_lat", 42.360444),
        camera_lon=raw.get("camera_lon", -71.089238),
        camera_alt_m=raw.get("camera_alt_m", 84.226575),
        video=video,
        ocr=_get("ocr", OcrConfig),
        adsb=_get("adsb", AdsbConfig),
        calibration=_get("calibration", CalibrationConfig),
        detection=_get("detection", DetectionConfig),
        aggregation=_get("aggregation", AggregationConfig),
    )
