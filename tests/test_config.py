"""Guard test: DetectionConfig / AggregationConfig defaults match the base production YAML.

Single-source-of-truth invariant: a bare ``DetectionConfig()`` (no YAML) must
honestly represent what the base production site config uses.  This test loads
``configs/mit_green_building.yaml`` and asserts that every detection/aggregation
field that has a non-trivial tuned value in the YAML equals the dataclass default.

If this test fails it means someone edited the YAML but forgot to sync the
dataclass default (or vice versa).  Fix both to agree, or — if the YAML was
intentionally diverged for a site-specific reason — suppress by documenting the
intentional asymmetry and adjusting the assertion here.
"""

from __future__ import annotations

from pathlib import Path

from concam.config import AggregationConfig, DetectionConfig, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_YAML = REPO_ROOT / "configs" / "mit_green_building.yaml"


def test_detection_and_aggregation_defaults_match_base_yaml() -> None:
    """Dataclass defaults must equal base-YAML values for centralized fields."""
    site = load_config(BASE_YAML)
    det = site.detection
    agg = site.aggregation
    bare_det = DetectionConfig()
    bare_agg = AggregationConfig()

    # --- DetectionConfig ---
    assert bare_det.roi_along_px == det.roi_along_px, (
        f"DetectionConfig.roi_along_px default ({bare_det.roi_along_px}) "
        f"!= base YAML ({det.roi_along_px})"
    )
    assert bare_det.roi_cross_px == det.roi_cross_px, (
        f"DetectionConfig.roi_cross_px default ({bare_det.roi_cross_px}) "
        f"!= base YAML ({det.roi_cross_px})"
    )
    assert bare_det.long_line_min_px == det.long_line_min_px, (
        f"DetectionConfig.long_line_min_px default ({bare_det.long_line_min_px}) "
        f"!= base YAML ({det.long_line_min_px})"
    )
    assert bare_det.hough_threshold == det.hough_threshold, (
        f"DetectionConfig.hough_threshold default ({bare_det.hough_threshold}) "
        f"!= base YAML ({det.hough_threshold})"
    )
    assert bare_det.hough_min_line_length == det.hough_min_line_length, (
        f"DetectionConfig.hough_min_line_length default ({bare_det.hough_min_line_length}) "
        f"!= base YAML ({det.hough_min_line_length})"
    )
    assert bare_det.angle_tolerance_deg == det.angle_tolerance_deg, (
        f"DetectionConfig.angle_tolerance_deg default ({bare_det.angle_tolerance_deg}) "
        f"!= base YAML ({det.angle_tolerance_deg})"
    )
    assert bare_det.score_length_norm_px == det.score_length_norm_px, (
        f"DetectionConfig.score_length_norm_px default ({bare_det.score_length_norm_px}) "
        f"!= base YAML ({det.score_length_norm_px})"
    )
    assert bare_det.preprocessing == det.preprocessing, (
        f"DetectionConfig.preprocessing default ({bare_det.preprocessing!r}) "
        f"!= base YAML ({det.preprocessing!r})"
    )
    assert bare_det.cross_grad_gain == det.cross_grad_gain, (
        f"DetectionConfig.cross_grad_gain default ({bare_det.cross_grad_gain}) "
        f"!= base YAML ({det.cross_grad_gain})"
    )

    # --- AggregationConfig ---
    assert bare_agg.detection_threshold == agg.detection_threshold, (
        f"AggregationConfig.detection_threshold default ({bare_agg.detection_threshold}) "
        f"!= base YAML ({agg.detection_threshold})"
    )
    assert bare_agg.smoothing_window_frames == agg.smoothing_window_frames, (
        f"AggregationConfig.smoothing_window_frames default ({bare_agg.smoothing_window_frames}) "
        f"!= base YAML ({agg.smoothing_window_frames})"
    )
