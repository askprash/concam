"""Property tests for concam.detection.geometry.candidate_geometry.

Uses hypothesis for property-based testing.  No snapshot / "busy" tests.
Properties asserted:

1. path_vec passthrough — returned path_vec == (cand['path_dx'], cand['path_dy']).
2. rect spans the whole crop — rect == Rect(0, 0, cw, ch).
3. extract_pad clamping — when roi.x >= extract_pad, center.x == pixel_x -
   (roi.x - extract_pad); when roi.x < extract_pad, center.x == pixel_x.
4. Translation equivariance — joint-shifting roi.x/roi.y AND pixel_x/pixel_y by
   the same (dx, dy) (keeping both roi coords >= extract_pad) leaves center_local
   and polygon unchanged.  This is the invariant that makes the crop-local
   coordinate system translation-invariant.
5. polygon shape (4, 2).
6. round-trip sanity — candidate_geometry output feeds into concam.detection.detect
   on a synthetic streak crop and fires (score > 0), confirming the geometry is
   consistent with the detector's coordinate expectations.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from concam.config import DetectionConfig
from concam.detection import detect
from concam.detection.geometry import CandidateGeometry, candidate_geometry
from concam.projection import Rect


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def candidate_dicts(
    draw,
    *,
    roi_x_min: int = 0,
    roi_x_max: int = 3000,
    roi_y_min: int = 0,
    roi_y_max: int = 2000,
) -> dict:
    """Generate a synthetic manifest candidate dict."""
    roi_x = draw(st.integers(roi_x_min, roi_x_max))
    roi_y = draw(st.integers(roi_y_min, roi_y_max))
    roi_w = draw(st.integers(10, 400))
    roi_h = draw(st.integers(10, 300))
    pixel_x = draw(st.floats(0.0, 4000.0, allow_nan=False, allow_infinity=False))
    pixel_y = draw(st.floats(0.0, 3000.0, allow_nan=False, allow_infinity=False))
    # path_vec: unit-length direction; draw angle, compute components.
    angle_deg = draw(st.floats(0.0, 360.0, allow_nan=False, allow_infinity=False))
    angle_rad = math.radians(angle_deg)
    path_dx = math.cos(angle_rad)
    path_dy = math.sin(angle_rad)
    return {
        "roi": {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h},
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "path_dx": path_dx,
        "path_dy": path_dy,
    }


@st.composite
def roi_params(draw):
    """Draw (roi_along_px, roi_cross_px, extract_pad, crop_shape)."""
    roi_along = draw(st.integers(60, 320))
    roi_cross = draw(st.integers(20, 120))
    extract_pad = draw(st.integers(0, 50))
    crop_h = draw(st.integers(10, 400))
    crop_w = draw(st.integers(10, 400))
    return roi_along, roi_cross, extract_pad, (crop_h, crop_w)


# ---------------------------------------------------------------------------
# Property 1: path_vec passthrough
# ---------------------------------------------------------------------------

@given(cand=candidate_dicts(), params=roi_params())
@settings(max_examples=200)
def test_path_vec_passthrough(cand: dict, params: tuple) -> None:
    """Returned path_vec equals (cand['path_dx'], cand['path_dy'])."""
    roi_along, roi_cross, extract_pad, crop_shape = params
    g = candidate_geometry(
        cand, crop_shape,
        roi_along_px=roi_along,
        roi_cross_px=roi_cross,
        extract_pad=extract_pad,
    )
    assert g.path_vec == (float(cand["path_dx"]), float(cand["path_dy"]))


# ---------------------------------------------------------------------------
# Property 2: rect spans the whole crop
# ---------------------------------------------------------------------------

@given(cand=candidate_dicts(), params=roi_params())
@settings(max_examples=200)
def test_rect_spans_whole_crop(cand: dict, params: tuple) -> None:
    """rect is always Rect(0, 0, cw, ch)."""
    roi_along, roi_cross, extract_pad, crop_shape = params
    crop_h, crop_w = crop_shape
    g = candidate_geometry(
        cand, crop_shape,
        roi_along_px=roi_along,
        roi_cross_px=roi_cross,
        extract_pad=extract_pad,
    )
    assert g.rect == Rect(x=0, y=0, w=crop_w, h=crop_h)


# ---------------------------------------------------------------------------
# Property 3: extract_pad clamping
# ---------------------------------------------------------------------------

@given(
    base=candidate_dicts(roi_x_min=0, roi_x_max=3000, roi_y_min=0, roi_y_max=2000),
    roi_along=st.integers(60, 320),
    roi_cross=st.integers(20, 120),
    extract_pad=st.integers(0, 60),
    crop_shape=st.tuples(st.integers(10, 400), st.integers(10, 400)),
)
@settings(max_examples=300)
def test_extract_pad_clamping(
    base: dict,
    roi_along: int,
    roi_cross: int,
    extract_pad: int,
    crop_shape: tuple,
) -> None:
    """center.x/y are computed from max(0, roi.x - extract_pad)."""
    g = candidate_geometry(
        base, crop_shape,
        roi_along_px=roi_along,
        roi_cross_px=roi_cross,
        extract_pad=extract_pad,
    )
    roi = base["roi"]
    expected_tl_x = max(0, int(roi["x"]) - extract_pad)
    expected_tl_y = max(0, int(roi["y"]) - extract_pad)
    expected_cx = float(base["pixel_x"]) - expected_tl_x
    expected_cy = float(base["pixel_y"]) - expected_tl_y
    assert g.center.x == expected_cx
    assert g.center.y == expected_cy

    # Specific sub-cases: test both branches.
    if roi["x"] >= extract_pad:
        # No clamp: center.x == pixel_x - (roi.x - extract_pad).
        assert g.center.x == float(base["pixel_x"]) - (int(roi["x"]) - extract_pad)
    else:
        # Clamp fires: full_tl_x == 0, so center.x == pixel_x.
        assert g.center.x == float(base["pixel_x"])


# ---------------------------------------------------------------------------
# Property 4: Translation equivariance
# ---------------------------------------------------------------------------

@given(
    cand=candidate_dicts(roi_x_min=50, roi_x_max=2000, roi_y_min=50, roi_y_max=1000),
    dx=st.integers(-40, 40),
    dy=st.integers(-40, 40),
    roi_along=st.integers(60, 320),
    roi_cross=st.integers(20, 120),
    extract_pad=st.integers(0, 40),
    crop_shape=st.tuples(st.integers(10, 400), st.integers(10, 400)),
)
@settings(max_examples=300)
def test_translation_equivariance(
    cand: dict,
    dx: int,
    dy: int,
    roi_along: int,
    roi_cross: int,
    extract_pad: int,
    crop_shape: tuple,
) -> None:
    """Jointly shifting roi and pixel by (dx, dy) leaves center and polygon unchanged.

    The invariant: center_local = pixel - max(0, roi.x - pad).  If we shift
    roi.x -> roi.x + dx AND pixel_x -> pixel_x + dx (and similarly for y),
    then max(0, roi.x + dx - pad) = roi.x + dx - pad (assuming both sides are
    above the clamp boundary), and center.x = pixel_x + dx - (roi.x + dx - pad)
    = pixel_x - (roi.x - pad) — unchanged.  Since center is unchanged, the
    polygon is also unchanged.
    """
    roi = cand["roi"]
    # Guard: both original and shifted roi coords must be >= extract_pad so the
    # clamp does NOT fire for either (otherwise the invariant breaks at the boundary).
    assume(roi["x"] >= extract_pad)
    assume(roi["y"] >= extract_pad)
    assume(roi["x"] + dx >= extract_pad)
    assume(roi["y"] + dy >= extract_pad)

    shifted_cand = {
        "roi": {
            "x": roi["x"] + dx,
            "y": roi["y"] + dy,
            "w": roi["w"],
            "h": roi["h"],
        },
        "pixel_x": cand["pixel_x"] + dx,
        "pixel_y": cand["pixel_y"] + dy,
        "path_dx": cand["path_dx"],
        "path_dy": cand["path_dy"],
    }

    g_orig = candidate_geometry(
        cand, crop_shape,
        roi_along_px=roi_along, roi_cross_px=roi_cross, extract_pad=extract_pad,
    )
    g_shifted = candidate_geometry(
        shifted_cand, crop_shape,
        roi_along_px=roi_along, roi_cross_px=roi_cross, extract_pad=extract_pad,
    )

    # Use isclose for center: adding an integer offset to a float and then
    # subtracting the same offset can differ by a few ULPs due to float
    # arithmetic order, so exact equality is too strict.
    import math as _math
    assert _math.isclose(g_shifted.center.x, g_orig.center.x, abs_tol=1e-9)
    assert _math.isclose(g_shifted.center.y, g_orig.center.y, abs_tol=1e-9)
    assert np.allclose(g_shifted.polygon, g_orig.polygon, atol=1e-4)


# ---------------------------------------------------------------------------
# Property 5: polygon shape
# ---------------------------------------------------------------------------

@given(cand=candidate_dicts(), params=roi_params())
@settings(max_examples=150)
def test_polygon_shape(cand: dict, params: tuple) -> None:
    """polygon is always (4, 2) float32."""
    roi_along, roi_cross, extract_pad, crop_shape = params
    g = candidate_geometry(
        cand, crop_shape,
        roi_along_px=roi_along,
        roi_cross_px=roi_cross,
        extract_pad=extract_pad,
    )
    assert g.polygon.shape == (4, 2)
    assert g.polygon.dtype == np.float32


# ---------------------------------------------------------------------------
# Property 6: round-trip sanity with detect()
# ---------------------------------------------------------------------------

def _make_streak_crop(
    width: int = 200,
    height: int = 120,
    center_x: float = 100.0,
    center_y: float = 60.0,
    path_dx: float = 1.0,
    path_dy: float = 0.0,
    streak_len: int = 120,
    thickness: int = 2,
    bg: int = 30,
    fg: int = 220,
) -> np.ndarray:
    """Create a synthetic BGR crop with a bright streak along path_vec."""
    crop = np.full((height, width, 3), bg, dtype=np.uint8)
    half = streak_len // 2
    pt1 = (int(round(center_x - half * path_dx)), int(round(center_y - half * path_dy)))
    pt2 = (int(round(center_x + half * path_dx)), int(round(center_y + half * path_dy)))
    cv2.line(crop, pt1, pt2, (fg, fg, fg), thickness)
    return crop


def _detect_config(roi_along: int = 120, roi_cross: int = 40) -> DetectionConfig:
    """Fixed-threshold config tuned for synthetic crops."""
    return DetectionConfig(
        canny_low=50,
        canny_high=150,
        hough_threshold=15,
        hough_min_line_length=15,
        hough_max_line_gap=5,
        roi_along_px=roi_along,
        roi_cross_px=roi_cross,
        use_adaptive_canny=False,
        angle_tolerance_deg=15.0,
        long_line_min_px=15.0,
        score_norm_count=2,
        score_fn="count",
        use_rotated_mask=True,
        blur_kernel=0,
        preprocessing="none",
    )


@pytest.mark.parametrize("angle_deg", [0, 30, 45, 90, 135])
def test_round_trip_streak_detection(angle_deg: int) -> None:
    """Geometry reconstructed via candidate_geometry drives detect() to score > 0.

    For each flight-path angle, build a synthetic crop with a bright streak,
    create a candidate dict that describes it, reconstruct geometry via
    candidate_geometry, feed into detect(), and assert the score is positive.
    This confirms the crop-local coordinate system is consistent with the
    detector's expectations.
    """
    crop_w, crop_h = 200, 120
    center_x, center_y = 100.0, 60.0
    roi_along, roi_cross = 120, 40
    extract_pad = 20

    angle_rad = math.radians(angle_deg)
    path_dx = math.cos(angle_rad)
    path_dy = math.sin(angle_rad)

    # Build the synthetic crop.
    crop = _make_streak_crop(
        width=crop_w, height=crop_h,
        center_x=center_x, center_y=center_y,
        path_dx=path_dx, path_dy=path_dy,
        streak_len=100,
    )

    # Build a candidate dict whose pixel_x/y and roi agree with the crop.
    # full_tl_x = max(0, roi_x - extract_pad).  We want center_local = center_x,y.
    # So: pixel_x = center_x + full_tl_x, roi_x = full_tl_x + extract_pad.
    full_tl_x = 50  # arbitrary: large enough not to clamp
    full_tl_y = 30
    roi_x = full_tl_x + extract_pad
    roi_y = full_tl_y + extract_pad
    cand = {
        "roi": {"x": roi_x, "y": roi_y, "w": 80, "h": 60},
        "pixel_x": center_x + full_tl_x,
        "pixel_y": center_y + full_tl_y,
        "path_dx": path_dx,
        "path_dy": path_dy,
    }

    g = candidate_geometry(
        cand, (crop_h, crop_w),
        roi_along_px=roi_along,
        roi_cross_px=roi_cross,
        extract_pad=extract_pad,
    )

    # Verify center_local is where we put the streak.
    assert abs(g.center.x - center_x) < 1e-9
    assert abs(g.center.y - center_y) < 1e-9

    cfg = _detect_config(roi_along=roi_along, roi_cross=roi_cross)
    result = detect(crop, g.rect, cfg, polygon=g.polygon, path_vec=g.path_vec)
    assert result.score > 0.0, (
        f"detect() returned score=0 for angle={angle_deg}°; "
        f"geometry: center=({g.center.x},{g.center.y}), "
        f"rect={g.rect}, polygon={g.polygon}"
    )
