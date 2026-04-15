"""Tests for the rotated-ROI contrail detector (concam.detection)."""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from concam.config import DetectionConfig
from concam.detection import DetectionResult, detect, grow_contrail_length
from concam.projection import PixelPoint, Rect, rotated_polygon


def _make_config(**overrides) -> DetectionConfig:
    """Config tuned so synthetic 200x200 test frames produce sensible scores."""
    base = dict(
        score_threshold=0.3,
        canny_low=50,
        canny_high=150,
        hough_threshold=20,
        hough_min_line_length=20,
        hough_max_line_gap=5,
        roi_padding=20,
        roi_along_px=120,
        roi_cross_px=40,
        # Synthetic tests use fixed-threshold mode because the percentile path
        # keys off real-sky statistics that don't exist in a uniform test fill.
        use_adaptive_canny=False,
        angle_tolerance_deg=10.0,
        long_line_min_px=20.0,
        score_norm_count=2,
        use_rotated_mask=True,
        blur_kernel=0,
    )
    base.update(overrides)
    return DetectionConfig(**base)


def _frame_with_line(
    width: int = 200,
    height: int = 200,
    pt1: tuple[int, int] = (10, 100),
    pt2: tuple[int, int] = (190, 100),
    thickness: int = 2,
    bg: int = 30,
    fg: int = 220,
) -> np.ndarray:
    frame = np.full((height, width), bg, dtype=np.uint8)
    cv2.line(frame, pt1, pt2, fg, thickness)
    return frame


# --- New-style tests (rotated mask + angle filter) ---------------------------


class TestRotatedROIDetector:
    """The new detector must accept / reject based on alignment with path_vec."""

    def test_line_along_path_scores_high(self):
        config = _make_config()
        frame = _frame_with_line(width=400, height=200,
                                 pt1=(50, 100), pt2=(350, 100), thickness=2)
        center = PixelPoint(x=200, y=100)
        path_vec = (1.0, 0.0)  # horizontal — aligns with the drawn line
        poly = rotated_polygon(center, path_vec, config)
        xs = poly[:, 0]; ys = poly[:, 1]
        roi = Rect(
            x=int(xs.min()), y=int(ys.min()),
            w=int(xs.max() - xs.min()) + 1,
            h=int(ys.max() - ys.min()) + 1,
        )
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        assert result.score > 0.0
        assert result.num_long_lines >= 1
        assert result.pixel_line is not None

    def test_line_perpendicular_to_path_scores_zero(self):
        config = _make_config()
        # Horizontal line in the frame, but path_vec is vertical.
        frame = _frame_with_line(width=400, height=200,
                                 pt1=(50, 100), pt2=(350, 100), thickness=2)
        center = PixelPoint(x=200, y=100)
        path_vec = (0.0, 1.0)  # vertical path, perpendicular to the drawn line
        poly = rotated_polygon(center, path_vec, config)
        xs = poly[:, 0]; ys = poly[:, 1]
        roi = Rect(
            x=max(0, int(xs.min())), y=max(0, int(ys.min())),
            w=int(xs.max() - xs.min()) + 1,
            h=int(ys.max() - ys.min()) + 1,
        )
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        # Rotated mask is vertical (20 px cross-track) so it barely overlaps the
        # horizontal line; any survivors fail the ±10° angle filter.
        assert result.score == 0.0
        assert result.num_long_lines == 0

    def test_blank_roi_scores_zero(self):
        config = _make_config()
        frame = np.full((200, 400), 60, dtype=np.uint8)
        center = PixelPoint(x=200, y=100)
        path_vec = (1.0, 0.0)
        poly = rotated_polygon(center, path_vec, config)
        xs = poly[:, 0]; ys = poly[:, 1]
        roi = Rect(
            x=int(xs.min()), y=int(ys.min()),
            w=int(xs.max() - xs.min()) + 1,
            h=int(ys.max() - ys.min()) + 1,
        )
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        assert result.score == 0.0
        assert result.pixel_line is None

    def test_discrete_gate_recoverable(self):
        """With score_norm_count=2, score>=1.0 iff num_long_lines>=2."""
        config = _make_config(score_norm_count=2)
        # Two parallel horizontal lines inside the rotated rect.
        frame = np.full((200, 400), 30, dtype=np.uint8)
        cv2.line(frame, (50, 95), (350, 95), 220, 2)
        cv2.line(frame, (50, 105), (350, 105), 220, 2)
        center = PixelPoint(x=200, y=100)
        path_vec = (1.0, 0.0)
        poly = rotated_polygon(center, path_vec, config)
        xs = poly[:, 0]; ys = poly[:, 1]
        roi = Rect(x=int(xs.min()), y=int(ys.min()),
                   w=int(xs.max() - xs.min()) + 1,
                   h=int(ys.max() - ys.min()) + 1)
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        # Two parallel long lines → discrete gate should fire.
        assert result.num_long_lines >= 2
        assert result.score >= 1.0 - 1e-6


# --- Legacy AABB-only tests kept as a safety net -----------------------------


@pytest.fixture
def legacy_config() -> DetectionConfig:
    return DetectionConfig(
        score_threshold=0.3,
        canny_low=50, canny_high=150,
        hough_threshold=20,
        hough_min_line_length=20,
        hough_max_line_gap=5,
        roi_padding=20,
        roi_along_px=120, roi_cross_px=40,
        use_adaptive_canny=False,
        angle_tolerance_deg=180.0,  # effectively no filter
        long_line_min_px=20.0,
        score_norm_count=1,
        use_rotated_mask=False,
        blur_kernel=0,
    )


class TestLegacyAABBPath:
    """Old-style call (no polygon / no path_vec) still exercises a valid path."""

    def test_blank_scores_zero(self, legacy_config: DetectionConfig):
        frame = np.zeros((200, 200), dtype=np.uint8)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, legacy_config)
        assert result.score == 0.0
        assert result.pixel_line is None

    def test_uniform_scores_zero(self, legacy_config: DetectionConfig):
        frame = np.full((200, 200), 128, dtype=np.uint8)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, legacy_config)
        assert result.score == 0.0

    def test_clear_line_scores_above_zero(self, legacy_config: DetectionConfig):
        frame = _frame_with_line()
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, legacy_config)
        assert result.score > 0.0
        assert result.num_long_lines >= 1

    def test_bgr_input(self, legacy_config: DetectionConfig):
        gray = _frame_with_line()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(bgr, roi, legacy_config)
        assert result.score > 0.0

    def test_zero_area_roi(self, legacy_config: DetectionConfig):
        frame = _frame_with_line()
        roi = Rect(x=100, y=100, w=0, h=0)
        result = detect(frame, roi, legacy_config)
        assert result.score == 0.0
        assert result.pixel_line is None

    def test_roi_fully_outside_frame(self, legacy_config: DetectionConfig):
        frame = _frame_with_line(width=200, height=200)
        roi = Rect(x=300, y=300, w=100, h=100)
        result = detect(frame, roi, legacy_config)
        assert result.score == 0.0

    def test_roi_partially_outside_frame(self, legacy_config: DetectionConfig):
        frame = _frame_with_line(width=200, height=200)
        roi = Rect(x=150, y=150, w=200, h=200)
        result = detect(frame, roi, legacy_config)
        assert isinstance(result, DetectionResult)

    def test_random_noise_does_not_crash(self, legacy_config: DetectionConfig):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, legacy_config)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0


# --- Temporal diff path ------------------------------------------------------


class TestTimestampExclusion:
    """Pixels inside timestamp_exclusion_region must not score."""

    def test_exclusion_region_suppresses_line_inside(self):
        """A bright line entirely inside the exclusion region scores zero."""
        # Full frame 400x200; exclusion covers rows 80:200, cols 200:400.
        config = _make_config(timestamp_exclusion_region=[80, 200, 200, 400])
        frame = np.full((200, 400), 30, dtype=np.uint8)
        # Bright horizontal line at row 150 — fully inside the exclusion region.
        cv2.line(frame, (210, 150), (390, 150), 220, 2)
        center = PixelPoint(x=300, y=150)
        path_vec = (1.0, 0.0)
        poly = rotated_polygon(center, path_vec, config)
        xs, ys = poly[:, 0], poly[:, 1]
        roi = Rect(
            x=int(xs.min()), y=int(ys.min()),
            w=int(xs.max() - xs.min()) + 1,
            h=int(ys.max() - ys.min()) + 1,
        )
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        assert result.score == 0.0

    def test_contrail_outside_exclusion_region_still_scores(self):
        """A contrail outside the exclusion region is unaffected."""
        # Exclusion covers only the far-right strip (cols 300-400).
        # The contrail is at row 100 spanning cols 50-250 — completely outside.
        config = _make_config(timestamp_exclusion_region=[0, 30, 300, 400])
        frame = np.full((200, 400), 30, dtype=np.uint8)
        cv2.line(frame, (50, 100), (250, 100), 220, 2)
        center = PixelPoint(x=150, y=100)
        path_vec = (1.0, 0.0)
        poly = rotated_polygon(center, path_vec, config)
        xs, ys = poly[:, 0], poly[:, 1]
        roi = Rect(
            x=int(xs.min()), y=int(ys.min()),
            w=int(xs.max() - xs.min()) + 1,
            h=int(ys.max() - ys.min()) + 1,
        )
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        assert result.score > 0.0
        assert result.pixel_line is not None

    def test_none_exclusion_region_has_no_effect(self):
        """timestamp_exclusion_region=None leaves the existing behaviour intact."""
        config = _make_config(timestamp_exclusion_region=None)
        frame = _frame_with_line(width=400, height=200,
                                  pt1=(50, 100), pt2=(350, 100), thickness=2)
        center = PixelPoint(x=200, y=100)
        path_vec = (1.0, 0.0)
        poly = rotated_polygon(center, path_vec, config)
        xs, ys = poly[:, 0], poly[:, 1]
        roi = Rect(
            x=int(xs.min()), y=int(ys.min()),
            w=int(xs.max() - xs.min()) + 1,
            h=int(ys.max() - ys.min()) + 1,
        )
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        assert result.score > 0.0


class TestTemporalDiff:
    """prev_frame should subtract static background; shape mismatch is silent."""

    def test_diff_suppresses_static_feature(self):
        config = _make_config(score_norm_count=1)
        # A horizontal line present in BOTH frames — diff should nearly zero it.
        static = _frame_with_line(width=400, height=200,
                                   pt1=(50, 100), pt2=(350, 100), thickness=2)
        center = PixelPoint(x=200, y=100)
        path_vec = (1.0, 0.0)
        poly = rotated_polygon(center, path_vec, config)
        xs = poly[:, 0]; ys = poly[:, 1]
        roi = Rect(x=int(xs.min()), y=int(ys.min()),
                   w=int(xs.max() - xs.min()) + 1,
                   h=int(ys.max() - ys.min()) + 1)
        result = detect(static, roi, config,
                        polygon=poly, path_vec=path_vec, prev_frame=static)
        # Identical frames → diff is all zeros → no edges → score 0.
        assert result.score == 0.0
        assert "diff" in result.method

    def test_shape_mismatch_falls_back(self):
        config = _make_config(score_norm_count=1)
        frame = _frame_with_line(width=400, height=200)
        mismatched = np.zeros((100, 100), dtype=np.uint8)
        center = PixelPoint(x=200, y=100)
        path_vec = (1.0, 0.0)
        poly = rotated_polygon(center, path_vec, config)
        xs = poly[:, 0]; ys = poly[:, 1]
        roi = Rect(x=int(xs.min()), y=int(ys.min()),
                   w=int(xs.max() - xs.min()) + 1,
                   h=int(ys.max() - ys.min()) + 1)
        result = detect(frame, roi, config,
                        polygon=poly, path_vec=path_vec,
                        prev_frame=mismatched)
        # prev_frame silently ignored; line still detected.
        assert result.score > 0.0
        assert "diff" not in result.method


# --- contrail_length_px and grow_contrail_length --------------------------------


class TestContrailLength:
    """contrail_length_px from detect() and grow_contrail_length() round-trips."""

    def _make_line_frame(self, width=600, height=200, line_len=400):
        """800×200 frame with a horizontal line of known length."""
        frame = np.full((height, width), 30, dtype=np.uint8)
        cx, cy = width // 2, height // 2
        x1 = cx - line_len // 2
        x2 = cx + line_len // 2
        cv2.line(frame, (x1, cy), (x2, cy), 220, 2)
        return frame, cx, cy

    def test_detect_returns_nonzero_length_for_clear_line(self):
        config = _make_config(roi_along_px=420, roi_cross_px=40, score_norm_count=1)
        frame, cx, cy = self._make_line_frame()
        path_vec = (1.0, 0.0)
        center = PixelPoint(x=cx, y=cy)
        poly = rotated_polygon(center, path_vec, config)
        xs, ys = poly[:, 0], poly[:, 1]
        roi = Rect(x=int(xs.min()), y=int(ys.min()),
                   w=int(xs.max()-xs.min())+1, h=int(ys.max()-ys.min())+1)
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        assert result.score > 0.0
        assert result.contrail_length_px > 0.0

    def test_blank_frame_gives_zero_length(self):
        config = _make_config()
        frame = np.full((200, 600), 30, dtype=np.uint8)
        path_vec = (1.0, 0.0)
        center = PixelPoint(x=300, y=100)
        poly = rotated_polygon(center, path_vec, config)
        xs, ys = poly[:, 0], poly[:, 1]
        roi = Rect(x=int(xs.min()), y=int(ys.min()),
                   w=int(xs.max()-xs.min())+1, h=int(ys.max()-ys.min())+1)
        result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
        assert result.score == 0.0
        assert result.contrail_length_px == 0.0

    def test_grow_finds_longer_extent_than_seed(self):
        """A 400-px line wider than the seed ROI (180 px) should grow to capture it."""
        config = _make_config(
            roi_along_px=120,
            roi_cross_px=40,
            roi_max_along_px=600,
            growth_step_px=20,
            score_norm_count=1,
            long_line_min_px=20.0,
        )
        frame, cx, cy = self._make_line_frame(width=600, height=200, line_len=400)
        path_vec = (1.0, 0.0)
        grown = grow_contrail_length(frame, (cx, cy), path_vec, config)
        # Seed ROI (120 px) cannot capture the 400-px line; growth should extend.
        assert grown > 150, f"Expected grown > 150 px, got {grown:.1f}"

    def test_grow_stops_at_contrail_edge_not_sky(self):
        """grow_contrail_length should not inflate when the line is narrower than max."""
        config = _make_config(
            roi_along_px=50,
            roi_cross_px=40,
            roi_max_along_px=800,
            growth_step_px=10,
            score_norm_count=1,
            long_line_min_px=20.0,
        )
        line_len = 100
        frame, cx, cy = self._make_line_frame(width=600, height=200, line_len=line_len)
        path_vec = (1.0, 0.0)
        grown = grow_contrail_length(frame, (cx, cy), path_vec, config)
        # Grown length must not exceed total frame width.
        assert grown < 600, f"Growth drifted into empty sky: {grown:.1f}"

    def test_grow_blank_frame_returns_zero(self):
        config = _make_config(
            roi_along_px=120, roi_max_along_px=600, growth_step_px=20
        )
        frame = np.full((200, 600), 30, dtype=np.uint8)
        grown = grow_contrail_length(frame, (300.0, 100.0), (1.0, 0.0), config)
        assert grown == 0.0
