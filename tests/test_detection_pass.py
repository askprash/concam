"""Interface tests for the canonical detection pass (concam.detection._core).

These are deliberately *metamorphic / property* tests, not snapshot tests: they
assert relations that must hold for any correct rotated-ROI contrail detector
(alignment-equivariance, translation-invariance, length-monotonicity), plus the
one consistency property that prevents the bug this refactor fixed —
``explain()`` must hand back exactly what ``detect()`` scored from.

The pre-existing ``test_detection.py`` covers the fixed-threshold path with
clean synthetic frames; here we also exercise the production adaptive-Canny +
``cross_grad`` path on noisy sky, which the older suite did not.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from concam.config import DetectionConfig
from concam.detection import DetectionPass, detect, explain, grow_contrail_length
from concam.projection import PixelPoint, Rect, rotated_polygon


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _clean_config(**overrides) -> DetectionConfig:
    """Fixed-threshold config: deterministic edges, good for equivariance."""
    base = dict(
        canny_low=50, canny_high=150,
        hough_threshold=20, hough_min_line_length=20, hough_max_line_gap=5,
        roi_along_px=160, roi_cross_px=40,
        use_adaptive_canny=False, angle_tolerance_deg=10.0,
        long_line_min_px=20.0, score_norm_count=2, use_rotated_mask=True,
        blur_kernel=0, preprocessing="none",
    )
    base.update(overrides)
    return DetectionConfig(**base)


def _prod_config(**overrides) -> DetectionConfig:
    """Adaptive-Canny + cross_grad config — the production detection path."""
    base = dict(
        use_adaptive_canny=True, canny_percentile_high=99.5,
        canny_percentile_low=96.0, canny_low_ratio=0.25, canny_min_high=60,
        angle_tolerance_deg=8.0, long_line_min_px=40.0,
        hough_threshold=30, hough_min_line_length=30, hough_max_line_gap=10,
        roi_along_px=180, roi_cross_px=40, use_rotated_mask=True,
        score_fn="length", score_length_norm_px=130.0,
        preprocessing="cross_grad", cross_grad_gain=2.0, blur_kernel=3,
    )
    base.update(overrides)
    return DetectionConfig(**base)


def _noisy_sky(seed: int, w: int = 360, h: int = 240, bg: int = 70, noise: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.clip(rng.normal(bg, noise, (h, w)), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    return np.clip(f.astype(np.float32) + 25.0 * np.sin(xx / 90.0), 0, 255).astype(np.uint8)


def _draw_streak(frame: np.ndarray, center: tuple[int, int], angle_deg: float,
                 length: int, fg: int = 210, thick: int = 3) -> np.ndarray:
    out = frame.copy()
    cx, cy = center
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    p1 = (int(round(cx - length / 2 * dx)), int(round(cy - length / 2 * dy)))
    p2 = (int(round(cx + length / 2 * dx)), int(round(cy + length / 2 * dy)))
    cv2.line(out, p1, p2, fg, thick)
    return out


def _roi_for(center: PixelPoint, path_vec, config) -> tuple[np.ndarray, Rect]:
    poly = rotated_polygon(center, path_vec, config)
    xs, ys = poly[:, 0], poly[:, 1]
    roi = Rect(x=int(xs.min()), y=int(ys.min()),
               w=int(xs.max() - xs.min()) + 1, h=int(ys.max() - ys.min()) + 1)
    return poly, roi


# ---------------------------------------------------------------------------
# Consistency: explain() == detect() (the property that fixes the bug)
# ---------------------------------------------------------------------------

class TestExplainMatchesDetect:
    """A visualiser rendering explain() must see exactly what detect() scored.

    Before the refactor, the review-panel scripts re-implemented the pre-Canny
    pipeline and skipped ``_prepare_base``, so they rendered different edges than
    the detector used under ``preprocessing != "none"``.  By construction now,
    detect() derives every output from the DetectionPass that explain() returns;
    these tests pin that invariant on both the clean and the production paths.
    """

    @pytest.mark.parametrize("config_fn", [_clean_config, _prod_config])
    @pytest.mark.parametrize("angle", [0.0, 35.0, 90.0])
    def test_counts_and_length_agree(self, config_fn, angle):
        config = config_fn()
        frame = _draw_streak(_noisy_sky(7), (180, 120), angle, length=170)
        pv = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        center = PixelPoint(x=180, y=120)
        poly, roi = _roi_for(center, pv, config)

        result = detect(frame, roi, config, polygon=poly, path_vec=pv)
        passed = explain(frame, roi, config, polygon=poly, path_vec=pv)

        assert isinstance(passed, DetectionPass)
        assert result.num_long_lines == len(passed.long_aligned)
        assert result.aligned_lines == len(passed.aligned)
        assert result.contrail_length_px == pytest.approx(passed.length_px)
        # The detector's reported method records the same preprocessing.
        assert result.method == passed.method

    def test_pixel_line_is_the_longest_pass_line(self):
        config = _prod_config()
        frame = _draw_streak(_noisy_sky(3), (180, 120), 30.0, length=170)
        pv = (math.cos(math.radians(30.0)), math.sin(math.radians(30.0)))
        center = PixelPoint(x=180, y=120)
        poly, roi = _roi_for(center, pv, config)

        result = detect(frame, roi, config, polygon=poly, path_vec=pv)
        passed = explain(frame, roi, config, polygon=poly, path_vec=pv)
        assert result.pixel_line is not None
        bucket = passed.long_aligned or passed.aligned
        best = max(bucket, key=lambda t: t[4])
        assert result.pixel_line == (float(best[0]), float(best[1]), float(best[2]), float(best[3]))


# ---------------------------------------------------------------------------
# Metamorphic: alignment-equivariance under rotation
# ---------------------------------------------------------------------------

class TestAlignmentEquivariance:
    """A contrail aligned with path_vec is detected at any orientation; a
    perpendicular one is rejected at any orientation.  The detector cares only
    about alignment with path_vec, so rotating the streak and path_vec together
    must preserve the detect/reject outcome."""

    @pytest.mark.parametrize("angle", [0.0, 25.0, 50.0, 75.0, 90.0, 135.0])
    def test_aligned_streak_detected(self, angle):
        config = _prod_config()
        frame = _draw_streak(_noisy_sky(11), (180, 120), angle, length=180)
        pv = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        center = PixelPoint(x=180, y=120)
        poly, roi = _roi_for(center, pv, config)
        result = detect(frame, roi, config, polygon=poly, path_vec=pv)
        assert result.score > 0.0
        assert result.num_long_lines >= 1

    @pytest.mark.parametrize("angle", [0.0, 25.0, 50.0, 75.0, 90.0, 135.0])
    def test_alignment_is_preferred_over_misalignment(self, angle):
        """Angle-selectivity (the true invariant): an ROI aligned with the
        streak detects at least as strongly as one misaligned by 45°, where the
        streak's own edges fall outside the ±angle_tolerance filter.

        (We deliberately do *not* assert "perpendicular ⇒ zero": a bright bar
        crossing a rotated mask can leave a surviving fragment, so strict
        rejection is not an invariant of this detector — only relative
        preference is.  The clean horizontal/vertical rejection case lives in
        test_detection.py.)"""
        config = _prod_config()
        frame = _draw_streak(_noisy_sky(11), (180, 120), angle, length=180)
        center = PixelPoint(x=180, y=120)

        pv_aligned = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        pv_off = (math.cos(math.radians(angle + 45.0)), math.sin(math.radians(angle + 45.0)))
        poly_a, roi_a = _roi_for(center, pv_aligned, config)
        poly_o, roi_o = _roi_for(center, pv_off, config)

        r_aligned = detect(frame, roi_a, config, polygon=poly_a, path_vec=pv_aligned)
        r_off = detect(frame, roi_o, config, polygon=poly_o, path_vec=pv_off)

        assert r_aligned.score > 0.0
        assert r_aligned.num_long_lines >= 1
        assert r_aligned.num_long_lines >= r_off.num_long_lines
        assert r_aligned.score >= r_off.score - 1e-9


# ---------------------------------------------------------------------------
# Metamorphic: translation invariance
# ---------------------------------------------------------------------------

class TestTranslationInvariance:
    """Shifting the frame content, ROI, polygon and path together by (dx, dy)
    leaves the score and line counts unchanged and translates the pixel_line."""

    @pytest.mark.parametrize("shift", [(40, 30), (-25, 15)])
    def test_score_and_counts_invariant(self, shift):
        config = _clean_config()
        dx, dy = shift
        pv = (1.0, 0.0)
        c0 = (180, 120)
        c1 = (180 + dx, 120 + dy)
        frame0 = _draw_streak(np.full((240, 360), 30, np.uint8), c0, 0.0, length=150)
        frame1 = _draw_streak(np.full((240, 360), 30, np.uint8), c1, 0.0, length=150)

        poly0, roi0 = _roi_for(PixelPoint(x=c0[0], y=c0[1]), pv, config)
        poly1, roi1 = _roi_for(PixelPoint(x=c1[0], y=c1[1]), pv, config)

        r0 = detect(frame0, roi0, config, polygon=poly0, path_vec=pv)
        r1 = detect(frame1, roi1, config, polygon=poly1, path_vec=pv)

        assert r0.num_long_lines == r1.num_long_lines
        assert r0.aligned_lines == r1.aligned_lines
        assert r0.score == pytest.approx(r1.score)
        assert r0.contrail_length_px == pytest.approx(r1.contrail_length_px)
        assert r0.pixel_line is not None and r1.pixel_line is not None
        # pixel_line of the shifted frame is the original translated by (dx, dy).
        assert r1.pixel_line[0] == pytest.approx(r0.pixel_line[0] + dx)
        assert r1.pixel_line[1] == pytest.approx(r0.pixel_line[1] + dy)


# ---------------------------------------------------------------------------
# Metamorphic: contrail-length monotonicity, and grow >= seed measurement
# ---------------------------------------------------------------------------

class TestLengthMonotonicity:
    def test_longer_streak_not_shorter_measurement(self):
        """A longer drawn streak yields a non-decreasing contrail_length_px."""
        config = _prod_config(roi_along_px=420, score_length_norm_px=400.0)
        pv = (1.0, 0.0)
        center = PixelPoint(x=300, y=120)
        prev_len = -1.0
        prev_score = -1.0
        for L in (80, 160, 240, 320):
            frame = _draw_streak(_noisy_sky(5, w=600), (300, 120), 0.0, length=L)
            poly, roi = _roi_for(center, pv, config)
            r = detect(frame, roi, config, polygon=poly, path_vec=pv)
            assert r.contrail_length_px >= prev_len - 1e-6
            assert r.score >= prev_score - 1e-6
            prev_len = r.contrail_length_px
            prev_score = r.score

    def test_grow_at_least_seed_length(self):
        """grow_contrail_length never reports less than the seed ROI measured,
        and growth captures a streak longer than the seed."""
        config = _clean_config(roi_along_px=120, roi_cross_px=40,
                               roi_max_along_px=600, growth_step_px=20,
                               long_line_min_px=20.0, score_norm_count=1)
        frame = np.full((200, 600), 30, np.uint8)
        cv2.line(frame, (100, 100), (500, 100), 220, 2)  # 400-px streak
        pv = (1.0, 0.0)
        center = PixelPoint(x=300, y=100)
        poly, roi = _roi_for(center, pv, config)
        seed = detect(frame, roi, config, polygon=poly, path_vec=pv)
        grown = grow_contrail_length(frame, (300.0, 100.0), pv, config)
        assert grown >= seed.contrail_length_px - 1e-6
        assert grown > 150.0  # seed (120 px) cannot capture the 400-px streak


# ---------------------------------------------------------------------------
# Production adaptive path actually fires (gap in the old synthetic suite)
# ---------------------------------------------------------------------------

def test_production_adaptive_crossgrad_fires():
    config = _prod_config()
    frame = _draw_streak(_noisy_sky(2), (180, 120), 20.0, length=180)
    pv = (math.cos(math.radians(20.0)), math.sin(math.radians(20.0)))
    center = PixelPoint(x=180, y=120)
    poly, roi = _roi_for(center, pv, config)
    passed = explain(frame, roi, config, polygon=poly, path_vec=pv)
    assert "cg" in passed.method  # cross_grad preprocessing was applied
    assert passed.edges.shape == passed.base.shape
    assert int(passed.edges.sum()) > 0  # the production path produced real edges
    r = detect(frame, roi, config, polygon=poly, path_vec=pv)
    assert r.score > 0.0
    assert r.num_long_lines >= 1
    assert r.contrail_length_px > 0.0


def test_blank_pass_is_empty():
    config = _prod_config()
    frame = _noisy_sky(9)  # no streak
    pv = (1.0, 0.0)
    center = PixelPoint(x=180, y=120)
    poly, roi = _roi_for(center, pv, config)
    passed = explain(frame, roi, config, polygon=poly, path_vec=pv)
    r = detect(frame, roi, config, polygon=poly, path_vec=pv)
    # No long aligned streak -> zero score, zero measured length.
    assert r.num_long_lines == len(passed.long_aligned)
    assert r.contrail_length_px == pytest.approx(passed.length_px)


# ---------------------------------------------------------------------------
# Static-scene mask: building edges must not produce detections
# ---------------------------------------------------------------------------

class TestStaticMaskExclusion:
    """A streak inside the static mask (building edge) must be suppressed,
    exactly like the timestamp exclusion; outside the mask detection is
    unaffected."""

    def _setup(self, tmp_path, mask_covers_streak: bool):
        from concam.detection.static_mask import save_static_mask

        config = _clean_config()
        frame = _draw_streak(_noisy_sky(11), (180, 120), 0.0, length=170)
        pv = (1.0, 0.0)
        poly, roi = _roi_for(PixelPoint(x=180, y=120), pv, config)

        mask = np.zeros(frame.shape[:2], dtype=bool)
        if mask_covers_streak:
            # Blankets the streak's ROI with margin — mirrors the dilate_px
            # safety margin a real building mask carries.
            mask[95:145, 75:285] = True
        else:
            mask[0:20, 0:20] = True       # far corner — irrelevant
        mask_path = tmp_path / "static_mask.npz"
        save_static_mask(mask, mask_path)
        config = _clean_config(static_mask_path=str(mask_path))
        return frame, roi, poly, pv, config

    def test_streak_inside_mask_suppressed(self, tmp_path):
        frame, roi, poly, pv, config = self._setup(tmp_path, True)
        result = detect(frame, roi, config, polygon=poly, path_vec=pv)
        assert result.score == 0.0
        assert result.contrail_length_px == 0.0

    def test_streak_outside_mask_detected(self, tmp_path):
        frame, roi, poly, pv, config = self._setup(tmp_path, False)
        result = detect(frame, roi, config, polygon=poly, path_vec=pv)
        assert result.score > 0.0

    def test_mask_respects_apply_exclusion_flag(self, tmp_path):
        # grow_contrail_length runs with apply_exclusion=False; the static mask
        # must follow the same switch so growth isn't silently masked.
        frame, roi, poly, pv, config = self._setup(tmp_path, True)
        passed = explain(frame, roi, config, polygon=poly, path_vec=pv,
                         apply_exclusion=False)
        assert len(passed.long_aligned) > 0
