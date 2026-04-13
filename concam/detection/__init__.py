"""Rotated-ROI contrail detector.

Rewrite of the original Canny+Hough module to match the techniques used by the
production sibling pipelines (camera-flight-overlay / groundcam_contrail_observatory):

  * Mask the **rotated** detection polygon, not an axis-aligned bbox, so Hough
    only sees the along-track strip.
  * Optional temporal frame-difference preprocessing (``prev_frame``). Per-frame
    fallback when no prior frame is cached — useful at 1 fps timelapse rates
    where frame-diff is mostly noise.
  * **Adaptive** Canny thresholds computed from percentiles of the masked pixel
    distribution rather than fixed 50 / 150.
  * **Angle-constrained** Hough scoring: lines must align with the flight-path
    vector within ``angle_tolerance_deg``.
  * Score is a continuous 0-1 value derived from the count of aligned long
    lines (normalised by ``score_norm_count``), preserving the pipeline's
    threshold-based aggregation contract while still exposing the discrete
    "≥2 long aligned lines" gate via ``result.num_long_lines``.

The legacy AABB-only / fixed-threshold path is kept as an opt-out via
``DetectionConfig.use_rotated_mask=False`` + ``use_adaptive_canny=False``; the
existing test_detection.py suite exercises that fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from concam.config import DetectionConfig
from concam.projection import Rect


@dataclass
class DetectionResult:
    """Result of running the contrail detector on one (frame, ROI) pair."""

    score: float  # 0-1 normalised detection confidence
    pixel_line: tuple[float, float, float, float] | None  # full-frame coords
    method: str
    num_long_lines: int = 0  # aligned lines ≥ long_line_min_px
    aligned_lines: int = 0  # aligned lines total (any length)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _angle180(dx: float, dy: float) -> float:
    """Line angle in [0, 180)."""
    return math.degrees(math.atan2(dy, dx)) % 180.0


def _angle_delta(a: float, b: float) -> float:
    """Unsigned acute angle between two orientations (both in [0, 180))."""
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def detect(
    frame: np.ndarray,
    roi: Rect,
    config: DetectionConfig,
    *,
    polygon: np.ndarray | None = None,
    path_vec: tuple[float, float] | None = None,
    prev_frame: np.ndarray | None = None,
) -> DetectionResult:
    """Run contrail detection within an oriented ROI.

    Args:
        frame: Full camera frame — (H, W) grayscale or (H, W, 3) BGR.
        roi: Axis-aligned bounding rectangle used as a cheap crop before
            rotated-mask application. Always required.
        config: DetectionConfig with thresholds / flags.
        polygon: Optional (4, 2) rotated polygon in full-frame pixel coords.
            When supplied and ``config.use_rotated_mask`` is True, Hough only
            sees edges inside this polygon.
        path_vec: Optional flight-path unit vector ``(vx, vy)``. When supplied,
            Hough lines are filtered to those within ``angle_tolerance_deg``
            of this direction.
        prev_frame: Optional prior frame for temporal-diff preprocessing. Must
            share the full frame's shape; otherwise silently ignored.

    Returns:
        DetectionResult with score in [0, 1], pixel_line in full-frame coords
        (or None), method string, and the aligned/long-aligned line counts.
    """
    method = "rotated_hough" if (
        polygon is not None and config.use_rotated_mask
    ) else "aabb_hough"

    gray = _to_gray(frame)
    h_fr, w_fr = gray.shape[:2]

    # Clip AABB to frame bounds.
    x1 = max(0, int(roi.x))
    y1 = max(0, int(roi.y))
    x2 = min(w_fr, int(roi.x + roi.w))
    y2 = min(h_fr, int(roi.y + roi.h))
    if x2 <= x1 or y2 <= y1:
        return DetectionResult(score=0.0, pixel_line=None, method=method)

    # Optional temporal diff at full-frame scale so the crop aligns.
    base = gray
    if prev_frame is not None:
        prev_gray = _to_gray(prev_frame)
        if prev_gray.shape == gray.shape:
            base = cv2.absdiff(gray, prev_gray)
            method = method + "_diff"

    # Light Gaussian blur to tame high-frequency noise.
    if config.blur_kernel and config.blur_kernel > 1:
        k = int(config.blur_kernel) | 1  # must be odd
        base = cv2.GaussianBlur(base, (k, k), 0)

    crop = base[y1:y2, x1:x2]
    if crop.size == 0:
        return DetectionResult(score=0.0, pixel_line=None, method=method)

    # Build the rotated-polygon mask in crop-local coords (or a full-crop mask
    # if we're falling back to AABB-only).
    ch, cw = crop.shape[:2]
    if polygon is not None and config.use_rotated_mask:
        local_poly = polygon.copy().astype(np.float32)
        local_poly[:, 0] -= x1
        local_poly[:, 1] -= y1
        local_poly_int = np.round(local_poly).astype(np.int32)
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(mask, [local_poly_int], 255)
        masked_values = crop[mask > 0]
    else:
        mask = None
        masked_values = crop.reshape(-1)

    if masked_values.size == 0:
        return DetectionResult(score=0.0, pixel_line=None, method=method)

    # Decide Canny thresholds.
    if config.use_adaptive_canny:
        p_hi = float(np.percentile(masked_values, config.canny_percentile_high))
        canny_high = max(int(round(p_hi)), int(config.canny_min_high))
        canny_low = max(1, int(round(canny_high * config.canny_low_ratio)))
        # Pixel floor: zero-out anything below the p_low percentile so Canny
        # doesn't respond to low-contrast sky texture.
        p_lo = float(np.percentile(masked_values, config.canny_percentile_low))
        floor = int(round(p_lo))
    else:
        canny_high = int(config.canny_high)
        canny_low = int(config.canny_low)
        floor = None

    # Apply the floor and optional mask before Canny.
    crop_for_canny = crop
    if mask is not None:
        crop_for_canny = cv2.bitwise_and(crop_for_canny, crop_for_canny, mask=mask)
    if floor is not None and floor > 0:
        _, crop_for_canny = cv2.threshold(crop_for_canny, floor, 255, cv2.THRESH_TOZERO)

    edges = cv2.Canny(crop_for_canny, canny_low, canny_high)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(config.hough_threshold),
        minLineLength=int(config.hough_min_line_length),
        maxLineGap=int(config.hough_max_line_gap),
    )
    if lines is None or len(lines) == 0:
        return DetectionResult(score=0.0, pixel_line=None, method=method)

    # Angle filter against the flight-path vector.
    tol = float(config.angle_tolerance_deg)
    min_long = float(config.long_line_min_px)
    if path_vec is not None:
        path_angle = _angle180(float(path_vec[0]), float(path_vec[1]))
    else:
        path_angle = None

    aligned: list[tuple[int, int, int, int, float]] = []
    long_aligned: list[tuple[int, int, int, int, float]] = []
    for ln in lines:
        lx1, ly1, lx2, ly2 = (int(v) for v in ln[0])
        length = float(math.hypot(lx2 - lx1, ly2 - ly1))
        if path_angle is not None:
            line_angle = _angle180(float(lx2 - lx1), float(ly2 - ly1))
            if _angle_delta(line_angle, path_angle) > tol:
                continue
        aligned.append((lx1, ly1, lx2, ly2, length))
        if length >= min_long:
            long_aligned.append((lx1, ly1, lx2, ly2, length))

    if not aligned:
        return DetectionResult(score=0.0, pixel_line=None, method=method)

    # Scoring contract: count of aligned long lines -> [0, 1] via score_norm_count.
    # When no long-aligned lines exist we still report the longest aligned
    # fragment (the detector "saw something" even if below the long-line gate),
    # but score scales from num_long_lines only so the discrete ≥2 gate is
    # recoverable as score >= 2 / score_norm_count.
    norm = max(1, int(config.score_norm_count))
    score = min(1.0, len(long_aligned) / float(norm))

    best_bucket = long_aligned if long_aligned else aligned
    best = max(best_bucket, key=lambda t: t[4])
    bx1, by1, bx2, by2, _ = best
    pixel_line = (
        float(bx1 + x1),
        float(by1 + y1),
        float(bx2 + x1),
        float(by2 + y1),
    )

    return DetectionResult(
        score=score,
        pixel_line=pixel_line,
        method=method,
        num_long_lines=len(long_aligned),
        aligned_lines=len(aligned),
    )
