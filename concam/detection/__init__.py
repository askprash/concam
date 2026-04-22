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
  * ``contrail_length_px``: along-track pixel span of the detected contrail
    streak, measured as the projection-onto-path_vec span of the endpoints of
    all aligned long Hough lines.
  * ``grow_contrail_length``: adaptive ROI growth that iteratively widens the
    along-track axis until the aligned-long-line count stops increasing, then
    returns the measured contrail pixel extent.

The legacy AABB-only / fixed-threshold path is kept as an opt-out via
``DetectionConfig.use_rotated_mask=False`` + ``use_adaptive_canny=False``; the
existing test_detection.py suite exercises that fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    # Along-track pixel span of aligned long lines (0.0 when no lines found).
    contrail_length_px: float = 0.0


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


def _prepare_base(
    frame: np.ndarray,
    config: DetectionConfig,
    *,
    path_vec: tuple[float, float] | None = None,
    prev_frame: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Apply temporal diff and spatial preprocessing to produce a detection base.

    Returns ``(base_gray, method_suffix)`` where ``method_suffix`` is appended
    to the detector method name to record which preprocessing was applied.
    """
    gray = _to_gray(frame)
    base = gray
    method_suffix = ""

    # Optional temporal diff at full-frame scale.
    if prev_frame is not None:
        prev_gray = _to_gray(prev_frame)
        if prev_gray.shape == gray.shape:
            base = cv2.absdiff(gray, prev_gray)
            method_suffix += "_diff"

    # Optional spatial pre-processing.
    preprocessing = getattr(config, "preprocessing", "none")
    if preprocessing == "local_contrast":
        sigma = float(getattr(config, "local_contrast_sigma", 25.0))
        bg = cv2.GaussianBlur(base.astype(np.float32), (0, 0), sigma)
        lc = base.astype(np.float32) - bg + 128.0
        base = np.clip(lc, 0, 255).astype(np.uint8)
        method_suffix += "_lc"
    elif preprocessing == "cross_grad" and path_vec is not None:
        px_v, py_v = float(path_vec[0]), float(path_vec[1])
        perp_x, perp_y = -py_v, px_v   # 90° rotation, unit length
        sx = cv2.Sobel(base.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(base.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        cross = np.abs(sx * perp_x + sy * perp_y)
        gain = float(getattr(config, "cross_grad_gain", 2.0))
        base = np.clip(cross * gain, 0, 255).astype(np.uint8)
        method_suffix += "_cg"

    # Light Gaussian blur to tame high-frequency noise.
    if config.blur_kernel and config.blur_kernel > 1:
        k = int(config.blur_kernel) | 1
        base = cv2.GaussianBlur(base, (k, k), 0)

    return base, method_suffix


def _canny_and_hough_in_polygon(
    base: np.ndarray,
    polygon: np.ndarray,
    config: DetectionConfig,
    x1: int,
    y1: int,
    w_fr: int,
    h_fr: int,
) -> tuple[list[tuple[int, int, int, int, float]], list[tuple[int, int, int, int, float]]]:
    """Run Canny + Hough within a rotated polygon crop.

    ``polygon`` must be in full-frame coords.  ``x1``, ``y1`` are the top-left
    corner of the crop already extracted from ``base`` (``base[y1:y2, x1:x2]``).

    Returns ``(aligned, long_aligned)`` line lists where each element is
    ``(lx1, ly1, lx2, ly2, length)`` in **full-frame** coordinates.
    Only lines within ``angle_tolerance_deg`` of the implicit path_vec are
    returned; the caller is responsible for setting ``path_vec=None`` → angle
    filter disabled by passing ``config.angle_tolerance_deg=90``.
    """
    x2 = min(w_fr, int(round(float(np.max(polygon[:, 0])))) + 1)
    y2 = min(h_fr, int(round(float(np.max(polygon[:, 1])))) + 1)
    crop = base[y1:y2, x1:x2]
    if crop.size == 0:
        return [], []

    ch, cw = crop.shape[:2]

    # Build rotated-polygon mask in crop-local coords.
    local_poly = polygon.copy().astype(np.float32)
    local_poly[:, 0] -= x1
    local_poly[:, 1] -= y1
    local_poly_int = np.round(local_poly).astype(np.int32)
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly_int], 255)
    masked_values = crop[mask > 0]

    if masked_values.size == 0:
        return [], []

    # Adaptive Canny thresholds.
    if config.use_adaptive_canny:
        p_hi = float(np.percentile(masked_values, config.canny_percentile_high))
        canny_high = max(int(round(p_hi)), int(config.canny_min_high))
        canny_low = max(1, int(round(canny_high * config.canny_low_ratio)))
        p_lo = float(np.percentile(masked_values, config.canny_percentile_low))
        floor = int(round(p_lo))
    else:
        canny_high = int(config.canny_high)
        canny_low = int(config.canny_low)
        floor = None

    crop_for_canny = cv2.bitwise_and(crop, crop, mask=mask)
    if floor is not None and floor > 0:
        _, crop_for_canny = cv2.threshold(crop_for_canny, floor, 255, cv2.THRESH_TOZERO)

    edges = cv2.Canny(crop_for_canny, canny_low, canny_high)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    lines_raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(config.hough_threshold),
        minLineLength=int(config.hough_min_line_length),
        maxLineGap=int(config.hough_max_line_gap),
    )
    return lines_raw, (x1, y1)


def _filter_lines(
    lines_raw,
    x1: int,
    y1: int,
    path_angle: float | None,
    config: DetectionConfig,
) -> tuple[list, list]:
    """Filter raw HoughLinesP output to aligned / long-aligned subsets.

    Returns ``(aligned, long_aligned)`` in full-frame coordinates.
    """
    if lines_raw is None or len(lines_raw) == 0:
        return [], []

    tol = float(config.angle_tolerance_deg)
    min_long = float(config.long_line_min_px)
    aligned = []
    long_aligned = []
    for ln in lines_raw:
        lx1, ly1, lx2, ly2 = (int(v) for v in ln[0])
        length = float(math.hypot(lx2 - lx1, ly2 - ly1))
        if path_angle is not None:
            line_angle = _angle180(float(lx2 - lx1), float(ly2 - ly1))
            if _angle_delta(line_angle, path_angle) > tol:
                continue
        aligned.append((lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1, length))
        if length >= min_long:
            long_aligned.append((lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1, length))
    return aligned, long_aligned


def _line_span_px(
    long_aligned: list[tuple[int, int, int, int, float]],
    path_vec: tuple[float, float],
) -> float:
    """Along-track span of long-aligned Hough line endpoints (full-frame coords)."""
    if not long_aligned:
        return 0.0
    vx, vy = float(path_vec[0]), float(path_vec[1])
    projections = (
        [vx * lx1 + vy * ly1 for lx1, ly1, lx2, ly2, _ in long_aligned]
        + [vx * lx2 + vy * ly2 for lx1, ly1, lx2, ly2, _ in long_aligned]
    )
    return float(max(projections) - min(projections))


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
        (or None), method string, the aligned/long-aligned line counts, and
        ``contrail_length_px`` (along-track span of aligned long lines).
    """
    method = "rotated_hough" if (
        polygon is not None and config.use_rotated_mask
    ) else "aabb_hough"

    base, method_suffix = _prepare_base(
        frame, config, path_vec=path_vec, prev_frame=prev_frame
    )
    method = method + method_suffix
    h_fr, w_fr = base.shape[:2]

    # Clip AABB to frame bounds.
    x1 = max(0, int(roi.x))
    y1 = max(0, int(roi.y))
    x2 = min(w_fr, int(roi.x + roi.w))
    y2 = min(h_fr, int(roi.y + roi.h))
    if x2 <= x1 or y2 <= y1:
        return DetectionResult(score=0.0, pixel_line=None, method=method)

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

    # Apply the floor, rotated mask, and timestamp exclusion region before Canny.
    crop_for_canny = crop
    if mask is not None:
        crop_for_canny = cv2.bitwise_and(crop_for_canny, crop_for_canny, mask=mask)
    if floor is not None and floor > 0:
        _, crop_for_canny = cv2.threshold(crop_for_canny, floor, 255, cv2.THRESH_TOZERO)

    # Zero out the burned-in timestamp region so its glyph edges cannot produce
    # aligned Hough lines.  The exclusion region is in full-frame coords
    # [y0, y1, x0, x1]; translate to crop-local coordinates before zeroing.
    excl = getattr(config, "timestamp_exclusion_region", None)
    if excl is not None and len(excl) == 4:
        ey0, ey1, ex0, ex1 = int(excl[0]), int(excl[1]), int(excl[2]), int(excl[3])
        # Translate to crop-local row/col indices.
        ry0 = max(0, ey0 - y1)
        ry1 = min(ch, ey1 - y1)
        cx0 = max(0, ex0 - x1)
        cx1 = min(cw, ex1 - x1)
        if ry1 > ry0 and cx1 > cx0:
            if crop_for_canny is crop:
                crop_for_canny = crop_for_canny.copy()
            crop_for_canny[ry0:ry1, cx0:cx1] = 0

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

    best_bucket = long_aligned if long_aligned else aligned
    best = max(best_bucket, key=lambda t: t[4])
    bx1, by1, bx2, by2, _ = best
    pixel_line = (
        float(bx1 + x1),
        float(by1 + y1),
        float(bx2 + x1),
        float(by2 + y1),
    )

    # Along-track pixel span from the seed ROI.  Requires path_vec; falls back
    # to the length of the single best Hough line when path_vec is unavailable.
    if path_vec is not None:
        length_px = _line_span_px(
            [(la[0]+x1, la[1]+y1, la[2]+x1, la[3]+y1, la[4]) for la in long_aligned],
            path_vec,
        )
    elif long_aligned:
        length_px = max(la[4] for la in long_aligned)
    else:
        length_px = 0.0

    # Scoring contract.  "length" (default) uses the along-track contrail pixel
    # span normalised by ``score_length_norm_px`` — continuous, does not
    # saturate on strong contrails.  "count" reproduces the legacy discrete
    # score that caused ~63% of April-8 episode peaks to hit 1.0.
    score_fn = getattr(config, "score_fn", "length")
    if score_fn == "count":
        norm = max(1, int(config.score_norm_count))
        score = min(1.0, len(long_aligned) / float(norm))
    else:
        norm_px = float(getattr(config, "score_length_norm_px", 130.0))
        if norm_px <= 0.0:
            norm_px = 130.0
        score = min(1.0, length_px / norm_px)

    return DetectionResult(
        score=score,
        pixel_line=pixel_line,
        method=method,
        num_long_lines=len(long_aligned),
        aligned_lines=len(aligned),
        contrail_length_px=length_px,
    )


def grow_contrail_length(
    frame: np.ndarray,
    center_xy: tuple[float, float],
    path_vec: tuple[float, float],
    config: DetectionConfig,
    *,
    prev_frame: np.ndarray | None = None,
) -> float:
    """Measure contrail pixel length by adaptive along-track ROI growth.

    Starting from ``config.roi_along_px``, grows the rotated rectangle along
    the flight-path vector (both directions) by ``config.growth_step_px`` per
    iteration until the count of aligned long Hough lines stops increasing or
    the total along-track dimension reaches ``config.roi_max_along_px``.

    Returns the along-track projection span of all aligned long Hough line
    endpoints at convergence, in full-frame pixel units.  Returns 0.0 when no
    aligned lines are found in the seed polygon.
    """
    base, _ = _prepare_base(frame, config, path_vec=path_vec, prev_frame=prev_frame)
    h_fr, w_fr = base.shape[:2]

    cx, cy = float(center_xy[0]), float(center_xy[1])
    vx, vy = float(path_vec[0]), float(path_vec[1])
    nx, ny = -vy, vx  # cross-track unit vector
    cross_half = float(getattr(config, "roi_cross_px", 40)) / 2.0
    path_angle = _angle180(vx, vy)

    growth_step = int(getattr(config, "growth_step_px", 20))
    max_along = int(getattr(config, "roi_max_along_px", 600))
    start_along = int(getattr(config, "roi_along_px", 180))

    best_length_px = 0.0
    prev_span = -1.0  # sentinel so first iteration never breaks early

    current_along = start_along
    while current_along <= max_along:
        along_half = current_along / 2.0

        # Rotated polygon at current along-track size.
        corners = np.array(
            [
                [cx - along_half * vx - cross_half * nx, cy - along_half * vy - cross_half * ny],
                [cx + along_half * vx - cross_half * nx, cy + along_half * vy - cross_half * ny],
                [cx + along_half * vx + cross_half * nx, cy + along_half * vy + cross_half * ny],
                [cx - along_half * vx + cross_half * nx, cy - along_half * vy + cross_half * ny],
            ],
            dtype=np.float32,
        )

        # AABB for the crop.
        xs = corners[:, 0]
        ys = corners[:, 1]
        bx1 = max(0, int(math.floor(float(xs.min()))))
        by1 = max(0, int(math.floor(float(ys.min()))))
        bx2 = min(w_fr, int(math.ceil(float(xs.max()))))
        by2 = min(h_fr, int(math.ceil(float(ys.max()))))
        if bx2 <= bx1 or by2 <= by1:
            break

        crop = base[by1:by2, bx1:bx2]
        ch, cw = crop.shape[:2]

        # Rotated mask in crop-local coords.
        local_poly = corners.copy()
        local_poly[:, 0] -= bx1
        local_poly[:, 1] -= by1
        local_poly_int = np.round(local_poly).astype(np.int32)
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(mask, [local_poly_int], 255)
        masked_values = crop[mask > 0]

        if masked_values.size == 0:
            break

        # Adaptive Canny.
        if config.use_adaptive_canny:
            p_hi = float(np.percentile(masked_values, config.canny_percentile_high))
            canny_high = max(int(round(p_hi)), int(config.canny_min_high))
            canny_low = max(1, int(round(canny_high * config.canny_low_ratio)))
            p_lo = float(np.percentile(masked_values, config.canny_percentile_low))
            floor = int(round(p_lo))
        else:
            canny_high = int(config.canny_high)
            canny_low = int(config.canny_low)
            floor = None

        crop_c = cv2.bitwise_and(crop, crop, mask=mask)
        if floor is not None and floor > 0:
            _, crop_c = cv2.threshold(crop_c, floor, 255, cv2.THRESH_TOZERO)

        edges = cv2.Canny(crop_c, canny_low, canny_high)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        lines_raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(config.hough_threshold),
            minLineLength=int(config.hough_min_line_length),
            maxLineGap=int(config.hough_max_line_gap),
        )

        # Angle-filter to aligned long lines in full-frame coords.
        tol = float(config.angle_tolerance_deg)
        min_long = float(config.long_line_min_px)
        long_lines: list[tuple[int, int, int, int]] = []
        if lines_raw is not None:
            for ln in lines_raw:
                lx1f, ly1f, lx2f, ly2f = (int(v) for v in ln[0])
                length = float(math.hypot(lx2f - lx1f, ly2f - ly1f))
                line_angle = _angle180(float(lx2f - lx1f), float(ly2f - ly1f))
                if _angle_delta(line_angle, path_angle) <= tol and length >= min_long:
                    long_lines.append((lx1f + bx1, ly1f + by1, lx2f + bx1, ly2f + by1))

        count = len(long_lines)

        # Compute the along-track span of all aligned long line endpoints.
        if count > 0:
            projections = (
                [vx * lx1 + vy * ly1 for lx1, ly1, lx2, ly2 in long_lines]
                + [vx * lx2 + vy * ly2 for lx1, ly1, lx2, ly2 in long_lines]
            )
            span = float(max(projections) - min(projections))
        else:
            span = 0.0

        # Stop growing when the span stops increasing.  Using span rather than
        # count handles the common case where a single Hough segment spans more
        # of the contrail as the ROI widens, without the count changing.
        if span <= prev_span and current_along > start_along:
            break

        best_length_px = max(best_length_px, span)
        prev_span = span
        current_along += 2 * growth_step

    return best_length_px
