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

Architecture
------------
The edge+Hough math lives in exactly one place — :func:`run_detection_pass` in
:mod:`concam.detection._core`, which returns a :class:`DetectionPass` carrying
every intermediate array.  :func:`detect` scores from it, :func:`explain`
returns it for visualisation, and :func:`grow_contrail_length` grows from it.
Render the same :class:`DetectionPass` you scored and a visualiser cannot drift
from the production detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from concam.config import DetectionConfig
from concam.projection import Rect

from concam.detection._core import (
    DetectionPass,
    _pass_on_base,
    _prepare_base,
    run_detection_pass,
)

__all__ = [
    "DetectionResult",
    "DetectionPass",
    "detect",
    "explain",
    "grow_contrail_length",
    "run_detection_pass",
]


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
    p = run_detection_pass(
        frame, roi, config,
        polygon=polygon,
        path_vec=path_vec,
        prev_frame=prev_frame,
        apply_exclusion=True,
    )

    if not p.aligned:
        return DetectionResult(
            score=0.0,
            pixel_line=None,
            method=p.method,
            num_long_lines=len(p.long_aligned),
            aligned_lines=len(p.aligned),
            contrail_length_px=p.length_px,
        )

    best_bucket = p.long_aligned if p.long_aligned else p.aligned
    best = max(best_bucket, key=lambda t: t[4])
    pixel_line = (float(best[0]), float(best[1]), float(best[2]), float(best[3]))
    length_px = p.length_px

    # Scoring contract.  "length" (default) uses the along-track contrail pixel
    # span normalised by ``score_length_norm_px`` — continuous, does not
    # saturate on strong contrails.  "count" reproduces the legacy discrete
    # score that caused ~63% of April-8 episode peaks to hit 1.0.
    score_fn = getattr(config, "score_fn", "length")
    if score_fn == "count":
        norm = max(1, int(config.score_norm_count))
        score = min(1.0, len(p.long_aligned) / float(norm))
    else:
        norm_px = float(getattr(config, "score_length_norm_px", 130.0))
        if norm_px <= 0.0:
            norm_px = 130.0
        score = min(1.0, length_px / norm_px)

    return DetectionResult(
        score=score,
        pixel_line=pixel_line,
        method=p.method,
        num_long_lines=len(p.long_aligned),
        aligned_lines=len(p.aligned),
        contrail_length_px=length_px,
    )


def explain(
    frame: np.ndarray,
    roi: Rect,
    config: DetectionConfig,
    *,
    polygon: np.ndarray | None = None,
    path_vec: tuple[float, float] | None = None,
    prev_frame: np.ndarray | None = None,
) -> DetectionPass:
    """Return the full :class:`DetectionPass` the detector produced for an ROI.

    This is :func:`detect`'s computation without the scoring step: the same
    base, mask, Canny thresholds, edges, and aligned/long-aligned line sets that
    :func:`detect` scored from.  Visualisers should render *this* — rendering the
    pass guarantees the panels show exactly what the detector saw, instead of a
    hand-rolled re-implementation that can silently diverge from production.
    """
    return run_detection_pass(
        frame, roi, config,
        polygon=polygon,
        path_vec=path_vec,
        prev_frame=prev_frame,
        apply_exclusion=True,
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
    iteration until the aligned-long-line span stops increasing or the total
    along-track dimension reaches ``config.roi_max_along_px``.

    Returns the along-track projection span of all aligned long Hough line
    endpoints at convergence, in full-frame pixel units.  Returns 0.0 when no
    aligned lines are found in the seed polygon.

    Note: growth runs the kernel with ``apply_exclusion=False`` — the
    timestamp-exclusion region is *not* applied here, matching this function's
    historical behaviour (only :func:`detect` excludes the overlay region).
    """
    h_fr, w_fr = frame.shape[:2]

    cx, cy = float(center_xy[0]), float(center_xy[1])
    vx, vy = float(path_vec[0]), float(path_vec[1])
    nx, ny = -vy, vx  # cross-track unit vector
    cross_half = float(getattr(config, "roi_cross_px", 40)) / 2.0

    growth_step = int(getattr(config, "growth_step_px", 20))
    max_along = int(getattr(config, "roi_max_along_px", 600))
    start_along = int(getattr(config, "roi_along_px", 180))

    # Max-case AABB bound (axis-aligned sum of along + cross half-extents)
    # so the preprocessing crop covers every growth iteration we'll see.
    max_radius = max_along / 2.0 + cross_half
    max_x1 = max(0, int(math.floor(cx - max_radius)))
    max_y1 = max(0, int(math.floor(cy - max_radius)))
    max_x2 = min(w_fr, int(math.ceil(cx + max_radius)))
    max_y2 = min(h_fr, int(math.ceil(cy + max_radius)))
    if max_x2 <= max_x1 or max_y2 <= max_y1:
        return 0.0

    # Prepare the detection base once over the max-case crop; each growth
    # iteration re-masks a sub-view of it (so Sobel/blur halos are consistent).
    base, _ = _prepare_base(
        frame, config,
        path_vec=path_vec,
        prev_frame=prev_frame,
        crop=(max_x1, max_y1, max_x2, max_y2),
    )

    best_length_px = 0.0
    prev_span = -1.0  # sentinel so first iteration never breaks early

    current_along = start_along
    while current_along <= max_along:
        along_half = current_along / 2.0

        # Rotated polygon at current along-track size (full-frame coords).
        corners = np.array(
            [
                [cx - along_half * vx - cross_half * nx, cy - along_half * vy - cross_half * ny],
                [cx + along_half * vx - cross_half * nx, cy + along_half * vy - cross_half * ny],
                [cx + along_half * vx + cross_half * nx, cy + along_half * vy + cross_half * ny],
                [cx - along_half * vx + cross_half * nx, cy - along_half * vy + cross_half * ny],
            ],
            dtype=np.float32,
        )

        xs = corners[:, 0]
        ys = corners[:, 1]
        bx1 = max(0, int(math.floor(float(xs.min()))))
        by1 = max(0, int(math.floor(float(ys.min()))))
        bx2 = min(w_fr, int(math.ceil(float(xs.max()))))
        by2 = min(h_fr, int(math.ceil(float(ys.max()))))
        if bx2 <= bx1 or by2 <= by1:
            break

        crop = base[by1 - max_y1 : by2 - max_y1, bx1 - max_x1 : bx2 - max_x1]

        p = _pass_on_base(
            crop,
            (bx1, by1),
            corners,
            config,
            method="grow",
            path_vec=path_vec,
            use_mask=True,
            apply_exclusion=False,
        )
        span = p.length_px

        # Stop growing when the span stops increasing.  Using span rather than
        # count handles the common case where a single Hough segment spans more
        # of the contrail as the ROI widens, without the count changing.
        if span <= prev_span and current_along > start_along:
            break

        best_length_px = max(best_length_px, span)
        prev_span = span
        current_along += 2 * growth_step

    return best_length_px
