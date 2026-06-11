"""Canonical detection pass: the single edge+Hough kernel the detector runs.

This module exists to end a self-duplication that had crept into the detector.
Before this, ``detect()`` and ``grow_contrail_length()`` *each* inlined the same
"adaptive Canny thresholds -> pixel floor -> rotated mask -> Canny -> angle-
constrained Hough" block, two now-deleted helpers (``_canny_and_hough_in_polygon``
and ``_filter_lines``) implemented a third, half-finished copy that nothing
called, and the visualisation scripts (``detection_review_panels``,
``filter_playground.ipynb``) re-implemented a *fourth* copy that silently skipped
``_prepare_base`` -- so the review panels rendered different edges than the
detector actually saw whenever ``preprocessing != "none"``.

The fix is one deep module.  :func:`run_detection_pass` is the only place the
edge+Hough math lives.  It returns a :class:`DetectionPass` carrying every
intermediate array the detector operated on.  ``detect()`` scores from it,
``grow_contrail_length()`` grows from it, and ``explain()`` renders it -- so a
visualiser cannot drift from production by construction.

Coordinate conventions (part of the interface)
----------------------------------------------
* ``frame``: full camera frame, (H, W) gray or (H, W, 3) BGR, uint8.
* ``base`` / ``edges`` / ``mask``: **crop-local** arrays, shape (y2-y1, x2-x1),
  where ``(x1, y1) == DetectionPass.crop_origin`` is the AABB top-left in
  full-frame pixels.
* ``aligned`` / ``long_aligned``: line tuples ``(x1, y1, x2, y2, length)`` in
  **full-frame** pixel coordinates.  ``length`` is the Euclidean segment length.
* ``length_px``: along-track span of ``long_aligned`` projected onto
  ``path_vec`` (full-frame units); falls back to the longest long-aligned
  segment length when ``path_vec is None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from concam.config import DetectionConfig
from concam.projection import Rect


# ---------------------------------------------------------------------------
# Geometry / angle helpers (single home; were duplicated in __init__).
# ---------------------------------------------------------------------------

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


def _prepare_base(
    frame: np.ndarray,
    config: DetectionConfig,
    *,
    path_vec: tuple[float, float] | None = None,
    prev_frame: np.ndarray | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, str]:
    """Apply temporal diff and spatial preprocessing to produce a detection base.

    Returns ``(base_gray, method_suffix)`` where ``method_suffix`` is appended
    to the detector method name to record which preprocessing was applied.

    When ``crop=(x1, y1, x2, y2)`` (full-frame pixel coords) is supplied, the
    spatial ops are applied on a halo-padded slice of the frame and the result
    is trimmed back, so ``base.shape == (y2-y1, x2-x1)``. The halo equals the
    widest stencil footprint across the enabled ops, so the interior pixels
    match a full-frame computation.
    """
    preprocessing = getattr(config, "preprocessing", "none")
    has_diff = (
        prev_frame is not None
        and prev_frame.shape[:2] == frame.shape[:2]
    )

    method_suffix = ""
    if has_diff:
        method_suffix += "_diff"
    if preprocessing == "local_contrast":
        method_suffix += "_lc"
    elif preprocessing == "cross_grad" and path_vec is not None:
        method_suffix += "_cg"

    pad = 0
    if preprocessing == "cross_grad" and path_vec is not None:
        pad = max(pad, 1)  # Sobel ksize=3
    if preprocessing == "local_contrast":
        sigma = float(getattr(config, "local_contrast_sigma", 25.0))
        pad = max(pad, int(math.ceil(3.0 * sigma)))
    if config.blur_kernel and config.blur_kernel > 1:
        k = int(config.blur_kernel) | 1
        pad = max(pad, k // 2)

    h_fr, w_fr = frame.shape[:2]
    if crop is not None:
        x1 = max(0, min(w_fr, int(crop[0])))
        y1 = max(0, min(h_fr, int(crop[1])))
        x2 = max(x1, min(w_fr, int(crop[2])))
        y2 = max(y1, min(h_fr, int(crop[3])))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((y2 - y1, x2 - x1), dtype=np.uint8), method_suffix
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w_fr, x2 + pad)
        py2 = min(h_fr, y2 + pad)
        gray = _to_gray(frame[py1:py2, px1:px2])
        base = gray
        if has_diff:
            prev_gray = _to_gray(prev_frame[py1:py2, px1:px2])
            base = cv2.absdiff(gray, prev_gray)
        trim_y = y1 - py1
        trim_x = x1 - px1
        trim_h = y2 - y1
        trim_w = x2 - x1
    else:
        gray = _to_gray(frame)
        base = gray
        if has_diff:
            prev_gray = _to_gray(prev_frame)
            base = cv2.absdiff(gray, prev_gray)
        trim_y = trim_x = 0
        trim_h, trim_w = base.shape[:2]

    if preprocessing == "local_contrast":
        sigma = float(getattr(config, "local_contrast_sigma", 25.0))
        base_f = base.astype(np.float32)
        bg = cv2.GaussianBlur(base_f, (0, 0), sigma)
        base = np.clip(base_f - bg + 128.0, 0, 255).astype(np.uint8)
    elif preprocessing == "cross_grad" and path_vec is not None:
        px_v, py_v = float(path_vec[0]), float(path_vec[1])
        perp_x, perp_y = -py_v, px_v   # 90° rotation, unit length
        base_f = base.astype(np.float32)
        sx = cv2.Sobel(base_f, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(base_f, cv2.CV_32F, 0, 1, ksize=3)
        cross = np.abs(sx * perp_x + sy * perp_y)
        gain = float(getattr(config, "cross_grad_gain", 2.0))
        base = np.clip(cross * gain, 0, 255).astype(np.uint8)

    if config.blur_kernel and config.blur_kernel > 1:
        k = int(config.blur_kernel) | 1
        base = cv2.GaussianBlur(base, (k, k), 0)

    if crop is not None:
        base = base[trim_y : trim_y + trim_h, trim_x : trim_x + trim_w]

    return base, method_suffix


# ---------------------------------------------------------------------------
# The detection pass
# ---------------------------------------------------------------------------

@dataclass
class DetectionPass:
    """Every intermediate the detector produced for one (frame, ROI) application.

    This is the shared output of the edge+Hough kernel.  ``detect()`` scores
    from it, ``grow_contrail_length()`` reads ``length_px`` from it, and
    ``explain()`` returns it for visualisation -- so callers can never render or
    score something other than what the detector actually computed.
    """

    method: str
    crop_origin: tuple[int, int]                 # (x1, y1) full-frame AABB top-left
    base: np.ndarray                             # preprocessed gray crop (crop-local)
    mask: np.ndarray | None                      # rotated-polygon mask (crop-local) or None
    canny_low: int
    canny_high: int
    floor: int | None                            # THRESH_TOZERO floor (adaptive) or None
    edges: np.ndarray                            # masked Canny edges (crop-local)
    # All line tuples are (x1, y1, x2, y2, length) in FULL-FRAME coordinates.
    raw_lines: list[tuple[int, int, int, int, float]] = field(default_factory=list)
    aligned: list[tuple[int, int, int, int, float]] = field(default_factory=list)
    long_aligned: list[tuple[int, int, int, int, float]] = field(default_factory=list)
    length_px: float = 0.0


def _empty_pass(method: str, crop_origin: tuple[int, int]) -> DetectionPass:
    z = np.zeros((0, 0), dtype=np.uint8)
    return DetectionPass(
        method=method,
        crop_origin=crop_origin,
        base=z,
        mask=None,
        canny_low=0,
        canny_high=0,
        floor=None,
        edges=z,
    )


def _pass_on_base(
    base: np.ndarray,
    crop_origin: tuple[int, int],
    polygon: np.ndarray | None,
    config: DetectionConfig,
    *,
    method: str,
    path_vec: tuple[float, float] | None,
    use_mask: bool,
    apply_exclusion: bool,
    frame_origin: tuple[int, int] = (0, 0),
) -> DetectionPass:
    """Run the edge+Hough kernel on an already-prepared base crop.

    ``base`` is crop-local with top-left ``crop_origin == (x1, y1)`` in
    full-frame coords.  ``polygon`` is in full-frame coords (translated to
    crop-local internally).  This is the *only* implementation of the detector's
    masking / adaptive-Canny / floor / Hough / angle-filter pipeline.

    ``frame_origin`` is the true full-frame position of the *input image's*
    (0, 0).  Production passes operate on the real full frame, so it stays
    (0, 0); crop-replay harnesses (which hand ``detect`` a padded crop "as if
    it were the full frame") must pass the crop's full-frame top-left so the
    full-frame-anchored exclusions — the timestamp region and the static-scene
    mask — are sliced at the pixels they actually cover.
    """
    x1, y1 = crop_origin
    # Absolute full-frame top-left of this base crop (for full-frame-anchored
    # exclusion lookups only; all other geometry stays input-image-relative).
    ax1 = x1 + int(frame_origin[0])
    ay1 = y1 + int(frame_origin[1])
    if base.size == 0:
        return _empty_pass(method, crop_origin)

    ch, cw = base.shape[:2]

    if polygon is not None and use_mask:
        local_poly = polygon.copy().astype(np.float32)
        local_poly[:, 0] -= x1
        local_poly[:, 1] -= y1
        local_poly_int = np.round(local_poly).astype(np.int32)
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(mask, [local_poly_int], 255)
        masked_values = base[mask > 0]
    else:
        mask = None
        masked_values = base.reshape(-1)

    if masked_values.size == 0:
        return _empty_pass(method, crop_origin)

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

    # Apply the floor and rotated mask before Canny.
    crop_for_canny = base
    if mask is not None:
        crop_for_canny = cv2.bitwise_and(crop_for_canny, crop_for_canny, mask=mask)
    if floor is not None and floor > 0:
        _, crop_for_canny = cv2.threshold(crop_for_canny, floor, 255, cv2.THRESH_TOZERO)

    # Zero out the burned-in timestamp region so its glyph edges cannot produce
    # aligned Hough lines.  Region is [y0, y1, x0, x1] in full-frame coords.
    if apply_exclusion:
        excl = getattr(config, "timestamp_exclusion_region", None)
        if excl is not None and len(excl) == 4:
            ey0, ey1, ex0, ex1 = int(excl[0]), int(excl[1]), int(excl[2]), int(excl[3])
            ry0 = max(0, ey0 - ay1)
            ry1 = min(ch, ey1 - ay1)
            cx0 = max(0, ex0 - ax1)
            cx1 = min(cw, ex1 - ax1)
            if ry1 > ry0 and cx1 > cx0:
                if crop_for_canny is base:
                    crop_for_canny = crop_for_canny.copy()
                crop_for_canny[ry0:ry1, cx0:cx1] = 0

    edges = cv2.Canny(crop_for_canny, canny_low, canny_high)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Suppress static-scene structure (buildings): persistent building edges
    # are the dominant false-positive source. Unlike the timestamp exclusion,
    # this is applied to the *edge map after Canny* — zeroing the input pixels
    # would manufacture fresh edges along the mask boundary, which are exactly
    # the aligned straight lines we are trying to remove. The mask is a
    # full-frame boolean npz built offline (scripts/build_static_mask.py) and
    # cached per-path, so this is a slice + assignment per pass.
    if apply_exclusion:
        mask_path = getattr(config, "static_mask_path", None)
        if mask_path:
            from concam.detection.static_mask import load_static_mask

            static = load_static_mask(mask_path)
            sub = static[ay1 : ay1 + ch, ax1 : ax1 + cw]
            if sub.any():
                # Pad to the crop shape in case the mask is smaller than the
                # frame (resolution mismatch) or the crop clips the frame edge.
                sm = np.zeros((ch, cw), dtype=bool)
                sm[: sub.shape[0], : sub.shape[1]] = sub
                edges[sm] = 0

    result = DetectionPass(
        method=method,
        crop_origin=crop_origin,
        base=base,
        mask=mask,
        canny_low=canny_low,
        canny_high=canny_high,
        floor=floor,
        edges=edges,
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(config.hough_threshold),
        minLineLength=int(config.hough_min_line_length),
        maxLineGap=int(config.hough_max_line_gap),
    )
    if lines is None or len(lines) == 0:
        return result

    tol = float(config.angle_tolerance_deg)
    min_long = float(config.long_line_min_px)
    path_angle = (
        _angle180(float(path_vec[0]), float(path_vec[1])) if path_vec is not None else None
    )

    for ln in lines:
        lx1, ly1, lx2, ly2 = (int(v) for v in ln[0])
        length = float(math.hypot(lx2 - lx1, ly2 - ly1))
        # Full-frame coordinates.
        fl = (lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1, length)
        result.raw_lines.append(fl)
        if path_angle is not None:
            line_angle = _angle180(float(lx2 - lx1), float(ly2 - ly1))
            if _angle_delta(line_angle, path_angle) > tol:
                continue
        result.aligned.append(fl)
        if length >= min_long:
            result.long_aligned.append(fl)

    if path_vec is not None:
        result.length_px = _line_span_px(result.long_aligned, path_vec)
    elif result.long_aligned:
        result.length_px = max(la[4] for la in result.long_aligned)
    else:
        result.length_px = 0.0

    return result


def run_detection_pass(
    frame: np.ndarray,
    roi: Rect,
    config: DetectionConfig,
    *,
    polygon: np.ndarray | None = None,
    path_vec: tuple[float, float] | None = None,
    prev_frame: np.ndarray | None = None,
    apply_exclusion: bool = True,
    frame_origin: tuple[int, int] = (0, 0),
) -> DetectionPass:
    """Run the canonical detection kernel over one oriented ROI.

    This prepares the detection base (temporal diff + spatial preprocessing,
    clipped to the ROI AABB) and runs the edge+Hough kernel.  It is the single
    source of truth for what the detector "sees"; ``detect``, ``explain``, and
    (per-iteration) ``grow_contrail_length`` all route through this kernel.

    ``frame_origin``: true full-frame coordinates of ``frame``'s (0, 0).
    Leave at (0, 0) when ``frame`` is the real full frame; crop-replay callers
    that pass a cached crop as the "frame" must supply the crop's full-frame
    top-left so the timestamp-exclusion region and the static-scene mask are
    anchored at the correct pixels (see ``_pass_on_base``).
    """
    use_mask = polygon is not None and config.use_rotated_mask
    method = ("rotated_hough" if use_mask else "aabb_hough")

    h_fr, w_fr = frame.shape[:2]
    x1 = max(0, int(roi.x))
    y1 = max(0, int(roi.y))
    x2 = min(w_fr, int(roi.x + roi.w))
    y2 = min(h_fr, int(roi.y + roi.h))

    base, method_suffix = _prepare_base(
        frame, config,
        path_vec=path_vec,
        prev_frame=prev_frame,
        crop=(x1, y1, x2, y2),
    )
    method = method + method_suffix

    if x2 <= x1 or y2 <= y1:
        return _empty_pass(method, (x1, y1))

    return _pass_on_base(
        base,
        (x1, y1),
        polygon,
        config,
        method=method,
        path_vec=path_vec,
        use_mask=use_mask,
        apply_exclusion=apply_exclusion,
        frame_origin=frame_origin,
    )
