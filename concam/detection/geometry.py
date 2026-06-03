"""Canonical geometry reconstruction for manifest candidates.

This module gives the "reconstruct detection geometry from a manifest candidate"
transformation one home.  Before this, the same logic was duplicated verbatim
across four scripts (``detection_validation_sweep.py``,
``detection_score_sweep.py``, ``detection_review_panels.py``,
``tune_from_episode_labels.py``) and also in ``notebooks/filter_playground.ipynb``
(which is not edited here; it still holds a local copy).

Coordinate convention (crop-local)
-----------------------------------
The returned :class:`CandidateGeometry` positions everything *as if the padded
crop were the full frame*:

* ``rect`` spans the whole crop — ``Rect(0, 0, cw, ch)``.
* ``center`` is the projected pixel point translated into crop-local coordinates:
  ``center.x = pixel_x - full_tl_x``, where ``full_tl_x = max(0, roi.x - extract_pad)``.
* ``polygon`` is the rotated detection rectangle in the same crop-local space.
  It is intentionally **not** clipped to the crop boundary — the mask inside
  ``detect()`` handles that.
* ``path_vec`` is the stored ``(path_dx, path_dy)`` unit vector, passed through unchanged.

A throwaway :class:`~concam.config.DetectionConfig` is constructed just to call
:func:`~concam.projection.rotated_polygon`.  This mirrors the original call sites
and is the simplest way to reach the polygon builder without duplicating its math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from concam.config import DetectionConfig
from concam.projection import PixelPoint, Rect, rotated_polygon


@dataclass(frozen=True)
class CandidateGeometry:
    """Crop-local geometry for one manifest candidate.

    All coordinates are in the padded-crop pixel space (top-left = (0, 0)).

    Attributes:
        rect: Axis-aligned bounding rectangle spanning the whole crop.
        polygon: (4, 2) float32 rotated detection polygon in crop-local coords.
        path_vec: Flight-path unit vector ``(vx, vy)``; passed through from the
            candidate's ``path_dx`` / ``path_dy`` fields.
        center: Projected pixel point in crop-local coordinates.
    """

    rect: Rect
    polygon: np.ndarray
    path_vec: tuple[float, float]
    center: PixelPoint


def candidate_geometry(
    cand: dict,
    crop_shape: tuple[int, int],
    *,
    roi_along_px: int,
    roi_cross_px: int,
    extract_pad: int = 20,
) -> CandidateGeometry:
    """Reconstruct the detection geometry for a stored manifest candidate.

    Args:
        cand: Manifest candidate dict.  Must contain keys ``roi`` (with sub-keys
            ``x``, ``y``), ``pixel_x``, ``pixel_y``, ``path_dx``, ``path_dy``.
        crop_shape: ``(height, width)`` of the saved ROI crop in pixels.
        roi_along_px: Along-track half-length of the rotated detection box.
        roi_cross_px: Cross-track half-width of the rotated detection box.
        extract_pad: Padding (px) added around the ROI AABB when the crop was
            saved.  The default (20) matches the pipeline's ``EXTRACT_PAD``.

    Returns:
        :class:`CandidateGeometry` with crop-local ``rect``, ``polygon``,
        ``path_vec``, and ``center``.

    Note:
        Building a throwaway :class:`~concam.config.DetectionConfig` just to
        reach :func:`~concam.projection.rotated_polygon` mirrors the original
        call sites.  Keeping it this way avoids duplicating the polygon math
        here and ensures the polygon is always built the same way as the
        production pipeline.
    """
    ch, cw = crop_shape
    roi = cand["roi"]

    # True full-frame top-left of the crop after edge clipping.
    full_tl_x = max(0, int(roi["x"]) - extract_pad)
    full_tl_y = max(0, int(roi["y"]) - extract_pad)

    center_local = PixelPoint(
        x=float(cand["pixel_x"]) - full_tl_x,
        y=float(cand["pixel_y"]) - full_tl_y,
    )
    path_vec = (float(cand["path_dx"]), float(cand["path_dy"]))

    # Build a throwaway DetectionConfig solely to call rotated_polygon, which
    # uses roi_along_px/roi_cross_px/roi_padding to compute the box half-extents.
    # roi_padding=20 is intentional: it is independent of whatever roi_padding
    # the projection stage used when the crop was extracted.
    _dummy_cfg = DetectionConfig(
        roi_along_px=roi_along_px,
        roi_cross_px=roi_cross_px,
        roi_padding=20,
    )
    poly = rotated_polygon(center_local, path_vec, _dummy_cfg)

    # rect spans the whole crop — the rotated polygon (not the rect) does the
    # along-track selection.  The polygon may extend beyond the crop; detect()
    # clips it via the mask.
    rect = Rect(x=0, y=0, w=cw, h=ch)

    return CandidateGeometry(rect=rect, polygon=poly, path_vec=path_vec, center=center_local)
