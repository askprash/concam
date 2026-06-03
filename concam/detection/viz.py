"""Detection visualisation utilities.

Two public functions:

``compose_grid``
    Pure tile compositor — no file I/O.  Takes a list of BGR uint8 arrays
    (tiles that may differ in size), pads each to the max tile dimensions,
    and lays them out in a column-major grid with a dark background fill.

``render_detection_panels``
    Converts a :class:`~concam.detection._core.DetectionPass` (as returned
    by :func:`~concam.detection.explain`) into a list of named BGR panels.
    Every panel is derived *exclusively* from the DetectionPass, so a panel
    cannot diverge from what :func:`~concam.detection.detect` actually
    computed — this is the guarantee that fixes the pre-Canny re-implementation
    bug in the old visualisation scripts (``detection_review_panels.py``,
    ``detection_validation_sweep.py``, ``filter_playground.ipynb``).

Coordinate note
---------------
DetectionPass.base / .edges / .mask are crop-local (origin = crop_origin).
DetectionPass.long_aligned / .aligned / .raw_lines are full-frame.
``render_detection_panels`` translates line coordinates from full-frame to
crop-local by subtracting crop_origin before drawing.
"""

from __future__ import annotations

import cv2
import numpy as np

from concam.detection._core import DetectionPass

__all__ = ["compose_grid", "render_detection_panels"]


# ---------------------------------------------------------------------------
# Grid compositor
# ---------------------------------------------------------------------------

def compose_grid(
    tiles: list[np.ndarray],
    cols: int,
    *,
    bg: int = 16,
    pad: int = 0,
) -> np.ndarray:
    """Composite *tiles* into a ``(rows, cols)`` grid image.

    Parameters
    ----------
    tiles:
        List of BGR uint8 arrays.  Tiles may differ in height/width; each is
        placed at the top-left of a cell whose dimensions are the *maximum*
        across all tiles.  The remainder of the cell is filled with *bg*.
    cols:
        Number of columns.  Incomplete last rows are padded with *bg* cells.
    bg:
        Scalar fill value for empty cell regions (default 16 — near-black).
    pad:
        Not currently used; reserved for future inter-cell padding.

    Returns
    -------
    np.ndarray
        Shape ``(rows * th, cols * tw, 3)``, dtype uint8.
    """
    if not tiles:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)

    # Pad each tile to the cell size.
    padded: list[np.ndarray] = []
    for t in tiles:
        if t.ndim == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        h, w = t.shape[:2]
        if h == th and w == tw:
            padded.append(t)
        else:
            cell = np.full((th, tw, 3), bg, dtype=np.uint8)
            cell[:h, :w] = t
            padded.append(cell)

    rows = (len(padded) + cols - 1) // cols
    grid = np.full((rows * th, cols * tw, 3), bg, dtype=np.uint8)
    for i, cell in enumerate(padded):
        r, c = i // cols, i % cols
        grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = cell
    return grid


# ---------------------------------------------------------------------------
# Panel renderer
# ---------------------------------------------------------------------------

def render_detection_panels(
    passed: DetectionPass,
    *,
    labels: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Render diagnostic BGR panels from a :class:`DetectionPass`.

    All panels are derived *only* from *passed* — the DetectionPass returned
    by :func:`~concam.detection.explain`.  Because explain() runs the same
    kernel as detect(), the panels cannot diverge from what the detector saw.

    Parameters
    ----------
    passed:
        A :class:`DetectionPass` as returned by
        :func:`~concam.detection.explain`.
    labels:
        When True (default), overlay text annotations (Canny thresholds,
        edge-pixel count, etc.) on the respective panels.

    Returns
    -------
    list of (name, bgr_image)
        Three panels, in order:

        ``"base"``
            The preprocessed grayscale crop (what Canny saw before thresholding
            and masking) converted to BGR.  Shape ``(h, w, 3)``.

        ``"edges"``
            The masked Canny edge map converted to BGR.  The single-channel
            value of this panel *equals* ``passed.edges`` by construction:
            ``panel[:, :, 0] == passed.edges`` (all three channels are
            identical since the input is grayscale).  This is the detector-
            saw-it guarantee — it is tested in
            ``tests/test_detection_viz.py``.

        ``"overlay"``
            The base image converted to BGR with aligned lines drawn in green
            and long-aligned lines drawn in brighter green.  Line coordinates
            are translated from full-frame to crop-local by subtracting
            ``passed.crop_origin``.  If ``passed.mask`` is not None, the
            polygon boundary is drawn in orange.
    """
    base = passed.base  # (h, w) uint8, crop-local
    edges = passed.edges  # (h, w) uint8, crop-local
    x0, y0 = passed.crop_origin

    if base.size == 0:
        blank = np.zeros((1, 1, 3), dtype=np.uint8)
        return [("base", blank.copy()), ("edges", blank.copy()), ("overlay", blank.copy())]

    # --- panel 1: base (preprocessed gray -> BGR) ---
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    if labels:
        ann = f"floor={passed.floor}  canny={passed.canny_low}/{passed.canny_high}"
        cv2.putText(
            base_bgr, ann, (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1, cv2.LINE_AA,
        )

    # --- panel 2: edges (Canny edges -> BGR) ---
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if labels:
        n_edge_px = int(edges.sum() // 255)
        cv2.putText(
            edges_bgr, f"{n_edge_px} edge px", (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1, cv2.LINE_AA,
        )

    # --- panel 3: overlay (base + lines + polygon outline) ---
    overlay = base_bgr.copy()

    # Draw polygon mask outline in orange if present.
    if passed.mask is not None:
        contours, _ = cv2.findContours(
            passed.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 180, 255), 1)

    # Draw aligned lines in green, translating full-frame -> crop-local.
    long_set = set(id(ln) for ln in passed.long_aligned)
    for ln in passed.aligned:
        fx1, fy1, fx2, fy2, _length = ln
        cx1, cy1 = fx1 - x0, fy1 - y0
        cx2, cy2 = fx2 - x0, fy2 - y0
        color = (50, 255, 50) if id(ln) in long_set else (50, 180, 50)
        cv2.line(overlay, (cx1, cy1), (cx2, cy2), color, 1)

    # Draw long-aligned lines thicker / brighter on top.
    for ln in passed.long_aligned:
        fx1, fy1, fx2, fy2, _length = ln
        cv2.line(overlay, (fx1 - x0, fy1 - y0), (fx2 - x0, fy2 - y0), (50, 255, 50), 2)

    return [
        ("base", base_bgr),
        ("edges", edges_bgr),
        ("overlay", overlay),
    ]
