"""Shared pixel preprocessing for the fixed-format timestamp OCR.

This is the single source of truth for how a frame is turned into normalized
per-slot glyphs.  Both the runtime reader (:mod:`concam.ocr.reader`) and the
one-shot template generator (``scripts/generate_ocr_templates.py``) import from
here.  Keeping them on the same code path is load-bearing: the template bank is
only valid if it was built with *exactly* the preprocessing the reader applies
at match time, so these helpers must never be copied and allowed to drift.

The slot layout constants are physical truths of the camera overlay font (a
single fixed font at a single fixed position), so they live alongside the
helpers rather than in config -- a font change means regenerating templates,
at which point this module changes too.
"""

from __future__ import annotations

import cv2
import numpy as np


# Slot x-coordinates (left edge of a 32px-wide character cell) inside the
# timestamp ROI, for the 18 characters of "MM/DD/YYYY" + "HH:MM:SS".
SLOT_X_STARTS = (
    128, 160, 192, 224, 256, 288, 320, 352, 384, 416,   # MM/DD/YYYY
    480, 512, 544, 576, 608, 640, 672, 704,             # HH:MM:SS
)
SLOT_KINDS = (
    "digit", "digit", "slash", "digit", "digit", "slash",
    "digit", "digit", "digit", "digit",
    "digit", "digit", "colon", "digit", "digit", "colon", "digit", "digit",
)
SLOT_WIDTH = 32
SLOT_Y_TOP = 12
SLOT_HEIGHT = 48

# Canonical normalized glyph size.
TEMPLATE_H = 44
TEMPLATE_W = 28


def crop_roi(frame: np.ndarray, region: tuple[int, int], position: str) -> np.ndarray:
    """Crop the overlay ROI of size ``region`` from a corner of ``frame``."""
    h, w = frame.shape[:2]
    rh, rw = region
    if position == "top_right":
        return frame[0:rh, w - rw : w]
    if position == "top_left":
        return frame[0:rh, 0:rw]
    if position == "bottom_right":
        return frame[h - rh : h, w - rw : w]
    if position == "bottom_left":
        return frame[h - rh : h, 0:rw]
    raise ValueError(f"unsupported timestamp_position: {position!r}")


def binarize(roi: np.ndarray, threshold: int = 200) -> np.ndarray:
    """Threshold the (possibly colour) ROI into a white-on-black mask."""
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def extract_slot(binary: np.ndarray, slot_idx: int) -> np.ndarray:
    """Crop the fixed cell for ``slot_idx`` from a binarized ROI.

    Clamps to the ROI bounds so an undersized ROI (e.g. a pre-cropped crop fed
    straight in) degrades to an empty slot instead of an index error.
    """
    x = SLOT_X_STARTS[slot_idx]
    bh, bw = binary.shape
    y0 = min(SLOT_Y_TOP, bh)
    y1 = min(SLOT_Y_TOP + SLOT_HEIGHT, bh)
    x0 = min(x, bw)
    x1 = min(x + SLOT_WIDTH, bw)
    return binary[y0:y1, x0:x1]


def normalize_glyph(slot: np.ndarray) -> np.ndarray:
    """Tight-crop a glyph from its slot and center-pad to the template size.

    Returns a ``float32`` array of shape ``(TEMPLATE_H, TEMPLATE_W)``.  This
    makes the template robust to small horizontal shifts frame-to-frame and to
    character-width variation (e.g. the narrower '1').
    """
    rows = np.where(slot.any(axis=1))[0]
    cols = np.where(slot.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=np.float32)
    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1
    cropped = slot[y0:y1, x0:x1]
    out = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=np.float32)
    ch, cw = cropped.shape
    if ch > TEMPLATE_H or cw > TEMPLATE_W:
        cropped = cv2.resize(cropped, (min(cw, TEMPLATE_W), min(ch, TEMPLATE_H)))
        ch, cw = cropped.shape
    dy = (TEMPLATE_H - ch) // 2
    dx = (TEMPLATE_W - cw) // 2
    out[dy : dy + ch, dx : dx + cw] = cropped.astype(np.float32)
    return out
