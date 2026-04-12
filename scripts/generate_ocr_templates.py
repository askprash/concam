"""One-shot script to extract character templates for the fixed-format OCR.

Reads a handful of frames from a raw camera segment whose timestamps we know
(because the segment filename encodes its UTC start and we know the frame rate)
and dumps a labelled template per glyph into ``concam/ocr/templates.npz``.

We only need each glyph once — the camera overlay uses a single fixed font at a
single fixed position.  Running this script is a calibration step; its output
is checked in alongside the source.

Usage::

    uv run python scripts/generate_ocr_templates.py \\
        --video /net/d16/data/contrail-camera/raw_segments_clean/2026-04-12_00-00-00.mp4 \\
        --segment-start 2026-04-12T00:00:00 \\
        --output concam/ocr/templates.npz

The defaults match the one-shot calibration we performed for the MIT Green
Building camera (overlay font fixed since install).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

# Slot x-coordinates (left edge of 32px-wide character cell) inside the
# 80x875 timestamp ROI, for the 19 characters of "MM/DD/YYYY HH:MM:SS".
# Derived from contour analysis of real frames.
SLOT_X_STARTS = [
    128, 160, 192, 224, 256, 288, 320, 352, 384, 416,   # MM/DD/YYYY
    480, 512, 544, 576, 608, 640, 672, 704,             # HH:MM:SS
]

SLOT_KIND = [
    "digit", "digit", "slash", "digit", "digit", "slash",
    "digit", "digit", "digit", "digit",
    "digit", "digit", "colon", "digit", "digit", "colon", "digit", "digit",
]

SLOT_WIDTH = 32
SLOT_Y_TOP = 12
SLOT_HEIGHT = 48

# Canonical normalized glyph size.
TEMPLATE_H = 44
TEMPLATE_W = 28

# Raw-segment frame rate.  Verified from the sample files.
FPS = 4.0


def _load_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    return frame


def _crop_roi(frame: np.ndarray, region: tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    region_h, region_w = region
    return frame[0:region_h, w - region_w : w]


def _binarize(roi: np.ndarray, threshold: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def _extract_slot(binary: np.ndarray, slot_idx: int) -> np.ndarray:
    x = SLOT_X_STARTS[slot_idx]
    return binary[SLOT_Y_TOP : SLOT_Y_TOP + SLOT_HEIGHT, x : x + SLOT_WIDTH]


def _normalize_glyph(slot: np.ndarray) -> np.ndarray:
    """Tight-crop a glyph from its slot and pad to the canonical template size.

    This makes the template robust to small horizontal shifts from frame to
    frame and to character-width variation (e.g., the narrower '1').
    """
    # Tight-crop to non-zero pixels.
    rows = np.where(slot.any(axis=1))[0]
    cols = np.where(slot.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        # Empty slot (shouldn't happen for known-good frames).
        return np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=np.uint8)
    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1
    cropped = slot[y0:y1, x0:x1]

    # Center-pad to the canonical size.
    out = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=np.uint8)
    ch, cw = cropped.shape
    if ch > TEMPLATE_H or cw > TEMPLATE_W:
        # Resize down if over-sized (rare; guard against font-scale surprises).
        cropped = cv2.resize(cropped, (min(cw, TEMPLATE_W), min(ch, TEMPLATE_H)))
        ch, cw = cropped.shape
    dy = (TEMPLATE_H - ch) // 2
    dx = (TEMPLATE_W - cw) // 2
    out[dy : dy + ch, dx : dx + cw] = cropped
    return out


def _expected_timestamp(
    segment_start: datetime, frame_idx: int, offset_seconds: float
) -> datetime:
    """Wall-clock timestamp displayed on ``frame_idx`` of the segment."""
    return segment_start + timedelta(seconds=offset_seconds + frame_idx / FPS)


def _collect_labels(segment_start: datetime, offset: float, frame_ids: list[int]):
    """Return ``{char: list[(frame_id, slot_idx)]}`` for templates to extract."""
    groups: dict[str, list[tuple[int, int]]] = {}
    for frame_id in frame_ids:
        ts = _expected_timestamp(segment_start, frame_id, offset)
        # Truncate sub-second; we aligned the offset so seconds are integer.
        text = ts.strftime("%m/%d/%Y %H:%M:%S")
        # Drop the space between date and time so our slot index lines up.
        compact = text[:10] + text[11:]
        assert len(compact) == 18, compact
        for slot_idx, ch in enumerate(compact):
            groups.setdefault(ch, []).append((frame_id, slot_idx))
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video",
        default="/net/d16/data/contrail-camera/raw_segments_clean/2026-04-12_00-00-00.mp4",
    )
    ap.add_argument(
        "--segment-start",
        default="2026-04-12T00:00:00",
        help="Wall-clock datetime at frame 0 (ISO format).",
    )
    ap.add_argument(
        "--offset-seconds",
        type=float,
        default=1.0,
        help="Seconds the camera's overlay clock is ahead of the segment start.",
    )
    ap.add_argument("--output", default="concam/ocr/templates.npz")
    ap.add_argument("--region-h", type=int, default=80)
    ap.add_argument("--region-w", type=int, default=875)
    ap.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Grayscale threshold for isolating white overlay text.",
    )
    args = ap.parse_args()

    segment_start = datetime.fromisoformat(args.segment_start)
    region = (args.region_h, args.region_w)

    # Frames cover all digits 0-9 plus / and :.  Chosen based on offset=1s:
    #   frame 0  -> 00:00:01  (covers 0,1)
    #   frame 4  -> 00:00:02  (covers 2)
    #   frame 8  -> 00:00:03  (covers 3)
    #   frame 12 -> 00:00:04  (covers 4)
    #   frame 16 -> 00:00:05  (covers 5)
    #   frame 20 -> 00:00:06  (covers 6)
    #   frame 24 -> 00:00:07  (covers 7)
    #   frame 28 -> 00:00:08  (covers 8)
    #   frame 32 -> 00:00:09  (covers 9)
    frame_ids = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    labels = _collect_labels(segment_start, args.offset_seconds, frame_ids)

    missing = set("0123456789/:") - set(labels.keys())
    if missing:
        raise RuntimeError(f"label set does not cover: {sorted(missing)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")

    # For each character we need, extract the glyph from the first frame that
    # covers it.
    templates: dict[str, np.ndarray] = {}
    try:
        for ch, occurrences in labels.items():
            frame_id, slot_idx = occurrences[0]
            frame = _load_frame(cap, frame_id)
            roi = _crop_roi(frame, region)
            binary = _binarize(roi, args.threshold)
            slot = _extract_slot(binary, slot_idx)
            templates[ch] = _normalize_glyph(slot)
            print(
                f"  ch={ch!r:4s} frame={frame_id:3d} slot={slot_idx:2d} "
                f"pixels={int(templates[ch].sum() // 255)}"
            )
    finally:
        cap.release()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # np.savez doesn't accept keys like '/' or ':'; encode them.
    safe_map = {"/": "slash", ":": "colon"}
    save_kwargs = {safe_map.get(k, k): v for k, v in templates.items()}
    np.savez(output, **save_kwargs)
    print(f"Wrote {len(templates)} templates to {output}")


if __name__ == "__main__":
    main()
