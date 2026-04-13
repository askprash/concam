"""Generate a single PNG review sheet of all detection-validation candidates.

Reads the manifest + context/ROI PNGs already written by
``detection_validation_extract.py`` and composes a dense grid where each
tile shows:

  * the 800x800 context patch (downscaled to 500x500) with ROI box + path
    arrow already overlaid at extract time
  * the native-resolution ROI crop scaled up 3x for detail inspection
  * a large numeric index in the top-left corner
  * callsign + UTC time + pixel position below

The goal is to let a human eyeball all N candidates in a single image and
reply with a compact list of positive/negative/skip indices that the
companion labels-from-list helper can turn into ``labels.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _compose_tile(context_bgr: np.ndarray, roi_bgr: np.ndarray, cand: dict) -> np.ndarray:
    # Top half: context patch downscaled to 500x500.
    ctx = cv2.resize(context_bgr, (500, 500))

    # Bottom: ROI crop scaled to height 160 (3x from ~50 px) preserving aspect.
    rh, rw = roi_bgr.shape[:2]
    target_rh = 160
    scale = target_rh / max(1, rh)
    target_rw = int(round(rw * scale))
    roi_resized = cv2.resize(roi_bgr, (target_rw, target_rh))
    roi_strip = np.full((target_rh, 500, 3), 16, dtype=np.uint8)
    off = max(0, (500 - target_rw) // 2)
    roi_strip[:, off : off + target_rw] = roi_resized[:, : min(target_rw, 500)]

    # Text footer (2 lines).
    footer = np.full((80, 500, 3), 32, dtype=np.uint8)
    line1 = f"#{cand['idx']:02d}  {cand['callsign']}  {cand['wall_time_utc'].split('+')[0]}"
    line2 = f"px ({cand['pixel_x']:.0f},{cand['pixel_y']:.0f})  roi {cand['roi']['w']}x{cand['roi']['h']}  frame {cand['frame_idx']}"
    cv2.putText(footer, line1, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(footer, line2, (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    tile = np.vstack([ctx, roi_strip, footer])

    # Draw a huge candidate index in the top-left corner of the context patch so
    # the reviewer can reference tiles without counting.
    cv2.rectangle(tile, (0, 0), (85, 55), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"{cand['idx']:02d}",
        (8, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (60, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return tile


def _compose_grid(tiles: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not tiles:
        raise ValueError("no tiles")
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * th, cols * tw, 3), 12, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = i // cols, i % cols
        grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = t
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--validation-dir", default=None,
                    help="overrides <output-dir>/validation/detection/<date>")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    if args.validation_dir:
        validation_dir = Path(args.validation_dir)
    else:
        if not args.date:
            raise SystemExit("--date is required unless --validation-dir is given")
        validation_dir = Path(args.output_dir) / "validation" / "detection" / args.date
    manifest_path = validation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    tiles: list[np.ndarray] = []
    for cand in manifest["candidates"]:
        ctx = cv2.imread(str(validation_dir / cand["context_png"]))
        roi = cv2.imread(str(validation_dir / cand["roi_png"]))
        if ctx is None or roi is None:
            print(f"  SKIP #{cand['idx']:02d}: missing PNGs")
            continue
        tiles.append(_compose_tile(ctx, roi, cand))

    grid = _compose_grid(tiles, cols=args.cols)
    out = validation_dir / "review_sheet.png"
    cv2.imwrite(str(out), grid)
    print(f"wrote {out}  ({grid.shape[1]}x{grid.shape[0]} px, {len(tiles)} tiles)")


if __name__ == "__main__":
    main()
