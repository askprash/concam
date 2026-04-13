"""Detection diagnosis sheet: why did the best-combo detector miss the contrails it missed?

For each labeled candidate, tile shows (left to right):
  1. 500x500 context patch with ROI box + flight-path arrow overlaid
  2. ROI crop at native resolution (upscaled to a readable size)
  3. Canny edges inside the ROI (same scale as #2)
  4. The ROI crop with the Hough line overlaid (if any)

Footer shows: label, score, PASS/FAIL against the recommended threshold,
callsign/time, and whether this case was a hit, miss (false neg), FP, or TN.

Intended reading: for contrails that scored 0, compare panel 1 (is the
contrail visible in the wider sky?) against panels 2 and 3 (did the ROI
capture it? did Canny pick up the gradient?). That tells you whether to
widen the ROI, reduce Canny thresholds, or accept the detector's recall ceiling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_sweep_module():
    path = Path(__file__).parent / "detection_validation_sweep.py"
    spec = importlib.util.spec_from_file_location("detection_validation_sweep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["detection_validation_sweep"] = module
    spec.loader.exec_module(module)
    return module


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = h / max(1, ih)
    return cv2.resize(img, (max(1, int(round(iw * scale))), h))


def _classify(label: str, score: float, threshold: float) -> tuple[str, tuple[int, int, int]]:
    passed = score >= threshold
    if label == "positive" and passed:
        return "HIT", (80, 220, 80)
    if label == "positive" and not passed:
        return "MISS", (80, 120, 240)
    if label == "negative" and passed:
        return "FP", (60, 60, 220)
    return "TN", (180, 180, 180)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--cols", type=int, default=2)
    args = ap.parse_args()

    sweep = _load_sweep_module()
    validation_dir = Path(args.output_dir) / "validation" / "detection" / args.date
    manifest = json.loads((validation_dir / "manifest.json").read_text())
    labels = {l["idx"]: l["label"] for l in json.loads((validation_dir / "labels.json").read_text())["labels"]}
    results = json.loads((validation_dir / "sweep_results.json").read_text())["results"]
    best = results[0]
    combo = sweep.Combo(**best["combo"])
    threshold = best["threshold"]

    # Sort so misses appear first (most diagnostic), then TNs, then HITs.
    ordered: list[dict] = sorted(
        manifest["candidates"],
        key=lambda c: {"positive": 0, "negative": 1}.get(labels[c["idx"]], 2),
    )
    # Re-sort more usefully: MISS (pos, low score) → HIT (pos, high score) → negatives
    def _score_of(c):
        roi_bgr = cv2.imread(str(validation_dir / c["roi_png"]))
        return sweep._score_roi(roi_bgr, combo)[0]

    scores_cache = {c["idx"]: _score_of(c) for c in manifest["candidates"]}

    def _rank(c):
        lbl = labels[c["idx"]]
        score = scores_cache[c["idx"]]
        # MISS first, then FP, then HIT, then TN; within each, worst cases first.
        kind, _ = _classify(lbl, score, threshold)
        order = {"MISS": 0, "FP": 1, "HIT": 2, "TN": 3}[kind]
        # Within MISS: higher scores are more interesting (closest to threshold).
        return (order, -score if kind == "MISS" else score)

    ordered = sorted(manifest["candidates"], key=_rank)

    tiles: list[np.ndarray] = []
    for c in ordered:
        lbl = labels[c["idx"]]
        context = cv2.imread(str(validation_dir / c["context_png"]))
        roi = cv2.imread(str(validation_dir / c["roi_png"]))
        score, line = sweep._score_roi(roi, combo)

        # Resize all panels to a common height.
        PANEL_H = 280
        context_panel = _pad_to_height(context, PANEL_H)
        roi_panel = _pad_to_height(roi, PANEL_H)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        edges = cv2.Canny(gray, combo.canny_low, combo.canny_high)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_panel = _pad_to_height(edges_bgr, PANEL_H)
        overlay = roi.copy()
        if line is not None:
            cv2.line(overlay, (line[0], line[1]), (line[2], line[3]), (80, 220, 80), 2)
        overlay_panel = _pad_to_height(overlay, PANEL_H)

        # Thin dividers between panels so boundaries are obvious.
        divider = np.full((PANEL_H, 3, 3), 60, dtype=np.uint8)
        row = np.hstack([
            context_panel, divider,
            roi_panel, divider,
            edges_panel, divider,
            overlay_panel,
        ])

        # Footer
        kind, color = _classify(lbl, score, threshold)
        footer_h = 100
        footer = np.full((footer_h, row.shape[1], 3), 24, dtype=np.uint8)
        line1 = f"#{c['idx']:02d}  [{kind}]  label={lbl}  score={score:.3f}  threshold={threshold:.3f}"
        line2 = f"{c['callsign']}  {c['wall_time_utc']}  px=({c['pixel_x']:.0f},{c['pixel_y']:.0f})  roi={c['roi']['w']}x{c['roi']['h']}"
        line3 = f"  panels: context 500x500 | ROI native | Canny edges | Hough overlay"
        cv2.putText(footer, line1, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(footer, line2, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(footer, line3, (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
        tile = np.vstack([row, footer])
        tiles.append(tile)

    # Compose into a 1-column grid (tall strip) so each case is easy to read.
    if not tiles:
        raise SystemExit("no tiles built")

    max_w = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[1] == max_w:
            padded.append(t)
        else:
            p = np.full((t.shape[0], max_w, 3), 12, dtype=np.uint8)
            p[:, : t.shape[1]] = t
            padded.append(p)

    gap = np.full((8, max_w, 3), 0, dtype=np.uint8)
    stacked = []
    for i, t in enumerate(padded):
        stacked.append(t)
        if i < len(padded) - 1:
            stacked.append(gap)
    diagnose = np.vstack(stacked)

    out_path = validation_dir / "diagnose_best_combo.png"
    cv2.imwrite(str(out_path), diagnose)
    print(f"wrote {out_path}  ({diagnose.shape[1]}x{diagnose.shape[0]} px, {len(tiles)} tiles)")
    print()
    print("Tile order: MISSes (ranked by score desc) → FPs → HITs → TNs.")
    print(f"Threshold: {threshold:.3f}  Best combo: {best['combo']}")


if __name__ == "__main__":
    main()
