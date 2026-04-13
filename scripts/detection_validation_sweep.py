"""Detection parameter sweep on human-labeled ROIs (PRD item 7 rewrite).

Consumes the manifest + labels.json produced by ``detection_validation_extract.py``
and sweeps parameters of the **rotated-ROI / angle-constrained** detector
(``concam.detection.detect``) against the real frames, ranking each combination
by how well its score distribution separates positives from negatives.

The sweep drives ``concam.detection.detect`` directly so sweep scores are
exactly what the pipeline will produce. Each candidate's ROI crop is loaded
from the manifest, the rotated polygon is reconstructed in crop-local coords
from the stored ``pixel_x/y`` + ``path_dx/dy``, and the detector is called
with ``polygon`` + ``path_vec`` so the rotated mask + angle filter apply.

Outputs:
  - ``sweep_report.md`` -- top-N parameter combinations with AUC, positive /
    negative score statistics, recommended threshold, and a ready-to-paste
    YAML snippet for ``configs/*.yaml``.
  - ``sweep_results.json`` -- the full result grid.
  - ``best_combo_visualisation.png`` -- for the top combo, one tile per labeled
    ROI showing crop + edges + detected line, colour-coded by label.

Usage::

    uv run python scripts/detection_validation_sweep.py \\
        --date 2026-04-08 \\
        --labels output/validation/detection/2026-04-08/labels.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from concam.config import DetectionConfig
from concam.detection import detect
from concam.projection import PixelPoint, Rect, rotated_polygon

# Parameter grid axes. 3^7 = 2187 combos × 20 ROIs × ~1 ms/detect() ≈ under a
# minute total. long_line_min_px is sweepable because the rotated ROI is only
# 120 px along-track; a 40 px floor (the sibling default) throws away shorter
# fragmented contrails.
CANNY_PCT_HIGH = (99.0, 99.5, 99.8)
CANNY_LOW_RATIO = (0.25, 0.4, 0.5)
HOUGH_THRESHOLD = (15, 30, 50)
HOUGH_MIN_LINE_LENGTH = (10, 20, 40)
HOUGH_MAX_LINE_GAP = (2, 5, 10)
ANGLE_TOLERANCE_DEG = (5.0, 8.0, 12.0)
LONG_LINE_MIN_PX = (15.0, 25.0, 40.0)

# Fixed knobs (pinned at sibling-pipeline defaults).
CANNY_MIN_HIGH = 60
CANNY_PCT_LOW = 96.0
SCORE_NORM_COUNT = 6
ROI_ALONG_PX = 120
ROI_CROSS_PX = 40


@dataclass
class Combo:
    canny_percentile_high: float
    canny_low_ratio: float
    hough_threshold: int
    hough_min_line_length: int
    hough_max_line_gap: int
    angle_tolerance_deg: float
    long_line_min_px: float

    def asdict(self) -> dict:
        return {
            "canny_percentile_high": self.canny_percentile_high,
            "canny_low_ratio": self.canny_low_ratio,
            "hough_threshold": self.hough_threshold,
            "hough_min_line_length": self.hough_min_line_length,
            "hough_max_line_gap": self.hough_max_line_gap,
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "long_line_min_px": self.long_line_min_px,
        }


def _combo_to_config(combo: Combo) -> DetectionConfig:
    return DetectionConfig(
        score_threshold=0.3,
        canny_low=50, canny_high=150,
        hough_threshold=int(combo.hough_threshold),
        hough_min_line_length=int(combo.hough_min_line_length),
        hough_max_line_gap=int(combo.hough_max_line_gap),
        roi_padding=20,
        roi_along_px=ROI_ALONG_PX,
        roi_cross_px=ROI_CROSS_PX,
        use_adaptive_canny=True,
        canny_percentile_low=CANNY_PCT_LOW,
        canny_percentile_high=float(combo.canny_percentile_high),
        canny_low_ratio=float(combo.canny_low_ratio),
        canny_min_high=CANNY_MIN_HIGH,
        angle_tolerance_deg=float(combo.angle_tolerance_deg),
        long_line_min_px=float(combo.long_line_min_px),
        score_norm_count=SCORE_NORM_COUNT,
        use_rotated_mask=True,
        blur_kernel=3,
    )


def _reconstruct_geometry(
    cand: dict, crop_shape: tuple[int, int], extract_pad: int = 20,
) -> tuple[Rect, np.ndarray, tuple[float, float], PixelPoint]:
    """Given the manifest candidate and its saved crop, return the Rect/polygon/
    path_vec/center needed to drive ``concam.detection.detect`` as if the crop
    itself were the full frame.

    Extract pads the crop by ``extract_pad`` on each side around the AABB (clipped
    to 0). To reproduce the correct crop-local coordinates we need both the
    crop's actual shape (to handle edge clipping) and the original AABB.
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

    # Rotated polygon built with the sweep's along/cross — independent of whatever
    # roi_padding value the projection stage used when the crop was extracted.
    dummy_cfg = DetectionConfig(
        roi_along_px=ROI_ALONG_PX, roi_cross_px=ROI_CROSS_PX, roi_padding=20,
    )
    poly = rotated_polygon(center_local, path_vec, dummy_cfg)

    # The Rect handed to detect() is the whole crop — the rotated mask does the
    # along-track selection. Use the crop's actual shape so we never reference
    # pixels outside it.
    rect = Rect(x=0, y=0, w=cw, h=ch)
    return rect, poly, path_vec, center_local


def _sweep(rois: list[tuple[dict, np.ndarray, str]]) -> list[dict]:
    """For each parameter combination, score every labeled ROI and summarise."""
    results: list[dict] = []
    # Pre-build per-ROI geometry once since it doesn't depend on the combo.
    roi_geoms = [
        (meta, crop, label, *_reconstruct_geometry(meta, crop.shape[:2]))
        for meta, crop, label in rois
    ]
    for pct_hi, lo_ratio, ht, hml, hmg, tol, lmin in itertools.product(
        CANNY_PCT_HIGH, CANNY_LOW_RATIO, HOUGH_THRESHOLD,
        HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP, ANGLE_TOLERANCE_DEG,
        LONG_LINE_MIN_PX,
    ):
        combo = Combo(pct_hi, lo_ratio, ht, hml, hmg, tol, lmin)
        cfg = _combo_to_config(combo)
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        for meta, crop, label, rect, poly, path_vec, _center in roi_geoms:
            result = detect(
                crop, rect, cfg, polygon=poly, path_vec=path_vec,
            )
            if label == "positive":
                pos_scores.append(result.score)
            elif label == "negative":
                neg_scores.append(result.score)
        auc = _mann_whitney_auc(pos_scores, neg_scores)
        threshold, youden_j = _best_threshold(pos_scores, neg_scores)
        results.append(
            {
                "combo": combo.asdict(),
                "auc": auc,
                "youden_j": youden_j,
                "threshold": threshold,
                "pos_median": statistics.median(pos_scores) if pos_scores else 0.0,
                "pos_min": min(pos_scores) if pos_scores else 0.0,
                "neg_median": statistics.median(neg_scores) if neg_scores else 0.0,
                "neg_max": max(neg_scores) if neg_scores else 0.0,
                "separation": (statistics.median(pos_scores) if pos_scores else 0.0)
                - (statistics.median(neg_scores) if neg_scores else 0.0),
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
            }
        )
    results.sort(key=lambda r: (r["auc"], r["youden_j"], r["separation"]), reverse=True)
    return results


def _mann_whitney_auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    if not pos or not neg:
        return 0.5, 0.0
    scores = sorted(set(pos + neg))
    if len(scores) == 1:
        return scores[0] - 1e-6, 0.0
    candidates = [(a + b) / 2 for a, b in zip(scores[:-1], scores[1:])]
    best_t = candidates[0]
    best_j = -1.0
    for t in candidates:
        tpr = sum(1 for p in pos if p >= t) / len(pos)
        fpr = sum(1 for n in neg if n >= t) / len(neg)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return best_t, best_j


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((h, h, 3), dtype=np.uint8)
    scale = h / max(1, ih)
    nw = max(1, int(round(iw * scale)))
    return cv2.resize(img, (nw, h))


def _compose_grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    if not tiles:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        h, w = t.shape[:2]
        p = np.full((th, tw, 3), 16, dtype=np.uint8)
        p[:h, :w] = t
        padded.append(p)
    rows = (len(padded) + cols - 1) // cols
    grid = np.full((rows * th, cols * tw, 3), 12, dtype=np.uint8)
    for i, t in enumerate(padded):
        r, c = i // cols, i % cols
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return grid


def _visualise_best_combo(
    best: dict,
    rois: list[tuple[dict, np.ndarray, str]],
    out_path: Path,
) -> None:
    combo = Combo(**best["combo"])
    cfg = _combo_to_config(combo)
    threshold = float(best["threshold"])
    tiles: list[np.ndarray] = []
    for meta, crop, label in rois:
        rect, poly, path_vec, _center = _reconstruct_geometry(meta, crop.shape[:2])
        result = detect(crop, rect, cfg, polygon=poly, path_vec=path_vec)

        vis_crop = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        # Draw the rotated polygon on the crop preview.
        overlay = vis_crop.copy()
        cv2.polylines(overlay, [poly.astype(np.int32)], True, (0, 180, 255), 1)
        if result.pixel_line is not None:
            x1, y1, x2, y2 = (int(v) for v in result.pixel_line)
            cv2.line(overlay, (x1, y1), (x2, y2), (80, 220, 80), 2)

        # Also render the masked-edge preview so humans can see what Canny saw.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        masked_vals = gray[mask > 0]
        if masked_vals.size:
            p_hi = float(np.percentile(masked_vals, combo.canny_percentile_high))
            canny_high = max(int(p_hi), CANNY_MIN_HIGH)
            canny_low = max(1, int(canny_high * combo.canny_low_ratio))
            p_lo = float(np.percentile(masked_vals, CANNY_PCT_LOW))
            gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
            _, gray_masked = cv2.threshold(gray_masked, int(p_lo), 255, cv2.THRESH_TOZERO)
            edges = cv2.Canny(gray_masked, canny_low, canny_high)
        else:
            edges = np.zeros_like(gray)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        combined = np.hstack([
            _pad_to_height(vis_crop, 160),
            _pad_to_height(edges_bgr, 160),
            _pad_to_height(overlay, 160),
        ])
        footer = np.full((80, combined.shape[1], 3), 32, dtype=np.uint8)
        color = (80, 220, 80) if label == "positive" else (60, 60, 220) if label == "negative" else (180, 180, 60)
        verdict = "PASS" if result.score >= threshold else "fail"
        line1 = (f"#{meta['idx']:02d} {label:<8} score={result.score:.3f} {verdict} "
                 f"(t={threshold:.3f}) long={result.num_long_lines}")
        line2 = f"{meta['callsign']} {meta['wall_time_utc']}"
        cv2.putText(footer, line1, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(footer, line2, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(np.vstack([combined, footer]))

    grid = _compose_grid(tiles, cols=4)
    cv2.imwrite(str(out_path), grid)


def _write_report(
    md_path: Path,
    date: str,
    labels_summary: dict,
    results: list[dict],
    top_n: int = 10,
) -> None:
    best = results[0]
    c = best["combo"]
    lines: list[str] = []
    lines.append(f"# Detection parameter sweep — {date}")
    lines.append("")
    lines.append(
        f"Positives: {labels_summary['positives']}  Negatives: {labels_summary['negatives']}  "
        f"Skipped: {labels_summary['skipped']}"
    )
    lines.append("")
    lines.append("Detector: rotated-ROI + adaptive percentile Canny + angle-constrained Hough "
                 "(concam.detection.detect).")
    lines.append("")
    lines.append("## Best parameter set")
    lines.append("")
    lines.append(f"- AUC (ROC): **{best['auc']:.3f}**  Youden's J: **{best['youden_j']:.3f}**")
    lines.append(f"- Recommended threshold: **{best['threshold']:.3f}**")
    lines.append(f"- Positive scores: median {best['pos_median']:.3f}, min {best['pos_min']:.3f}")
    lines.append(f"- Negative scores: median {best['neg_median']:.3f}, max {best['neg_max']:.3f}")
    lines.append("")
    lines.append("```yaml")
    lines.append("detection:")
    lines.append(f"  canny_percentile_high: {c['canny_percentile_high']}")
    lines.append(f"  canny_percentile_low: {CANNY_PCT_LOW}")
    lines.append(f"  canny_low_ratio: {c['canny_low_ratio']}")
    lines.append(f"  canny_min_high: {CANNY_MIN_HIGH}")
    lines.append(f"  hough_threshold: {c['hough_threshold']}")
    lines.append(f"  hough_min_line_length: {c['hough_min_line_length']}")
    lines.append(f"  hough_max_line_gap: {c['hough_max_line_gap']}")
    lines.append(f"  angle_tolerance_deg: {c['angle_tolerance_deg']}")
    lines.append(f"  long_line_min_px: {c['long_line_min_px']}")
    lines.append(f"  score_norm_count: {SCORE_NORM_COUNT}")
    lines.append(f"  roi_along_px: {ROI_ALONG_PX}")
    lines.append(f"  roi_cross_px: {ROI_CROSS_PX}")
    lines.append("  use_adaptive_canny: true")
    lines.append("  use_rotated_mask: true")
    lines.append("aggregation:")
    lines.append(f"  detection_threshold: {best['threshold']:.3f}")
    lines.append("```")
    lines.append("")
    lines.append(f"## Top {top_n} parameter combinations")
    lines.append("")
    lines.append("| rank | pct_hi | lo_r | h_thr | min_len | max_gap | tol | l_min | AUC | J | threshold | pos_med | neg_med |")
    lines.append("|------|--------|------|-------|---------|---------|-----|-------|-----|---|-----------|---------|---------|")
    for i, r in enumerate(results[:top_n], 1):
        c = r["combo"]
        lines.append(
            f"| {i} | {c['canny_percentile_high']} | {c['canny_low_ratio']} | "
            f"{c['hough_threshold']} | {c['hough_min_line_length']} | "
            f"{c['hough_max_line_gap']} | {c['angle_tolerance_deg']} | "
            f"{c['long_line_min_px']} | "
            f"{r['auc']:.3f} | {r['youden_j']:.3f} | {r['threshold']:.3f} | "
            f"{r['pos_median']:.3f} | {r['neg_median']:.3f} |"
        )
    lines.append("")
    lines.append("## Go/no-go decision")
    lines.append("")
    go = best["auc"] >= 0.85 and best["youden_j"] >= 0.6
    investigate = best["auc"] >= 0.75 and not go
    status = "GO" if go else ("INVESTIGATE" if investigate else "NO-GO")
    lines.append(
        f"Auto-assessment: **{status}** — AUC {best['auc']:.3f}, Youden's J {best['youden_j']:.3f}. "
        f"Target for this rewrite: AUC ≥ 0.85."
    )
    lines.append("")
    lines.append(
        "If AUC < 0.85, re-label batch 2 (already extracted under "
        "`output/validation/detection/2026-04-08-batch2/`) to expand the training set "
        "before giving up on the rotated-ROI + Hough path. If the combined set still "
        "underperforms, the next levers are (a) color-space preprocessing (HSV V / LAB L "
        "/ whiteness) and (b) Frangi ridge filter — both are deferred items from the "
        "detection redesign survey."
    )
    md_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--labels", required=True, help="labels.json from the labeller HTML")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    validation_dir = Path(args.output_dir) / "validation" / "detection" / args.date
    manifest_path = validation_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing {manifest_path}. Run detection_validation_extract.py first."
        )

    manifest = json.loads(manifest_path.read_text())
    labels = json.loads(Path(args.labels).read_text())

    label_by_idx = {entry["idx"]: entry["label"] for entry in labels["labels"]}
    rois: list[tuple[dict, np.ndarray, str]] = []
    for cand in manifest["candidates"]:
        label = label_by_idx.get(cand["idx"])
        if label is None:
            continue
        roi_path = validation_dir / cand["roi_png"]
        crop = cv2.imread(str(roi_path))
        if crop is None:
            print(f"  WARN: failed to read {roi_path}")
            continue
        rois.append((cand, crop, label))

    positives = sum(1 for _, _, lbl in rois if lbl == "positive")
    negatives = sum(1 for _, _, lbl in rois if lbl == "negative")
    skipped = sum(1 for _, _, lbl in rois if lbl == "skip")

    print(f"Labeled ROIs: {len(rois)} (positive={positives}, negative={negatives}, skip={skipped})")
    if positives < 3 or negatives < 2:
        raise SystemExit(
            "Need at least 3 positives and 2 negatives for a meaningful sweep. "
            "Label more ROIs in the labeller HTML and re-export labels.json."
        )

    scoring_rois = [(m, r, l) for m, r, l in rois if l in ("positive", "negative")]
    total_combos = (
        len(CANNY_PCT_HIGH) * len(CANNY_LOW_RATIO) * len(HOUGH_THRESHOLD)
        * len(HOUGH_MIN_LINE_LENGTH) * len(HOUGH_MAX_LINE_GAP) * len(ANGLE_TOLERANCE_DEG)
        * len(LONG_LINE_MIN_PX)
    )
    print(f"Running sweep over {total_combos} parameter combinations...")
    results = _sweep(scoring_rois)
    print(
        f"Top AUC: {results[0]['auc']:.3f}   threshold: {results[0]['threshold']:.3f}   "
        f"combo: {results[0]['combo']}"
    )

    json_path = validation_dir / "sweep_results.json"
    json_path.write_text(json.dumps({"date": args.date, "results": results}, indent=2))
    md_path = validation_dir / "sweep_report.md"
    _write_report(
        md_path,
        args.date,
        {"positives": positives, "negatives": negatives, "skipped": skipped},
        results,
        top_n=args.top_n,
    )
    vis_path = validation_dir / "best_combo_visualisation.png"
    _visualise_best_combo(results[0], scoring_rois, vis_path)

    print()
    print(f"  Report        : {md_path}")
    print(f"  Full results  : {json_path}")
    print(f"  Visualisation : {vis_path}")


if __name__ == "__main__":
    main()
