"""Detection parameter sweep on human-labeled ROIs (PRD item 6).

Consumes the manifest + labels.json produced by the labeller HTML that
``detection_validation_extract.py`` writes, then sweeps Canny + Hough
parameters against the real frames and ranks each combination by how well
the resulting score distribution separates positives from negatives.

Inputs per candidate:
  - ROI PNG (the oriented bounding box crop saved by the extract script)
  - Label from the human: positive / negative / skip

Outputs:
  - ``sweep_report.md`` -- top-N parameter combinations with AUC,
    positive/negative score statistics, recommended threshold, and a
    ready-to-paste YAML snippet.
  - ``sweep_results.json`` -- the full result grid for downstream analysis.
  - ``best_combo_visualisation.png`` -- for the best parameter set,
    one tile per labeled ROI showing crop, Canny edges, detected line,
    and the score (coloured by label). Lets a human eyeball whether the
    top-ranked combo actually produces visually sensible lines.

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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Parameter grid. 3^5 = 243 combinations is cheap (runs in seconds on 20 ROIs).
# Values chosen to bracket the current config defaults.
CANNY_LOW = (30, 50, 80)
CANNY_HIGH = (100, 150, 200)
HOUGH_THRESHOLD = (20, 40, 60)
HOUGH_MIN_LINE_LENGTH = (20, 40, 60)
HOUGH_MAX_LINE_GAP = (5, 10, 20)


@dataclass
class Combo:
    canny_low: int
    canny_high: int
    hough_threshold: int
    hough_min_line_length: int
    hough_max_line_gap: int

    def asdict(self) -> dict:
        return {
            "canny_low": self.canny_low,
            "canny_high": self.canny_high,
            "hough_threshold": self.hough_threshold,
            "hough_min_line_length": self.hough_min_line_length,
            "hough_max_line_gap": self.hough_max_line_gap,
        }


def _score_roi(roi_bgr: np.ndarray, combo: Combo) -> tuple[float, tuple | None]:
    """Run Canny+Hough on an ROI image and return (normalised score, best line).

    Mirrors ``concam.detection.detect`` so sweep scores are directly comparable
    to pipeline scores. The ROI here already *is* the crop (no bounding-box
    translation needed), so we operate on the whole image.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr
    edges = cv2.Canny(gray, combo.canny_low, combo.canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=combo.hough_threshold,
        minLineLength=combo.hough_min_line_length,
        maxLineGap=combo.hough_max_line_gap,
    )
    h, w = gray.shape[:2]
    diag = float(np.hypot(w, h))
    if lines is None or diag < 1:
        return 0.0, None
    best_len = 0.0
    best_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length > best_len:
            best_len = length
            best_line = (int(x1), int(y1), int(x2), int(y2))
    return min(1.0, best_len / diag), best_line


def _mann_whitney_auc(pos: list[float], neg: list[float]) -> float:
    """AUC via Mann-Whitney U (equivalent to ROC-AUC for two groups).

    Returns 0.5 when pos or neg is empty (undefined), otherwise P(score_pos >
    score_neg) with ties counted as 0.5.
    """
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
    """Return (threshold, Youden's J) maximising TPR - FPR.

    Picks from the set of score midpoints between consecutive unique scores,
    which is the standard way to enumerate candidate ROC thresholds.
    """
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


def _sweep(rois: list[tuple[dict, np.ndarray, str]]) -> list[dict]:
    """For each parameter combination, score every labeled ROI and summarise."""
    results: list[dict] = []
    for cl, ch, ht, hml, hmg in itertools.product(
        CANNY_LOW, CANNY_HIGH, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP
    ):
        if cl >= ch:
            continue  # Canny requires low < high
        combo = Combo(cl, ch, ht, hml, hmg)
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        for meta, roi_bgr, label in rois:
            score, _ = _score_roi(roi_bgr, combo)
            if label == "positive":
                pos_scores.append(score)
            elif label == "negative":
                neg_scores.append(score)
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
    # Sort by AUC descending, then Youden's J, then separation; tiebreaker keeps stable.
    results.sort(key=lambda r: (r["auc"], r["youden_j"], r["separation"]), reverse=True)
    return results


def _visualise_best_combo(
    best: dict,
    rois: list[tuple[dict, np.ndarray, str]],
    out_path: Path,
) -> None:
    """One tile per ROI showing crop + Canny edges + detected line, coloured by label."""
    combo = Combo(**best["combo"])
    threshold = best["threshold"]
    tiles: list[np.ndarray] = []
    for meta, roi_bgr, label in rois:
        score, line = _score_roi(roi_bgr, combo)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr
        edges = cv2.Canny(gray, combo.canny_low, combo.canny_high)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        crop = roi_bgr if roi_bgr.ndim == 3 else cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
        overlay = crop.copy()
        if line is not None:
            cv2.line(overlay, (line[0], line[1]), (line[2], line[3]), (80, 220, 80), 2)
        combined = np.hstack([_pad_to_height(crop, 160), _pad_to_height(edges_bgr, 160), _pad_to_height(overlay, 160)])

        # Footer with metadata
        footer = np.full((80, combined.shape[1], 3), 32, dtype=np.uint8)
        color = (80, 220, 80) if label == "positive" else (60, 60, 220) if label == "negative" else (180, 180, 60)
        line1 = f"#{meta['idx']:02d} {label:<8} score={score:.3f} {'PASS' if score >= threshold else 'fail'} (t={threshold:.3f})"
        line2 = f"{meta['callsign']} {meta['wall_time_utc']}"
        cv2.putText(footer, line1, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(footer, line2, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        tile = np.vstack([combined, footer])
        tiles.append(tile)

    grid = _compose_grid(tiles, cols=4)
    cv2.imwrite(str(out_path), grid)


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    """Scale img to height h, preserving aspect ratio."""
    ih, iw = img.shape[:2]
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


def _write_report(
    md_path: Path,
    date: str,
    labels_summary: dict,
    results: list[dict],
    top_n: int = 10,
) -> None:
    best = results[0]
    combo = best["combo"]

    lines: list[str] = []
    lines.append(f"# Detection parameter sweep — {date}")
    lines.append("")
    lines.append(f"Positives: {labels_summary['positives']}  Negatives: {labels_summary['negatives']}  Skipped: {labels_summary['skipped']}")
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
    lines.append(f"  canny_low: {combo['canny_low']}")
    lines.append(f"  canny_high: {combo['canny_high']}")
    lines.append(f"  hough_threshold: {combo['hough_threshold']}")
    lines.append(f"  hough_min_line_length: {combo['hough_min_line_length']}")
    lines.append(f"  hough_max_line_gap: {combo['hough_max_line_gap']}")
    lines.append("aggregation:")
    lines.append(f"  detection_threshold: {best['threshold']:.3f}")
    lines.append("```")
    lines.append("")
    lines.append(f"## Top {top_n} parameter combinations")
    lines.append("")
    lines.append("| rank | canny_low | canny_high | hough_thr | min_line | max_gap | AUC | J | threshold | pos_med | neg_med |")
    lines.append("|------|-----------|------------|-----------|----------|---------|-----|---|-----------|---------|---------|")
    for i, r in enumerate(results[:top_n], 1):
        c = r["combo"]
        lines.append(
            f"| {i} | {c['canny_low']} | {c['canny_high']} | {c['hough_threshold']} | "
            f"{c['hough_min_line_length']} | {c['hough_max_line_gap']} | "
            f"{r['auc']:.3f} | {r['youden_j']:.3f} | {r['threshold']:.3f} | "
            f"{r['pos_median']:.3f} | {r['neg_median']:.3f} |"
        )
    lines.append("")
    lines.append("## Go/no-go decision")
    lines.append("")
    go = best["auc"] >= 0.75 and best["youden_j"] >= 0.5
    lines.append(
        f"Auto-assessment: **{'GO' if go else 'INVESTIGATE'}** — "
        f"AUC {'≥' if best['auc'] >= 0.75 else '<'} 0.75 and Youden's J {'≥' if best['youden_j'] >= 0.5 else '<'} 0.5."
    )
    lines.append("")
    lines.append(
        "If AUC < 0.75, the Hough+Canny path is not separating positives from "
        "negatives well enough and a different detector (e.g. CNN-based contrail "
        "classifier) should be investigated. Record the final decision here."
    )

    md_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--labels", required=True, help="labels.json produced by the labeller HTML")
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
    labels_path = Path(args.labels)
    labels = json.loads(labels_path.read_text())

    # Join labels with manifest by idx.
    label_by_idx = {entry["idx"]: entry["label"] for entry in labels["labels"]}
    rois: list[tuple[dict, np.ndarray, str]] = []
    for cand in manifest["candidates"]:
        label = label_by_idx.get(cand["idx"])
        if label is None:
            continue  # unlabeled
        roi_path = validation_dir / cand["roi_png"]
        roi_bgr = cv2.imread(str(roi_path))
        if roi_bgr is None:
            print(f"  WARN: failed to read {roi_path}")
            continue
        rois.append((cand, roi_bgr, label))

    positives = sum(1 for _, _, lbl in rois if lbl == "positive")
    negatives = sum(1 for _, _, lbl in rois if lbl == "negative")
    skipped = sum(1 for _, _, lbl in rois if lbl == "skip")

    print(f"Labeled ROIs: {len(rois)} (positive={positives}, negative={negatives}, skip={skipped})")
    if positives < 3 or negatives < 2:
        raise SystemExit(
            "Need at least 3 positives and 2 negatives for a meaningful sweep. "
            "Label more ROIs in the labeller HTML and re-export labels.json."
        )

    print(f"Running sweep over {len(CANNY_LOW) * len(CANNY_HIGH) * len(HOUGH_THRESHOLD) * len(HOUGH_MIN_LINE_LENGTH) * len(HOUGH_MAX_LINE_GAP)} parameter combinations...")
    # Skip skip-labeled ROIs from scoring (label != positive and != negative).
    scoring_rois = [(m, r, l) for m, r, l in rois if l in ("positive", "negative")]
    results = _sweep(scoring_rois)
    print(f"Top AUC: {results[0]['auc']:.3f}   threshold: {results[0]['threshold']:.3f}   combo: {results[0]['combo']}")

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
