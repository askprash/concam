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
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import DetectionConfig, load_config
from concam.detection import detect, explain
from concam.detection.geometry import candidate_geometry
from concam.detection.metrics import mann_whitney_auc, rank_metric, youden_threshold
from concam.detection.viz import compose_grid, render_detection_panels
from concam.pipeline import resolve_video_path
from concam.video import decode_frames

# Parameter grid axes. 3^7 = 2187 combos × 20 ROIs × ~1 ms/detect() ≈ under a
# minute total. long_line_min_px is sweepable because the rotated ROI is only
# a 40 px floor (the sibling default) throws away shorter fragmented contrails.
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
# Default ROI dimensions: read from the loaded DetectionConfig at runtime.
# These module-level sentinels are overridden in main() once the config is loaded.
_DEFAULT_ROI_ALONG_PX = 180
_DEFAULT_ROI_CROSS_PX = 40

# ROI dimension sub-grid for --roi-sweep mode (item 20).
ROI_SWEEP_ALONG = (120, 180, 240, 320)
ROI_SWEEP_CROSS = (40, 60, 80)


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


def _combo_to_config(
    combo: Combo,
    preprocessing: str = "none",
    roi_along: int = _DEFAULT_ROI_ALONG_PX,
    roi_cross: int = _DEFAULT_ROI_CROSS_PX,
) -> DetectionConfig:
    return DetectionConfig(
        canny_low=50, canny_high=150,
        hough_threshold=int(combo.hough_threshold),
        hough_min_line_length=int(combo.hough_min_line_length),
        hough_max_line_gap=int(combo.hough_max_line_gap),
        roi_padding=20,
        roi_along_px=roi_along,
        roi_cross_px=roi_cross,
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
        preprocessing=preprocessing,
    )



def _extract_prev_crops(
    manifest: dict,
    video_path: Path,
    rois_meta: list[dict],
    pad: int = 20,
    upscale_to: tuple[int, int] | None = None,
) -> dict[int, np.ndarray]:
    """Decode the frame at frame_idx-1 for each candidate and return a {cand_idx: crop} dict.

    The crop is taken at the same ROI + pad as the saved roi_png so that calling
    ``detect(prev_crop, ...)`` produces a same-shaped temporal diff image.
    Candidates with frame_idx=0 are skipped (no previous frame).

    Frame decoding delegates to ``concam.video.decode_frames`` (seek-per-frame
    strategy); this function retains the crop logic which is script-specific.
    """
    # Build seek_frame_idx -> list[cand_idx] map.
    seek_to_cands: dict[int, list[int]] = {}
    for meta in rois_meta:
        fidx = meta.get("frame_idx", 0)
        if fidx > 0:
            seek_to_cands.setdefault(fidx - 1, []).append(meta["idx"])

    if not seek_to_cands:
        return {}

    # Decode all needed frames in one call.
    decoded_frames = decode_frames(video_path, list(seek_to_cands.keys()), upscale_to=upscale_to)

    prev_crops: dict[int, np.ndarray] = {}
    for seek_fidx, cand_idxs in seek_to_cands.items():
        arr = decoded_frames.get(seek_fidx)
        if arr is None:
            continue
        fh, fw = arr.shape[:2]
        for cand_idx in cand_idxs:
            meta = next(m for m in rois_meta if m["idx"] == cand_idx)
            roi = meta["roi"]
            x1 = max(0, roi["x"] - pad)
            y1 = max(0, roi["y"] - pad)
            x2 = min(fw, roi["x"] + roi["w"] + pad)
            y2 = min(fh, roi["y"] + roi["h"] + pad)
            prev_crops[cand_idx] = arr[y1:y2, x1:x2].copy()
    return prev_crops


def _sweep(
    rois: list[tuple[dict, np.ndarray, str]],
    preprocessing: str = "none",
    prev_crops: dict[int, np.ndarray] | None = None,
    roi_along: int = _DEFAULT_ROI_ALONG_PX,
    roi_cross: int = _DEFAULT_ROI_CROSS_PX,
) -> list[dict]:
    """For each parameter combination, score every labeled ROI and summarise."""
    results: list[dict] = []
    # Pre-build per-ROI geometry once since it doesn't depend on the combo.
    roi_geoms = [
        (meta, crop, label,
         candidate_geometry(meta, crop.shape[:2], roi_along_px=roi_along, roi_cross_px=roi_cross))
        for meta, crop, label in rois
    ]
    for pct_hi, lo_ratio, ht, hml, hmg, tol, lmin in itertools.product(
        CANNY_PCT_HIGH, CANNY_LOW_RATIO, HOUGH_THRESHOLD,
        HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP, ANGLE_TOLERANCE_DEG,
        LONG_LINE_MIN_PX,
    ):
        combo = Combo(pct_hi, lo_ratio, ht, hml, hmg, tol, lmin)
        cfg = _combo_to_config(combo, preprocessing=preprocessing, roi_along=roi_along, roi_cross=roi_cross)
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        for meta, crop, label, g in roi_geoms:
            prev_frame = prev_crops.get(meta["idx"]) if prev_crops else None
            result = detect(
                crop, g.rect, cfg, polygon=g.polygon, path_vec=g.path_vec,
                prev_frame=prev_frame, frame_origin=g.frame_origin,
            )
            if label == "positive":
                pos_scores.append(result.score)
            elif label == "negative":
                neg_scores.append(result.score)
        auc = mann_whitney_auc(pos_scores, neg_scores)
        threshold, youden_j = youden_threshold(pos_scores, neg_scores)
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
    results.sort(
        key=lambda r: (rank_metric(r["auc"]), rank_metric(r["youden_j"]), r["separation"]),
        reverse=True,
    )
    return results


def _score_combo_on_rois(
    rois: list[tuple[dict, np.ndarray, str]],
    combo: Combo,
    preprocessing: str,
    roi_along: int,
    roi_cross: int,
    prev_crops: dict[int, np.ndarray] | None = None,
) -> tuple[float, float, float, list[float], list[float]]:
    """Score a single parameter combo on the labeled ROIs.

    Returns (auc, youden_j, threshold, pos_scores, neg_scores).
    Used by the ROI dimension sub-grid sweep where other axes are frozen.
    """
    roi_geoms = [
        (meta, crop, label,
         candidate_geometry(meta, crop.shape[:2], roi_along_px=roi_along, roi_cross_px=roi_cross))
        for meta, crop, label in rois
    ]
    cfg = _combo_to_config(combo, preprocessing=preprocessing, roi_along=roi_along, roi_cross=roi_cross)
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for meta, crop, label, g in roi_geoms:
        prev_frame = prev_crops.get(meta["idx"]) if prev_crops else None
        result = detect(crop, g.rect, cfg, polygon=g.polygon, path_vec=g.path_vec,
                        prev_frame=prev_frame, frame_origin=g.frame_origin)
        if label == "positive":
            pos_scores.append(result.score)
        elif label == "negative":
            neg_scores.append(result.score)
    auc = mann_whitney_auc(pos_scores, neg_scores)
    threshold, youden_j = youden_threshold(pos_scores, neg_scores)
    return auc, youden_j, threshold, pos_scores, neg_scores


def _roi_dimension_sweep(
    rois: list[tuple[dict, np.ndarray, str]],
    frozen_combo: Combo,
    preprocessing: str,
    roi_alongs: tuple[int, ...] = ROI_SWEEP_ALONG,
    roi_crosses: tuple[int, ...] = ROI_SWEEP_CROSS,
    prev_crops: dict[int, np.ndarray] | None = None,
) -> list[dict]:
    """Sweep roi_along_px × roi_cross_px with all other axes frozen at frozen_combo.

    Returns a list of dicts with keys: roi_along, roi_cross, auc, youden_j,
    threshold, pos_median, neg_median, pos_scores, neg_scores.
    Sorted by (auc, youden_j) descending.
    """
    results: list[dict] = []
    n_cells = len(roi_alongs) * len(roi_crosses)
    print(f"  ROI dimension sub-grid: {len(roi_alongs)} along × {len(roi_crosses)} cross = {n_cells} cells")
    for along in roi_alongs:
        for cross in roi_crosses:
            auc, youden_j, threshold, pos_scores, neg_scores = _score_combo_on_rois(
                rois, frozen_combo, preprocessing, along, cross, prev_crops=prev_crops,
            )
            print(
                f"    roi_along={along:3d}  roi_cross={cross:2d}  "
                f"AUC={auc:.3f}  J={youden_j:.3f}  t={threshold:.3f}  "
                f"pos_med={statistics.median(pos_scores) if pos_scores else 0:.3f}  "
                f"neg_med={statistics.median(neg_scores) if neg_scores else 0:.3f}"
            )
            results.append({
                "roi_along": along,
                "roi_cross": cross,
                "auc": auc,
                "youden_j": youden_j,
                "threshold": threshold,
                "pos_median": statistics.median(pos_scores) if pos_scores else 0.0,
                "pos_min": min(pos_scores) if pos_scores else 0.0,
                "neg_median": statistics.median(neg_scores) if neg_scores else 0.0,
                "neg_max": max(neg_scores) if neg_scores else 0.0,
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
            })
    results.sort(
        key=lambda r: (rank_metric(r["auc"]), rank_metric(r["youden_j"])),
        reverse=True,
    )
    return results


def _write_roi_sweep_report(
    md_path: Path,
    date: str,
    labels_summary: dict,
    results: list[dict],
    frozen_combo: Combo,
    preprocessing: str,
    baseline_auc: float,
    baseline_j: float,
) -> None:
    """Write roi_dimension_sweep_report.md."""
    best = results[0]
    lines: list[str] = []
    lines.append(f"# ROI dimension sweep — {date}")
    lines.append("")
    lines.append(
        f"Positives: {labels_summary['positives']}  "
        f"Negatives: {labels_summary['negatives']}  "
        f"Skipped: {labels_summary['skipped']}"
    )
    lines.append("")
    lines.append(
        f"All other hyperparameters frozen at best cross_grad combo from prior sweep. "
        f"Preprocessing: **{preprocessing}**."
    )
    lines.append("")
    lines.append("## Frozen hyperparameters")
    lines.append("")
    lines.append("```yaml")
    lines.append(f"  canny_percentile_high: {frozen_combo.canny_percentile_high}")
    lines.append(f"  canny_low_ratio: {frozen_combo.canny_low_ratio}")
    lines.append(f"  hough_threshold: {frozen_combo.hough_threshold}")
    lines.append(f"  hough_min_line_length: {frozen_combo.hough_min_line_length}")
    lines.append(f"  hough_max_line_gap: {frozen_combo.hough_max_line_gap}")
    lines.append(f"  angle_tolerance_deg: {frozen_combo.angle_tolerance_deg}")
    lines.append(f"  long_line_min_px: {frozen_combo.long_line_min_px}")
    lines.append("```")
    lines.append("")
    lines.append("## Results grid (AUC / Youden-J)")
    lines.append("")

    # Build unique sorted lists for the table header.
    alongs = sorted(set(r["roi_along"] for r in results))
    crosses = sorted(set(r["roi_cross"] for r in results))
    lookup = {(r["roi_along"], r["roi_cross"]): r for r in results}

    # Header row
    header = "| roi_along \\ roi_cross |" + "".join(f" {c:2d} |" for c in crosses)
    sep = "|----------------------|" + "".join("------|" for _ in crosses)
    lines.append(header)
    lines.append(sep)
    for along in alongs:
        cells = []
        for cross in crosses:
            r = lookup.get((along, cross))
            if r:
                marker = " ★" if r["roi_along"] == best["roi_along"] and r["roi_cross"] == best["roi_cross"] else ""
                cells.append(f" {r['auc']:.3f}/{r['youden_j']:.3f}{marker} |")
            else:
                cells.append(" — |")
        lines.append(f"| {along:3d}px                |" + "".join(cells))
    lines.append("")
    lines.append(f"*(★ = best cell)*")
    lines.append("")

    lines.append("## Best (along, cross) pair")
    lines.append("")
    lines.append(f"- **roi_along_px = {best['roi_along']}**  roi_cross_px = {best['roi_cross']}")
    lines.append(f"- AUC: **{best['auc']:.3f}** (best prior combo: {baseline_auc:.3f})")
    lines.append(f"- Youden's J: **{best['youden_j']:.3f}** (baseline: {baseline_j:.3f})")
    lines.append(f"- Recommended threshold: {best['threshold']:.3f}")
    lines.append(f"- Positive scores: median {best['pos_median']:.3f}, min {best['pos_min']:.3f}")
    lines.append(f"- Negative scores: median {best['neg_median']:.3f}, max {best['neg_max']:.3f}")
    lines.append("")
    lines.append("```yaml")
    lines.append("detection:")
    lines.append(f"  roi_along_px: {best['roi_along']}")
    lines.append(f"  roi_cross_px: {best['roi_cross']}")
    lines.append(f"  preprocessing: {preprocessing}")
    lines.append(f"  canny_percentile_high: {frozen_combo.canny_percentile_high}")
    lines.append(f"  canny_percentile_low: {CANNY_PCT_LOW}")
    lines.append(f"  canny_low_ratio: {frozen_combo.canny_low_ratio}")
    lines.append(f"  canny_min_high: {CANNY_MIN_HIGH}")
    lines.append(f"  hough_threshold: {frozen_combo.hough_threshold}")
    lines.append(f"  hough_min_line_length: {frozen_combo.hough_min_line_length}")
    lines.append(f"  hough_max_line_gap: {frozen_combo.hough_max_line_gap}")
    lines.append(f"  angle_tolerance_deg: {frozen_combo.angle_tolerance_deg}")
    lines.append(f"  long_line_min_px: {frozen_combo.long_line_min_px}")
    lines.append(f"  score_norm_count: {SCORE_NORM_COUNT}")
    lines.append("  use_adaptive_canny: true")
    lines.append("  use_rotated_mask: true")
    lines.append("aggregation:")
    lines.append(f"  detection_threshold: {best['threshold']:.3f}")
    lines.append("```")
    md_path.write_text("\n".join(lines))


# mann_whitney_auc and youden_threshold live in concam.detection.metrics.


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((h, h, 3), dtype=np.uint8)
    scale = h / max(1, ih)
    nw = max(1, int(round(iw * scale)))
    return cv2.resize(img, (nw, h))



def _visualise_best_combo(
    best: dict,
    rois: list[tuple[dict, np.ndarray, str]],
    out_path: Path,
    preprocessing: str = "none",
    prev_crops: dict[int, np.ndarray] | None = None,
    roi_along: int = _DEFAULT_ROI_ALONG_PX,
    roi_cross: int = _DEFAULT_ROI_CROSS_PX,
) -> None:
    combo = Combo(**best["combo"])
    cfg = _combo_to_config(combo, preprocessing=preprocessing, roi_along=roi_along, roi_cross=roi_cross)
    threshold = float(best["threshold"])
    tiles: list[np.ndarray] = []
    for meta, crop, label in rois:
        g = candidate_geometry(meta, crop.shape[:2], roi_along_px=roi_along, roi_cross_px=roi_cross)
        prev_frame = prev_crops.get(meta["idx"]) if prev_crops else None

        # Obtain the exact DetectionPass the detector used so our panels show
        # what the detector actually saw (not a hand-rolled re-implementation).
        passed = explain(crop, g.rect, cfg, polygon=g.polygon, path_vec=g.path_vec,
                         prev_frame=prev_frame, frame_origin=g.frame_origin)
        result = detect(crop, g.rect, cfg, polygon=g.polygon, path_vec=g.path_vec,
                        prev_frame=prev_frame, frame_origin=g.frame_origin)

        vis_crop = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        # Draw the rotated polygon on the crop preview.
        overlay = vis_crop.copy()
        cv2.polylines(overlay, [g.polygon.astype(np.int32)], True, (0, 180, 255), 1)
        if result.pixel_line is not None:
            x1, y1, x2, y2 = (int(v) for v in result.pixel_line)
            cv2.line(overlay, (x1, y1), (x2, y2), (80, 220, 80), 2)

        # Use render_detection_panels to get the exact edges the detector computed.
        panels = render_detection_panels(passed, labels=False)
        edges_bgr = next(img for name, img in panels if name == "edges")

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

    grid = compose_grid(tiles, cols=4, bg=12)
    cv2.imwrite(str(out_path), grid)


def _write_report(
    md_path: Path,
    date: str,
    labels_summary: dict,
    results: list[dict],
    top_n: int = 10,
    preprocessing: str = "none",
    use_prev_frame: bool = False,
    roi_along: int = _DEFAULT_ROI_ALONG_PX,
    roi_cross: int = _DEFAULT_ROI_CROSS_PX,
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
    pp_desc = preprocessing if preprocessing != "none" else "none (raw grayscale)"
    prev_desc = " + temporal diff (prev_frame)" if use_prev_frame else ""
    lines.append(
        f"Detector: rotated-ROI + adaptive percentile Canny + angle-constrained Hough "
        f"(concam.detection.detect).  Preprocessing: **{pp_desc}**{prev_desc}."
    )
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
    lines.append(f"  roi_along_px: {roi_along}")
    lines.append(f"  roi_cross_px: {roi_cross}")
    lines.append("  use_adaptive_canny: true")
    lines.append("  use_rotated_mask: true")
    if preprocessing != "none":
        lines.append(f"  preprocessing: {preprocessing}")
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
    ap.add_argument(
        "--config",
        default="configs/mit_green_building.yaml",
        help="Site YAML config. ROI defaults (--roi-along/--roi-cross) are taken "
             "from detection.roi_along_px / detection.roi_cross_px in this file.",
    )
    ap.add_argument(
        "--preprocessing",
        default="none",
        choices=["none", "local_contrast", "cross_grad"],
        help="Spatial preprocessing applied before Canny (default: none). "
             "'local_contrast' subtracts a large-sigma Gaussian to suppress cloud backgrounds. "
             "'cross_grad' uses the Sobel gradient perpendicular to the flight path.",
    )
    ap.add_argument(
        "--use-prev-frame",
        action="store_true",
        help="Decode the preceding video frame as temporal diff input for each candidate. "
             "Requires the video file to be accessible (from manifest or --video).",
    )
    ap.add_argument(
        "--video",
        default=None,
        help="Override the video path from the manifest (useful when the stored path is stale).",
    )
    ap.add_argument(
        "--upscale-to-calibration",
        action="store_true",
        help="Bilinearly upscale decoded frames to 3840×2160 (4K calibration space) before "
             "cropping. Use for sub-4K videos (e.g. Oct 2025 720p archive).",
    )
    # ROI dimension overrides (single values; use --roi-sweep for the full sub-grid).
    # Defaults are set after config loading below; we use None as a sentinel here.
    ap.add_argument("--roi-along", type=int, default=None,
                    help="roi_along_px override (default: detection.roi_along_px from --config)")
    ap.add_argument("--roi-cross", type=int, default=None,
                    help="roi_cross_px override (default: detection.roi_cross_px from --config)")
    # ROI dimension sub-grid sweep (item 20). Freezes other axes at best combo from a prior run.
    ap.add_argument(
        "--roi-sweep",
        action="store_true",
        help="Run a roi_along × roi_cross sub-grid sweep with all other axes frozen at the "
             "best combo loaded from --best-combo-from. Outputs roi_dimension_sweep_report.md.",
    )
    ap.add_argument(
        "--best-combo-from",
        default=None,
        help="Path to a sweep_results_*.json whose top combo will be frozen for --roi-sweep. "
             "If omitted and --roi-sweep is set, defaults to sweep_results_cross_grad.json for "
             "the same date.",
    )
    args = ap.parse_args()

    # Load the site config so ROI defaults track production, not stale constants.
    site = load_config(args.config)
    det_cfg = site.detection
    roi_along = args.roi_along if args.roi_along is not None else det_cfg.roi_along_px
    roi_cross = args.roi_cross if args.roi_cross is not None else det_cfg.roi_cross_px

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

    # Optionally decode prev-frame crops for temporal diff preprocessing.
    prev_crops: dict[int, np.ndarray] | None = None
    if args.use_prev_frame:
        video_path = Path(args.video) if args.video else Path(manifest.get("video", ""))
        if not video_path.exists():
            raise SystemExit(
                f"Video not found at {video_path}. "
                "Pass --video <path> to override the manifest's stored video path."
            )
        upscale_to = (3840, 2160) if args.upscale_to_calibration else None
        print(f"Decoding prev-frame crops from {video_path} ...")
        scoring_meta = [m for m, _, _ in scoring_rois]
        prev_crops = _extract_prev_crops(manifest, video_path, scoring_meta, upscale_to=upscale_to)
        loaded = sum(1 for m in scoring_meta if m["idx"] in prev_crops)
        print(f"  Loaded {loaded}/{len(scoring_meta)} prev-frame crops.")

    # --- ROI dimension sub-grid sweep mode (--roi-sweep) ---
    if args.roi_sweep:
        best_combo_path = args.best_combo_from
        if best_combo_path is None:
            best_combo_path = str(validation_dir / "sweep_results_cross_grad.json")
        best_combo_path = Path(best_combo_path)
        if not best_combo_path.exists():
            raise SystemExit(
                f"Missing best-combo file: {best_combo_path}. "
                "Run the standard sweep first (--preprocessing cross_grad) or pass "
                "--best-combo-from <path>."
            )
        prior = json.loads(best_combo_path.read_text())
        frozen_dict = prior["results"][0]["combo"]
        frozen_combo = Combo(**frozen_dict)
        baseline_auc = prior["results"][0]["auc"]
        baseline_j = prior["results"][0]["youden_j"]
        print(
            f"Frozen combo (from {best_combo_path.name}): AUC={baseline_auc:.3f}  J={baseline_j:.3f}"
        )
        print(f"  {frozen_dict}")
        print()
        print(f"Running ROI dimension sub-grid: {ROI_SWEEP_ALONG} × {ROI_SWEEP_CROSS} ...")
        roi_results = _roi_dimension_sweep(
            scoring_rois, frozen_combo, args.preprocessing,
            roi_alongs=ROI_SWEEP_ALONG, roi_crosses=ROI_SWEEP_CROSS,
            prev_crops=prev_crops,
        )
        best_roi = roi_results[0]
        print()
        print(
            f"Best ROI: along={best_roi['roi_along']}  cross={best_roi['roi_cross']}  "
            f"AUC={best_roi['auc']:.3f}  J={best_roi['youden_j']:.3f}  "
            f"(best prior combo: AUC={baseline_auc:.3f}  J={baseline_j:.3f})"
        )
        roi_json_path = validation_dir / "roi_dimension_sweep_results.json"
        roi_json_path.write_text(
            json.dumps(
                {
                    "date": args.date,
                    "preprocessing": args.preprocessing,
                    "frozen_combo": frozen_dict,
                    "baseline_auc": baseline_auc,
                    "baseline_j": baseline_j,
                    "results": roi_results,
                },
                indent=2,
            )
        )
        roi_md_path = validation_dir / "roi_dimension_sweep_report.md"
        _write_roi_sweep_report(
            roi_md_path, args.date,
            {"positives": positives, "negatives": negatives, "skipped": skipped},
            roi_results, frozen_combo, args.preprocessing,
            baseline_auc, baseline_j,
        )
        # Visualise the best ROI combo using the frozen hyperparams.
        roi_vis_path = validation_dir / "best_roi_combo_visualisation.png"
        best_roi_as_sweep_result = {"combo": frozen_dict, "threshold": best_roi["threshold"]}
        _visualise_best_combo(
            best_roi_as_sweep_result, scoring_rois, roi_vis_path,
            preprocessing=args.preprocessing, prev_crops=prev_crops,
            roi_along=best_roi["roi_along"], roi_cross=best_roi["roi_cross"],
        )
        print()
        print(f"  Report        : {roi_md_path}")
        print(f"  Full results  : {roi_json_path}")
        print(f"  Visualisation : {roi_vis_path}")
        return

    # --- Standard full-combo sweep mode ---
    # roi_along / roi_cross already resolved from config + CLI overrides above.

    # Build output filename suffix so multiple preprocessing runs don't overwrite each other.
    suffix = args.preprocessing if args.preprocessing != "none" else "none"
    if args.use_prev_frame:
        suffix = f"diff_{suffix}" if suffix != "none" else "diff"
    if roi_along != det_cfg.roi_along_px or roi_cross != det_cfg.roi_cross_px:
        suffix = f"{suffix}_along{roi_along}_cross{roi_cross}"

    total_combos = (
        len(CANNY_PCT_HIGH) * len(CANNY_LOW_RATIO) * len(HOUGH_THRESHOLD)
        * len(HOUGH_MIN_LINE_LENGTH) * len(HOUGH_MAX_LINE_GAP) * len(ANGLE_TOLERANCE_DEG)
        * len(LONG_LINE_MIN_PX)
    )
    pp_label = args.preprocessing + (" +diff" if args.use_prev_frame else "")
    print(
        f"Running sweep over {total_combos} parameter combinations "
        f"(preprocessing={pp_label}, roi={roi_along}×{roi_cross})..."
    )
    results = _sweep(
        scoring_rois, preprocessing=args.preprocessing, prev_crops=prev_crops,
        roi_along=roi_along, roi_cross=roi_cross,
    )
    print(
        f"Top AUC: {results[0]['auc']:.3f}   threshold: {results[0]['threshold']:.3f}   "
        f"combo: {results[0]['combo']}"
    )

    json_path = validation_dir / f"sweep_results_{suffix}.json"
    json_path.write_text(json.dumps({"date": args.date, "preprocessing": suffix, "results": results}, indent=2))
    md_path = validation_dir / f"sweep_report_{suffix}.md"
    _write_report(
        md_path,
        args.date,
        {"positives": positives, "negatives": negatives, "skipped": skipped},
        results,
        top_n=args.top_n,
        preprocessing=args.preprocessing,
        use_prev_frame=args.use_prev_frame,
        roi_along=roi_along,
        roi_cross=roi_cross,
    )
    vis_path = validation_dir / f"best_combo_visualisation_{suffix}.png"
    _visualise_best_combo(
        results[0], scoring_rois, vis_path,
        preprocessing=args.preprocessing, prev_crops=prev_crops,
        roi_along=roi_along, roi_cross=roi_cross,
    )

    print()
    print(f"  Report        : {md_path}")
    print(f"  Full results  : {json_path}")
    print(f"  Visualisation : {vis_path}")


if __name__ == "__main__":
    main()
