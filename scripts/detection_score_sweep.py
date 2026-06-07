"""Score-function sweep for PRD item 25.

Loads the April-8 (or any dated) labeled candidate set, runs
``concam.detection.detect`` once per candidate at the current production config,
captures raw per-detection measurements (``num_long_lines``, ``aligned_lines``,
``contrail_length_px``), and then sweeps alternative score functions
analytically over those measurements.  This avoids re-running ``detect()`` per
scoring variant.

The goal is to pick a continuous score function that (a) preserves ROC AUC
against the baseline ``min(1, num_long_lines / 6)`` and (b) produces a
well-spread score distribution on the all-day run (90th-percentile score < 0.9
on the labeled positive set), so the score can act as a real confidence signal
for downstream consumers instead of bottoming out at 1.000.

Usage::

    uv run python scripts/detection_score_sweep.py \\
        --date 2026-04-08 \\
        --labels output/validation/detection/2026-04-08/labels.json \\
        --config configs/mit_green_building.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import DetectionConfig, load_config
from concam.detection import detect
from concam.detection.geometry import candidate_geometry
from concam.detection.metrics import mann_whitney_auc, rank_metric, youden_threshold


# mann_whitney_auc and youden_threshold live in concam.detection.metrics.



def _load_candidates(date: str, labels_path: Path) -> tuple[list[dict], dict]:
    with labels_path.open() as f:
        labels = json.load(f)
    manifest_path = labels_path.parent / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    label_by_idx = {l["idx"]: l for l in labels["labels"]}
    rois_dir = labels_path.parent
    candidates = []
    for cand in manifest["candidates"]:
        idx = cand["idx"]
        if idx not in label_by_idx:
            continue
        label = label_by_idx[idx]["label"]
        if label not in ("positive", "negative"):
            continue
        crop_path = rois_dir / cand["roi_png"]
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            continue
        candidates.append({"meta": cand, "crop": crop, "label": label})
    return candidates, manifest


# -----------------------------------------------------------------------------
# Score-function catalogue.  Each fn takes (num_long_lines, aligned_lines,
# contrail_length_px, roi_along_px, roi_cross_px, long_line_min_px) and returns
# a score in [0, 1].  Names describe the underlying math so the report is
# self-explanatory.
# -----------------------------------------------------------------------------

# Fixed reference normalization constants for the analytical score-function
# catalogue below.  These are intentionally NOT pulled from the loaded config
# at runtime: each function in SCORE_FNS is a distinct mathematical variant
# whose denominator is meant to be a stable reference (e.g. "length / diagonal
# of the 180×40 ROI").  Changing them per-run would alter what each function
# *means*, defeating the comparative analysis.
#
# These values MUST match configs/mit_green_building.yaml detection section.
# If production roi_along_px / roi_cross_px / long_line_min_px are ever
# re-tuned, update these constants and re-run the sweep to regenerate
# interpretable reference scores.
#
# Verified against configs/mit_green_building.yaml on 2026-06-07.
ROI_ALONG = 180   # == det.roi_along_px in the base YAML
ROI_CROSS = 40    # == det.roi_cross_px in the base YAML
LONG_MIN = 25.0   # == det.long_line_min_px in the base YAML
ROI_DIAG = math.hypot(ROI_ALONG, ROI_CROSS)


def _baseline(*, nll: int, **kw) -> float:
    return min(1.0, nll / 6.0)


def _norm_10(*, nll: int, **kw) -> float:
    return min(1.0, nll / 10.0)


def _norm_12(*, nll: int, **kw) -> float:
    return min(1.0, nll / 12.0)


def _sigmoid(*, nll: int, **kw) -> float:
    # 1 - exp(-n/k): k=4 → k lines gives score ~0.63, well-spread ramp.
    if nll <= 0:
        return 0.0
    return 1.0 - math.exp(-nll / 4.0)


def _length_over_diag(*, length_px: float, **kw) -> float:
    return min(1.0, length_px / ROI_DIAG)


def _length_over_along(*, length_px: float, **kw) -> float:
    return min(1.0, length_px / float(ROI_ALONG))


def _length_over_diag_scaled(*, length_px: float, **kw) -> float:
    # Scale so p99 (~ diag*0.65) maps near 1.0.
    return min(1.0, length_px / (ROI_DIAG * 0.70))


def _longest_single(*, length_px: float, long_min: float, **kw) -> float:
    # Fraction of ROI along-axis covered by longest single aligned line, but
    # length_px is already the span across all aligned long lines.  Treat as
    # approximation of longest streak.
    # (A tighter implementation would return max single line length from
    # detect() but that requires wiring another field through; deferring.)
    return min(1.0, length_px / float(ROI_ALONG))


def _combined(*, nll: int, length_px: float, **kw) -> float:
    # Hybrid: average of saturation-safe count score and length score.
    s_count = min(1.0, nll / 10.0)
    s_len = min(1.0, length_px / ROI_DIAG)
    return 0.5 * s_count + 0.5 * s_len


def _length_scaled_by_min(*, length_px: float, long_min: float, **kw) -> float:
    # Contrail length measured in multiples of long_line_min_px; capped at 6.
    if long_min <= 0:
        return 0.0
    return min(1.0, length_px / (long_min * 6.0))


SCORE_FNS: dict[str, Callable] = {
    "baseline_norm6": _baseline,
    "norm10": _norm_10,
    "norm12": _norm_12,
    "sigmoid_k4": _sigmoid,
    "length_over_diag": _length_over_diag,
    "length_over_along": _length_over_along,
    "length_over_diag_scaled0.7": _length_over_diag_scaled,
    "length_over_6xlong_min": _length_scaled_by_min,
    "combined_count_length": _combined,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--config", default="configs/mit_green_building.yaml")
    ap.add_argument("--output-dir", default="output/validation/detection")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    site = load_config(args.config)
    det_cfg = site.detection

    candidates, _ = _load_candidates(args.date, labels_path)
    pos = [c for c in candidates if c["label"] == "positive"]
    neg = [c for c in candidates if c["label"] == "negative"]
    print(f"Loaded {len(candidates)} labeled candidates: {len(pos)} pos / {len(neg)} neg")

    # Step 1: run detect() once per candidate at the production config.
    records = []
    for cand in candidates:
        meta = cand["meta"]
        crop = cand["crop"]
        g = candidate_geometry(
            meta, crop.shape[:2],
            roi_along_px=det_cfg.roi_along_px,
            roi_cross_px=det_cfg.roi_cross_px,
        )
        r = detect(crop, g.rect, det_cfg, polygon=g.polygon, path_vec=g.path_vec)
        records.append(
            {
                "idx": meta["idx"],
                "callsign": meta["callsign"],
                "label": cand["label"],
                "score": r.score,
                "num_long_lines": r.num_long_lines,
                "aligned_lines": r.aligned_lines,
                "contrail_length_px": r.contrail_length_px,
            }
        )

    # Step 2: evaluate score functions analytically.
    results = []
    for name, fn in SCORE_FNS.items():
        pos_scores, neg_scores = [], []
        for rec in records:
            s = fn(
                nll=rec["num_long_lines"],
                aligned=rec["aligned_lines"],
                length_px=rec["contrail_length_px"],
                long_min=LONG_MIN,
            )
            if rec["label"] == "positive":
                pos_scores.append(s)
            else:
                neg_scores.append(s)
        auc = mann_whitney_auc(pos_scores, neg_scores)
        threshold, youden_j = youden_threshold(pos_scores, neg_scores)
        p90 = float(np.percentile(pos_scores, 90)) if pos_scores else 0.0
        pos_saturated = sum(1 for s in pos_scores if s >= 1.0)
        results.append(
            {
                "name": name,
                "auc": auc,
                "youden_j": youden_j,
                "threshold": threshold,
                "pos_median": statistics.median(pos_scores) if pos_scores else 0.0,
                "pos_p90": p90,
                "pos_saturated": pos_saturated,
                "pos_n": len(pos_scores),
                "neg_median": statistics.median(neg_scores) if neg_scores else 0.0,
                "neg_max": max(neg_scores) if neg_scores else 0.0,
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
            }
        )

    # Rank by (auc, youden_j, -pos_saturated_fraction) so we prefer scores that
    # discriminate positives/negatives AND leave headroom on the positive set.
    results.sort(
        key=lambda r: (rank_metric(r["auc"]), rank_metric(r["youden_j"]), -r["pos_saturated"]),
        reverse=True,
    )

    # Report.
    out_dir = Path(args.output_dir) / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "score_sweep_report.md"
    json_path = out_dir / "score_sweep_results.json"

    with json_path.open("w") as f:
        json.dump({"records": records, "results": results}, f, indent=2, default=float)

    lines = [
        f"# Score-function sweep — {args.date}",
        "",
        f"Positives: {len(pos)}  Negatives: {len(neg)}",
        "",
        f"Detector config: {args.config} "
        f"(roi_along={det_cfg.roi_along_px}, roi_cross={det_cfg.roi_cross_px}, "
        f"long_line_min_px={det_cfg.long_line_min_px}, preprocessing={det_cfg.preprocessing})",
        "",
        "## Raw per-candidate measurements",
        "",
        "| idx | callsign | label | score | long | aligned | length_px |",
        "|-----|----------|-------|-------|------|---------|-----------|",
    ]
    for rec in records:
        lines.append(
            f"| {rec['idx']:3d} | {rec['callsign']:8s} | {rec['label']} | "
            f"{rec['score']:.3f} | {rec['num_long_lines']} | {rec['aligned_lines']} | "
            f"{rec['contrail_length_px']:.1f} |"
        )
    lines += ["", "## Score-function comparison", ""]
    lines += [
        "| fn | AUC | Youden-J | threshold | pos_med | pos_p90 | pos_sat@1.0 | neg_med | neg_max |",
        "|----|-----|----------|-----------|---------|---------|-------------|---------|---------|",
    ]
    for r in results:
        lines.append(
            f"| `{r['name']}` | {r['auc']:.3f} | {r['youden_j']:.3f} | {r['threshold']:.3f} | "
            f"{r['pos_median']:.3f} | {r['pos_p90']:.3f} | "
            f"{r['pos_saturated']}/{r['pos_n']} | "
            f"{r['neg_median']:.3f} | {r['neg_max']:.3f} |"
        )
    lines += ["", "## Interpretation", ""]
    baseline = next(r for r in results if r["name"] == "baseline_norm6")
    lines.append(
        f"Baseline (`min(1, nll/6)`): AUC={baseline['auc']:.3f}, "
        f"{baseline['pos_saturated']}/{baseline['pos_n']} positives saturate at 1.0."
    )
    top = results[0]
    lines.append(
        f"Top by joint (AUC, J, headroom): **`{top['name']}`** AUC={top['auc']:.3f} "
        f"J={top['youden_j']:.3f} pos_sat={top['pos_saturated']}/{top['pos_n']}."
    )
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
