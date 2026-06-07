"""Hyperparameter sweep for the contrail detector against multi-reviewer labels.

Sweeps four knobs the user approved for the 2026-04-09 HPO pass:

  * detect-time:    cross_grad_gain  × canny_percentile_high × roi_along_px
                    (5 × 5 × 4 = 100 combos requiring a detect() replay)
  * post-hoc:       aggregation.detection_threshold (5 thresholds applied to
                    the same per-episode peak scores)

  → 500 (combo, threshold) cells, but only 100 detect() replays.

Inputs:
  - One or more labels JSON files (the prash / thendo / reviewer-1 exports
    in labels/<date>_<reviewer>.json).
  - A public bundle manifest.json (episode_id → frames + ADS-B path).
  - The pipeline's projections.jsonl for that date.
  - The raw timelapse video.

Output: a directory with sweep_report.md (top combos by AUC/Youden-J), the
full results JSON, and the recommended YAML snippet for the winner.

Usage::

    uv run python scripts/detection_hpo.py \\
        --date 2026-04-09 \\
        --labels labels/2026-04-09_prash.json labels/2026-04-09_thendo.json \\
                 labels/2026-04-09_reviewer-1.json \\
        --manifest ~/public_html/concam/2026-04-09/manifest.json \\
        --out-dir output/validation/detection/2026-04-09/hpo
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import helpers from the single-reviewer tune script. They handle frame
# sampling, crop caching, and detect() replay; we only swap the sweep axes.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from tune_from_episode_labels import (  # noqa: E402
    build_targets,
    extract_crops,
    load_labels,
    load_projections_index,
    score_target,
)

from concam.config import DetectionConfig, load_config  # noqa: E402
from concam.detection.metrics import mann_whitney_auc, rank_metric, youden_at  # noqa: E402


# 4-knob grid the user approved. detection_threshold is post-hoc: it filters
# per-episode peak scores into TP/FP/FN buckets without re-running detect().
CROSS_GRAD_GAIN = (0.5, 0.75, 1.0, 1.5, 2.0)
CANNY_PCT_HIGH = (99.0, 99.3, 99.5, 99.7, 99.9)
ROI_ALONG_PX = (120, 180, 240, 320)
DETECTION_THRESHOLDS = (0.05, 0.083, 0.12, 0.18, 0.25)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--labels", required=True, type=Path, nargs="+",
                   help="One or more labels JSON files (prash / thendo / reviewer-1).")
    p.add_argument("--manifest", required=True, type=Path,
                   help="Public bundle manifest.json (episode_id → frames + ADS-B path).")
    p.add_argument("--projections", type=Path,
                   help="output/<date>/projections.jsonl (default inferred from --date).")
    p.add_argument("--video", type=Path,
                   help="Raw video path (default inferred from --date).")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs" / "mit_green_building.yaml")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory to write sweep_report.md and sweep_results.json into.")
    p.add_argument("--frames-per-episode", type=int, default=8)
    p.add_argument("--crop-pad-px", type=int, default=200,
                   help="Pad around each per-frame ROI when caching crops. Must exceed "
                        "max(roi_along_px)/2 + safety margin so the rotated polygon "
                        "stays inside the cached crop at every roi_along we sweep.")
    return p.parse_args()


def merge_labels(paths: list[Path]) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Merge multiple {episode_id: label} dicts.

    Returns (unified_label, sources_per_episode). Conflicts use the first
    file's label and emit a warning; ``sources_per_episode`` lets the caller
    report which reviewers covered which episodes.
    """
    unified: dict[int, str] = {}
    sources: dict[int, list[str]] = {}
    for path in paths:
        labels = load_labels(path)
        labeler = json.loads(path.read_text()).get("labeler_id", path.stem)
        for ep_id, label in labels.items():
            sources.setdefault(ep_id, []).append(labeler)
            if ep_id in unified and unified[ep_id] != label:
                print(f"[hpo] CONFLICT episode {ep_id}: "
                      f"{unified[ep_id]} (kept) vs {label} ({labeler})",
                      file=sys.stderr)
                continue
            unified[ep_id] = label
    return unified, sources


def build_config_for_combo(
    base: DetectionConfig, gain: float, pct_high: float, roi_along: int
) -> DetectionConfig:
    """Return a DetectionConfig with the three detect-time knobs overridden.

    score_length_norm_px and roi_cross_px stay at base values so the sweep
    isolates the 3 detect-time knobs. preprocessing is forced to cross_grad
    (the prod default).
    """
    return replace(
        base,
        preprocessing="cross_grad",
        cross_grad_gain=gain,
        canny_percentile_high=pct_high,
        roi_along_px=roi_along,
    )



def evaluate_combo(
    per_episode_peak: dict[int, float],
    labels: dict[int, str],
    thresholds: tuple[float, ...],
) -> dict:
    """Compute AUC + per-threshold (TP, FP, FN, Youden-J) for one combo."""
    pos = [s for eid, s in per_episode_peak.items() if labels[eid] == "contrail"]
    neg = [s for eid, s in per_episode_peak.items() if labels[eid] == "no_contrail"]
    a = mann_whitney_auc(pos, neg)
    per_thr = []
    for thr in thresholds:
        tp = sum(1 for s in pos if s >= thr)
        fp = sum(1 for s in neg if s >= thr)
        fn = len(pos) - tp
        per_thr.append({
            "threshold": thr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": len(neg) - fp,
            "youden_j": youden_at(pos, neg, thr),
        })
    return {
        "auc": a,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "pos_mean": statistics.mean(pos) if pos else 0.0,
        "neg_mean": statistics.mean(neg) if neg else 0.0,
        "neg_max": max(neg) if neg else 0.0,
        "pos_min": min(pos) if pos else 0.0,
        "per_threshold": per_thr,
    }


def main() -> int:
    args = parse_args()

    projections_path = args.projections or (
        REPO_ROOT / "output" / args.date / "projections.jsonl")
    video_path = args.video or Path(
        f"/net/d16/data/contrail-camera/{args.date.replace('-', '_')}_0000_2359.mp4")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hpo] merging labels from {len(args.labels)} files")
    labels, sources = merge_labels(args.labels)
    n_pos = sum(1 for v in labels.values() if v == "contrail")
    n_neg = sum(1 for v in labels.values() if v == "no_contrail")
    print(f"[hpo] unified: {len(labels)} episodes ({n_pos} contrail / {n_neg} no_contrail)")

    print(f"[hpo] manifest: {args.manifest}")
    manifest = json.loads(args.manifest.read_text())

    print(f"[hpo] projections: {projections_path}")
    proj_idx = load_projections_index(projections_path)

    targets = build_targets(labels, manifest, proj_idx, args.frames_per_episode)
    print(f"[hpo] sampled {len(targets)} frames across {len(labels)} episodes")

    print(f"[hpo] extracting crops from {video_path} (pad={args.crop_pad_px})")
    crops = extract_crops(video_path, targets, args.crop_pad_px)
    kept = [i for i in range(len(targets)) if i in crops]
    print(f"[hpo] cached {len(kept)} crops "
          f"({len(targets) - len(kept)} missed — frame_idx past end of video?)")

    base_cfg = load_config(args.config).detection

    n_combos = len(CROSS_GRAD_GAIN) * len(CANNY_PCT_HIGH) * len(ROI_ALONG_PX)
    print(f"[hpo] running {n_combos} detect-config combos × {len(DETECTION_THRESHOLDS)} "
          f"thresholds = {n_combos * len(DETECTION_THRESHOLDS)} ranking cells")

    results: list[dict] = []
    for i, gain in enumerate(CROSS_GRAD_GAIN):
        for pct in CANNY_PCT_HIGH:
            for roi_along in ROI_ALONG_PX:
                cfg = build_config_for_combo(base_cfg, gain, pct, roi_along)
                per_ep_peak: dict[int, float] = {}
                for ti in kept:
                    t = targets[ti]
                    s = score_target(t, crops[ti], cfg)
                    eid = t["episode_id"]
                    per_ep_peak[eid] = max(per_ep_peak.get(eid, 0.0), s)
                metrics = evaluate_combo(per_ep_peak, labels, DETECTION_THRESHOLDS)
                results.append({
                    "cross_grad_gain": gain,
                    "canny_percentile_high": pct,
                    "roi_along_px": roi_along,
                    **metrics,
                })
        print(f"[hpo] {(i + 1) * len(CANNY_PCT_HIGH) * len(ROI_ALONG_PX)}/"
              f"{n_combos} combos done")

    # Best combo per threshold by Youden-J; AUC tiebreak.
    def rank_key(r: dict, thr_idx: int) -> tuple:
        pt = r["per_threshold"][thr_idx]
        return (-pt["youden_j"], -r["auc"], pt["fp"], -pt["tp"])

    # Single overall ranking: pick each combo's best Youden-J across the 5
    # thresholds, sort by that. Ties broken by AUC.
    for r in results:
        best_thr = max(r["per_threshold"], key=lambda pt: pt["youden_j"])
        r["best_threshold"] = best_thr["threshold"]
        r["best_youden_j"] = best_thr["youden_j"]
        r["best_tp"] = best_thr["tp"]
        r["best_fp"] = best_thr["fp"]
        r["best_fn"] = best_thr["fn"]

    results.sort(key=lambda r: (-rank_metric(r["best_youden_j"]), -rank_metric(r["auc"]), r["best_fp"]))

    # Baseline at production defaults.
    baseline_cfg = base_cfg
    base_peak: dict[int, float] = {}
    for ti in kept:
        t = targets[ti]
        s = score_target(t, crops[ti], baseline_cfg)
        eid = t["episode_id"]
        base_peak[eid] = max(base_peak.get(eid, 0.0), s)
    baseline_metrics = evaluate_combo(base_peak, labels, DETECTION_THRESHOLDS)
    base_best_thr = max(baseline_metrics["per_threshold"],
                        key=lambda pt: pt["youden_j"])

    # Write outputs.
    out_json = args.out_dir / "sweep_results.json"
    out_json.write_text(json.dumps({
        "date": args.date,
        "labels_merged_from": [str(p) for p in args.labels],
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_kept_frames": len(kept),
        "grid": {
            "cross_grad_gain": list(CROSS_GRAD_GAIN),
            "canny_percentile_high": list(CANNY_PCT_HIGH),
            "roi_along_px": list(ROI_ALONG_PX),
            "detection_thresholds": list(DETECTION_THRESHOLDS),
        },
        "baseline": {
            "cross_grad_gain": baseline_cfg.cross_grad_gain,
            "canny_percentile_high": baseline_cfg.canny_percentile_high,
            "roi_along_px": baseline_cfg.roi_along_px,
            **baseline_metrics,
        },
        "results": results,
    }, indent=2) + "\n")

    out_md = args.out_dir / "sweep_report.md"
    lines = [
        f"# HPO sweep — {args.date}",
        "",
        f"- **Label files merged:** {', '.join(p.name for p in args.labels)}",
        f"- **Unified label set:** {n_pos} contrail / {n_neg} no_contrail "
        f"({n_pos + n_neg} strict labels)",
        f"- **Frames per episode:** {args.frames_per_episode}",
        f"- **Combos evaluated:** {n_combos} detect-config × "
        f"{len(DETECTION_THRESHOLDS)} thresholds",
        "",
        "## Baseline (current production config)",
        "",
        f"- `cross_grad_gain={baseline_cfg.cross_grad_gain}`, "
        f"`canny_percentile_high={baseline_cfg.canny_percentile_high}`, "
        f"`roi_along_px={baseline_cfg.roi_along_px}`",
        f"- AUC: **{baseline_metrics['auc']:.3f}**",
        f"- Best Youden-J: **{base_best_thr['youden_j']:.3f}** at "
        f"threshold={base_best_thr['threshold']} "
        f"(TP={base_best_thr['tp']}/{baseline_metrics['n_pos']}, "
        f"FP={base_best_thr['fp']}/{baseline_metrics['n_neg']})",
        "",
        "## Top 10 combos by best-Youden-J (tie-break: AUC, FP)",
        "",
        "| rank | gain | pct_high | roi_along | thr | AUC | YJ | TP | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results[:10], 1):
        lines.append(
            f"| {i} | {r['cross_grad_gain']} | {r['canny_percentile_high']} | "
            f"{r['roi_along_px']} | {r['best_threshold']} | "
            f"{r['auc']:.3f} | {r['best_youden_j']:.3f} | "
            f"{r['best_tp']} | {r['best_fp']} | {r['best_fn']} |"
        )

    best = results[0]
    lines += [
        "",
        "## Recommended YAML snippet (for `configs/mit_green_building.tuned.yaml`)",
        "",
        "```yaml",
        "detection:",
        f"  cross_grad_gain: {best['cross_grad_gain']}",
        f"  canny_percentile_high: {best['canny_percentile_high']}",
        f"  roi_along_px: {best['roi_along_px']}",
        "aggregation:",
        f"  detection_threshold: {best['best_threshold']}",
        "```",
        "",
        "## Caveats",
        "",
        f"- Tuned on a single date (2026-04-09) with merged labels from "
        f"{len(args.labels)} reviewers.",
        "- Combos within ~1-2 FP of each other are noise on a sample this size; "
        "prefer combos where multiple top-10 rows agree on each axis.",
        "- detection_threshold is post-hoc: changing it after the sweep is fine, "
        "but the per-episode score histogram itself depends on the three detect-time "
        "knobs.",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    # Also print a brief summary to stdout.
    print()
    print(f"Baseline: AUC={baseline_metrics['auc']:.3f}, "
          f"best_YJ={base_best_thr['youden_j']:.3f} @ thr={base_best_thr['threshold']}")
    print(f"Best combo: gain={best['cross_grad_gain']}, "
          f"pct_high={best['canny_percentile_high']}, "
          f"roi_along={best['roi_along_px']}, thr={best['best_threshold']}")
    print(f"  AUC={best['auc']:.3f}, YJ={best['best_youden_j']:.3f}, "
          f"TP={best['best_tp']}/{best['n_pos']}, FP={best['best_fp']}/{best['n_neg']}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
