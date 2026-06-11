"""Hyperparameter sweep for the contrail detector against episode labels.

Sweep axes (June-2026 reliable-label retune):

  * detect-time:    preprocessing variant × canny_percentile_high × roi_along_px
                    (10 × 4 × 3 = 120 combos requiring a detect() replay).
                    The preprocessing axis covers every mode the kernel
                    supports — "none", "local_contrast" (× sigma) and
                    "cross_grad" (× gain, on the finest grid: it is the
                    production default and the most promising mode).
  * post-hoc:       aggregation.detection_threshold (5 thresholds applied to
                    the same per-episode peak scores)

  → 600 (combo, threshold) cells, but only 120 detect() replays.

Labels come from EITHER:
  - ``--reliable-labels labels/derived/reliable_labels.json`` (preferred): the
    cross-generation consensus set whose episode IDs match the *current*
    public manifests (ADR-0003, docs/label_reliability.md); or
  - ``--labels`` <one or more raw reviewer exports> (legacy; only valid when
    every file is in the current manifest's episode-ID space).

Episodes are filtered to a daylight window on their manifest ``onset``
(``--daylight-utc``, default 11:00–22:30 UTC; pass ``all`` to disable).

Other inputs:
  - A public bundle manifest.json (episode_id → frames + ADS-B path).
  - The pipeline's projections.jsonl for that date.
  - The raw timelapse video.

The static-scene mask stays exactly as configured in the base YAML (i.e. ON
for production configs); crop replays anchor it correctly via the
``frame_origin`` plumbing in ``concam.detection``.

Output: a directory with sweep_report.md (top combos by AUC/Youden-J), the
full results JSON, and the recommended YAML snippet for the winner.

Usage::

    uv run python scripts/detection_hpo.py \\
        --date 2026-04-08 \\
        --reliable-labels labels/derived/reliable_labels.json \\
        --manifest ~/public_html/concam/2026-04-08/manifest.json \\
        --out-dir output/hpo/2026-04-08
"""

from __future__ import annotations

import argparse
import datetime
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


# --- Sweep grid (June-2026 reliable-label daytime retune) -------------------
#
# Preprocessing variants: every mode DetectionConfig supports, each with its
# own knob grid.  (variant_id, config-overrides) pairs; the variant_id is the
# stable key the holdout-selection step matches combos on.
#
#   * cross_grad gain (0.5, 0.75, 1.0, 1.25, 1.5, 2.0): finest grid, centred
#     on the production value 1.0. Provenance: the 04-09 reviewer-1 sweep
#     (configs/mit_green_building.yaml notes, 2026-04-22) found gain 2.0
#     saturates bright cloud edges (FP 12/15) while 0.5 starts losing faint
#     contrails; 0.75/1.25 close the gap around the production optimum.
#   * local_contrast sigma (15, 25, 40): brackets the DetectionConfig default
#     (25.0, the only value ever run); ±~60% probes whether the background
#     subtraction scale matters before investing in a finer grid.
#   * none: no-preprocessing control.
#
# Frangi / NRBR / DoG / tophat / CLAHE exist in concam/detection/transforms.py
# but are NOT wired into the kernel's `preprocessing` option, so they are
# deliberately not swept here (wiring them in is a separate feature).
PREPROC_VARIANTS: tuple[tuple[str, dict], ...] = (
    ("none", {"preprocessing": "none"}),
    ("local_contrast(s=15)",
     {"preprocessing": "local_contrast", "local_contrast_sigma": 15.0}),
    ("local_contrast(s=25)",
     {"preprocessing": "local_contrast", "local_contrast_sigma": 25.0}),
    ("local_contrast(s=40)",
     {"preprocessing": "local_contrast", "local_contrast_sigma": 40.0}),
    ("cross_grad(g=0.5)", {"preprocessing": "cross_grad", "cross_grad_gain": 0.5}),
    ("cross_grad(g=0.75)", {"preprocessing": "cross_grad", "cross_grad_gain": 0.75}),
    ("cross_grad(g=1)", {"preprocessing": "cross_grad", "cross_grad_gain": 1.0}),
    ("cross_grad(g=1.25)", {"preprocessing": "cross_grad", "cross_grad_gain": 1.25}),
    ("cross_grad(g=1.5)", {"preprocessing": "cross_grad", "cross_grad_gain": 1.5}),
    ("cross_grad(g=2)", {"preprocessing": "cross_grad", "cross_grad_gain": 2.0}),
)
# Shared detect-time axes.  canny pct brackets production 99.5 with finer
# spacing than the May round; 99.9 dropped (99.5 already beat 99.8 on the
# 04-09 reviewer-1 set — see configs/mit_green_building.yaml).  roi_along
# brackets production 180; 320 dropped to keep the grid at 120 combos
# (~6 h budget at ~20 ms/detect on the ~5.5 k daylight crops of 04-08+04-09)
# and because 320/2 = 160 px leaves no safety margin inside the 200 px crop
# pad.  Other knobs (blur_kernel, angle_tolerance_deg, long_line_min_px,
# hough_*) stay at base-config values; score_length_norm_px is not swept
# because score = min(1, length/norm) — below saturation it is a pure score
# rescaling, already covered by the post-hoc threshold axis.
CANNY_PCT_HIGH = (99.0, 99.3, 99.5, 99.7)
ROI_ALONG_PX = (120, 180, 240)
# detection_threshold is post-hoc: it buckets per-episode peak scores into
# TP/FP/FN without re-running detect().
DETECTION_THRESHOLDS = (0.05, 0.083, 0.12, 0.18, 0.25)

# Daylight window default. The crop-extract convention for April at Boston is
# civil daylight ~10:30–23:30 UTC (scripts/detection_validation_extract.py);
# the 2026-06 labeled-episode review found the detector strongest roughly
# 12:00–20:00 UTC. 11:00–22:30 trims the grazing-light dawn/dusk hours while
# keeping most of the labeled volume (04-08: 357/565, 04-09: 329/469).
DEFAULT_DAYLIGHT_UTC = "11:00,22:30"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--labels", type=Path, nargs="+",
                     help="One or more raw reviewer label exports (legacy; episode "
                          "IDs must be in the CURRENT manifest's ID space).")
    grp.add_argument("--reliable-labels", type=Path,
                     help="labels/derived/reliable_labels.json — the consensus "
                          "label set keyed {date: {episode_id: {label, ...}}} "
                          "(ADR-0003); --date selects the day.")
    p.add_argument("--daylight-utc", default=DEFAULT_DAYLIGHT_UTC,
                   help="UTC window HH:MM,HH:MM; only episodes whose manifest "
                        "onset falls inside it are tuned/evaluated. "
                        f"Default {DEFAULT_DAYLIGHT_UTC}; pass 'all' to disable.")
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


def load_reliable_labels(
    path: Path, date: str
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Load one date's labels from the consensus reliable-label set.

    ``labels/derived/reliable_labels.json`` is keyed
    ``{"labels": {date: {episode_id: {"label", "labelers", "votes"}}}}`` with
    episode IDs in the CURRENT public-manifest ID space (ADR-0003). Returns
    (``{episode_id: label}``, ``{episode_id: labelers}``) — the same shape as
    :func:`merge_labels` — keeping only definite contrail / no_contrail labels.
    """
    data = json.loads(path.read_text())
    day = (data.get("labels") or {}).get(date)
    if day is None:
        raise SystemExit(
            f"[hpo] {path} has no labels for {date} "
            f"(available: {sorted((data.get('labels') or {}).keys())})")
    unified: dict[int, str] = {}
    sources: dict[int, list[str]] = {}
    for ep_id, rec in day.items():
        label = rec.get("label")
        if label not in ("contrail", "no_contrail"):
            continue
        unified[int(ep_id)] = label
        sources[int(ep_id)] = list(rec.get("labelers", []))
    return unified, sources


def parse_daylight_window(
    spec: str,
) -> tuple[datetime.time, datetime.time] | None:
    """Parse 'HH:MM,HH:MM' into (start, end) UTC times; 'all' disables."""
    if spec.strip().lower() in ("all", ""):
        return None
    start_s, end_s = spec.split(",")
    h0, m0 = (int(v) for v in start_s.strip().split(":"))
    h1, m1 = (int(v) for v in end_s.strip().split(":"))
    return datetime.time(h0, m0), datetime.time(h1, m1)


def filter_daylight(
    labels: dict[int, str],
    manifest: dict,
    window: tuple[datetime.time, datetime.time] | None,
) -> dict[int, str]:
    """Keep only episodes whose manifest onset time-of-day is inside ``window``.

    Onsets are compared in UTC (manifest onsets carry +00:00 offsets).
    Labeled episodes missing from the manifest are dropped here too — they
    would be skipped later by build_targets anyway, but dropping them up front
    keeps the reported positive/negative counts honest.
    """
    if window is None:
        return dict(labels)
    start, end = window
    eps_by_id = {int(e["episode_id"]): e for e in manifest["episodes"]}
    out: dict[int, str] = {}
    for ep_id, label in labels.items():
        ep = eps_by_id.get(ep_id)
        if ep is None:
            continue
        onset = datetime.datetime.fromisoformat(ep["onset"])
        if onset.tzinfo is not None:
            onset = onset.astimezone(datetime.timezone.utc)
        if start <= onset.time() <= end:
            out[ep_id] = label
    return out


def build_config_for_combo(
    base: DetectionConfig, variant_overrides: dict, pct_high: float, roi_along: int
) -> DetectionConfig:
    """Return a DetectionConfig with the detect-time sweep knobs overridden.

    ``variant_overrides`` sets the preprocessing mode + its knob (see
    PREPROC_VARIANTS); pct_high / roi_along are the shared axes. Everything
    else — including static_mask_path, so the static-scene mask stays ON —
    keeps the base-config value.
    """
    return replace(
        base,
        canny_percentile_high=pct_high,
        roi_along_px=roi_along,
        **variant_overrides,
    )


def variant_id_for_config(cfg: DetectionConfig) -> str:
    """Canonical variant string for a config (matches PREPROC_VARIANTS ids)."""
    if cfg.preprocessing == "cross_grad":
        return f"cross_grad(g={cfg.cross_grad_gain:g})"
    if cfg.preprocessing == "local_contrast":
        return f"local_contrast(s={cfg.local_contrast_sigma:g})"
    return cfg.preprocessing



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

    if args.reliable_labels is not None:
        print(f"[hpo] loading reliable labels for {args.date} from {args.reliable_labels}")
        labels, sources = load_reliable_labels(args.reliable_labels, args.date)
        label_sources = [f"{args.reliable_labels}#{args.date}"]
    else:
        print(f"[hpo] merging labels from {len(args.labels)} files")
        labels, sources = merge_labels(args.labels)
        label_sources = [str(p) for p in args.labels]

    print(f"[hpo] manifest: {args.manifest}")
    manifest = json.loads(args.manifest.read_text())

    window = parse_daylight_window(args.daylight_utc)
    n_all = len(labels)
    labels = filter_daylight(labels, manifest, window)
    if window is not None:
        print(f"[hpo] daylight window {args.daylight_utc} UTC: "
              f"kept {len(labels)}/{n_all} labeled episodes")
    n_pos = sum(1 for v in labels.values() if v == "contrail")
    n_neg = sum(1 for v in labels.values() if v == "no_contrail")
    print(f"[hpo] tuning set: {len(labels)} episodes "
          f"({n_pos} contrail / {n_neg} no_contrail)")
    if not labels:
        raise SystemExit("[hpo] no labeled episodes left after filtering — aborting")

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

    n_combos = len(PREPROC_VARIANTS) * len(CANNY_PCT_HIGH) * len(ROI_ALONG_PX)
    print(f"[hpo] running {n_combos} detect-config combos × {len(DETECTION_THRESHOLDS)} "
          f"thresholds = {n_combos * len(DETECTION_THRESHOLDS)} ranking cells")

    results: list[dict] = []
    for i, (variant, overrides) in enumerate(PREPROC_VARIANTS):
        for pct in CANNY_PCT_HIGH:
            for roi_along in ROI_ALONG_PX:
                cfg = build_config_for_combo(base_cfg, overrides, pct, roi_along)
                per_ep_peak: dict[int, float] = {}
                for ti in kept:
                    t = targets[ti]
                    s = score_target(t, crops[ti], cfg, extract_pad=args.crop_pad_px)
                    eid = t["episode_id"]
                    per_ep_peak[eid] = max(per_ep_peak.get(eid, 0.0), s)
                metrics = evaluate_combo(per_ep_peak, labels, DETECTION_THRESHOLDS)
                results.append({
                    "variant": variant,
                    "preprocessing": cfg.preprocessing,
                    "cross_grad_gain": cfg.cross_grad_gain,
                    "local_contrast_sigma": cfg.local_contrast_sigma,
                    "canny_percentile_high": pct,
                    "roi_along_px": roi_along,
                    **metrics,
                })
        print(f"[hpo] {(i + 1) * len(CANNY_PCT_HIGH) * len(ROI_ALONG_PX)}/"
              f"{n_combos} combos done", flush=True)

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
        s = score_target(t, crops[ti], baseline_cfg, extract_pad=args.crop_pad_px)
        eid = t["episode_id"]
        base_peak[eid] = max(base_peak.get(eid, 0.0), s)
    baseline_metrics = evaluate_combo(base_peak, labels, DETECTION_THRESHOLDS)
    base_best_thr = max(baseline_metrics["per_threshold"],
                        key=lambda pt: pt["youden_j"])

    # Write outputs.
    out_json = args.out_dir / "sweep_results.json"
    out_json.write_text(json.dumps({
        "date": args.date,
        "labels_merged_from": label_sources,
        "daylight_utc": args.daylight_utc,
        "n_labeled_total": n_all,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_kept_frames": len(kept),
        "grid": {
            "preproc_variants": [v for v, _ in PREPROC_VARIANTS],
            "canny_percentile_high": list(CANNY_PCT_HIGH),
            "roi_along_px": list(ROI_ALONG_PX),
            "detection_thresholds": list(DETECTION_THRESHOLDS),
        },
        "baseline": {
            "variant": variant_id_for_config(baseline_cfg),
            "preprocessing": baseline_cfg.preprocessing,
            "cross_grad_gain": baseline_cfg.cross_grad_gain,
            "local_contrast_sigma": baseline_cfg.local_contrast_sigma,
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
        f"- **Labels:** {', '.join(Path(s).name for s in label_sources)}",
        f"- **Daylight window (UTC, onset):** {args.daylight_utc} "
        f"({len(labels)}/{n_all} labeled episodes kept)",
        f"- **Tuning set:** {n_pos} contrail / {n_neg} no_contrail "
        f"({n_pos + n_neg} strict labels)",
        f"- **Frames per episode:** {args.frames_per_episode}",
        f"- **Combos evaluated:** {n_combos} detect-config × "
        f"{len(DETECTION_THRESHOLDS)} thresholds",
        "",
        "## Baseline (current production config)",
        "",
        f"- `{variant_id_for_config(baseline_cfg)}`, "
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
        "| rank | variant | pct_high | roi_along | thr | AUC | YJ | TP | FP | FN |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results[:10], 1):
        lines.append(
            f"| {i} | {r['variant']} | {r['canny_percentile_high']} | "
            f"{r['roi_along_px']} | {r['best_threshold']} | "
            f"{r['auc']:.3f} | {r['best_youden_j']:.3f} | "
            f"{r['best_tp']} | {r['best_fp']} | {r['best_fn']} |"
        )

    best = results[0]
    best_yaml = [
        "detection:",
        f"  preprocessing: {best['preprocessing']}",
    ]
    if best["preprocessing"] == "cross_grad":
        best_yaml.append(f"  cross_grad_gain: {best['cross_grad_gain']}")
    elif best["preprocessing"] == "local_contrast":
        best_yaml.append(f"  local_contrast_sigma: {best['local_contrast_sigma']}")
    best_yaml += [
        f"  canny_percentile_high: {best['canny_percentile_high']}",
        f"  roi_along_px: {best['roi_along_px']}",
        "aggregation:",
        f"  detection_threshold: {best['best_threshold']}",
    ]
    lines += [
        "",
        "## Recommended YAML snippet (for `configs/mit_green_building.tuned.yaml`)",
        "",
        "```yaml",
        *best_yaml,
        "```",
        "",
        "## Caveats",
        "",
        f"- Tuned on a single date ({args.date}) within the {args.daylight_utc} UTC "
        "daylight window; night/dawn behaviour is unmeasured here.",
        "- Combos within ~1-2 FP of each other are noise on a sample this size; "
        "prefer combos where multiple top-10 rows agree on each axis.",
        "- detection_threshold is post-hoc: changing it after the sweep is fine, "
        "but the per-episode score histogram itself depends on the detect-time "
        "knobs.",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    # Also print a brief summary to stdout.
    print()
    print(f"Baseline: AUC={baseline_metrics['auc']:.3f}, "
          f"best_YJ={base_best_thr['youden_j']:.3f} @ thr={base_best_thr['threshold']}")
    print(f"Best combo: {best['variant']}, "
          f"pct_high={best['canny_percentile_high']}, "
          f"roi_along={best['roi_along_px']}, thr={best['best_threshold']}")
    print(f"  AUC={best['auc']:.3f}, YJ={best['best_youden_j']:.3f}, "
          f"TP={best['best_tp']}/{best['n_pos']}, FP={best['best_fp']}/{best['n_neg']}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
