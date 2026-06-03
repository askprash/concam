"""Tune detection params against episode-level reviewer labels.

Inputs:
  - labels/<date>_<reviewer>.json (episode_id → contrail/no_contrail)
  - ~/public_html/concam/<date>/manifest.json (episode_id → frames + geometry)
  - output/<date>/projections.jsonl (per-frame ROI + path vector)
  - /net/d16/data/contrail-camera/<date>.mp4 (raw video)

For each labeled episode we sample N evenly-spaced frames, cache a padded
crop around each frame's ROI, then sweep a small grid of detector parameters
(``cross_grad_gain``, ``canny_percentile_high``, ``score_length_norm_px``).
Per combo we replay ``concam.detection.detect`` on every cached crop,
aggregate to one peak score per labeled episode, and rank combos by
false-positive count at threshold=0.3 (lower = better) with AUC as a
tiebreaker.

Why episode-level and not frame-level? Reviewers tag whole flight passes
("is there a contrail anywhere in this pass?"), not individual ROIs.
An episode peaks through its strongest frame, so replaying on a thin sample
across the pass gives a faithful proxy for the produced peak_score.

Usage::

    uv run python scripts/tune_from_episode_labels.py \\
        --date 2026-04-15 \\
        --labels labels/2026-04-15_reviewer-1.json \\
        --manifest ~/public_html/concam/2026-04-15/manifest.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import DetectionConfig, load_config
from concam.detection import detect
from concam.detection.geometry import candidate_geometry
from concam.detection.metrics import mann_whitney_auc
from concam.video import decode_frames_sequential


# Sweep grid. Keep this small — the default ~20 labels × 8 frames/episode ≈ 160
# detect() calls per combo, so a 40-combo grid finishes in seconds.
CROSS_GRAD_GAIN = (0.5, 1.0, 1.5, 2.0, 3.0)
CANNY_PCT_HIGH = (99.5, 99.8, 99.9)
SCORE_LENGTH_NORM_PX = (130.0, 200.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path,
                   help="public manifest.json (has episode_id + per-frame wall_times)")
    p.add_argument("--projections", type=Path,
                   help="output/<date>/projections.jsonl (default inferred from --date)")
    p.add_argument("--video", type=Path,
                   help="raw video path (default inferred from --date)")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs" / "mit_green_building.yaml")
    p.add_argument("--frames-per-episode", type=int, default=8,
                   help="evenly-spaced frames to sample per episode")
    p.add_argument("--episode-threshold", type=float, default=0.3,
                   help="score threshold that classifies an episode as auto-detected")
    p.add_argument("--crop-pad-px", type=int, default=40,
                   help="padding around each ROI when caching crops; cross_grad Sobel "
                        "uses a 3x3 neighborhood so ≥2 px is safe — default is generous")
    return p.parse_args()


def load_labels(path: Path) -> dict[int, str]:
    """Return {episode_id: 'contrail' | 'no_contrail'}; skip 'unsure'."""
    data = json.loads(path.read_text())
    out: dict[int, str] = {}
    for entry in data["labels"]:
        lbl = entry.get("label")
        if lbl in ("contrail", "no_contrail"):
            out[int(entry["episode_id"])] = lbl
    return out


def load_projections_index(path: Path) -> dict[tuple[str, str], dict]:
    """Index projections.jsonl by (transponder_id, wall_time_utc) → full row."""
    idx: dict[tuple[str, str], dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            idx[(row["transponder_id"], row["wall_time_utc"])] = row
    return idx


def sample_frame_times(frames: list[dict], k: int) -> list[str]:
    """Pick k evenly-spaced wall_time_utc values across the episode's frames."""
    n = len(frames)
    if n == 0:
        return []
    if n <= k:
        return [f["wall_time_utc"] for f in frames]
    step = (n - 1) / (k - 1) if k > 1 else 0
    picks = [frames[int(round(i * step))]["wall_time_utc"] for i in range(k)]
    # De-dup in case rounding collides.
    seen = set()
    out = []
    for t in picks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_targets(
    labels: dict[int, str],
    manifest: dict,
    proj_idx: dict[tuple[str, str], dict],
    frames_per_episode: int,
) -> list[dict]:
    """Return one target per (episode, sampled-frame) with everything detect() needs."""
    eps_by_id = {int(e["episode_id"]): e for e in manifest["episodes"]}
    video_start = np.datetime64(manifest["video"]["start_utc"].replace("+00:00", ""))
    seconds_per_frame = float(manifest["video"]["seconds_per_frame"])

    targets: list[dict] = []
    missing_proj = 0
    for episode_id, label in labels.items():
        ep = eps_by_id.get(episode_id)
        if ep is None:
            print(f"[warn] episode {episode_id} not in manifest, skipping", file=sys.stderr)
            continue
        tid = ep["transponder_id"]
        frames = ep.get("frames", [])
        for wt in sample_frame_times(frames, frames_per_episode):
            proj = proj_idx.get((tid, wt))
            if proj is None:
                missing_proj += 1
                continue
            # Compute frame_idx: seconds since video start, divided by seconds_per_frame.
            dt = (np.datetime64(wt.replace("+00:00", "")) - video_start) \
                / np.timedelta64(1, "s")
            frame_idx = int(round(float(dt) / seconds_per_frame))
            targets.append({
                "episode_id": episode_id,
                "label": label,
                "transponder_id": tid,
                "wall_time_utc": wt,
                "frame_idx": frame_idx,
                "pixel_x": proj["pixel_x"],
                "pixel_y": proj["pixel_y"],
                "path_dx": proj["path_dx"],
                "path_dy": proj["path_dy"],
                "roi": proj["roi"],
            })
    if missing_proj:
        print(f"[warn] {missing_proj} frame(s) had no projection row (skipped)",
              file=sys.stderr)
    return targets


def extract_crops(video_path: Path, targets: list[dict], pad: int) -> dict[int, np.ndarray]:
    """Decode the video once; cache a padded BGR crop per target indexed by its list position.

    Scanning sequentially + emitting-on-match is much faster than seeking per
    target when targets are spread across the whole day.

    Frame decoding delegates to ``concam.video.decode_frames_sequential``; this
    function retains the per-target crop logic which is script-specific.
    """
    # Group targets by frame_idx to handle episodes whose sampled frames collide.
    by_frame: dict[int, list[int]] = {}
    for i, t in enumerate(targets):
        by_frame.setdefault(t["frame_idx"], []).append(i)
    if not by_frame:
        return {}

    # Decode all wanted frames in one sequential pass.
    decoded = decode_frames_sequential(video_path, list(by_frame.keys()))

    crops: dict[int, np.ndarray] = {}
    for fidx, tis in by_frame.items():
        img = decoded.get(fidx)
        if img is None:
            continue
        h, w = img.shape[:2]
        for ti in tis:
            roi = targets[ti]["roi"]
            x0 = max(0, int(roi["x"]) - pad)
            y0 = max(0, int(roi["y"]) - pad)
            x1 = min(w, int(roi["x"]) + int(roi["w"]) + pad)
            y1 = min(h, int(roi["y"]) + int(roi["h"]) + pad)
            crops[ti] = img[y0:y1, x0:x1].copy()
            targets[ti]["_crop_tl"] = (x0, y0)
            targets[ti]["_crop_shape"] = (y1 - y0, x1 - x0)
    return crops


def build_config(base: DetectionConfig, gain: float, pct_high: float,
                 score_norm: float) -> DetectionConfig:
    """Copy base config with sweep values overridden. preprocessing stays cross_grad."""
    return replace(
        base,
        preprocessing="cross_grad",
        cross_grad_gain=gain,
        canny_percentile_high=pct_high,
        score_length_norm_px=score_norm,
    )


def score_target(
    target: dict,
    crop: np.ndarray,
    config: DetectionConfig,
    extract_pad: int = 40,
) -> float:
    """Run detect() on one cached crop; return its score (0 on any failure).

    ``extract_pad`` must match the pad used when the crop was extracted
    (i.e. ``args.crop_pad_px``; default 40).  It is forwarded to
    :func:`~concam.detection.geometry.candidate_geometry` so that the
    crop-local coordinates are computed consistently with the crop boundary.
    """
    g = candidate_geometry(
        target, crop.shape[:2],
        roi_along_px=config.roi_along_px,
        roi_cross_px=config.roi_cross_px,
        extract_pad=extract_pad,
    )
    result = detect(crop, g.rect, config, polygon=g.polygon, path_vec=g.path_vec)
    return float(result.score)



def main() -> int:
    args = parse_args()

    projections_path = args.projections or (
        REPO_ROOT / "output" / args.date / "projections.jsonl")
    video_path = args.video or Path(
        f"/net/d16/data/contrail-camera/{args.date.replace('-', '_')}_0000_2359.mp4")

    print(f"[tune] loading labels from {args.labels}")
    labels = load_labels(args.labels)
    n_pos = sum(1 for v in labels.values() if v == "contrail")
    n_neg = sum(1 for v in labels.values() if v == "no_contrail")
    print(f"[tune] {len(labels)} episodes labeled "
          f"({n_pos} contrail / {n_neg} no_contrail)")

    print(f"[tune] loading manifest from {args.manifest}")
    manifest = json.loads(args.manifest.read_text())

    print(f"[tune] indexing projections from {projections_path}")
    proj_idx = load_projections_index(projections_path)

    targets = build_targets(labels, manifest, proj_idx, args.frames_per_episode)
    print(f"[tune] sampling {len(targets)} frames across {len(labels)} episodes")

    print(f"[tune] extracting crops from {video_path}")
    crops = extract_crops(video_path, targets, args.crop_pad_px)
    kept = [i for i in range(len(targets)) if i in crops]
    print(f"[tune] cached {len(kept)} crops "
          f"({len(targets) - len(kept)} missed — frame_idx past end of video?)")

    base_cfg = load_config(args.config).detection

    # Sweep.
    results = []
    for gain in CROSS_GRAD_GAIN:
        for pct in CANNY_PCT_HIGH:
            for norm in SCORE_LENGTH_NORM_PX:
                cfg = build_config(base_cfg, gain, pct, norm)
                per_episode_peak: dict[int, float] = {}
                for ti in kept:
                    t = targets[ti]
                    s = score_target(t, crops[ti], cfg, extract_pad=args.crop_pad_px)
                    ep_id = t["episode_id"]
                    per_episode_peak[ep_id] = max(per_episode_peak.get(ep_id, 0.0), s)
                pos_scores = [s for eid, s in per_episode_peak.items()
                              if labels[eid] == "contrail"]
                neg_scores = [s for eid, s in per_episode_peak.items()
                              if labels[eid] == "no_contrail"]
                fp = sum(1 for s in neg_scores if s >= args.episode_threshold)
                tp = sum(1 for s in pos_scores if s >= args.episode_threshold)
                fn = len(pos_scores) - tp
                results.append({
                    "gain": gain,
                    "pct_high": pct,
                    "score_norm": norm,
                    "auc": mann_whitney_auc(pos_scores, neg_scores),
                    "fp": fp,
                    "tp": tp,
                    "fn": fn,
                    "pos_mean": statistics.mean(pos_scores) if pos_scores else 0.0,
                    "neg_mean": statistics.mean(neg_scores) if neg_scores else 0.0,
                    "neg_max": max(neg_scores) if neg_scores else 0.0,
                })

    # Baseline (production defaults) for comparison.
    baseline_cfg = base_cfg
    baseline_peak: dict[int, float] = {}
    for ti in kept:
        t = targets[ti]
        s = score_target(t, crops[ti], baseline_cfg, extract_pad=args.crop_pad_px)
        ep_id = t["episode_id"]
        baseline_peak[ep_id] = max(baseline_peak.get(ep_id, 0.0), s)
    b_pos = [s for eid, s in baseline_peak.items() if labels[eid] == "contrail"]
    b_neg = [s for eid, s in baseline_peak.items() if labels[eid] == "no_contrail"]
    b_fp = sum(1 for s in b_neg if s >= args.episode_threshold)
    b_tp = sum(1 for s in b_pos if s >= args.episode_threshold)

    # Rank: fewest FP first, then highest AUC, then highest TP.
    results.sort(key=lambda r: (r["fp"], -r["auc"], -r["tp"]))

    print()
    print(f"Baseline (production defaults: gain={baseline_cfg.cross_grad_gain}, "
          f"pct_high={baseline_cfg.canny_percentile_high}, "
          f"score_norm={baseline_cfg.score_length_norm_px})")
    print(f"  AUC={mann_whitney_auc(b_pos, b_neg):.3f}  TP={b_tp}/{len(b_pos)}  "
          f"FP={b_fp}/{len(b_neg)}  neg_max={max(b_neg) if b_neg else 0:.3f}")
    print()
    print("Top 10 by (FP↓, AUC↑, TP↑):")
    hdr = f"{'gain':>6} {'pct':>6} {'norm':>6} | {'AUC':>5} {'TP':>4} {'FP':>4} " \
          f"{'FN':>4} {'neg_max':>8} {'pos_mean':>9} {'neg_mean':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results[:10]:
        print(f"{r['gain']:>6.2f} {r['pct_high']:>6.2f} {r['score_norm']:>6.0f} | "
              f"{r['auc']:>5.3f} {r['tp']:>4d} {r['fp']:>4d} {r['fn']:>4d} "
              f"{r['neg_max']:>8.3f} {r['pos_mean']:>9.3f} {r['neg_mean']:>9.3f}")

    print()
    best = results[0]
    print("Recommended YAML (paste under `detection:` in configs/...):")
    print(f"  cross_grad_gain: {best['gain']}")
    print(f"  canny_percentile_high: {best['pct_high']}")
    print(f"  score_length_norm_px: {best['score_norm']}")
    print()
    print("Caveats:")
    print(f"  - Tuned on {len(b_pos)} positives + {len(b_neg)} negatives from a single "
          "reviewer on a single day. Rankings within 1-2 FP of each other are noise.")
    print(f"  - ROC-AUC on ≤5 positives is very high-variance; prefer combos that "
          "agree across multiple top-10 rows.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
