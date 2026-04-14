"""Score every daylight flight on a given date with the current detector and
extract the top-N as candidates for human verification.

Unlike the bucket-sampled approach, this runs detect() on every unique
transponder in the daylight window so the candidate set is biased toward frames
where the detector already sees something contrail-like.  The user then verifies
true positives vs false positives in the labelling notebook.

Usage::

    uv run python scripts/detection_score_select.py \\
        --date 2025-10-19 \\
        --video /net/d16/data/contrail-camera/2025_10_19_1200_2359_av1.mp4 \\
        --seconds-per-frame 0.1667 \\
        --upscale-to-calibration \\
        --top-n 25

    # Append high-scoring candidates to an existing batch:
    uv run python scripts/detection_score_select.py \\
        --date 2025-10-19 ... \\
        --append-to-manifest output/validation/detection/2025-10-19/manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import DetectionConfig, load_config
from concam.detection import detect
from concam.projection import PixelPoint, Rect, rotated_polygon

# Re-use helpers from the extraction script rather than duplicating them.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from detection_validation_extract import (
    Candidate,
    _decode_frames,
    _extract_context_crop,
    _extract_roi_crop,
    _load_frame_zero_anchor,
    _load_projections,
    _annotate_tile,
    _compose_grid,
    _video_meta,
    _write_labeller_html,
)


def _sample_pings_per_transponder(
    projections: list[dict],
    daylight_start: str,
    daylight_end: str,
    samples_per_flight: int = 1,
) -> list[dict]:
    """Return up to ``samples_per_flight`` projection rows per unique transponder.

    Rows are sampled evenly across the transponder's full time window within the
    daylight filter so that a contrail visible only briefly (at formation or
    dissipation) has a chance to be included.  With samples_per_flight=1 the
    single row chosen is the ping closest to the image centre (original behaviour).
    """
    # Collect all daylight pings per transponder.
    by_tid: dict[str, list[dict]] = {}
    for row in projections:
        t = row["wall_time_utc"][11:16]
        if not (daylight_start <= t <= daylight_end):
            continue
        by_tid.setdefault(row["transponder_id"], []).append(row)

    cx, cy = 3840 / 2, 2160 / 2
    result: list[dict] = []
    for tid, rows in by_tid.items():
        rows_sorted = sorted(rows, key=lambda r: r["wall_time_utc"])
        if samples_per_flight <= 1:
            best = min(rows_sorted, key=lambda r: math.hypot(r["pixel_x"] - cx, r["pixel_y"] - cy))
            result.append(best)
        else:
            n = len(rows_sorted)
            # Evenly-spaced indices across the full transit window.
            if n <= samples_per_flight:
                sampled = rows_sorted
            else:
                idxs = [int(round(k * (n - 1) / (samples_per_flight - 1)))
                        for k in range(samples_per_flight)]
                sampled = [rows_sorted[i] for i in idxs]
            result.extend(sampled)
    return result


def _score_row(
    frame: np.ndarray,
    row: dict,
    config: DetectionConfig,
) -> float:
    """Run detect() on a single projection row and return the score."""
    roi_d = row["roi"]
    roi = Rect(x=roi_d["x"], y=roi_d["y"], w=roi_d["w"], h=roi_d["h"])
    center = PixelPoint(x=float(row["pixel_x"]), y=float(row["pixel_y"]))
    path_vec = (float(row["path_dx"]), float(row["path_dy"]))
    poly = rotated_polygon(center, path_vec, config)
    result = detect(frame, roi, config, polygon=poly, path_vec=path_vec)
    return result.score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--config", default="configs/mit_green_building.yaml")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--video", default=None)
    ap.add_argument("--seconds-per-frame", type=float, default=1.0)
    ap.add_argument("--daylight-utc", default="10:30,23:30")
    ap.add_argument("--top-n", type=int, default=25,
                    help="number of highest-scoring candidates to keep")
    ap.add_argument("--samples-per-flight", type=int, default=4,
                    help="number of frames to sample across each flight's transit window "
                         "(default 4 = start, 1/3, 2/3, end); best-scored frame per flight is kept")
    ap.add_argument("--score-threshold", type=float, default=0.0,
                    help="only keep candidates with score >= this (default 0 = keep all, sorted by score)")
    ap.add_argument("--upscale-to-calibration", action="store_true")
    ap.add_argument("--append-to-manifest", default=None,
                    help="path to existing manifest.json; new candidates are appended with higher idx values")
    args = ap.parse_args()

    import datetime
    site_config = load_config(args.config)
    det_config = site_config.detection

    output_dir = Path(args.output_dir) / args.date
    projections_path = output_dir / "projections.jsonl"
    ocr_path = output_dir / "ocr.jsonl"
    if not projections_path.exists():
        raise SystemExit(f"Missing {projections_path}")
    if not ocr_path.exists():
        raise SystemExit(f"Missing {ocr_path}")

    val_dir = Path(args.output_dir) / "validation" / "detection" / args.date
    rois_dir = val_dir / "rois"
    rois_dir.mkdir(parents=True, exist_ok=True)

    projections = _load_projections(projections_path)
    anchor_utc = _load_frame_zero_anchor(ocr_path)

    daylight_start, daylight_end = args.daylight_utc.split(",")
    daylight_start = daylight_start.strip()[:5]   # HH:MM
    daylight_end = daylight_end.strip()[:5]

    # Load existing manifest for append mode.
    existing_candidates: list[dict] = []
    if args.append_to_manifest:
        ap_path = Path(args.append_to_manifest)
        if ap_path.exists():
            existing_candidates = json.loads(ap_path.read_text()).get("candidates", [])
            print(f"Appending to existing manifest with {len(existing_candidates)} candidates.")
    start_idx = max((c["idx"] for c in existing_candidates), default=-1) + 1
    exclude_tids = {c["transponder_id"] for c in existing_candidates}

    all_rows = _sample_pings_per_transponder(
        projections, daylight_start, daylight_end,
        samples_per_flight=args.samples_per_flight,
    )
    rows = [r for r in all_rows if r["transponder_id"] not in exclude_tids]
    unique_tids = len({r["transponder_id"] for r in rows})
    print(f"Scoring {len(rows)} frames across {unique_tids} unique daylight transponders "
          f"({args.samples_per_flight} samples/flight)...")

    video_path = Path(args.video) if args.video else (
        Path(site_config.video.root) /
        site_config.video.timelapse_glob.format(date=datetime.date.fromisoformat(args.date))
    )
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    duration_s, total_frames = _video_meta(video_path)
    upscale_to = tuple(int(v) for v in site_config.calibration.calibration_resolution) \
        if args.upscale_to_calibration else None

    # Compute frame index for each row.
    def _frame_idx(row: dict) -> int:
        t = datetime.datetime.fromisoformat(row["wall_time_utc"])
        return int(round((t - anchor_utc).total_seconds() / args.seconds_per_frame))

    frame_indices = [_frame_idx(r) for r in rows]
    print(f"Decoding {len(frame_indices)} frames from {video_path.name}...")
    frames = _decode_frames(video_path, frame_indices, total_frames, duration_s,
                            upscale_to=upscale_to)

    # Score each row.
    scored: list[tuple[float, dict, int]] = []
    for row, fidx in zip(rows, frame_indices):
        frame = frames.get(fidx)
        if frame is None:
            continue
        score = _score_row(frame, row, det_config)
        scored.append((score, row, fidx))

    scored.sort(key=lambda t: t[0], reverse=True)

    # Keep only the best-scored frame per transponder so one chatty flight
    # can't fill the entire top-N list.
    seen_tids: set[str] = set()
    best_per_tid: list[tuple[float, dict, int]] = []
    for s, r, f in scored:
        tid = r["transponder_id"]
        if tid not in seen_tids:
            seen_tids.add(tid)
            best_per_tid.append((s, r, f))

    # Filter and take top-N.
    selected = [(s, r, f) for s, r, f in best_per_tid if s >= args.score_threshold]
    selected = selected[: args.top_n]

    print(f"\nTop {len(selected)} scored candidates (score_threshold={args.score_threshold}):")
    for score, row, _ in selected:
        print(f"  {score:.3f}  {row['callsign']:<10}  {row['wall_time_utc'][11:19]} UTC  "
              f"px=({row['pixel_x']:.0f},{row['pixel_y']:.0f})")

    # Build Candidate objects and extract crops.
    tiles: list[np.ndarray] = []
    manifest_candidates: list[dict] = []
    for new_idx, (score, row, fidx) in enumerate(selected):
        cand = Candidate(
            idx=start_idx + new_idx,
            frame_idx=fidx,
            wall_time_utc=row["wall_time_utc"],
            callsign=row["callsign"],
            transponder_id=row["transponder_id"],
            pixel_x=float(row["pixel_x"]),
            pixel_y=float(row["pixel_y"]),
            roi=row["roi"],
            path_dx=float(row["path_dx"]),
            path_dy=float(row["path_dy"]),
        )
        frame = frames[fidx]
        roi_crop = _extract_roi_crop(frame, cand.roi, pad=20)
        context_crop = _extract_context_crop(frame, cand, context_size=800)
        roi_png = rois_dir / f"roi_{cand.idx:02d}.png"
        ctx_png = rois_dir / f"context_{cand.idx:02d}.png"
        cv2.imwrite(str(roi_png), roi_crop)
        cv2.imwrite(str(ctx_png), context_crop)
        tiles.append(_annotate_tile(roi_crop, cand))
        manifest_candidates.append({
            "idx": cand.idx,
            "frame_idx": cand.frame_idx,
            "wall_time_utc": cand.wall_time_utc,
            "callsign": cand.callsign,
            "transponder_id": cand.transponder_id,
            "pixel_x": cand.pixel_x,
            "pixel_y": cand.pixel_y,
            "roi": cand.roi,
            "path_dx": cand.path_dx,
            "path_dy": cand.path_dy,
            "roi_png": f"rois/{roi_png.name}",
            "context_png": f"rois/{ctx_png.name}",
            "detection_score": score,
        })

    grid = _compose_grid(tiles, cols=5)
    grid_path = val_dir / "candidate_grid_scored.png"
    cv2.imwrite(str(grid_path), grid)

    all_candidates = existing_candidates + manifest_candidates
    import datetime as _dt
    manifest = {
        "schema_version": 1,
        "date": args.date,
        "video": str(video_path),
        "anchor_utc": anchor_utc.isoformat(),
        "daylight_utc": args.daylight_utc,
        "candidates": all_candidates,
    }
    manifest_path = val_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _write_labeller_html(val_dir / "labeller.html", manifest_path.name, len(all_candidates))

    print(f"\n  Grid     : {grid_path}")
    print(f"  Manifest : {manifest_path}  ({len(all_candidates)} total candidates)")
    print(f"  New ROIs : {rois_dir}")


if __name__ == "__main__":
    main()
