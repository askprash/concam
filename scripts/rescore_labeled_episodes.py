#!/usr/bin/env python3
"""Re-score labeled episodes with and without the static-scene mask.

Measures the mask's real effect on detector output without re-running whole
days: for every reliably-labeled episode, re-run `detect()` on its top-K
above-threshold frames (the mask only *removes* edges, so scores are
monotonically <= the unmasked scores — episodes that never fired stay at 0
and need no decode).

Frame resolution follows the production pipeline exactly (see
docs/pts_drift_bug.md and the eval-phase OCR scout):
  * wall_time -> frame_idx via the ocr.jsonl inverse map keyed on verbatim
    second-resolution ISO strings (drift-collision seconds map to TWO frames;
    re-score both and take the max, as production scored both);
  * `detect()` gets `prev_frame` = decode-order frame at frame_idx - 1;
  * geometry (Rect / rotated polygon / path_vec) is rebuilt from
    projections.jsonl just as concam/pipeline/stages.py run_detect_stage does.

The no-mask arm doubles as the harness's validation: it must reproduce the
stored detections.jsonl scores (report lists any mismatch > 1e-6).

Usage:
    uv run python scripts/rescore_labeled_episodes.py --date 2026-04-09 \\
        --out output/validation/mask_rescore/2026-04-09.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mit_green_building.yaml"
RELIABLE = REPO_ROOT / "labels" / "derived" / "reliable_labels.json"
PUBLIC_ROOT = Path.home() / "public_html" / "concam"
TOP_K = 5


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _second_key(iso: str) -> str:
    return (
        datetime.datetime.fromisoformat(iso).replace(microsecond=0).isoformat()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--video", type=Path, default=None)
    args = ap.parse_args()

    from concam.config import load_config
    from concam.detection import detect
    from concam.projection import PixelPoint, Rect, rotated_polygon
    from concam.video import decode_frames

    site = load_config(args.config)
    det_masked = site.detection
    det_nomask = dataclasses.replace(det_masked, static_mask_path=None)
    threshold = site.aggregation.detection_threshold

    video = args.video or Path(
        f"/net/d16/data/contrail-camera/{args.date.replace('-', '_')}_0000_2359.mp4"
    )
    out_dir = REPO_ROOT / "output" / args.date

    labels = json.loads(RELIABLE.read_text())["labels"].get(args.date, {})
    manifest = json.loads(
        (PUBLIC_ROOT / args.date / "manifest.json").read_text()
    )
    eps = {str(e["episode_id"]): e for e in manifest["episodes"]}

    # wall second -> [frame_idx], production source of truth.
    inv: dict[str, list[int]] = defaultdict(list)
    for rec in _iter_jsonl(out_dir / "ocr.jsonl"):
        inv[_second_key(rec["wall_time_utc"])].append(rec["frame_idx"])

    # (tid, wall second) -> projection record.
    proj_index: dict[tuple[str, str], dict] = {}
    for rec in _iter_jsonl(out_dir / "projections.jsonl"):
        proj_index[(rec["transponder_id"], _second_key(rec["wall_time_utc"]))] = rec

    # Select work: top-K above-threshold frames per labeled episode.
    work = []  # (eid, label, tid, wall_key, stored_score)
    skipped_zero = 0
    for eid, lab in labels.items():
        ep = eps.get(eid)
        if ep is None:
            continue
        strong = [f for f in ep["frames"] if (f.get("score") or 0.0) >= threshold]
        if not strong:
            skipped_zero += 1
            continue
        strong.sort(key=lambda f: -f["score"])
        for f in strong[: args.top_k]:
            work.append((eid, lab["label"], ep["transponder_id"],
                         _second_key(f["wall_time_utc"]), float(f["score"])))

    frame_ids = sorted({
        fi for _, _, _, wall, _ in work for fi in inv.get(wall, [])
    })
    decode_ids = sorted({fi for f in frame_ids for fi in (f - 1, f) if fi >= 0})
    print(f"[rescore] {args.date}: {len(labels)} labeled, {skipped_zero} never "
          f"fired (skip), {len(work)} frame-tasks, {len(decode_ids)} decodes")

    frames = decode_frames(video, decode_ids)
    print(f"[rescore] decoded {len(frames)}/{len(decode_ids)}")

    results: dict[str, dict] = {}
    mismatches = 0
    for eid, label, tid, wall, stored in work:
        proj = proj_index.get((tid, wall))
        if proj is None:
            continue
        roi = Rect(x=proj["roi"]["x"], y=proj["roi"]["y"],
                   w=proj["roi"]["w"], h=proj["roi"]["h"])
        center = PixelPoint(x=float(proj["pixel_x"]), y=float(proj["pixel_y"]))
        path_vec = (float(proj["path_dx"]), float(proj["path_dy"]))

        best_nomask = 0.0
        best_masked = 0.0
        for fi in inv.get(wall, []):
            frame = frames.get(fi)
            if frame is None:
                continue
            prev = frames.get(fi - 1)
            poly = rotated_polygon(center, path_vec, det_nomask)
            r0 = detect(frame, roi, det_nomask, polygon=poly,
                        path_vec=path_vec, prev_frame=prev)
            r1 = detect(frame, roi, det_masked, polygon=poly,
                        path_vec=path_vec, prev_frame=prev)
            best_nomask = max(best_nomask, r0.score)
            best_masked = max(best_masked, r1.score)

        entry = results.setdefault(eid, {
            "label": label, "stored_peak": 0.0,
            "nomask_peak": 0.0, "masked_peak": 0.0,
        })
        entry["stored_peak"] = max(entry["stored_peak"], stored)
        entry["nomask_peak"] = max(entry["nomask_peak"], best_nomask)
        entry["masked_peak"] = max(entry["masked_peak"], best_masked)
        if abs(best_nomask - stored) > 1e-6:
            mismatches += 1

    # Episode-level summary at the production threshold.
    def detected(v):
        return v >= threshold

    summary = {"fp_killed": 0, "fp_total": 0, "tp_killed": 0, "tp_total": 0}
    for entry in results.values():
        if not detected(entry["stored_peak"]):
            continue
        if entry["label"] == "no_contrail":
            summary["fp_total"] += 1
            if not detected(entry["masked_peak"]):
                summary["fp_killed"] += 1
        elif entry["label"] == "contrail":
            summary["tp_total"] += 1
            if not detected(entry["masked_peak"]):
                summary["tp_killed"] += 1

    payload = {
        "date": args.date,
        "threshold": threshold,
        "top_k": args.top_k,
        "frame_task_nomask_mismatches": mismatches,
        "summary": summary,
        "episodes": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"[rescore] nomask-vs-stored mismatched frame-tasks: {mismatches}")
    print(f"[rescore] FP killed {summary['fp_killed']}/{summary['fp_total']}, "
          f"TP killed {summary['tp_killed']}/{summary['tp_total']}")
    print(f"[rescore] wrote {args.out}")


if __name__ == "__main__":
    main()
