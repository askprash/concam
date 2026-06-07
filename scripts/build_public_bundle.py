#!/usr/bin/env python3
"""Build a public-review bundle where every in-scene flight is a sidebar episode.

The production ``concam bundle`` only exposes flights whose detection score
passed the operational threshold.  External reviewers need to see *all*
in-scene flights so they can catch detector false negatives.  This script
synthesizes one sidebar episode per flight pass by walking
``projections.jsonl`` (for the in-scene window) and ``detections.jsonl``
(for per-frame scores + pixel lines), regardless of score.

Episodes are split on gaps larger than ``max_gap_seconds`` (mirrors the
aggregation stage) so a flight re-entering the scene gets a fresh entry.

Reuses the existing per-labeler manifest for ``flight_tracks``,
``image_size``, ``video`` metadata, and the labeler.html asset.

Usage:
    uv run python scripts/build_public_bundle.py \\
        --date 2026-04-09 \\
        --source-bundle output/2026-04-09/bundles/prash \\
        --out-dir public_bundle/2026-04-09
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

from concam.bundle import calibration_block
from concam.config import load_config
from concam.projection import load_calibration

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mit_green_building.yaml"
LABELER_TEMPLATE = REPO_ROOT / "concam" / "bundle" / "templates" / "labeler.html"


def _parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _split_on_gap(frames: list[dict], max_gap_seconds: float) -> list[list[dict]]:
    if not frames:
        return []
    frames = sorted(frames, key=lambda r: r["t"])
    runs: list[list[dict]] = [[frames[0]]]
    for cur in frames[1:]:
        prev = runs[-1][-1]
        if (cur["t"] - prev["t"]).total_seconds() > max_gap_seconds:
            runs.append([cur])
        else:
            runs[-1].append(cur)
    return runs


def _build_flight_tracks_with_altitude(projections_path: Path) -> dict[str, dict]:
    """Like concam.bundle._build_flight_tracks but carries per-ping altitude.

    The labeler uses ``alt_baro_m`` to render the flight level on the on-screen
    dot label (FL = round(alt_baro_ft / 100)). ``alt_m`` is the effective
    altitude used for projection and is kept as a fallback when barometric is
    missing.
    """
    tracks: dict[str, dict] = {}
    for rec in _iter_jsonl(projections_path):
        tid = rec["transponder_id"]
        entry = tracks.setdefault(
            tid,
            {
                "callsign": rec.get("callsign") or tid,
                "transponder_id": tid,
                "pings": [],
            },
        )
        entry["pings"].append(
            {
                "wall_time_utc": rec["wall_time_utc"],
                "pixel_x": rec["pixel_x"],
                "pixel_y": rec["pixel_y"],
                "alt_m": rec.get("alt_m"),
                "alt_baro_m": rec.get("alt_baro_m"),
            }
        )
    for entry in tracks.values():
        entry["pings"].sort(key=lambda p: p["wall_time_utc"])
    return tracks


def build_manifest(
    source_manifest: dict,
    projections_path: Path,
    detections_path: Path,
    max_gap_seconds: float,
    detection_threshold: float,
) -> dict:
    per_tid_frames: dict[str, list[dict]] = {}
    per_tid_callsign: dict[str, str] = {}

    for rec in _iter_jsonl(projections_path):
        tid = rec["transponder_id"]
        per_tid_callsign.setdefault(tid, rec.get("callsign") or tid)
        per_tid_frames.setdefault(tid, []).append(
            {
                "t": _parse_iso(rec["wall_time_utc"]),
                "wall_time_utc": rec["wall_time_utc"],
                "score": 0.0,
                "pixel_line": None,
            }
        )

    # Overlay detection scores/lines onto the frame dicts, keyed by
    # (transponder_id, wall_time_utc).
    det_index: dict[tuple[str, str], dict] = {}
    for rec in _iter_jsonl(detections_path):
        det_index[(rec["transponder_id"], rec["wall_time_utc"])] = rec

    for tid, frames in per_tid_frames.items():
        for f in frames:
            det = det_index.get((tid, f["wall_time_utc"]))
            if det is None:
                continue
            f["score"] = float(det.get("score") or 0.0)
            f["pixel_line"] = det.get("pixel_line")

    episodes_out: list[dict] = []
    next_eid = 1
    for tid, frames in per_tid_frames.items():
        for run in _split_on_gap(frames, max_gap_seconds):
            peak = max(run, key=lambda f: f["score"])
            onset = run[0]["t"]
            end = run[-1]["t"]
            episodes_out.append(
                {
                    "episode_id": next_eid,
                    "callsign": per_tid_callsign[tid],
                    "transponder_id": tid,
                    "onset": onset.isoformat(),
                    "end": end.isoformat(),
                    "frame_count": len(run),
                    "peak_score": float(peak["score"]),
                    "peak_pixel_line": peak["pixel_line"],
                    "peak_contrail_length_m": None,
                    "is_overlap": False,
                    "frames": [
                        {
                            "wall_time_utc": f["wall_time_utc"],
                            "score": float(f["score"]),
                            "pixel_line": f["pixel_line"],
                        }
                        for f in run
                    ],
                }
            )
            next_eid += 1

    episodes_out.sort(key=lambda e: (e["onset"], e["episode_id"]))

    manifest = {
        "schema_version": source_manifest.get("schema_version", 1),
        "date": source_manifest["date"],
        "labeler_id": "public-review",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "video": source_manifest["video"],
        "image_size": source_manifest["image_size"],
        "detection_threshold": detection_threshold,
        "overlap_episode_ids": [],
        "episodes": episodes_out,
        "flight_tracks": source_manifest["flight_tracks"],
    }
    return manifest


def _merge_altitudes_into_flight_tracks(
    flight_tracks: dict[str, dict],
    projections_path: Path,
) -> None:
    """Attach alt_m / alt_baro_m to each ping by (tid, wall_time_utc) key."""
    alt_index: dict[tuple[str, str], dict] = {}
    for rec in _iter_jsonl(projections_path):
        alt_index[(rec["transponder_id"], rec["wall_time_utc"])] = {
            "alt_m": rec.get("alt_m"),
            "alt_baro_m": rec.get("alt_baro_m"),
        }
    for tid, track in flight_tracks.items():
        for ping in track["pings"]:
            alt = alt_index.get((tid, ping["wall_time_utc"]))
            if alt is not None:
                ping["alt_m"] = alt["alt_m"]
                ping["alt_baro_m"] = alt["alt_baro_m"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--source-bundle", required=True, type=Path,
                    help="Existing bundle dir with manifest.json + labeler.html")
    ap.add_argument("--projections",  type=Path, default=None)
    ap.add_argument("--detections", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    site = load_config(args.config)
    projections = args.projections or (
        Path("output") / args.date / "projections.jsonl"
    )
    detections = args.detections or (
        Path("output") / args.date / "detections.jsonl"
    )

    source_manifest_path = args.source_bundle / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())

    manifest = build_manifest(
        source_manifest=source_manifest,
        projections_path=projections,
        detections_path=detections,
        max_gap_seconds=site.aggregation.max_gap_seconds,
        detection_threshold=site.aggregation.detection_threshold,
    )
    # Rebuild flight_tracks from the current projections.jsonl rather than
    # trusting the source bundle's snapshot — the pipeline may have been
    # re-run since the bundle was generated, in which case the source manifest
    # misses any flights added by the rerun.
    manifest["flight_tracks"] = _build_flight_tracks_with_altitude(projections)
    manifest["calibration"] = calibration_block(load_calibration(site.calibration))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest))
    # Copy labeler.html from the canonical template so UI updates take effect
    # on the next bundle run without needing a full pipeline rebuild.
    shutil.copy2(LABELER_TEMPLATE, args.out_dir / "labeler.html")

    positives = sum(
        1 for ep in manifest["episodes"]
        if ep["peak_score"] >= manifest["detection_threshold"]
    )
    print(f"[{args.date}] wrote {args.out_dir}")
    print(f"  episodes={len(manifest['episodes'])} (detector-positive={positives})")
    print(f"  flight_tracks={len(manifest['flight_tracks'])}")


if __name__ == "__main__":
    main()
