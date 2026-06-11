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

from concam.adsb import _haversine_km
from concam.bundle import calibration_block, exclusion_regions_block
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


def compact_pings(pings: list[dict], *, step_s: float = 2.0) -> list[dict]:
    """Compact + thin a track's ping list for the manifest.

    The verbose ping schema dominated manifest size (61 MB of a 91 MB
    manifest on 2026-04-09: 341k pings with 16-digit floats, duplicated
    altitude fields, and 28-byte ISO timestamps). Compact schema per ping:

      t          epoch milliseconds (int) — labeler uses Date.parse anyway
      x, y       pixel position, 1 decimal (sub-0.1px precision is noise)
      alt_baro_m rounded int; alt_m emitted ONLY when barometric is missing
                 (mirrors the labeler's fallback order)
      dist_km    unchanged; omitted when null

    Thinning keeps one ping per ``step_s`` seconds plus the final ping —
    ADS-B is upsampled to 1 Hz by linear interpolation, so intermediate
    pings carry no extra information at overlay scale. The labeler accepts
    both this and the legacy verbose schema.
    """
    out: list[dict] = []
    last_kept: float | None = None
    for i, p in enumerate(pings):
        t = _parse_iso(p["wall_time_utc"]).timestamp()
        is_last = i == len(pings) - 1
        if last_kept is not None and not is_last and (t - last_kept) < step_s:
            continue
        last_kept = t
        rec: dict = {
            "t": int(round(t * 1000)),
            "x": round(float(p["pixel_x"]), 1),
            "y": round(float(p["pixel_y"]), 1),
        }
        alt_baro = p.get("alt_baro_m")
        alt = p.get("alt_m")
        if alt_baro is not None:
            rec["alt_baro_m"] = int(round(alt_baro))
        elif alt is not None:
            rec["alt_m"] = int(round(alt))
        if p.get("dist_km") is not None:
            rec["dist_km"] = p["dist_km"]
        out.append(rec)
    return out


def _build_flight_tracks_with_altitude(
    projections_path: Path,
    site_lat: float,
    site_lon: float,
) -> dict[str, dict]:
    """Like concam.bundle._build_flight_tracks but carries per-ping altitude and distance.

    The labeler uses ``alt_baro_m`` to render the flight level on the on-screen
    dot label (FL = round(alt_baro_ft / 100)). ``alt_m`` is the effective
    altitude used for projection and is kept as a fallback when barometric is
    missing.

    ``dist_km`` is the great-circle distance from the camera site to the ping's
    lat/lon position, rounded to 2 decimal places. It is ``None`` when the
    projection record lacks ``lat``/``lon`` (older pipeline data).
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
        lat = rec.get("lat")
        lon = rec.get("lon")
        if lat is not None and lon is not None:
            dist_km = round(_haversine_km(site_lat, site_lon, lat, lon), 2)
        else:
            dist_km = None
        entry["pings"].append(
            {
                "wall_time_utc": rec["wall_time_utc"],
                "pixel_x": rec["pixel_x"],
                "pixel_y": rec["pixel_y"],
                "alt_m": rec.get("alt_m"),
                "alt_baro_m": rec.get("alt_baro_m"),
                "dist_km": dist_km,
            }
        )
    for entry in tracks.values():
        entry["pings"].sort(key=lambda p: p["wall_time_utc"])
    return tracks


def sustained_overlap_ids(
    episodes: list[dict],
    flight_tracks: dict[str, dict],
    *,
    sep_px: float = 100.0,
    sample_step_s: float = 5.0,
    min_overlap_s: float = 10.0,
) -> set:
    """Episode ids whose pixel tracks run sustained-parallel to another's.

    Two flights tens of km apart in 3D can project onto near-identical pixel
    tracks, so one physical contrail gets credited to both. A pair is flagged
    when the *median* pixel separation over their temporal overlap is
    <= ``sep_px`` — the median keeps transient crossings (perpendicular
    tracks) unflagged. Same-transponder pairs are skipped.

    sep_px=100 provenance: 2026-04-09 sensitivity sweep flagged 8/17/30/40% of
    665 episodes at 60/100/150/200 px; the eval-phase attribution analysis put
    per-frame two-flight ambiguity at 1.9% within 100 px and confirmed
    double-credited detections at <=150 px midpoint separation. 100 px flags
    genuinely confusable pairs without drowning the sidebar in badges; not yet
    verified against adjudicated double-credit ground truth.
    """
    import statistics

    def _positions(tid):
        track = flight_tracks.get(tid)
        if not track:
            return [], []
        times, xs, ys = [], [], []
        for p in track["pings"]:
            times.append(_parse_iso(p["wall_time_utc"]).timestamp())
            xs.append(float(p["pixel_x"]))
            ys.append(float(p["pixel_y"]))
        return times, list(zip(xs, ys))

    def _at(times, pos, t):
        # Linear interpolation along the ping list (times are sorted).
        import bisect
        i = bisect.bisect_left(times, t)
        if i <= 0:
            return pos[0]
        if i >= len(times):
            return pos[-1]
        t0, t1 = times[i - 1], times[i]
        f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (pos[i - 1][0] + f * (pos[i][0] - pos[i - 1][0]),
                pos[i - 1][1] + f * (pos[i][1] - pos[i - 1][1]))

    spans = []
    for ep in episodes:
        t0 = _parse_iso(ep["onset"]).timestamp()
        t1 = _parse_iso(ep["end"]).timestamp()
        spans.append((t0, t1, ep))
    spans.sort(key=lambda s: s[0])

    cache: dict[str, tuple] = {}
    flagged: set = set()
    for i, (a0, a1, ea) in enumerate(spans):
        for b0, b1, eb in spans[i + 1:]:
            if b0 > a1:
                break  # sorted by onset — no later episode overlaps either
            if ea["transponder_id"] == eb["transponder_id"]:
                continue
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo < min_overlap_s:
                continue
            for tid in (ea["transponder_id"], eb["transponder_id"]):
                if tid not in cache:
                    cache[tid] = _positions(tid)
            ta, pa = cache[ea["transponder_id"]]
            tb, pb = cache[eb["transponder_id"]]
            if not pa or not pb:
                continue
            n = max(3, int((hi - lo) / sample_step_s))
            seps = []
            for k in range(n + 1):
                t = lo + (hi - lo) * k / n
                xa, ya = _at(ta, pa, t)
                xb, yb = _at(tb, pb, t)
                seps.append(((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5)
            if statistics.median(seps) <= sep_px:
                flagged.add(ea["episode_id"])
                flagged.add(eb["episode_id"])
    return flagged


def build_manifest(
    source_manifest: dict,
    projections_path: Path,
    detections_path: Path,
    max_gap_seconds: float,
    detection_threshold: float,
    site_lat: float = 42.360444,
    site_lon: float = -71.089238,
) -> dict:
    per_tid_frames: dict[str, list[dict]] = {}
    per_tid_callsign: dict[str, str] = {}
    # Per-transponder list of (wall_time_utc, dist_km) for closest-approach
    # computation. dist_km may be None when lat/lon is absent in older data.
    per_tid_dist: dict[str, list[tuple[dt.datetime, float | None]]] = {}

    for rec in _iter_jsonl(projections_path):
        tid = rec["transponder_id"]
        per_tid_callsign.setdefault(tid, rec.get("callsign") or tid)
        t = _parse_iso(rec["wall_time_utc"])
        per_tid_frames.setdefault(tid, []).append(
            {
                "t": t,
                "wall_time_utc": rec["wall_time_utc"],
                "score": 0.0,
                "pixel_line": None,
            }
        )
        lat = rec.get("lat")
        lon = rec.get("lon")
        if lat is not None and lon is not None:
            dist_km: float | None = round(
                _haversine_km(site_lat, site_lon, lat, lon), 2
            )
        else:
            dist_km = None
        per_tid_dist.setdefault(tid, []).append((t, dist_km))

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
            # Closest-approach: minimum dist_km over this flight's pings whose
            # wall_time_utc falls within [onset, end]. Skips pings with None
            # dist_km (older data missing lat/lon). None when no pings qualify.
            in_window_dists = [
                d
                for t, d in per_tid_dist.get(tid, [])
                if onset <= t <= end and d is not None
            ]
            closest_approach_km: float | None = (
                round(min(in_window_dists), 2) if in_window_dists else None
            )
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
                    "closest_approach_km": closest_approach_km,
                    # Only frames carrying signal: zero-score lineless entries
                    # were 88% of frame rows and the labeler never reads them
                    # (detection lines need pixel_line; the forming-now check
                    # needs score >= threshold; absent == score 0).
                    "frames": [
                        {
                            "wall_time_utc": f["wall_time_utc"],
                            "score": float(f["score"]),
                            "pixel_line": f["pixel_line"],
                        }
                        for f in run
                        if f["score"] > 0.0 or f["pixel_line"]
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
        site_lat=site.adsb.site_lat,
        site_lon=site.adsb.site_lon,
    )
    # Rebuild flight_tracks from the current projections.jsonl rather than
    # trusting the source bundle's snapshot — the pipeline may have been
    # re-run since the bundle was generated, in which case the source manifest
    # misses any flights added by the rerun.
    manifest["flight_tracks"] = _build_flight_tracks_with_altitude(
        projections,
        site_lat=site.adsb.site_lat,
        site_lon=site.adsb.site_lon,
    )
    # Flag sustained pixel-track overlaps so reviewers see when two flights
    # could be claiming the same physical contrail. Keyed on pixel separation,
    # not 3D km — closest_approach_km misses line-of-sight coincidence (two
    # flights 80 km apart can project onto near-identical pixel tracks).
    overlap_ids = sustained_overlap_ids(
        manifest["episodes"], manifest["flight_tracks"]
    )
    for ep in manifest["episodes"]:
        ep["is_overlap"] = ep["episode_id"] in overlap_ids
    manifest["overlap_episode_ids"] = sorted(overlap_ids)
    # Compact AFTER overlap flagging — sustained_overlap_ids reads the verbose
    # ping schema. This is the manifest-size lever: 91 MB -> ~10 MB.
    for track in manifest["flight_tracks"].values():
        track["pings"] = compact_pings(track["pings"])
    manifest["calibration"] = calibration_block(load_calibration(site.calibration))
    excl = exclusion_regions_block(site.detection)
    if excl is not None:
        manifest["exclusion_regions"] = excl

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
