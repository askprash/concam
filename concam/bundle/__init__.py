"""Generate per-labeler annotation bundles from pipeline outputs.

A bundle is a self-contained directory that a student labeler opens in a
browser.  It contains:

- ``manifest.json`` — every episode assigned to the labeler plus the
  per-frame pixel geometry needed to draw overlays (ADS-B track points,
  detected line segments, score timelines).
- ``labeler.html`` — the labeling UI (items 12/13 flesh this out).

The video is NOT copied.  The manifest references it via a relative path so
that the browser can load it from the original location (e.g. an NFS mount)
without inflating the bundle.

Assignment across labelers is deterministic: sorting episodes by (onset,
episode_id) and picking evenly-spaced indices for the overlap set produces
the same assignment every run, which is required so that inter-rater
agreement can be re-computed on the same overlapping episodes.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av

from concam.projection import Calibration
from concam.storage import Database

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_LABELER_TEMPLATE = _TEMPLATES_DIR / "labeler.html"


@dataclass(frozen=True)
class Assignment:
    """Episode-id assignment for a single labeler."""

    labeler_id: str
    episode_ids: tuple[int, ...]
    overlap_episode_ids: frozenset[int]


def assign_episodes(
    episode_ids: Iterable[int],
    labelers: list[str],
    overlap_fraction: float,
) -> dict[str, Assignment]:
    """Deterministically assign episodes to labelers.

    The overlap set is picked as ``round(N * overlap_fraction)`` evenly-spaced
    positions through the sorted episode list.  Remaining episodes are
    distributed round-robin.  Every episode is labeled by at least one
    labeler; overlap episodes are labeled by all labelers.
    """
    if not labelers:
        raise ValueError("At least one labeler is required.")
    if not (0.0 <= overlap_fraction <= 1.0):
        raise ValueError(f"overlap_fraction must be in [0, 1]; got {overlap_fraction}")

    sorted_ids = sorted(set(episode_ids))
    n = len(sorted_ids)
    n_overlap = int(round(n * overlap_fraction))
    n_overlap = min(n_overlap, n)

    # Evenly spaced indices across the sorted list.
    if n_overlap == 0 or n == 0:
        overlap_positions: set[int] = set()
    else:
        overlap_positions = {int(k * n / n_overlap) for k in range(n_overlap)}

    overlap_ids = sorted(sorted_ids[i] for i in overlap_positions)
    overlap_set = frozenset(overlap_ids)
    remaining = [sorted_ids[i] for i in range(n) if i not in overlap_positions]

    per_labeler: dict[str, list[int]] = {lbl: list(overlap_ids) for lbl in labelers}
    for i, eid in enumerate(remaining):
        per_labeler[labelers[i % len(labelers)]].append(eid)

    return {
        lbl: Assignment(
            labeler_id=lbl,
            episode_ids=tuple(sorted(ids)),
            overlap_episode_ids=overlap_set,
        )
        for lbl, ids in per_labeler.items()
    }


def _load_episode_rows(
    db_path: Path, date: datetime.date
) -> list[dict]:
    """Read episode rows from DuckDB for a given date.

    Only rows with ``labeler_id IS NULL`` are returned — these are the raw
    pipeline outputs available for labeling.  Rows already carrying a
    labeler_id are someone else's completed work and should not be re-bundled.
    """
    with Database(db_path) as db:
        rows = db.query(
            """
            SELECT episode_id, callsign, transponder_id,
                   onset, end_time, frame_count,
                   peak_score, peak_line_x1, peak_line_y1,
                   peak_line_x2, peak_line_y2,
                   peak_contrail_length_m
            FROM contrail_episodes
            WHERE date = ? AND labeler_id IS NULL
            ORDER BY onset, episode_id
            """,
            [date],
        )
    out: list[dict] = []
    for r in rows:
        line = None
        if r[7] is not None:
            line = [float(r[7]), float(r[8]), float(r[9]), float(r[10])]
        out.append(
            {
                "episode_id": int(r[0]),
                "callsign": r[1],
                "transponder_id": r[2],
                "onset": _as_iso(r[3]),
                "end": _as_iso(r[4]),
                "frame_count": int(r[5]),
                "peak_score": float(r[6]),
                "peak_pixel_line": line,
                "peak_contrail_length_m": float(r[11]) if r[11] is not None else None,
            }
        )
    return out


def _as_iso(value) -> str:
    """Normalize a DuckDB timestamp to a UTC ISO string.

    DuckDB returns TIMESTAMPTZ values in the connection's local timezone.
    Manifests use a single canonical time representation (UTC) so the labeler
    UI can compare wall_time_utc strings across episodes, detections, and
    flight tracks directly.
    """
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(datetime.timezone.utc).isoformat()
    return str(value)


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _first_frame_wall_time(ocr_path: Path) -> str | None:
    """Return the UTC ISO timestamp of frame_idx=0, or None if unavailable."""
    if not ocr_path.exists():
        return None
    for rec in _iter_jsonl(ocr_path):
        if rec.get("frame_idx") == 0:
            return rec.get("wall_time_utc")
    return None


def _build_flight_tracks(projections_path: Path) -> dict[str, dict]:
    """Group projections by transponder_id into per-flight pixel tracks."""
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
            }
        )
    for entry in tracks.values():
        entry["pings"].sort(key=lambda p: p["wall_time_utc"])
    return tracks


def _build_detections_by_flight(
    detections_path: Path,
) -> dict[str, list[dict]]:
    """Group detection records by transponder_id, sorted by time."""
    by_tid: dict[str, list[dict]] = defaultdict(list)
    for rec in _iter_jsonl(detections_path):
        by_tid[rec["transponder_id"]].append(
            {
                "wall_time_utc": rec["wall_time_utc"],
                "score": rec["score"],
                "pixel_line": rec["pixel_line"],
            }
        )
    for lst in by_tid.values():
        lst.sort(key=lambda d: d["wall_time_utc"])
    return by_tid


def _frames_within(
    detections: list[dict], onset_iso: str, end_iso: str
) -> list[dict]:
    """Select detection records falling within [onset, end] (inclusive)."""
    return [d for d in detections if onset_iso <= d["wall_time_utc"] <= end_iso]


def _probe_video_fps(video_path: Path) -> float:
    """Return the encoded frame-rate of the video (e.g. 30.0 for 30fps)."""
    try:
        with av.open(str(video_path)) as c:
            vs = next((s for s in c.streams if s.type == "video"), None)
            if vs is not None:
                fps = float(vs.average_rate)
                if fps > 0:
                    return fps
    except Exception:
        pass
    return 30.0  # safe fallback for the MIT timelapse


def calibration_block(calib: Calibration) -> dict:
    """Serialize calibration into a JSON-friendly dict for the labeler UI.

    The browser needs the camera intrinsics + extrinsics to back-project a
    pixel into a 3D ray when measuring contrail length at a user-supplied
    altitude. ``camera_alt_m`` is the camera's altitude above the WGS84
    ellipsoid in metres (the ENU origin's height); the labeler subtracts it
    from the contrail altitude to get z in ENU.
    """
    return {
        "camera_matrix": calib.camera_matrix.tolist(),
        "distortion_coefficients": calib.distortion_coefficients.flatten().tolist(),
        "rotation": calib.rotation.tolist(),
        "translation": calib.translation.flatten().tolist(),
        "camera_alt_m": float(calib.camera_gps[2]),
        "calibration_resolution": list(calib.calibration_resolution),
    }


def _symlink_video(video_path: Path, bundle_dir: Path) -> str:
    """Symlink the video into the bundle directory and return the link name.

    Serving a bundle over HTTP requires all assets to be under the server
    root.  A filesystem-relative path that traverses above the server root
    (e.g. ``../../../../net/…``) is blocked by browsers.  Symlinking the
    video into the bundle dir makes it accessible as a simple filename.
    """
    link = bundle_dir / ("video" + video_path.suffix)
    target = video_path.resolve()
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    return link.name


def build_manifest(
    *,
    date: datetime.date,
    labeler_id: str,
    assignment: Assignment,
    episode_rows: list[dict],
    flight_tracks: dict[str, dict],
    detections_by_flight: dict[str, list[dict]],
    video_path: Path,
    bundle_dir: Path,
    image_size: tuple[int, int],
    seconds_per_frame: float,
    video_start_utc: str | None = None,
    detection_threshold: float = 0.3,
    calibration: Calibration | None = None,
) -> dict:
    """Assemble the manifest dict for a single labeler.

    ``video_start_utc`` is the UTC wall time of frame 0 — the labeler UI uses
    it to map ``video.currentTime`` to absolute wall time, so overlays stay
    aligned even when the timelapse does not start at local midnight.
    ``detection_threshold`` is copied from ``AggregationConfig`` so the UI can
    highlight auto-detected flights with the same cutoff the pipeline used.
    """
    assigned_ids = set(assignment.episode_ids)
    overlap_ids = set(assignment.overlap_episode_ids)

    episodes_out: list[dict] = []
    for row in episode_rows:
        if row["episode_id"] not in assigned_ids:
            continue
        tid = row["transponder_id"]
        frames = _frames_within(
            detections_by_flight.get(tid, []),
            row["onset"],
            row["end"],
        )
        episodes_out.append(
            {
                **row,
                "is_overlap": row["episode_id"] in overlap_ids,
                "frames": frames,
            }
        )
    episodes_out.sort(key=lambda e: (e["onset"], e["episode_id"]))

    video_block: dict = {
        "path": _symlink_video(video_path, bundle_dir),
        "fps": _probe_video_fps(video_path),
        "seconds_per_frame": seconds_per_frame,
    }
    if video_start_utc is not None:
        video_block["start_utc"] = video_start_utc

    manifest: dict = {
        "schema_version": 1,
        "date": date.isoformat(),
        "labeler_id": labeler_id,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "video": video_block,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "detection_threshold": float(detection_threshold),
        "overlap_episode_ids": sorted(overlap_ids),
        "episodes": episodes_out,
        "flight_tracks": flight_tracks,
    }
    if calibration is not None:
        manifest["calibration"] = calibration_block(calibration)
    return manifest


def generate_bundles(
    *,
    date: datetime.date,
    labelers: list[str],
    overlap_fraction: float,
    db_path: Path,
    projections_path: Path,
    detections_path: Path,
    video_path: Path,
    image_size: tuple[int, int],
    output_dir: Path,
    seconds_per_frame: float = 1.0,
    ocr_path: Path | None = None,
    detection_threshold: float = 0.3,
    calibration: Calibration | None = None,
) -> dict[str, Path]:
    """Generate one bundle directory per labeler.

    Returns a mapping of ``labeler_id -> bundle_dir``.

    If ``ocr_path`` is provided, the wall-time of frame 0 is read from it and
    stored in each manifest as ``video.start_utc``. This lets the labeler UI
    map video playhead time to absolute wall time without assuming the video
    starts at UTC midnight.
    """
    if not labelers:
        raise ValueError("At least one labeler is required.")

    episode_rows = _load_episode_rows(db_path, date)
    episode_ids = [row["episode_id"] for row in episode_rows]
    assignments = assign_episodes(episode_ids, labelers, overlap_fraction)

    flight_tracks = _build_flight_tracks(projections_path)
    detections_by_flight = _build_detections_by_flight(detections_path)

    video_start_utc = _first_frame_wall_time(ocr_path) if ocr_path else None

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dirs: dict[str, Path] = {}
    for labeler, assignment in assignments.items():
        bundle_dir = output_dir / labeler
        bundle_dir.mkdir(parents=True, exist_ok=True)

        manifest = build_manifest(
            date=date,
            labeler_id=labeler,
            assignment=assignment,
            episode_rows=episode_rows,
            flight_tracks=flight_tracks,
            detections_by_flight=detections_by_flight,
            video_path=video_path,
            bundle_dir=bundle_dir,
            image_size=image_size,
            seconds_per_frame=seconds_per_frame,
            video_start_utc=video_start_utc,
            detection_threshold=detection_threshold,
            calibration=calibration,
        )
        with open(bundle_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        shutil.copy(_LABELER_TEMPLATE, bundle_dir / "labeler.html")
        bundle_dirs[labeler] = bundle_dir
        logger.info(
            "bundle: labeler=%s episodes=%d overlap=%d -> %s",
            labeler,
            len(assignment.episode_ids),
            len(assignment.overlap_episode_ids),
            bundle_dir,
        )
    return bundle_dirs
