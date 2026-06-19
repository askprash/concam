"""Individual pipeline stages: ocr, adsb, project, detect, aggregate, store.

Each stage is a single function that reads the prior stage's output file(s)
and writes its own output file.  The CLI composes these in order and may
skip stages via ``--from-stage`` so long as the earlier outputs already
exist on disk.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import av
import numpy as np

from concam.adsb import Flight, Ping, load_flights
from concam.aggregation import Episode, FrameResult, aggregate_episodes
from concam.config import SiteConfig
from concam.detection import detect, grow_contrail_length
from concam.ocr import FixedFormatTimestampReader, TrustButVerifyTracker
from concam.projection import (
    PixelPoint,
    Rect,
    load_calibration,
    oriented_roi,
    project_pixel_to_meters,
    project_pings,
    rotated_polygon,
)
from concam.storage import Database

logger = logging.getLogger(__name__)

STAGES: tuple[str, ...] = ("ocr", "adsb", "project", "detect", "aggregate", "store")


def stage_paths(output_dir: Path, date: datetime.date) -> dict[str, Path]:
    """Return the canonical per-stage artefact paths under output_dir/date/."""
    base = Path(output_dir) / date.isoformat()
    return {
        "base": base,
        "ocr": base / "ocr.jsonl",
        "adsb": base / "adsb.json",
        "projections": base / "projections.jsonl",
        "detections": base / "detections.jsonl",
        "episodes": base / "episodes.jsonl",
        "store": base / "pipeline.duckdb",
    }


def iter_video_frames(video_path: Path) -> Iterator[np.ndarray]:
    """Yield (frame_idx, BGR ndarray) for each decoded frame.

    Uses PyAV (libavcodec) so we can decode AV1 (daily timelapse) and H.265
    (raw segments) with the same code path.  OpenCV's bundled FFmpeg lacks AV1.
    """
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="bgr24")
    finally:
        container.close()


def resolve_video_path(video_config, date: datetime.date) -> Path:
    """Resolve the timelapse video path for a given date using the config glob."""
    root = Path(video_config.root)
    # The config glob uses {date:%Y_%m_%d}; format it with the date.
    filename = video_config.timelapse_glob.format(date=date)
    candidate = root / filename
    if candidate.exists():
        return candidate
    # Fallback: treat as a glob under root.
    matches = sorted(root.glob(filename))
    if not matches:
        raise FileNotFoundError(
            f"No video file matching {filename!r} under {root}"
        )
    return matches[0]


# --- OCR stage -----------------------------------------------------------

def run_ocr_stage(
    video_path: Path,
    date: datetime.date,
    site_config: SiteConfig,
    out_path: Path,
    max_frames: int | None = None,
    seconds_per_frame: float = 1.0,
) -> int:
    """Read every frame of the video and write per-frame timestamps as jsonl.

    Returns the number of frames processed.

    seconds_per_frame is the wall-clock time between consecutive video frames:
    1.0 for the daily timelapse (1 frame per real-second), 0.25 for raw
    segments (4 fps). Used by the trust-but-verify tracker for sanity checks.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reader = FixedFormatTimestampReader(site_config.ocr)

    tz = ZoneInfo(site_config.timezone)
    seed_local = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
    tracker = TrustButVerifyTracker(
        start_local=seed_local,
        frame_interval_seconds=seconds_per_frame,
        seconds_per_frame=seconds_per_frame,
    )

    # Arm 1: out-of-day OCR-date gate.  ``read.parsed_dt`` is the naive *local*
    # wall time parsed straight off the burned-in overlay.  A day's timelapse
    # legitimately spans local-midnight of ``date`` through local-midnight of
    # ``date + 1`` (verified against the clean 2026-04-08 output: local wall
    # dates {2026-04-08, 2026-04-09}).  We allow +-1 day of slack around the
    # processed day so a real midnight rollover and any DST edge are never
    # rejected; anything outside that window is a confident OCR year/day misread
    # (e.g. 2026-04-11 -> 2026-04-14 or 8026-04-11).  Such reads are forced
    # invalid so the tracker PROJECTS through them instead of anchoring on
    # garbage, which is what produced the silent post-midday detection cliff.
    allowed_lo = date - datetime.timedelta(days=1)
    allowed_hi = date + datetime.timedelta(days=1)
    date_gate_rejected = 0
    date_gate_samples: list[tuple[int, str]] = []

    frames_written = 0
    with open(out_path, "w") as f:
        for frame_idx, frame in enumerate(iter_video_frames(video_path)):
            if max_frames is not None and frame_idx >= max_frames:
                break

            read = reader.read(frame)
            ocr_dt = read.parsed_dt if read.ok else None
            ocr_valid = read.ok
            if ocr_dt is not None and not (
                allowed_lo <= ocr_dt.date() <= allowed_hi
            ):
                # Out-of-day OCR date: drop it so the tracker projects through.
                date_gate_rejected += 1
                if len(date_gate_samples) < 5:
                    date_gate_samples.append(
                        (frame_idx, ocr_dt.date().isoformat())
                    )
                ocr_dt = None
                ocr_valid = False
            wall_local, status = tracker.validate(
                frame_num=frame_idx,
                ocr_ts=ocr_dt,
                is_valid=ocr_valid,
            )
            # wall_local is naive; interpret as local-time and convert to UTC.
            wall_utc = wall_local.replace(tzinfo=tz).astimezone(
                datetime.timezone.utc
            )

            record = {
                "frame_idx": frame_idx,
                "wall_time_utc": wall_utc.isoformat(),
                "confidence": round(read.confidence, 4),
                "method": read.method,
                "ocr_status": read.status,
                "tracker_status": status,
            }
            f.write(json.dumps(record) + "\n")
            frames_written += 1
            if (frame_idx + 1) % 1000 == 0:
                logger.info("OCR: %d frames processed", frame_idx + 1)

    # Observability: never let rejected frames vanish silently.  A non-zero
    # count here is the early-warning signal for the detection-time bug
    # (GitHub #1) — frames whose OCR date fell outside [day-1, day+1].
    if date_gate_rejected:
        logger.warning(
            "OCR: rejected %d/%d frames with out-of-day OCR dates "
            "(allowed window %s..%s); sample [(frame_idx, ocr_date)]: %s",
            date_gate_rejected,
            frames_written,
            allowed_lo.isoformat(),
            allowed_hi.isoformat(),
            date_gate_samples,
        )
    if tracker.date_rejected_count:
        logger.warning(
            "OCR: tracker refused %d contender reads on calendar-date grounds "
            "(date jump > %d day(s) vs trusted timeline)",
            tracker.date_rejected_count,
            tracker.max_reanchor_date_delta_days,
        )

    return frames_written


# --- ADS-B stage ---------------------------------------------------------

def _flight_to_dict(flight: Flight) -> dict:
    return {
        "callsign": flight.callsign,
        "transponder_id": flight.transponder_id,
        "aircraft_type": flight.aircraft_type,
        "orig": flight.orig,
        "dest": flight.dest,
        "pings": [
            {
                "time": p.time.isoformat(),
                "lat": p.lat,
                "lon": p.lon,
                "alt_m": p.alt_m,
                "alt_gnss_m": p.alt_gnss_m,
                "alt_baro_m": p.alt_baro_m,
                "alt_source": p.alt_source,
            }
            for p in flight.pings
        ],
    }


def _flight_from_dict(d: dict) -> Flight:
    pings = [
        Ping(
            time=datetime.datetime.fromisoformat(p["time"]),
            lat=p["lat"],
            lon=p["lon"],
            alt_m=p["alt_m"],
            alt_gnss_m=p.get("alt_gnss_m"),
            alt_baro_m=p.get("alt_baro_m"),
            alt_source=p.get("alt_source", "gnss"),
        )
        for p in d["pings"]
    ]
    return Flight(
        callsign=d["callsign"],
        transponder_id=d["transponder_id"],
        aircraft_type=d.get("aircraft_type"),
        orig=d.get("orig"),
        dest=d.get("dest"),
        pings=pings,
    )


def run_adsb_stage(
    date: datetime.date,
    site_config: SiteConfig,
    out_path: Path,
) -> int:
    """Load flights from feder and serialize to JSON.  Returns flight count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flights = load_flights(date, site_config.adsb, timezone=site_config.timezone)
    with open(out_path, "w") as f:
        json.dump([_flight_to_dict(fl) for fl in flights], f)
    return len(flights)


def load_adsb_file(path: Path) -> list[Flight]:
    with open(path) as f:
        raw = json.load(f)
    return [_flight_from_dict(d) for d in raw]


# --- Project stage -------------------------------------------------------

def run_project_stage(
    adsb_path: Path,
    site_config: SiteConfig,
    out_path: Path,
) -> int:
    """Project every ping of every flight to pixel coordinates.

    Writes one jsonl record per (ping_time, flight) that lands inside the frame.
    Pings behind the camera or outside the image bounds are dropped.
    Returns the number of records written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flights = load_adsb_file(adsb_path)
    calib = load_calibration(site_config.calibration)
    det_config = site_config.detection
    image_size = tuple(site_config.calibration.calibration_resolution)

    written = 0
    with open(out_path, "w") as f:
        for flight in flights:
            if len(flight.pings) == 0:
                continue
            projected = project_pings(flight.pings, calib)
            # Build neighbour index so the path_vector uses the closest ping
            # that also projects inside the frame — otherwise a short leg where
            # the next ping is behind the camera would raise.
            valid_indices = [i for i, p in enumerate(projected) if p is not None]
            if not valid_indices:
                continue
            valid_set = set(valid_indices)
            for i in valid_indices:
                pt = projected[i]
                # Find a neighbour (prefer next, fall back to previous).
                neighbour = None
                for j in range(i + 1, len(projected)):
                    if j in valid_set:
                        neighbour = projected[j]
                        break
                if neighbour is None:
                    for j in range(i - 1, -1, -1):
                        if j in valid_set:
                            neighbour = projected[j]
                            break
                if neighbour is None:
                    # Single-point flight: assume a degenerate horizontal path.
                    path = (1.0, 0.0)
                else:
                    dx = neighbour.x - pt.x
                    dy = neighbour.y - pt.y
                    mag = (dx * dx + dy * dy) ** 0.5
                    if mag < 1e-9:
                        path = (1.0, 0.0)
                    else:
                        path = (dx / mag, dy / mag)

                roi = oriented_roi(pt, path, det_config, image_size=image_size)
                if roi.w <= 0 or roi.h <= 0:
                    continue

                ping = flight.pings[i]
                record = {
                    "wall_time_utc": ping.time.astimezone(
                        datetime.timezone.utc
                    ).isoformat(),
                    "callsign": flight.callsign,
                    "transponder_id": flight.transponder_id,
                    "pixel_x": pt.x,
                    "pixel_y": pt.y,
                    "path_dx": path[0],
                    "path_dy": path[1],
                    "roi": {"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h},
                    # GPS position included so the detect stage can compute
                    # physical contrail length without re-reading ADS-B data.
                    "lat": ping.lat,
                    "lon": ping.lon,
                    "alt_m": ping.alt_m,
                    "alt_baro_m": ping.alt_baro_m,
                }
                f.write(json.dumps(record) + "\n")
                written += 1
    return written


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- Detect stage --------------------------------------------------------

def run_detect_stage(
    video_path: Path,
    ocr_path: Path,
    projections_path: Path,
    site_config: SiteConfig,
    out_path: Path,
    max_frames: int | None = None,
) -> int:
    """For each frame, run the detector on every flight that is over the frame.

    Iterates the video sequentially (seeking AV1 is expensive).  For each frame
    index, looks up its wall-clock time from the OCR output, then finds all
    projections that match that wall-clock second.  Runs the detector per
    (frame, flight) pair and streams results to jsonl.

    Each detection record carries ``contrail_length_px`` (seed ROI span),
    ``contrail_length_grown_px`` (after adaptive growth), and
    ``contrail_length_m`` (physical metres via pinhole-ray geometry).

    Returns the number of detection records written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load calibration for physical-length conversion.
    calib = load_calibration(site_config.calibration)

    # Build frame_idx -> wall_time_utc map from ocr.jsonl.
    frame_time: dict[int, str] = {}
    for rec in _iter_jsonl(ocr_path):
        frame_time[rec["frame_idx"]] = rec["wall_time_utc"]

    # Group projections by wall_time_utc.
    by_second: dict[str, list[dict]] = defaultdict(list)
    for rec in _iter_jsonl(projections_path):
        # Normalize to a second-resolution key (strip microseconds).
        ts = datetime.datetime.fromisoformat(rec["wall_time_utc"]).replace(
            microsecond=0
        )
        key = ts.isoformat()
        by_second[key].append(rec)

    det_config = site_config.detection

    written = 0
    prev_frame: np.ndarray | None = None
    # buffering=1 selects line-buffered mode: each \n-terminated write is
    # flushed immediately, which prevents records from being smashed together
    # on the same line if the process is killed between frame writes.
    with open(out_path, "w", buffering=1) as f:
        for frame_idx, frame in enumerate(iter_video_frames(video_path)):
            if max_frames is not None and frame_idx >= max_frames:
                break

            wall = frame_time.get(frame_idx)
            if wall is None:
                prev_frame = frame
                continue

            # Round to second resolution for the join.
            ts = datetime.datetime.fromisoformat(wall).replace(microsecond=0)
            key = ts.isoformat()
            candidates = by_second.get(key, ())
            for proj in candidates:
                roi = Rect(
                    x=proj["roi"]["x"],
                    y=proj["roi"]["y"],
                    w=proj["roi"]["w"],
                    h=proj["roi"]["h"],
                )
                # Rebuild the rotated polygon from the projection record so the
                # detector can mask the along-track strip. Projections written
                # before this schema field existed fall back to AABB-only mode.
                poly = None
                path_vec: tuple[float, float] | None = None
                center: PixelPoint | None = None
                if "pixel_x" in proj and "path_dx" in proj:
                    center = PixelPoint(x=float(proj["pixel_x"]),
                                        y=float(proj["pixel_y"]))
                    path_vec = (float(proj["path_dx"]), float(proj["path_dy"]))
                    poly = rotated_polygon(center, path_vec, det_config)
                result = detect(
                    frame, roi, det_config,
                    polygon=poly, path_vec=path_vec,
                    prev_frame=prev_frame,
                )

                # Adaptive length measurement: grow the ROI when a signal exists.
                grown_length_px = 0.0
                contrail_length_m: float | None = None
                if result.score > 0.0 and center is not None and path_vec is not None:
                    grown_length_px = grow_contrail_length(
                        frame,
                        (center.x, center.y),
                        path_vec,
                        det_config,
                        prev_frame=prev_frame,
                    )
                    # Physical length via pinhole-ray geometry.
                    # Use barometric altitude when available (more reliable than GNSS).
                    alt_m = float(proj.get("alt_baro_m") or proj.get("alt_m") or 0.0)
                    if alt_m > 0.0 and grown_length_px > 0.0:
                        # Build a minimal Ping for projection (only lat/lon/alt used).
                        ping_for_length = Ping(
                            time=ts,
                            lat=float(proj["lat"]),
                            lon=float(proj["lon"]),
                            alt_m=alt_m,
                        )
                        contrail_length_m = project_pixel_to_meters(
                            grown_length_px, ping_for_length, calib
                        )

                record = {
                    "wall_time_utc": wall,
                    "callsign": proj["callsign"],
                    "transponder_id": proj["transponder_id"],
                    "score": result.score,
                    "pixel_line": list(result.pixel_line)
                    if result.pixel_line is not None
                    else None,
                    "method": result.method,
                    "num_long_lines": result.num_long_lines,
                    "aligned_lines": result.aligned_lines,
                    "contrail_length_px": result.contrail_length_px,
                    "contrail_length_grown_px": grown_length_px,
                    "contrail_length_m": contrail_length_m,
                }
                f.write(json.dumps(record) + "\n")
                written += 1

            prev_frame = frame
            if (frame_idx + 1) % 1000 == 0:
                logger.info("DETECT: %d frames scanned", frame_idx + 1)

    return written


# --- Aggregate stage -----------------------------------------------------

def run_aggregate_stage(
    detections_path: Path,
    site_config: SiteConfig,
    out_path: Path,
) -> int:
    """Group detections into episodes, writing serialized episodes to jsonl."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_results: list[FrameResult] = []
    for rec in _iter_jsonl(detections_path):
        time = datetime.datetime.fromisoformat(rec["wall_time_utc"])
        line = rec["pixel_line"]
        frame_results.append(
            FrameResult(
                time=time,
                callsign=rec["callsign"],
                transponder_id=rec["transponder_id"],
                score=rec["score"],
                pixel_line=tuple(line) if line is not None else None,
                contrail_length_m=rec.get("contrail_length_m"),
            )
        )
    episodes = aggregate_episodes(frame_results, site_config.aggregation)
    with open(out_path, "w") as f:
        for ep in episodes:
            record = {
                "callsign": ep.callsign,
                "transponder_id": ep.transponder_id,
                "onset": ep.onset.isoformat(),
                "end": ep.end.isoformat(),
                "peak_score": ep.peak_score,
                "peak_pixel_line": list(ep.peak_pixel_line)
                if ep.peak_pixel_line is not None
                else None,
                "frame_count": ep.frame_count,
                "peak_contrail_length_m": ep.peak_contrail_length_m,
            }
            f.write(json.dumps(record) + "\n")
    return len(episodes)


def load_episodes_file(path: Path) -> list[Episode]:
    episodes: list[Episode] = []
    for rec in _iter_jsonl(path):
        line = rec["peak_pixel_line"]
        episodes.append(
            Episode(
                callsign=rec["callsign"],
                transponder_id=rec["transponder_id"],
                onset=datetime.datetime.fromisoformat(rec["onset"]),
                end=datetime.datetime.fromisoformat(rec["end"]),
                peak_score=rec["peak_score"],
                peak_pixel_line=tuple(line) if line is not None else None,
                frame_count=rec["frame_count"],
                peak_contrail_length_m=rec.get("peak_contrail_length_m"),
            )
        )
    return episodes


# --- Store stage ---------------------------------------------------------

def run_store_stage(
    episodes_path: Path,
    date: datetime.date,
    db_path: Path,
) -> int:
    """Write episodes to DuckDB.  Returns the number of rows inserted.

    Re-running for the same date deletes any unlabeled rows for that date
    first, so re-runs produce a deterministic row count.  Rows that already
    carry labeler input are preserved.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes_file(episodes_path)
    with Database(db_path) as db:
        db.create_schema()
        db.con.execute(
            "DELETE FROM contrail_episodes WHERE date = ? AND labeler_id IS NULL",
            [date],
        )
        db.insert_episodes(episodes, date=date, start_id=1)
    return len(episodes)
