"""Build a stratified 35-candidate label batch for detector-redesign research (PRD item 28).

The detector-redesign work (PRD items 29+) needs ground-truth labels on
2026-04-09 to drive bulk-AUC evaluation of filter chains in the upcoming
``filter_playground.ipynb``. The April-8 label set (7 positives / 13 negatives)
is too small and too clear-blue-sky to generalise.

This script is the machine half of item 28: read the cached April-9
projections + detections, pick 35 flight-peaks stratified across altitude,
ground-distance, and sun-elevation bands, and emit a concam-bundle-shaped
labeler directory the user opens in a browser. The human half is the user
labeling ≥30 of the 35 candidates in ``labeler.html`` and exporting
``labels.json``.

Strata (default quotas)::

    10 high-cirrus   :  10_400 ≤ alt_baro_m ≤ 12_200  and  sun_alt > 15°
    10 mid-cruise    :   8_500 ≤ alt_baro_m < 10_400  and  sun_alt > 15°
    10 wide-radius   :  60 ≤ ground_distance_km ≤ 150 and  sun_alt > 15°
     5 marginal      :  sun_alt < 15°  (dusk / dawn / night)

Each flight is used at most once across strata (strata are applied in the
order above; a TID taken by an earlier stratum is excluded from later ones).

A peak for a flight is the ping closest to image center (3840/2, 2160/2),
which matches the convention used by ``scripts/detection_validation_extract.py``.

The bundle produced here is self-contained and can be served via
``scripts/serve_bundle.py``.  A sidecar ``candidates.json`` records
per-candidate geometry (ROI, path vector, altitude, sun angle, stratum,
frame_idx) so item 29's bulk-AUC bench can re-run ``detect()`` on the
user-labeled set without reparsing the bundle.

Usage::

    uv run python scripts/build_april9_label_batch.py --date 2026-04-09
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from concam.bundle import (
    Assignment,
    _LABELER_TEMPLATE,
    _build_detections_by_flight,
    _build_flight_tracks,
    _first_frame_wall_time,
    build_manifest,
)
from concam.config import load_config
from concam.pipeline.stages import resolve_video_path
from concam.sun import sun_alt_deg

logger = logging.getLogger("build_april9_label_batch")

# MIT Green Building camera site (configs/mit_green_building.yaml:10-11).
SITE_LAT = 42.360444
SITE_LON = -71.089238

# 4K calibration (configs/mit_green_building.yaml:66).
IMAGE_W = 3840
IMAGE_H = 2160
FOV_CX = IMAGE_W / 2
FOV_CY = IMAGE_H / 2

# Inspection window around each candidate peak.  21 s is long enough that the
# labeler sees 10 s before and after the peak — useful for dim/fading contrails.
HALF_WINDOW_S = 10

# Daily timelapse encodes 1 real-second per frame.
SECONDS_PER_FRAME = 1.0

DEFAULT_QUOTAS = (10, 10, 10, 5)
STRATA_ORDER = ("high_cirrus", "mid_cruise", "wide_radius", "marginal")


@dataclass
class FlightPeak:
    """Per-flight representative ping (closest to FOV center)."""

    transponder_id: str
    callsign: str
    proj: dict  # the projection record at the peak
    time: datetime.datetime
    alt_baro_m: float
    ground_distance_km: float
    sun_alt: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) pairs in km."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_flight_peaks(
    projections_path: Path,
    *,
    time_window: tuple[datetime.datetime, datetime.datetime] | None = None,
) -> list[FlightPeak]:
    """Reduce ``projections.jsonl`` to one representative ping per flight.

    If ``time_window`` is given, drop pings outside ``[start, end]`` before
    selecting each flight's FOV-center peak.  This prevents candidates from
    being picked at times the video does not cover (e.g. before frame 0 or
    after the last encoded frame).
    """
    best: dict[str, tuple[dict, float]] = {}
    for rec in _iter_jsonl(projections_path):
        if time_window is not None:
            t = datetime.datetime.fromisoformat(rec["wall_time_utc"])
            if not (time_window[0] <= t <= time_window[1]):
                continue
        tid = rec["transponder_id"]
        d = math.hypot(rec["pixel_x"] - FOV_CX, rec["pixel_y"] - FOV_CY)
        if tid not in best or d < best[tid][1]:
            best[tid] = (rec, d)

    peaks: list[FlightPeak] = []
    for tid, (rec, _) in best.items():
        t = datetime.datetime.fromisoformat(rec["wall_time_utc"])
        baro = rec.get("alt_baro_m")
        alt = float(baro) if baro is not None else float(rec["alt_m"])
        peaks.append(
            FlightPeak(
                transponder_id=tid,
                callsign=rec["callsign"],
                proj=rec,
                time=t,
                alt_baro_m=alt,
                ground_distance_km=haversine_km(
                    SITE_LAT, SITE_LON, rec["lat"], rec["lon"]
                ),
                sun_alt=sun_alt_deg(t, SITE_LAT, SITE_LON),
            )
        )
    peaks.sort(key=lambda p: p.time)
    return peaks


def _center_distance(p: FlightPeak) -> float:
    return math.hypot(p.proj["pixel_x"] - FOV_CX, p.proj["pixel_y"] - FOV_CY)


def time_bucketed_select(
    peaks: list[FlightPeak], n: int, bucket_minutes: int = 30
) -> list[FlightPeak]:
    """Pick up to ``n`` peaks spread across the day.

    One peak per time bucket (keep the one closest to FOV center per bucket),
    then evenly downsample by index if too many buckets survive.
    """
    if not peaks or n <= 0:
        return []
    buckets: dict[int, FlightPeak] = {}
    for p in peaks:
        key = (p.time.hour * 60 + p.time.minute) // bucket_minutes
        if key not in buckets or _center_distance(p) < _center_distance(buckets[key]):
            buckets[key] = p
    picked = [buckets[k] for k in sorted(buckets)]
    if len(picked) <= n:
        return picked
    if n == 1:
        return [picked[len(picked) // 2]]
    idxs = [int(round(k * (len(picked) - 1) / (n - 1))) for k in range(n)]
    # Dedup idx collisions (possible when n is close to len(picked)).
    seen: set[int] = set()
    out: list[FlightPeak] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(picked[i])
    return out


def _high_cirrus(p: FlightPeak) -> bool:
    return 10_400 <= p.alt_baro_m <= 12_200 and p.sun_alt > 15


def _mid_cruise(p: FlightPeak) -> bool:
    return 8_500 <= p.alt_baro_m < 10_400 and p.sun_alt > 15


def _wide_radius(p: FlightPeak) -> bool:
    return 60 <= p.ground_distance_km <= 150 and p.sun_alt > 15


def _marginal(p: FlightPeak) -> bool:
    return p.sun_alt < 15


STRATUM_FILTERS = {
    "high_cirrus": _high_cirrus,
    "mid_cruise": _mid_cruise,
    "wide_radius": _wide_radius,
    "marginal": _marginal,
}


def stratify(
    peaks: list[FlightPeak], quotas: tuple[int, int, int, int]
) -> dict[str, list[FlightPeak]]:
    """Split ``peaks`` into 4 disjoint strata.

    Strata are applied in ``STRATA_ORDER``.  A TID picked by an earlier
    stratum is excluded from later ones, so every candidate is unique.
    """
    used: set[str] = set()
    result: dict[str, list[FlightPeak]] = {}
    for name, n in zip(STRATA_ORDER, quotas):
        pool = [p for p in peaks if p.transponder_id not in used and STRATUM_FILTERS[name](p)]
        picked = time_bucketed_select(pool, n)
        used.update(p.transponder_id for p in picked)
        result[name] = picked
        if len(picked) < n:
            logger.warning(
                "stratum %s: wanted %d, got %d (pool size=%d) — manifest will be short",
                name,
                n,
                len(picked),
                len(pool),
            )
    return result


def _lookup_detection(
    detections: list[dict], target_iso: str, tolerance_s: float = 1.5
) -> dict | None:
    """Find the detection record nearest (by wall time) to ``target_iso``.

    Used to populate ``peak_score`` / ``peak_pixel_line`` on synthetic
    episodes so the labeler shows the existing detector's take alongside
    each candidate.  Returns None if no detection is within ``tolerance_s``.
    """
    if not detections:
        return None
    target = datetime.datetime.fromisoformat(target_iso)
    best = None
    best_delta = math.inf
    for rec in detections:
        t = datetime.datetime.fromisoformat(rec["wall_time_utc"])
        delta = abs((t - target).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = rec
    return best if best_delta <= tolerance_s else None


def _episode_row(
    episode_id: int, peak: FlightPeak, detection: dict | None
) -> dict:
    onset_dt = peak.time - datetime.timedelta(seconds=HALF_WINDOW_S)
    end_dt = peak.time + datetime.timedelta(seconds=HALF_WINDOW_S)
    # Canonicalise to UTC to match projections / detections exactly.
    if onset_dt.tzinfo is None:
        onset_dt = onset_dt.replace(tzinfo=datetime.timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=datetime.timezone.utc)

    peak_score = 0.0
    peak_line = None
    peak_length_m = None
    if detection is not None:
        peak_score = float(detection.get("score") or 0.0)
        peak_line = detection.get("pixel_line")
        peak_length_m = detection.get("contrail_length_m")

    return {
        "episode_id": episode_id,
        "callsign": peak.callsign,
        "transponder_id": peak.transponder_id,
        "onset": onset_dt.isoformat(),
        "end": end_dt.isoformat(),
        "frame_count": int(round(2 * HALF_WINDOW_S / SECONDS_PER_FRAME)) + 1,
        "peak_score": peak_score,
        "peak_pixel_line": peak_line,
        "peak_contrail_length_m": peak_length_m,
    }


def _candidate_sidecar_row(
    episode_id: int,
    stratum: str,
    peak: FlightPeak,
    frame0_anchor: datetime.datetime,
) -> dict:
    """Per-candidate seed for item-29 bulk-AUC reuse.

    Carries the bits the bundle manifest doesn't expose (ROI, path vector)
    plus stratum metadata so the bulk-AUC bench knows where each label came
    from.
    """
    peak_frame_idx = int(round((peak.time - frame0_anchor).total_seconds() / SECONDS_PER_FRAME))
    return {
        "episode_id": episode_id,
        "stratum": stratum,
        "callsign": peak.callsign,
        "transponder_id": peak.transponder_id,
        "peak_wall_time": peak.proj["wall_time_utc"],
        "peak_frame_idx": peak_frame_idx,
        "pixel_x": float(peak.proj["pixel_x"]),
        "pixel_y": float(peak.proj["pixel_y"]),
        "path_dx": float(peak.proj["path_dx"]),
        "path_dy": float(peak.proj["path_dy"]),
        "roi": peak.proj["roi"],
        "alt_baro_m": peak.alt_baro_m,
        "ground_distance_km": peak.ground_distance_km,
        "sun_alt_deg": peak.sun_alt,
    }


def _video_time_window(ocr_path: Path) -> tuple[datetime.datetime, datetime.datetime] | None:
    """Return the [first_frame, last_frame] UTC times from the OCR cache.

    Used to filter candidates to moments the video actually covers — pings
    outside this window would seek outside the video and the labeler would
    display nothing.
    """
    if not ocr_path.exists():
        return None
    first_iso: str | None = None
    last_iso: str | None = None
    with open(ocr_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iso = rec.get("wall_time_utc")
            if iso is None:
                continue
            if first_iso is None:
                first_iso = iso
            last_iso = iso
    if first_iso is None or last_iso is None:
        return None
    return (
        datetime.datetime.fromisoformat(first_iso),
        datetime.datetime.fromisoformat(last_iso),
    )


def build_batch(
    *,
    date: datetime.date,
    labeler_id: str,
    output_dir: Path,
    projections_path: Path,
    detections_path: Path,
    ocr_path: Path,
    video_path: Path,
    quotas: tuple[int, int, int, int],
    detection_threshold: float,
) -> dict:
    """End-to-end: stratify peaks and emit manifest + labeler + sidecar."""
    time_window = _video_time_window(ocr_path)
    if time_window is not None:
        logger.info(
            "restricting to video window %s -> %s",
            time_window[0].isoformat(),
            time_window[1].isoformat(),
        )
    peaks = load_flight_peaks(projections_path, time_window=time_window)
    logger.info("loaded %d flight peaks from %s", len(peaks), projections_path.name)

    strata = stratify(peaks, quotas)
    for name in STRATA_ORDER:
        logger.info("  stratum %-12s  n=%d", name, len(strata[name]))

    # Sort final candidate set chronologically for deterministic episode_ids.
    ordered: list[tuple[str, FlightPeak]] = []
    for name in STRATA_ORDER:
        for p in strata[name]:
            ordered.append((name, p))
    ordered.sort(key=lambda pair: pair[1].time)
    if not ordered:
        raise RuntimeError(
            "No candidates selected.  Check that the date's projections.jsonl "
            "exists and that the altitude / radius / sun-alt filters are not all "
            "rejecting the available flights."
        )

    detections_by_flight = _build_detections_by_flight(detections_path)
    flight_tracks = _build_flight_tracks(projections_path)

    frame0_iso = _first_frame_wall_time(ocr_path)
    frame0_anchor = (
        datetime.datetime.fromisoformat(frame0_iso)
        if frame0_iso
        else datetime.datetime.combine(date, datetime.time(0), tzinfo=datetime.timezone.utc)
    )

    episode_rows: list[dict] = []
    sidecar_rows: list[dict] = []
    for idx, (stratum, peak) in enumerate(ordered, start=1):
        det = _lookup_detection(
            detections_by_flight.get(peak.transponder_id, []),
            peak.proj["wall_time_utc"],
        )
        episode_rows.append(_episode_row(idx, peak, det))
        sidecar_rows.append(_candidate_sidecar_row(idx, stratum, peak, frame0_anchor))

    assignment = Assignment(
        labeler_id=labeler_id,
        episode_ids=tuple(row["episode_id"] for row in episode_rows),
        overlap_episode_ids=frozenset(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        date=date,
        labeler_id=labeler_id,
        assignment=assignment,
        episode_rows=episode_rows,
        flight_tracks=flight_tracks,
        detections_by_flight=detections_by_flight,
        video_path=video_path,
        bundle_dir=output_dir,
        image_size=(IMAGE_W, IMAGE_H),
        seconds_per_frame=SECONDS_PER_FRAME,
        video_start_utc=frame0_iso,
        detection_threshold=detection_threshold,
    )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    shutil.copy(_LABELER_TEMPLATE, output_dir / "labeler.html")

    sidecar = {
        "schema_version": 1,
        "date": date.isoformat(),
        "labeler_id": labeler_id,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "quotas": dict(zip(STRATA_ORDER, quotas)),
        "counts": {name: len(strata[name]) for name in STRATA_ORDER},
        "site": {"lat": SITE_LAT, "lon": SITE_LON},
        "candidates": sidecar_rows,
    }
    with open(output_dir / "candidates.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    return {"manifest": manifest, "candidates": sidecar}


def _parse_quotas(s: str) -> tuple[int, int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4 or any(n < 0 for n in parts):
        raise argparse.ArgumentTypeError(
            "--quota expects four non-negative integers (cirrus,cruise,wide,marginal)"
        )
    return tuple(parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument("--labeler-id", default="prash")
    parser.add_argument(
        "--output-root",
        default="output/validation/detection",
        help="Root dir for label-batch output (default: output/validation/detection)",
    )
    parser.add_argument(
        "--quota",
        type=_parse_quotas,
        default=DEFAULT_QUOTAS,
        help="Comma-separated: cirrus,cruise,wide,marginal (default: 10,10,10,5)",
    )
    parser.add_argument(
        "--config",
        default="configs/mit_green_building.yaml",
        help="Site YAML config (for video-path resolution + detection threshold)",
    )
    parser.add_argument(
        "--pipeline-output",
        default="output",
        help="Pipeline output root containing <date>/projections.jsonl etc.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    date = datetime.date.fromisoformat(args.date)
    config = load_config(args.config)
    video_path = resolve_video_path(config.video, date)

    pipeline_dir = Path(args.pipeline_output) / args.date
    projections_path = pipeline_dir / "projections.jsonl"
    detections_path = pipeline_dir / "detections.jsonl"
    ocr_path = pipeline_dir / "ocr.jsonl"
    for p in (projections_path, ocr_path):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")

    output_dir = Path(args.output_root) / args.date / "label_batch"
    result = build_batch(
        date=date,
        labeler_id=args.labeler_id,
        output_dir=output_dir,
        projections_path=projections_path,
        detections_path=detections_path,
        ocr_path=ocr_path,
        video_path=video_path,
        quotas=args.quota,
        detection_threshold=float(config.aggregation.detection_threshold),
    )
    logger.info(
        "wrote %d-episode bundle to %s (candidates.json sidecar included)",
        len(result["manifest"]["episodes"]),
        output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
