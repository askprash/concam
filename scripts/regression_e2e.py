"""End-to-end regression baseline for a single date.

Runs three things PRD item 17 wants:

  1. **Determinism** — re-runs the numeric stages (``detect`` + ``aggregate``)
     against the cached OCR / ADS-B / projections of ``output/<date>/`` and
     diff-compares the new ``detections.jsonl`` and ``episodes.jsonl`` against
     the originals. Any byte-level drift fails the check. OCR / ADS-B /
     projection stages are pure file / geodesic math and cheap to re-derive
     separately; the risky floating-point code lives in detect+aggregate.
  2. **Baseline metrics** — writes ``metrics.json`` with flight count, ping
     count, detection score histogram, above-threshold detection count,
     episode count, above-threshold episode count, and the peak-score /
     near-threshold episode lists. Future regressions can diff this file.
  3. **Spot-check panels** — renders a 4-panel PNG per episode for the top 5
     by peak_score and the 5 episodes closest to the aggregation threshold,
     so a human can eyeball whether high-score episodes are real contrails
     and near-threshold episodes are the right calls. The script does not
     judge correctness — that is the human's job — but it captures the
     evidence a reviewer needs.

Usage::

    uv run python scripts/regression_e2e.py --date 2026-04-08

Assumes ``output/<date>/`` already contains a fresh pipeline run. If in
doubt, ``uv run concam run --date <date>`` first, then this script.
"""

from __future__ import annotations

import argparse
import datetime
import filecmp
import json
import logging
import shutil
import statistics
import sys
from pathlib import Path

import av
import cv2
import duckdb
import matplotlib.pyplot as plt
import numpy as np

# Make ``concam`` importable when this script is run from a clean checkout.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import load_config
from concam.pipeline import (  # noqa: E402
    resolve_video_path,
    run_aggregate_stage,
    run_detect_stage,
    run_store_stage,
    stage_paths,
)
from concam.projection import PixelPoint, rotated_polygon  # noqa: E402

logger = logging.getLogger("regression_e2e")

DEFAULT_CONFIG = REPO_ROOT / "configs" / "mit_green_building.yaml"

# Histogram bins for detection scores (inclusive lower, exclusive upper bound
# except the last bin). Matched to the aggregation threshold (0.083) and the
# discrete score grid produced by ``score_norm_count=6`` (0, 1/6, 2/6, ...).
SCORE_BINS: list[tuple[float, float]] = [
    (0.0, 0.083),
    (0.083, 0.167),
    (0.167, 0.334),
    (0.334, 0.501),
    (0.501, 0.668),
    (0.668, 1.001),
]


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- Determinism check --------------------------------------------------

def _rerun_numeric_stages(
    date: datetime.date,
    source_dir: Path,
    target_dir: Path,
    video_path: Path,
    site_config,
) -> None:
    """Re-run detect+aggregate+store using cached OCR/ADSB/projections.

    Copies the upstream outputs into ``target_dir`` so the new stage outputs
    write alongside them (the CLI's stage path layout assumes a single
    per-date dir), then invokes the three stage functions directly.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ocr.jsonl", "adsb.json", "projections.jsonl"):
        src = source_dir / name
        dst = target_dir / name
        if dst.exists():
            dst.unlink()
        # Hardlink rather than copy — same filesystem, byte-identical.
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)

    target_date_dir = target_dir.parent
    paths = stage_paths(target_date_dir, date)

    n_det = run_detect_stage(
        video_path=video_path,
        ocr_path=paths["ocr"],
        projections_path=paths["projections"],
        site_config=site_config,
        out_path=paths["detections"],
    )
    logger.info("rerun: wrote %d detections", n_det)

    n_ep = run_aggregate_stage(
        detections_path=paths["detections"],
        site_config=site_config,
        out_path=paths["episodes"],
    )
    logger.info("rerun: wrote %d episodes", n_ep)

    if paths["store"].exists():
        paths["store"].unlink()
    n_st = run_store_stage(
        episodes_path=paths["episodes"],
        date=date,
        db_path=paths["store"],
    )
    logger.info("rerun: inserted %d episodes into duckdb", n_st)


def _diff_jsonl(path_a: Path, path_b: Path) -> dict:
    """Compare two jsonl files. Returns a summary dict with identical/diff counts."""
    if filecmp.cmp(str(path_a), str(path_b), shallow=False):
        return {"identical": True, "lines": sum(1 for _ in open(path_a))}

    lines_a = list(_iter_jsonl(path_a))
    lines_b = list(_iter_jsonl(path_b))
    n_diff = 0
    first_diff_idx = None
    first_diff_preview = None
    for i, (a, b) in enumerate(zip(lines_a, lines_b)):
        if a != b:
            n_diff += 1
            if first_diff_idx is None:
                first_diff_idx = i
                first_diff_preview = {"a": a, "b": b}
    return {
        "identical": False,
        "lines_a": len(lines_a),
        "lines_b": len(lines_b),
        "differing_records": n_diff,
        "first_diff_idx": first_diff_idx,
        "first_diff_preview": first_diff_preview,
    }


# --- Baseline metrics ---------------------------------------------------

def _score_histogram(detections: list[dict]) -> list[dict]:
    bins = [{"lo": lo, "hi": hi, "count": 0} for (lo, hi) in SCORE_BINS]
    for rec in detections:
        s = rec["score"]
        for b in bins:
            if b["lo"] <= s < b["hi"]:
                b["count"] += 1
                break
    return bins


def _summarise_duckdb(db_path: Path) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row_count = con.execute("SELECT COUNT(*) FROM contrail_episodes").fetchone()[0]
        peak_stats = con.execute(
            "SELECT MIN(peak_score), MAX(peak_score), AVG(peak_score) FROM contrail_episodes"
        ).fetchone()
        # peak_contrail_length_m was added in PRD item 21; column may be absent on old DBs.
        try:
            len_stats = con.execute(
                "SELECT COUNT(peak_contrail_length_m), "
                "       MIN(peak_contrail_length_m), "
                "       MAX(peak_contrail_length_m), "
                "       AVG(peak_contrail_length_m) "
                "FROM contrail_episodes WHERE peak_contrail_length_m IS NOT NULL"
            ).fetchone()
            length_summary = {
                "episodes_with_length": len_stats[0],
                "length_m_min": round(len_stats[1], 1) if len_stats[1] is not None else None,
                "length_m_max": round(len_stats[2], 1) if len_stats[2] is not None else None,
                "length_m_mean": round(len_stats[3], 1) if len_stats[3] is not None else None,
            }
        except Exception:
            length_summary = None
    finally:
        con.close()
    result = {
        "row_count": row_count,
        "peak_score_min": peak_stats[0],
        "peak_score_max": peak_stats[1],
        "peak_score_mean": peak_stats[2],
    }
    if length_summary is not None:
        result["peak_contrail_length_m"] = length_summary
    return result


def _collect_metrics(
    source_dir: Path, aggregation_threshold: float
) -> dict:
    adsb = json.loads((source_dir / "adsb.json").read_text())
    flight_count = len(adsb)
    ping_count = sum(len(f["pings"]) for f in adsb)

    ocr_records = list(_iter_jsonl(source_dir / "ocr.jsonl"))
    frame_count = len(ocr_records)
    ocr_ok = sum(1 for r in ocr_records if r.get("ocr_status") == "ok")

    proj_count = sum(1 for _ in _iter_jsonl(source_dir / "projections.jsonl"))

    detections = list(_iter_jsonl(source_dir / "detections.jsonl"))
    det_count = len(detections)
    hit_count = sum(1 for d in detections if d["score"] >= aggregation_threshold)
    scores = [d["score"] for d in detections if d["score"] > 0]
    score_hist = _score_histogram(detections)

    # Contrail length stats from detections (PRD item 21).
    lengths_m = [
        d["contrail_length_m"]
        for d in detections
        if d.get("contrail_length_m") is not None and d["contrail_length_m"] > 0
    ]
    length_hist_bins = [
        (0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, float("inf"))
    ]
    length_hist: list[dict] | None = None
    if lengths_m:
        bins = [{"lo": lo, "hi": hi, "count": 0} for (lo, hi) in length_hist_bins]
        for lm in lengths_m:
            for b in bins:
                if b["lo"] <= lm < b["hi"]:
                    b["count"] += 1
                    break
        length_hist = bins

    episodes = list(_iter_jsonl(source_dir / "episodes.jsonl"))
    ep_count = len(episodes)
    ep_above = sum(1 for e in episodes if e["peak_score"] >= aggregation_threshold)

    db_summary = _summarise_duckdb(source_dir / "pipeline.duckdb")

    out: dict = {
        "flights": flight_count,
        "pings": ping_count,
        "frames": frame_count,
        "ocr_ok": ocr_ok,
        "projections": proj_count,
        "detections": det_count,
        "detections_above_threshold": hit_count,
        "detection_hit_rate": (hit_count / det_count) if det_count else 0.0,
        "detection_score_histogram": score_hist,
        "detection_score_mean_nonzero": statistics.mean(scores) if scores else 0.0,
        "detection_score_max": max((d["score"] for d in detections), default=0.0),
        "episodes": ep_count,
        "episodes_above_threshold": ep_above,
        "aggregation_threshold": aggregation_threshold,
        "duckdb": db_summary,
    }
    if length_hist is not None:
        import statistics as _stats
        out["contrail_length_m_count"] = len(lengths_m)
        out["contrail_length_m_mean"] = round(_stats.mean(lengths_m), 1)
        out["contrail_length_m_max"] = round(max(lengths_m), 1)
        out["contrail_length_m_histogram"] = length_hist
    return out


# --- Episode selection for spot-check -----------------------------------

def _pick_episodes(
    episodes: list[dict], threshold: float, n_top: int = 5, n_threshold: int = 5,
) -> dict[str, list[dict]]:
    above = [e for e in episodes if e["peak_score"] >= threshold]
    top_sorted = sorted(above, key=lambda e: -e["peak_score"])
    top = top_sorted[:n_top]

    # "Near threshold" = above-threshold episodes whose peak_score is closest
    # to the threshold (i.e. weakest accepted detections). These are the ones
    # most likely to be marginal and the most informative to eyeball.
    nearest = sorted(above, key=lambda e: abs(e["peak_score"] - threshold))
    near = nearest[:n_threshold]

    # Deduplicate — an episode can appear in both lists on small sets.
    seen: set[tuple[str, str]] = set()
    unique_near = []
    for e in near:
        key = (e["callsign"], e["onset"])
        if key in {(t["callsign"], t["onset"]) for t in top}:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_near.append(e)

    return {"top": top, "near_threshold": unique_near}


# --- Spot-check panels --------------------------------------------------

def _find_peak_frame(
    detections_by_flight: dict[tuple[str, str], list[dict]], episode: dict
) -> dict | None:
    """Return the detection record for this episode's peak frame."""
    key = (episode["callsign"], episode["transponder_id"])
    candidates = detections_by_flight.get(key, [])
    onset = datetime.datetime.fromisoformat(episode["onset"])
    end = datetime.datetime.fromisoformat(episode["end"])
    best = None
    for d in candidates:
        t = datetime.datetime.fromisoformat(d["wall_time_utc"])
        if not (onset <= t <= end):
            continue
        if best is None or d["score"] > best["score"]:
            best = d
    return best


def _wall_to_frame_idx(ocr_records: list[dict], wall: str) -> int | None:
    wall_dt = datetime.datetime.fromisoformat(wall).replace(microsecond=0)
    for rec in ocr_records:
        rec_dt = datetime.datetime.fromisoformat(rec["wall_time_utc"]).replace(
            microsecond=0
        )
        if rec_dt == wall_dt:
            return rec["frame_idx"]
    return None


def _decode_single_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
        total = int(stream.frames) if stream.frames else int(
            round(duration_s * float(stream.average_rate or 30))
        )
        if total == 0:
            return None
        target_time_s = (frame_idx / total) * duration_s
        target_pts = int(target_time_s / float(time_base))
        container.seek(target_pts, stream=stream, any_frame=False, backward=True)
        decoded = None
        for frame in container.decode(stream):
            decoded = frame
            if frame.pts is not None and frame.pts >= target_pts:
                break
        if decoded is None:
            return None
        return decoded.to_ndarray(format="bgr24")
    finally:
        container.close()


def _render_panel(
    episode: dict,
    peak_det: dict,
    frame: np.ndarray,
    projection: dict | None,
    det_config,
    label: str,
    out_path: Path,
) -> None:
    crop_pad = 200
    if projection is not None:
        cx = float(projection["pixel_x"])
        cy = float(projection["pixel_y"])
    elif peak_det.get("pixel_line"):
        x1, y1, x2, y2 = peak_det["pixel_line"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
    else:
        cx = frame.shape[1] / 2
        cy = frame.shape[0] / 2

    h, w = frame.shape[:2]
    x0 = max(0, int(cx - crop_pad))
    y0 = max(0, int(cy - crop_pad))
    x1c = min(w, int(cx + crop_pad))
    y1c = min(h, int(cy + crop_pad))
    crop = frame[y0:y1c, x0:x1c].copy()
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    overlay = crop_rgb.copy()
    if projection is not None and "path_dx" in projection:
        poly = rotated_polygon(
            PixelPoint(x=cx, y=cy),
            (float(projection["path_dx"]), float(projection["path_dy"])),
            det_config,
        )
        poly_local = poly.copy()
        poly_local[:, 0] -= x0
        poly_local[:, 1] -= y0
        pts = poly_local.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 200, 0), thickness=2)

    if peak_det.get("pixel_line"):
        px1, py1, px2, py2 = peak_det["pixel_line"]
        cv2.line(
            overlay,
            (int(px1) - x0, int(py1) - y0),
            (int(px2) - x0, int(py2) - y0),
            (0, 255, 0),
            thickness=3,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(crop_rgb)
    axes[0].set_title(
        f"{episode['callsign']} — peak {episode['peak_score']:.3f} — {label}\n"
        f"onset {episode['onset']}  frames={episode['frame_count']}"
    )
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"ROI polygon (orange) + detection line (green)\n"
        f"aligned_lines={peak_det.get('aligned_lines')} "
        f"num_long_lines={peak_det.get('num_long_lines')}"
    )
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --- Driver -------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="April-8 style E2E regression baseline.")
    parser.add_argument("--date", required=True, type=str, help="UTC date YYYY-MM-DD.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Site YAML config path.")
    parser.add_argument("--output-dir", default="output",
                        help="Pipeline output root.")
    parser.add_argument("--regression-dir", default="output/validation/regression",
                        help="Where to write the regression artifacts.")
    parser.add_argument("--skip-rerun", action="store_true",
                        help="Skip the determinism re-run (for quick metric-only passes).")
    parser.add_argument("--video", default=None,
                        help="Override video path (else resolved from config).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    date = datetime.date.fromisoformat(args.date)
    site_config = load_config(args.config)
    source_dir = Path(args.output_dir) / args.date
    regression_root = Path(args.regression_dir) / args.date
    regression_root.mkdir(parents=True, exist_ok=True)

    required = [
        "ocr.jsonl", "adsb.json", "projections.jsonl",
        "detections.jsonl", "episodes.jsonl", "pipeline.duckdb",
    ]
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"Source pipeline dir {source_dir} is missing: {missing}. "
            f"Run `uv run concam run --date {args.date}` first."
        )

    threshold = site_config.aggregation.detection_threshold
    print(f"[regression] date={args.date}  threshold={threshold}")
    print(f"[regression] source={source_dir}")
    print(f"[regression] output={regression_root}")

    # ---- Determinism ----
    determinism: dict = {}
    prev_metrics_path = regression_root / "metrics.json"
    if args.skip_rerun:
        # Preserve any earlier verdict so --skip-rerun can be used to refresh
        # metrics/panels after the expensive re-run has already passed.
        prev_det = None
        if prev_metrics_path.exists():
            try:
                prev_det = json.loads(prev_metrics_path.read_text()).get("determinism")
            except (json.JSONDecodeError, OSError):
                prev_det = None
        if prev_det and not prev_det.get("skipped"):
            determinism = {**prev_det, "reused_from": str(prev_metrics_path)}
            print(f"[regression] determinism re-run SKIPPED "
                  f"(reusing prior verdict: pass={determinism.get('pass')})")
        else:
            determinism = {"skipped": True}
            print("[regression] determinism re-run SKIPPED (--skip-rerun)")
    else:
        rerun_dir = regression_root / "run2" / args.date
        if args.video:
            video_path = Path(args.video)
        else:
            video_path = resolve_video_path(site_config.video, date)
        print(f"[regression] re-running detect+aggregate+store against {video_path} ...")
        _rerun_numeric_stages(
            date=date,
            source_dir=source_dir,
            target_dir=rerun_dir,
            video_path=video_path,
            site_config=site_config,
        )
        det_diff = _diff_jsonl(
            source_dir / "detections.jsonl", rerun_dir / "detections.jsonl"
        )
        ep_diff = _diff_jsonl(
            source_dir / "episodes.jsonl", rerun_dir / "episodes.jsonl"
        )
        determinism = {
            "detections": det_diff,
            "episodes": ep_diff,
            "rerun_dir": str(rerun_dir),
        }
        ok = det_diff.get("identical") and ep_diff.get("identical")
        determinism["pass"] = bool(ok)
        status = "IDENTICAL" if ok else "DRIFT"
        print(f"[regression] determinism: {status}")
        if not ok:
            print(f"[regression] detections diff: {det_diff}")
            print(f"[regression] episodes diff:   {ep_diff}")

    # ---- Metrics ----
    metrics = _collect_metrics(source_dir, threshold)
    print(f"[regression] flights={metrics['flights']}  "
          f"episodes={metrics['episodes']}  "
          f"above_threshold={metrics['episodes_above_threshold']}  "
          f"detections={metrics['detections']}  "
          f"hit_rate={metrics['detection_hit_rate']:.4f}")

    # ---- Spot-check panels ----
    episodes = list(_iter_jsonl(source_dir / "episodes.jsonl"))
    picks = _pick_episodes(episodes, threshold)
    detections = list(_iter_jsonl(source_dir / "detections.jsonl"))
    det_by_flight: dict[tuple[str, str], list[dict]] = {}
    for d in detections:
        det_by_flight.setdefault((d["callsign"], d["transponder_id"]), []).append(d)
    proj_by_flight: dict[tuple[str, str], list[dict]] = {}
    for p in _iter_jsonl(source_dir / "projections.jsonl"):
        proj_by_flight.setdefault((p["callsign"], p["transponder_id"]), []).append(p)
    ocr_records = list(_iter_jsonl(source_dir / "ocr.jsonl"))

    video_path: Path | None = None
    try:
        video_path = Path(args.video) if args.video else resolve_video_path(
            site_config.video, date
        )
    except FileNotFoundError as e:
        print(f"[regression] WARNING: video unavailable for panels: {e}")

    panels_dir = regression_root / "panels"
    panels_dir.mkdir(exist_ok=True, parents=True)

    panel_records = []
    if video_path is not None:
        for label, group in (("top", picks["top"]), ("near_threshold", picks["near_threshold"])):
            for i, ep in enumerate(group, start=1):
                peak = _find_peak_frame(det_by_flight, ep)
                if peak is None:
                    print(f"[regression] no peak detection for {ep['callsign']} {ep['onset']}")
                    continue
                frame_idx = _wall_to_frame_idx(ocr_records, peak["wall_time_utc"])
                if frame_idx is None:
                    print(f"[regression] no OCR frame for {peak['wall_time_utc']}")
                    continue
                frame = _decode_single_frame(video_path, frame_idx)
                if frame is None:
                    print(f"[regression] frame decode failed @ idx={frame_idx}")
                    continue

                flight_projs = proj_by_flight.get(
                    (ep["callsign"], ep["transponder_id"]), []
                )
                peak_wall = peak["wall_time_utc"]
                projection = next(
                    (p for p in flight_projs if p["wall_time_utc"] == peak_wall),
                    None,
                )
                out_path = panels_dir / f"{label}_{i:02d}_{ep['callsign']}_{frame_idx}.png"
                _render_panel(
                    episode=ep,
                    peak_det=peak,
                    frame=frame,
                    projection=projection,
                    det_config=site_config.detection,
                    label=label,
                    out_path=out_path,
                )
                panel_records.append({
                    "label": label,
                    "index": i,
                    "callsign": ep["callsign"],
                    "transponder_id": ep["transponder_id"],
                    "onset": ep["onset"],
                    "peak_score": ep["peak_score"],
                    "peak_wall_time": peak_wall,
                    "frame_idx": frame_idx,
                    "panel": str(out_path.relative_to(regression_root)),
                })
                print(f"[regression] wrote {out_path.name}")

    # ---- Persist ----
    metrics_out = {
        "date": args.date,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "determinism": determinism,
        "metrics": metrics,
        "spot_check": {
            "top": [{"callsign": e["callsign"], "peak_score": e["peak_score"],
                     "onset": e["onset"]} for e in picks["top"]],
            "near_threshold": [{"callsign": e["callsign"], "peak_score": e["peak_score"],
                                "onset": e["onset"]}
                               for e in picks["near_threshold"]],
            "panels": panel_records,
        },
    }
    (regression_root / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2, default=str)
    )
    print(f"[regression] wrote metrics -> {regression_root / 'metrics.json'}")

    # ---- Report ----
    _write_report(regression_root / "report.md", metrics_out)
    print(f"[regression] wrote report  -> {regression_root / 'report.md'}")
    return 0 if determinism.get("pass", True) or args.skip_rerun else 1


def _write_report(path: Path, out: dict) -> None:
    m = out["metrics"]
    det = out["determinism"]
    lines: list[str] = []
    lines.append(f"# Regression baseline — {out['date']}")
    lines.append("")
    lines.append(f"Generated: {out['generated_at']}")
    lines.append("")
    lines.append("## Determinism")
    if det.get("skipped"):
        lines.append("- Skipped (--skip-rerun).")
    else:
        ok = det.get("pass")
        verdict = "PASS (byte-identical re-run)" if ok else "FAIL (drift detected)"
        lines.append(f"- **{verdict}**")
        lines.append(f"- detections diff: `{det['detections']}`")
        lines.append(f"- episodes diff:   `{det['episodes']}`")
        lines.append(f"- rerun dir:       `{det['rerun_dir']}`")
    lines.append("")
    lines.append("## Baseline metrics")
    lines.append(f"- Flights: **{m['flights']}**")
    lines.append(f"- Pings: {m['pings']}")
    lines.append(f"- Frames (OCR): {m['frames']} ({m['ocr_ok']} ok)")
    lines.append(f"- Projections: {m['projections']}")
    lines.append(f"- Detections: **{m['detections']}** "
                 f"({m['detections_above_threshold']} ≥ {m['aggregation_threshold']})")
    lines.append(f"- Detection hit rate: {m['detection_hit_rate']:.4f}")
    lines.append(f"- Max detection score: {m['detection_score_max']:.3f}")
    lines.append(f"- Mean non-zero score: {m['detection_score_mean_nonzero']:.3f}")
    lines.append(f"- Episodes: **{m['episodes']}** "
                 f"({m['episodes_above_threshold']} ≥ {m['aggregation_threshold']})")
    lines.append("")
    lines.append("### Detection score histogram")
    lines.append("| lo | hi | count |")
    lines.append("|----|----|-------|")
    for b in m["detection_score_histogram"]:
        lines.append(f"| {b['lo']:.3f} | {b['hi']:.3f} | {b['count']} |")
    lines.append("")
    if "contrail_length_m_histogram" in m:
        lines.append("### Contrail length histogram (detection-level, length_m > 0)")
        lines.append("| lo (m) | hi (m) | count |")
        lines.append("|--------|--------|-------|")
        for b in m["contrail_length_m_histogram"]:
            hi_str = f"{b['hi']:.0f}" if b["hi"] != float("inf") else "∞"
            lines.append(f"| {b['lo']:.0f} | {hi_str} | {b['count']} |")
        lines.append(f"- Detections with length: **{m['contrail_length_m_count']}**  "
                     f"mean {m['contrail_length_m_mean']:.0f} m  "
                     f"max {m['contrail_length_m_max']:.0f} m")
        lines.append("")
    lines.append("### DuckDB")
    d = m["duckdb"]
    lines.append(f"- rows: {d['row_count']}")
    lines.append(f"- peak_score min/mean/max: "
                 f"{d['peak_score_min']:.3f} / {d['peak_score_mean']:.3f} / "
                 f"{d['peak_score_max']:.3f}")
    if "peak_contrail_length_m" in d and d["peak_contrail_length_m"]:
        pl = d["peak_contrail_length_m"]
        lines.append(f"- peak_contrail_length_m (episodes with value): "
                     f"{pl['episodes_with_length']} episodes  "
                     f"min {pl['length_m_min']:.0f} m  "
                     f"mean {pl['length_m_mean']:.0f} m  "
                     f"max {pl['length_m_max']:.0f} m")
    lines.append("")
    lines.append("## Spot-check episodes")
    lines.append("")
    lines.append("### Top 5 by peak_score")
    lines.append("| # | callsign | peak | onset |")
    lines.append("|---|----------|------|-------|")
    for i, e in enumerate(out["spot_check"]["top"], start=1):
        lines.append(f"| {i} | {e['callsign']} | {e['peak_score']:.3f} | {e['onset']} |")
    lines.append("")
    lines.append(f"### 5 nearest to threshold ({m['aggregation_threshold']})")
    lines.append("| # | callsign | peak | onset |")
    lines.append("|---|----------|------|-------|")
    for i, e in enumerate(out["spot_check"]["near_threshold"], start=1):
        lines.append(f"| {i} | {e['callsign']} | {e['peak_score']:.3f} | {e['onset']} |")
    lines.append("")
    lines.append("### Panels")
    for rec in out["spot_check"]["panels"]:
        lines.append(f"- `{rec['panel']}` — {rec['label']} #{rec['index']} "
                     f"{rec['callsign']} @ frame {rec['frame_idx']} (score {rec['peak_score']:.3f})")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
