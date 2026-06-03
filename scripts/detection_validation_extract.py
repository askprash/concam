"""Detection validation: extract candidate (frame, flight) pairs for human labeling (PRD item 6).

The detection go/no-go gate in the PRD requires ~20 real frame-flight pairs
with ground-truth positive/negative labels. This script handles the machine
half of that workflow:

  1. Read the cached ``projections.jsonl`` for the date (full-day coverage
     is already produced by ``concam run``; the project stage is fast and
     doesn't depend on detection output).
  2. Filter to daylight hours at the camera site (Boston: roughly 10:30 UTC
     to 23:30 UTC in April; overridable via ``--daylight-utc``).
  3. Select N candidates that are well-spread in time and pixel location
     so the labeled set samples the full FOV and full day, not a single
     corridor.
  4. Seek the daily timelapse via PyAV, extract the oriented ROI crop plus
     a wider context patch showing where the ROI sits in the sky.
  5. Save per-candidate PNGs, a manifest, a summary grid PNG, and a tiny
     HTML labeller that writes ``labels.json`` on download.

The human labels the candidates (contrail / no_contrail / skip) and the
companion ``detection_validation_sweep.py`` ingests ``labels.json`` to
tune Canny+Hough parameters against the real score distributions.

Usage::

    uv run python scripts/detection_validation_extract.py --date 2026-04-08
    uv run python scripts/detection_validation_extract.py \\
        --date 2026-04-09 --num-candidates 20 --daylight-utc 10:30,23:30
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from concam.config import load_config
from concam.video import decode_frames


@dataclass
class Candidate:
    """One (frame, flight) pair selected for labeling."""

    idx: int
    frame_idx: int
    wall_time_utc: str
    callsign: str
    transponder_id: str
    pixel_x: float
    pixel_y: float
    roi: dict  # {x, y, w, h}
    path_dx: float
    path_dy: float


def _load_projections(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_frame_zero_anchor(ocr_path: Path) -> datetime.datetime:
    """Return wall time of frame 0 from the OCR cache.

    The pipeline writes ``ocr.jsonl`` sorted by frame_idx, so line 0 is frame 0.
    If the OCR cache is absent or frame 0 is missing we raise; the caller
    decides how to surface that to the user.
    """
    with ocr_path.open() as f:
        first = json.loads(f.readline())
    if first.get("frame_idx") != 0:
        raise ValueError(
            f"expected frame_idx=0 on first line of {ocr_path}, got {first.get('frame_idx')!r}"
        )
    return datetime.datetime.fromisoformat(first["wall_time_utc"])


def _parse_hhmm(s: str) -> datetime.time:
    h, m = s.split(":")
    return datetime.time(hour=int(h), minute=int(m))


def _pick_forced_candidates(
    projections: list[dict],
    callsigns: list[str],
    anchor_utc: datetime.datetime,
    seconds_per_frame: float,
    start_idx: int = 0,
) -> list[Candidate]:
    """Return one Candidate per forced callsign.

    Each entry in ``callsigns`` may optionally include a UTC target timestamp in
    the form ``CALLSIGN@HH:MM:SS``.  When a timestamp is given, the ping closest
    to that time is chosen.  Without a timestamp the ping closest to the image
    centre is used instead.  Callsigns not found in projections are skipped with
    a warning.
    """
    # Parse optional @HH:MM:SS suffix.
    parsed: list[tuple[str, datetime.datetime | None]] = []
    for entry in callsigns:
        if "@" in entry:
            cs_part, time_part = entry.split("@", 1)
            h, m, s = (int(x) for x in time_part.strip().split(":"))
            target_dt = anchor_utc.replace(hour=h, minute=m, second=s, microsecond=0)
            parsed.append((cs_part.strip(), target_dt))
        else:
            parsed.append((entry.strip(), None))

    cs_keys = {cs.upper().replace(" ", "") for cs, _ in parsed}
    by_cs: dict[str, list[dict]] = {}
    for row in projections:
        cs = row["callsign"].upper().replace(" ", "")
        if cs in cs_keys:
            by_cs.setdefault(cs, []).append(row)

    cx, cy = 3840 / 2, 2160 / 2
    candidates: list[Candidate] = []
    idx = start_idx
    for cs, target_dt in parsed:
        key = cs.upper().replace(" ", "")
        rows = by_cs.get(key)
        if not rows:
            print(f"  WARN: callsign {cs!r} not found in projections — skipping")
            continue
        if target_dt is not None:
            best = min(rows, key=lambda r: abs(
                (datetime.datetime.fromisoformat(r["wall_time_utc"]) - target_dt).total_seconds()
            ))
            delta = abs((datetime.datetime.fromisoformat(best["wall_time_utc"]) - target_dt).total_seconds())
            print(f"  {cs}: closest ping at {best['wall_time_utc'][11:19]} UTC  (Δ{delta:.0f}s)")
        else:
            best = min(rows, key=lambda r: math.hypot(r["pixel_x"] - cx, r["pixel_y"] - cy))
        t = datetime.datetime.fromisoformat(best["wall_time_utc"])
        frame_idx = int(round((t - anchor_utc).total_seconds() / seconds_per_frame))
        candidates.append(
            Candidate(
                idx=idx,
                frame_idx=frame_idx,
                wall_time_utc=best["wall_time_utc"],
                callsign=best["callsign"],
                transponder_id=best["transponder_id"],
                pixel_x=float(best["pixel_x"]),
                pixel_y=float(best["pixel_y"]),
                roi=best["roi"],
                path_dx=float(best["path_dx"]),
                path_dy=float(best["path_dy"]),
            )
        )
        idx += 1
    return candidates


def _pick_candidates(
    projections: list[dict],
    anchor_utc: datetime.datetime,
    daylight_start: datetime.time,
    daylight_end: datetime.time,
    n: int,
    seconds_per_frame: float,
    bucket_minutes: int = 30,
    exclude_transponders: set[str] | None = None,
) -> list[Candidate]:
    """Pick N daylight candidates well-spread in time and pixel space.

    We bucket by 30-minute time windows and pick the most-central projection
    per bucket (closest to FOV center), then trim to N using even-index slicing.
    Each distinct transponder_id can be selected at most once so the labeled
    set isn't dominated by a single long-crossing flight.
    """
    # Filter to daylight
    daylight_rows: list[dict] = []
    for row in projections:
        t = datetime.datetime.fromisoformat(row["wall_time_utc"])
        local_time = t.time()  # naive UTC; compare against UTC daylight bounds
        if daylight_start <= local_time <= daylight_end:
            daylight_rows.append(row)

    if not daylight_rows:
        raise RuntimeError(
            f"no projections in daylight window {daylight_start}-{daylight_end} UTC; "
            f"got {len(projections)} rows total"
        )

    # Bucket by time windows (default 30 min); within each bucket keep the ping
    # closest to image center (so labels aren't biased toward edge-of-frame geometry).
    buckets: dict[int, dict] = {}  # key: minute_bucket, value: (row, distance_to_center)
    cx, cy = 3840 / 2, 2160 / 2
    exclude = exclude_transponders or set()
    for row in daylight_rows:
        if row["transponder_id"] in exclude:
            continue
        t = datetime.datetime.fromisoformat(row["wall_time_utc"])
        bucket = (t.hour * 60 + t.minute) // bucket_minutes
        dist = math.hypot(row["pixel_x"] - cx, row["pixel_y"] - cy)
        if bucket not in buckets or dist < buckets[bucket][1]:
            buckets[bucket] = (row, dist)

    # Pick one row per transponder at most (prefer earlier buckets), trim to N.
    seen_tids: set[str] = set()
    picked: list[dict] = []
    for bucket in sorted(buckets):
        row = buckets[bucket][0]
        tid = row["transponder_id"]
        if tid in seen_tids:
            continue
        seen_tids.add(tid)
        picked.append(row)

    if len(picked) == 0:
        raise RuntimeError("no candidates after de-duplication by transponder_id")

    if len(picked) > n:
        # Evenly-spaced downsample so we keep full-day coverage.
        idxs = [int(round(k * (len(picked) - 1) / (n - 1))) for k in range(n)]
        picked = [picked[i] for i in idxs]

    candidates: list[Candidate] = []
    for i, row in enumerate(picked):
        t = datetime.datetime.fromisoformat(row["wall_time_utc"])
        frame_idx = int(round((t - anchor_utc).total_seconds() / seconds_per_frame))
        candidates.append(
            Candidate(
                idx=i,
                frame_idx=frame_idx,
                wall_time_utc=row["wall_time_utc"],
                callsign=row["callsign"],
                transponder_id=row["transponder_id"],
                pixel_x=float(row["pixel_x"]),
                pixel_y=float(row["pixel_y"]),
                roi=row["roi"],
                path_dx=float(row["path_dx"]),
                path_dy=float(row["path_dy"]),
            )
        )
    return candidates



def _extract_roi_crop(frame: np.ndarray, roi: dict, pad: int = 20) -> np.ndarray:
    """Crop the oriented ROI from the frame with optional padding for context."""
    h, w = frame.shape[:2]
    x1 = max(0, roi["x"] - pad)
    y1 = max(0, roi["y"] - pad)
    x2 = min(w, roi["x"] + roi["w"] + pad)
    y2 = min(h, roi["y"] + roi["h"] + pad)
    return frame[y1:y2, x1:x2].copy()


def _extract_context_crop(
    frame: np.ndarray, cand: Candidate, context_size: int = 800
) -> np.ndarray:
    """Crop a wider context patch (context_size x context_size) centered on the pixel,
    with the ROI drawn as a red box and the path vector as a yellow arrow."""
    h, w = frame.shape[:2]
    half = context_size // 2
    cx = int(round(cand.pixel_x))
    cy = int(round(cand.pixel_y))
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    crop = frame[y1:y2, x1:x2].copy()

    # Draw the ROI rectangle in crop-local coordinates.
    rx1 = cand.roi["x"] - x1
    ry1 = cand.roi["y"] - y1
    rx2 = rx1 + cand.roi["w"]
    ry2 = ry1 + cand.roi["h"]
    cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (60, 60, 255), 2)

    # Draw path arrow (fixed 80 px length)
    ax1 = cx - x1
    ay1 = cy - y1
    ax2 = int(round(ax1 + 80 * cand.path_dx))
    ay2 = int(round(ay1 + 80 * cand.path_dy))
    cv2.arrowedLine(crop, (ax1, ay1), (ax2, ay2), (60, 220, 220), 2, tipLength=0.25)

    # Center dot
    cv2.circle(crop, (ax1, ay1), 4, (100, 255, 100), -1)
    return crop


def _annotate_tile(
    roi_crop: np.ndarray, cand: Candidate, target_w: int = 500
) -> np.ndarray:
    """Tile = ROI crop on top, 3 lines of metadata below."""
    h, w = roi_crop.shape[:2]
    scale = target_w / max(1, w)
    resized = cv2.resize(roi_crop, (target_w, max(1, int(round(h * scale)))))
    text_h = 90
    rh = resized.shape[0]
    tile = np.full((rh + text_h, target_w, 3), 32, dtype=np.uint8)
    tile[:rh] = resized

    line1 = f"#{cand.idx:02d}  {cand.callsign}  frame={cand.frame_idx}"
    line2 = f"utc {cand.wall_time_utc.split('+')[0]}"
    line3 = f"px ({cand.pixel_x:.0f},{cand.pixel_y:.0f})  roi {cand.roi['w']}x{cand.roi['h']}"
    cv2.putText(tile, line1, (8, rh + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(tile, line2, (8, rh + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(tile, line3, (8, rh + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return tile


def _compose_grid(tiles: list[np.ndarray], cols: int = 5) -> np.ndarray:
    if not tiles:
        raise ValueError("no tiles to compose")
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    padded: list[np.ndarray] = []
    for t in tiles:
        h, w = t.shape[:2]
        if h == th and w == tw:
            padded.append(t)
        else:
            pad = np.full((th, tw, 3), 32, dtype=np.uint8)
            pad[:h, :w] = t
            padded.append(pad)
    rows = (len(padded) + cols - 1) // cols
    grid = np.full((rows * th, cols * tw, 3), 16, dtype=np.uint8)
    for i, t in enumerate(padded):
        r = i // cols
        c = i % cols
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return grid


def _write_labeller_html(
    path: Path,
    manifest_name: str,
    num_candidates: int,
    manifest_dict: dict | None = None,
    base_dir: Path | None = None,
) -> None:
    """Self-contained HTML labeller — manifest and images are inlined as data URIs.

    When ``manifest_dict`` and ``base_dir`` are supplied, the manifest JSON and
    all referenced ROI/context images are embedded directly in the HTML so the
    file works in JupyterHub, file://, or any context that blocks fetch().
    Falls back to a fetch()-based approach when manifest data is not available
    (backward-compatible with old call sites that only pass path + name + count).
    """
    import base64

    # Build an inlined manifest with data-URI images when possible.
    if manifest_dict is not None and base_dir is not None:
        inlined = dict(manifest_dict)
        inlined_candidates = []
        for c in manifest_dict.get("candidates", []):
            ic = dict(c)
            for key in ("roi_png", "context_png"):
                rel = c.get(key, "")
                img_path = base_dir / rel if rel else None
                if img_path and img_path.exists():
                    b64 = base64.b64encode(img_path.read_bytes()).decode()
                    ic[key] = f"data:image/png;base64,{b64}"
            inlined_candidates.append(ic)
        inlined["candidates"] = inlined_candidates
        manifest_js = "const MANIFEST = " + json.dumps(inlined) + ";"
        init_js = "function init() { manifest = MANIFEST; render(); }"
    else:
        manifest_js = ""
        init_js = f"""async function init() {{
  const res = await fetch("{manifest_name}");
  manifest = await res.json();
  render();
}}"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Detection validation labeller</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; margin: 1rem; }}
h1 {{ font-size: 1.2rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 0.75rem; }}
.card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 0.5rem; }}
.card img {{ width: 100%; height: auto; display: block; border-radius: 3px; }}
.card .meta {{ font-size: 0.75rem; color: #aaa; margin-top: 0.3rem; }}
.card.positive {{ background: #17301a; border-color: #2d5; }}
.card.negative {{ background: #301818; border-color: #d25; }}
.card.skip     {{ background: #2a2a14; border-color: #db5; }}
.controls {{ margin-top: 0.3rem; display: flex; gap: 0.4rem; flex-wrap: wrap; }}
.controls label {{ cursor: pointer; font-size: 0.8rem; }}
textarea {{ width: 100%; margin-top: 0.3rem; background: #111; color: #eee; border: 1px solid #333; }}
.toolbar {{ position: sticky; top: 0; background: #111; padding: 0.5rem 0; z-index: 10; display: flex; gap: 0.75rem; align-items: center; border-bottom: 1px solid #333; }}
button {{ background: #2a5; color: #fff; border: 0; padding: 0.4rem 0.8rem; border-radius: 4px; cursor: pointer; }}
button:hover {{ background: #3c6; }}
.small {{ font-size: 0.75rem; color: #888; }}
</style></head>
<body>
<h1>Detection validation labeller ({num_candidates} candidates)</h1>
<div class="toolbar">
<button id="download">Download labels.json</button>
<span id="progress" class="small">0 / {num_candidates}</span>
<span class="small">Tip: "skip" for cases where the flight track is off-screen or the overlay is wrong.</span>
</div>
<div id="grid" class="grid"></div>
<script>
{manifest_js}
const STORAGE_KEY = "concam-detection-labels";
let manifest = null;
let state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");

{init_js}
function updateProgress() {{
  const n = manifest.candidates.filter(c => state[c.idx]?.label).length;
  document.getElementById("progress").textContent = n + " / " + manifest.candidates.length;
}}
function save() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); updateProgress(); }}
function render() {{
  const g = document.getElementById("grid");
  g.innerHTML = "";
  for (const c of manifest.candidates) {{
    const cur = state[c.idx] || {{}};
    const card = document.createElement("div");
    card.className = "card" + (cur.label ? " " + cur.label : "");
    card.innerHTML = `
      <img src="${{c.roi_png}}" alt="ROI ${{c.idx}}"/>
      <img src="${{c.context_png}}" alt="context ${{c.idx}}" style="margin-top:0.3rem"/>
      <div class="meta">#${{String(c.idx).padStart(2, '0')}} ${{c.callsign}} · frame ${{c.frame_idx}} · ${{c.wall_time_utc}}</div>
      <div class="controls">
        <label><input type="radio" name="l${{c.idx}}" value="positive" ${{cur.label === 'positive' ? 'checked' : ''}}/> contrail</label>
        <label><input type="radio" name="l${{c.idx}}" value="negative" ${{cur.label === 'negative' ? 'checked' : ''}}/> clear</label>
        <label><input type="radio" name="l${{c.idx}}" value="skip"     ${{cur.label === 'skip'     ? 'checked' : ''}}/> skip</label>
      </div>
      <textarea data-idx="${{c.idx}}" placeholder="notes (optional)">${{cur.notes || ''}}</textarea>`;
    card.querySelectorAll('input[type=radio]').forEach(i => i.addEventListener('change', (e) => {{
      state[c.idx] = state[c.idx] || {{}};
      state[c.idx].label = e.target.value;
      save();
      render();
    }}));
    card.querySelector('textarea').addEventListener('input', (e) => {{
      state[c.idx] = state[c.idx] || {{}};
      state[c.idx].notes = e.target.value;
      save();
    }});
    g.appendChild(card);
  }}
  updateProgress();
}}
document.getElementById('download').addEventListener('click', () => {{
  const labels = manifest.candidates
    .filter(c => state[c.idx]?.label)
    .map(c => ({{
      idx: c.idx,
      frame_idx: c.frame_idx,
      wall_time_utc: c.wall_time_utc,
      callsign: c.callsign,
      transponder_id: c.transponder_id,
      label: state[c.idx].label,
      notes: state[c.idx].notes || ''
    }}));
  const payload = {{ date: manifest.date, generated_at: new Date().toISOString(), labels }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels.json';
  a.click();
}});
init();
</script></body></html>
"""
    path.write_text(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    ap.add_argument("--config", default="configs/mit_green_building.yaml")
    ap.add_argument("--output-dir", default="output", help="where cached pipeline artefacts live")
    ap.add_argument("--validation-dir", default=None, help="output/validation/detection/<date> by default")
    ap.add_argument("--num-candidates", type=int, default=20)
    ap.add_argument(
        "--daylight-utc",
        default="10:30,23:30",
        help="UTC daylight window as HH:MM,HH:MM (Boston April default).",
    )
    ap.add_argument("--seconds-per-frame", type=float, default=1.0)
    ap.add_argument("--video", default=None, help="override the timelapse path")
    ap.add_argument("--bucket-minutes", type=int, default=30,
                    help="time-window granularity for candidate spacing (smaller = denser)")
    ap.add_argument("--exclude-manifest", default=None,
                    help="path to an existing manifest.json; its transponder_ids will be excluded (use to extract a fresh batch)")
    ap.add_argument("--force-callsigns", default=None,
                    help="comma-separated callsigns to always include (e.g. confirmed contrail flights); "
                         "bypass bucket sampling for these; combined with --append-to-manifest to extend an existing batch")
    ap.add_argument("--append-to-manifest", default=None,
                    help="path to an existing manifest.json; new candidates are appended with higher idx values "
                         "and existing ROI PNGs are preserved")
    ap.add_argument(
        "--upscale-to-calibration",
        action="store_true",
        help=(
            "If the video is smaller than the calibration resolution, bilinearly "
            "upscale every decoded frame to match. Needed for archive dates "
            "captured at 720p/1080p when the calibration is 4K."
        ),
    )
    args = ap.parse_args()

    date = datetime.date.fromisoformat(args.date)
    config = load_config(args.config)
    output_dir = Path(args.output_dir) / date.isoformat()
    projections_path = output_dir / "projections.jsonl"
    ocr_path = output_dir / "ocr.jsonl"
    if not projections_path.exists():
        raise SystemExit(
            f"Missing {projections_path}. Run `uv run concam run --date {args.date}` "
            f"through the project stage first."
        )
    if not ocr_path.exists():
        raise SystemExit(
            f"Missing {ocr_path}. Need OCR for frame 0 anchor; "
            f"run `uv run concam run --date {args.date} --max-frames 1` at minimum."
        )

    validation_dir = Path(args.validation_dir) if args.validation_dir else Path(args.output_dir) / "validation" / "detection" / date.isoformat()
    rois_dir = validation_dir / "rois"
    rois_dir.mkdir(parents=True, exist_ok=True)

    projections = _load_projections(projections_path)
    anchor_utc = _load_frame_zero_anchor(ocr_path)
    daylight_start_s, daylight_end_s = args.daylight_utc.split(",")
    daylight_start = _parse_hhmm(daylight_start_s)
    daylight_end = _parse_hhmm(daylight_end_s)

    exclude_tids: set[str] = set()
    if args.exclude_manifest:
        ex = json.loads(Path(args.exclude_manifest).read_text())
        exclude_tids = {c["transponder_id"] for c in ex["candidates"]}
        print(f"Excluding {len(exclude_tids)} transponder_ids from {args.exclude_manifest}")

    # Load existing manifest for append mode — existing candidates are preserved as-is.
    existing_manifest: dict | None = None
    existing_candidates: list[dict] = []
    if args.append_to_manifest:
        ap_path = Path(args.append_to_manifest)
        if ap_path.exists():
            existing_manifest = json.loads(ap_path.read_text())
            existing_candidates = existing_manifest.get("candidates", [])
            print(f"Appending to existing manifest with {len(existing_candidates)} candidates.")
        else:
            print(f"  WARN: --append-to-manifest path not found ({ap_path}); starting fresh.")
    start_idx = max((c["idx"] for c in existing_candidates), default=-1) + 1

    # Forced callsigns — bypass bucket sampling entirely for these.
    forced: list[Candidate] = []
    if args.force_callsigns:
        cs_list = [c.strip() for c in args.force_callsigns.split(",") if c.strip()]
        forced = _pick_forced_candidates(
            projections, cs_list, anchor_utc, args.seconds_per_frame, start_idx=start_idx,
        )
        print(f"Forced callsigns: selected {len(forced)} candidates ({[c.callsign for c in forced]}).")
        # Exclude their transponder IDs from the bucket pass.
        exclude_tids |= {c.transponder_id for c in forced}
        start_idx += len(forced)

    # Bucket-sampled candidates fill remaining slots (up to --num-candidates total new candidates).
    n_bucket = max(0, args.num_candidates - len(forced))
    if n_bucket > 0:
        bucket_cands = _pick_candidates(
            projections,
            anchor_utc,
            daylight_start,
            daylight_end,
            n_bucket,
            args.seconds_per_frame,
            bucket_minutes=args.bucket_minutes,
            exclude_transponders=exclude_tids,
        )
        # Re-index bucket candidates to follow forced ones.
        bucket_cands = [
            Candidate(
                idx=start_idx + i,
                frame_idx=c.frame_idx,
                wall_time_utc=c.wall_time_utc,
                callsign=c.callsign,
                transponder_id=c.transponder_id,
                pixel_x=c.pixel_x,
                pixel_y=c.pixel_y,
                roi=c.roi,
                path_dx=c.path_dx,
                path_dy=c.path_dy,
            )
            for i, c in enumerate(bucket_cands)
        ]
    else:
        bucket_cands = []

    candidates = forced + bucket_cands
    total_new = len(candidates)
    print(f"Selected {total_new} new candidates ({len(forced)} forced + {len(bucket_cands)} bucket-sampled).")

    video_path = Path(args.video) if args.video else Path(config.video.root) / config.video.timelapse_glob.format(date=date)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    upscale_to: tuple[int, int] | None = None
    if args.upscale_to_calibration:
        upscale_to = tuple(int(v) for v in config.calibration.calibration_resolution)
        print(f"Upscaling decoded frames to calibration resolution {upscale_to}.")

    print(f"Decoding {len(candidates)} frames from {video_path}...")
    frames = decode_frames(
        video_path, [c.frame_idx for c in candidates],
        upscale_to=upscale_to,
    )

    tiles: list[np.ndarray] = []
    manifest_candidates: list[dict] = []
    for cand in candidates:
        frame = frames.get(cand.frame_idx)
        if frame is None:
            print(f"  SKIP #{cand.idx:02d}: frame {cand.frame_idx} decode failed")
            continue
        roi_crop = _extract_roi_crop(frame, cand.roi, pad=20)
        context_crop = _extract_context_crop(frame, cand, context_size=800)
        roi_png = rois_dir / f"roi_{cand.idx:02d}.png"
        context_png = rois_dir / f"context_{cand.idx:02d}.png"
        cv2.imwrite(str(roi_png), roi_crop)
        cv2.imwrite(str(context_png), context_crop)
        tiles.append(_annotate_tile(roi_crop, cand))
        manifest_candidates.append(
            {
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
                "context_png": f"rois/{context_png.name}",
            }
        )

    grid = _compose_grid(tiles, cols=5) if tiles else None
    grid_path = validation_dir / "candidate_grid.png"
    if grid is not None:
        cv2.imwrite(str(grid_path), grid)

    all_candidates = existing_candidates + manifest_candidates
    manifest = {
        "schema_version": 1,
        "date": date.isoformat(),
        "video": str(video_path),
        "anchor_utc": anchor_utc.isoformat(),
        "daylight_utc": args.daylight_utc,
        "candidates": all_candidates,
    }
    manifest_path = validation_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    html_path = validation_dir / "labeller.html"
    _write_labeller_html(
        html_path, manifest_path.name, len(all_candidates),
        manifest_dict=manifest, base_dir=validation_dir,
    )

    print()
    print(f"  Grid     : {grid_path}")
    print(f"  Manifest : {manifest_path}")
    print(f"  Labeller : {html_path}")
    print(f"  ROIs     : {rois_dir} ({2 * len(all_candidates)} total PNGs)")
    print()
    print(f"Next: open {html_path} in a browser, label each candidate, download labels.json, then")
    print(f"  uv run python scripts/detection_validation_sweep.py --date {args.date} \\")
    print(f"      --labels {validation_dir}/labels.json")


if __name__ == "__main__":
    main()
