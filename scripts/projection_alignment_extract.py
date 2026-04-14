"""Projection alignment validation: render real-frame overlays for 5 flyovers (PRD item 13).

The ADS-B → pixel projection unit tests check landmark accuracy within ~80 px
for buildings on the ground. They do NOT verify that overhead flights land on
the visible aircraft in the video — a sign-flip or small extrinsics error
would still pass landmark tests while putting the overlay in the wrong
quadrant. If the overlay can't be trusted, labels made against it are garbage
regardless of detector quality, so this script prepares the machine half of
the go/no-go gate:

  1. Group the cached ``projections.jsonl`` by transponder_id into flight
     trajectories, filter to daylight hours and in-frame sequences that are
     long enough to contain a visible aircraft silhouette.
  2. Pick N well-separated flyovers covering different parts of the FOV so
     a systematic distortion is visible across the image.
  3. For each flyover sample M frames at fixed spacing, seek the daily
     timelapse, and render an overlay image: full projected track as a
     polyline plus a crosshair on the ping that corresponds to the current
     frame. Images are downscaled for the browser.
  4. Emit ``manifest.json``, ``overview.png`` (one tile per flyover), and a
     self-contained ``labeller.html`` that lets a human click the *actual*
     aircraft pixel in each frame and downloads ``labels.json``.

The companion ``projection_alignment_analyze.py`` ingests ``labels.json``
and computes the offset vectors + go/no-go verdict against the 100 px
threshold in PRD item 13.

Usage::

    uv run python scripts/projection_alignment_extract.py --date 2026-04-09
    uv run python scripts/projection_alignment_extract.py \\
        --date 2026-04-09 --num-flyovers 5 --frames-per-flyover 8
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np

from concam.config import load_config


@dataclass
class Flyover:
    """One contiguous in-frame sequence of pings for a single flight."""

    idx: int
    transponder_id: str
    callsign: str
    pings: list[dict] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if len(self.pings) < 2:
            return 0.0
        t0 = datetime.datetime.fromisoformat(self.pings[0]["wall_time_utc"])
        t1 = datetime.datetime.fromisoformat(self.pings[-1]["wall_time_utc"])
        return (t1 - t0).total_seconds()

    @property
    def pixel_span(self) -> float:
        xs = [p["pixel_x"] for p in self.pings]
        ys = [p["pixel_y"] for p in self.pings]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    @property
    def center_px(self) -> tuple[float, float]:
        xs = [p["pixel_x"] for p in self.pings]
        ys = [p["pixel_y"] for p in self.pings]
        return sum(xs) / len(xs), sum(ys) / len(ys)


def _load_projections(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_frame_zero_anchor(ocr_path: Path) -> datetime.datetime:
    """Read frame 0's wall_time_utc from ocr.jsonl. Required to map wall times back to frame indices."""
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


def _group_flyovers(
    projections: list[dict],
    daylight_start: datetime.time,
    daylight_end: datetime.time,
    image_size: tuple[int, int],
    min_duration_s: float,
    min_pixel_span_px: float,
    max_gap_s: float = 3.0,
) -> list[Flyover]:
    """Build one Flyover per contiguous, in-frame, daylight chunk per transponder.

    A flight may enter/exit/re-enter the FOV across the day. Each contiguous
    in-frame run becomes its own candidate so we pick geometrically-distinct
    crossings rather than averaging over an aircraft that loops back.
    """
    width, height = image_size

    by_tid: dict[str, list[dict]] = {}
    for row in projections:
        t = datetime.datetime.fromisoformat(row["wall_time_utc"])
        if not (daylight_start <= t.time() <= daylight_end):
            continue
        px = row["pixel_x"]
        py = row["pixel_y"]
        # The projection stage already writes ROIs that may lie outside the frame
        # (projected yet clipped) — keep only pings whose pixel centre is inside.
        if not (0 <= px < width and 0 <= py < height):
            continue
        by_tid.setdefault(row["transponder_id"], []).append(row)

    flyovers: list[Flyover] = []
    for tid, rows in by_tid.items():
        rows.sort(key=lambda r: r["wall_time_utc"])
        # Split on time gaps — the upsampler emits 1 s pings, so anything >max_gap is a new crossing.
        runs: list[list[dict]] = [[rows[0]]]
        for prev, cur in zip(rows, rows[1:]):
            dt = (
                datetime.datetime.fromisoformat(cur["wall_time_utc"])
                - datetime.datetime.fromisoformat(prev["wall_time_utc"])
            ).total_seconds()
            if dt > max_gap_s:
                runs.append([cur])
            else:
                runs[-1].append(cur)

        for run in runs:
            fly = Flyover(
                idx=-1,  # filled in by caller
                transponder_id=tid,
                callsign=run[0]["callsign"],
                pings=run,
            )
            if fly.duration_s >= min_duration_s and fly.pixel_span >= min_pixel_span_px:
                flyovers.append(fly)
    return flyovers


def _pick_spread_flyovers(flyovers: list[Flyover], n: int, image_size: tuple[int, int]) -> list[Flyover]:
    """Greedy selection: pick flyovers whose center pixels are far apart in the FOV.

    Farthest-point sampling so a systematic radial distortion is exposed by
    covering the corners/edges rather than bunching around image center.
    """
    if not flyovers:
        return []
    if len(flyovers) <= n:
        for i, f in enumerate(flyovers):
            f.idx = i
        return flyovers

    width, height = image_size
    remaining = list(flyovers)
    # Seed with the flyover whose midpoint sits closest to the FOV corners, so
    # the first pick is off-center and the spread metric below has something
    # meaningful to push away from.
    cx, cy = width / 2, height / 2
    remaining.sort(key=lambda f: -math.hypot(f.center_px[0] - cx, f.center_px[1] - cy))
    picked: list[Flyover] = [remaining.pop(0)]

    while len(picked) < n and remaining:
        # Pick the flyover whose center is farthest from the nearest already-picked center.
        def min_dist(f: Flyover) -> float:
            return min(
                math.hypot(f.center_px[0] - p.center_px[0], f.center_px[1] - p.center_px[1])
                for p in picked
            )

        remaining.sort(key=min_dist, reverse=True)
        picked.append(remaining.pop(0))

    picked.sort(key=lambda f: f.pings[0]["wall_time_utc"])
    for i, f in enumerate(picked):
        f.idx = i
    return picked


def _sample_frame_indices(
    fly: Flyover,
    anchor_utc: datetime.datetime,
    seconds_per_frame: float,
    frames_per_flyover: int,
) -> list[dict]:
    """Pick M pings evenly spaced across the flyover's duration."""
    pings = fly.pings
    if len(pings) <= frames_per_flyover:
        chosen = list(pings)
    else:
        idxs = [int(round(k * (len(pings) - 1) / (frames_per_flyover - 1))) for k in range(frames_per_flyover)]
        chosen = [pings[i] for i in idxs]
    out = []
    for p in chosen:
        t = datetime.datetime.fromisoformat(p["wall_time_utc"])
        frame_idx = int(round((t - anchor_utc).total_seconds() / seconds_per_frame))
        out.append({
            "frame_idx": frame_idx,
            "wall_time_utc": p["wall_time_utc"],
            "pixel_x": float(p["pixel_x"]),
            "pixel_y": float(p["pixel_y"]),
        })
    return out


def _video_meta(video_path: Path) -> tuple[float, int]:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
        frames = int(stream.frames) if stream.frames else int(round(duration_s * float(stream.average_rate or 30)))
        return duration_s, frames
    finally:
        container.close()


def _decode_frames(
    video_path: Path,
    frame_indices: list[int],
    total_frames: int,
    duration_s: float,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for target_idx in sorted(set(frame_indices)):
            target_time_s = (target_idx / total_frames) * duration_s if total_frames else 0.0
            target_pts = int(target_time_s / float(time_base))
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            decoded = None
            for frame in container.decode(stream):
                decoded = frame
                if frame.pts is not None and frame.pts >= target_pts:
                    break
            if decoded is not None:
                out[target_idx] = decoded.to_ndarray(format="bgr24")
    finally:
        container.close()
    return out


def _render_overlay(
    frame: np.ndarray,
    full_track_px: list[tuple[float, float]],
    current_px: tuple[float, float],
    scale: float,
) -> np.ndarray:
    """Downscale the frame and draw the projected track + current-ping crosshair.

    The track is drawn in a muted color; the *current* ping is a bright
    crosshair + circle so the labeller knows which projected position they
    should be matching against the visible aircraft.
    """
    h, w = frame.shape[:2]
    new_size = (int(round(w * scale)), int(round(h * scale)))
    img = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # Full track polyline.
    pts = np.array(
        [[int(round(px * scale)), int(round(py * scale))] for px, py in full_track_px],
        dtype=np.int32,
    )
    if len(pts) >= 2:
        cv2.polylines(img, [pts], isClosed=False, color=(200, 200, 60), thickness=2, lineType=cv2.LINE_AA)

    # Current position crosshair.
    cx = int(round(current_px[0] * scale))
    cy = int(round(current_px[1] * scale))
    cv2.circle(img, (cx, cy), 18, (60, 220, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx - 30, cy), (cx + 30, cy), (60, 220, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 30), (cx, cy + 30), (60, 220, 255), 2, cv2.LINE_AA)
    return img


def _compose_overview(flyovers: list[Flyover], first_overlay: dict[int, np.ndarray], cols: int = 3) -> np.ndarray:
    """Build a small grid showing one overlay tile per flyover for at-a-glance triage."""
    tiles = []
    for fly in flyovers:
        key = fly.idx
        img = first_overlay.get(key)
        if img is None:
            continue
        tile_w = 560
        scale = tile_w / img.shape[1]
        tile = cv2.resize(img, (tile_w, int(round(img.shape[0] * scale))))
        label_h = 28
        th, tw = tile.shape[:2]
        framed = np.full((th + label_h, tw, 3), 24, dtype=np.uint8)
        framed[:th] = tile
        cv2.putText(
            framed,
            f"#{fly.idx:02d} {fly.callsign} {fly.pings[0]['wall_time_utc'].split('+')[0]}",
            (8, th + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        tiles.append(framed)
    if not tiles:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        h, w = t.shape[:2]
        if h == th and w == tw:
            padded.append(t)
        else:
            pad = np.full((th, tw, 3), 24, dtype=np.uint8)
            pad[:h, :w] = t
            padded.append(pad)
    rows = (len(padded) + cols - 1) // cols
    grid = np.full((rows * th, cols * tw, 3), 16, dtype=np.uint8)
    for i, t in enumerate(padded):
        r, c = divmod(i, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return grid


def _write_labeller_html(path: Path, manifest_name: str, num_flyovers: int, num_frames: int) -> None:
    """Self-contained HTML where the labeller clicks the visible aircraft on each frame.

    Click coordinates are stored in *image pixel space* (manifest.image_size),
    not PNG-display space, so the offset analysis can compare directly against
    the projected pixels in the manifest.
    """
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Projection alignment labeller</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; margin: 1rem; }}
h1 {{ font-size: 1.2rem; }}
.flyover {{ margin-bottom: 2rem; border: 1px solid #333; border-radius: 6px; padding: 0.75rem; background: #161616; }}
.flyover h2 {{ font-size: 1rem; margin: 0 0 0.3rem 0; }}
.frames {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 0.5rem; }}
.frame {{ background: #0c0c0c; border: 1px solid #2a2a2a; border-radius: 4px; padding: 0.3rem; position: relative; }}
.frame img {{ width: 100%; height: auto; cursor: crosshair; display: block; border-radius: 3px; }}
.frame .meta {{ font-size: 0.7rem; color: #aaa; margin-top: 0.2rem; display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; }}
.frame .meta button {{ background: #444; color: #eee; border: 0; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.7rem; cursor: pointer; }}
.frame .meta button:hover {{ background: #666; }}
.frame.labeled {{ border-color: #2d5; }}
.frame.notvisible {{ border-color: #db5; }}
.toolbar {{ position: sticky; top: 0; background: #111; padding: 0.5rem 0; z-index: 10; display: flex; gap: 0.75rem; align-items: center; border-bottom: 1px solid #333; margin-bottom: 1rem; }}
button.primary {{ background: #2a5; color: #fff; border: 0; padding: 0.4rem 0.8rem; border-radius: 4px; cursor: pointer; }}
button.primary:hover {{ background: #3c6; }}
.small {{ font-size: 0.75rem; color: #888; }}
.click-marker {{ position: absolute; width: 18px; height: 18px; border-radius: 50%; border: 2px solid #2f5; background: rgba(47, 232, 80, 0.25); pointer-events: none; transform: translate(-50%, -50%); }}
</style></head>
<body>
<h1>Projection alignment labeller ({num_flyovers} flyovers, {num_frames} frames)</h1>
<div class="toolbar">
<button class="primary" id="download">Download labels.json</button>
<span id="progress" class="small">0 / {num_frames}</span>
<span class="small">Click the visible aircraft in each frame. Use "not visible" if it isn't in the sky (clouds/glare/tiny).</span>
</div>
<div id="root"></div>
<script>
const MANIFEST_URL = "{manifest_name}";
const STORAGE_KEY = "concam-projection-alignment";
let manifest = null;
let state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");

async function init() {{
  manifest = await fetch(MANIFEST_URL).then(r => r.json());
  render();
}}
function frameKey(flyIdx, frameIdx) {{ return flyIdx + ':' + frameIdx; }}
function progressCount() {{
  let n = 0;
  for (const fly of manifest.flyovers) {{
    for (const fr of fly.frames) {{
      const s = state[frameKey(fly.idx, fr.frame_idx)];
      if (s && (s.visible === false || (typeof s.click_x === 'number'))) n++;
    }}
  }}
  return n;
}}
function save() {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  document.getElementById('progress').textContent = progressCount() + ' / ' + manifest.total_frames;
}}
function render() {{
  const root = document.getElementById('root');
  root.innerHTML = '';
  const [imgW, imgH] = manifest.image_size;
  for (const fly of manifest.flyovers) {{
    const sec = document.createElement('div');
    sec.className = 'flyover';
    sec.innerHTML = `<h2>#${{String(fly.idx).padStart(2, '0')}} ${{fly.callsign}} (${{fly.transponder_id}}) · ${{fly.frames.length}} frames</h2>`;
    const grid = document.createElement('div');
    grid.className = 'frames';
    for (const fr of fly.frames) {{
      const key = frameKey(fly.idx, fr.frame_idx);
      const cur = state[key] || {{}};
      const card = document.createElement('div');
      card.className = 'frame' + (cur.visible === false ? ' notvisible' : (typeof cur.click_x === 'number' ? ' labeled' : ''));
      card.innerHTML = `
        <img src="${{fr.overlay_png}}" alt="flyover ${{fly.idx}} frame ${{fr.frame_idx}}"/>
        <div class="meta">
          <span>f=${{fr.frame_idx}}</span>
          <span>${{fr.wall_time_utc.split('+')[0]}}</span>
          <span>proj=(${{fr.pixel_x.toFixed(0)}},${{fr.pixel_y.toFixed(0)}})</span>
          <button data-action="notvisible">not visible</button>
          <button data-action="clear">clear</button>
        </div>`;
      const img = card.querySelector('img');
      img.addEventListener('click', (e) => {{
        const rect = img.getBoundingClientRect();
        const relX = (e.clientX - rect.left) / rect.width;
        const relY = (e.clientY - rect.top) / rect.height;
        state[key] = {{
          click_x: relX * imgW,
          click_y: relY * imgH,
          visible: true,
          labeled_at: new Date().toISOString(),
        }};
        save();
        render();
      }});
      card.querySelector('[data-action=notvisible]').addEventListener('click', () => {{
        state[key] = {{visible: false, labeled_at: new Date().toISOString()}};
        save();
        render();
      }});
      card.querySelector('[data-action=clear]').addEventListener('click', () => {{
        delete state[key];
        save();
        render();
      }});
      if (typeof cur.click_x === 'number') {{
        // Show a marker at the click position once the image has loaded.
        img.addEventListener('load', () => {{
          const m = document.createElement('div');
          m.className = 'click-marker';
          const w = img.clientWidth;
          const h = img.clientHeight;
          m.style.left = (cur.click_x / imgW) * w + 'px';
          m.style.top = (3 + (cur.click_y / imgH) * h) + 'px';
          card.appendChild(m);
        }}, {{once: true}});
      }}
      grid.appendChild(card);
    }}
    sec.appendChild(grid);
    root.appendChild(sec);
  }}
  document.getElementById('progress').textContent = progressCount() + ' / ' + manifest.total_frames;
}}
document.getElementById('download').addEventListener('click', () => {{
  const labels = [];
  for (const fly of manifest.flyovers) {{
    for (const fr of fly.frames) {{
      const key = frameKey(fly.idx, fr.frame_idx);
      const s = state[key];
      if (!s) continue;
      labels.push({{
        flyover_idx: fly.idx,
        transponder_id: fly.transponder_id,
        callsign: fly.callsign,
        frame_idx: fr.frame_idx,
        wall_time_utc: fr.wall_time_utc,
        projected_pixel_x: fr.pixel_x,
        projected_pixel_y: fr.pixel_y,
        visible: s.visible !== false,
        click_x: typeof s.click_x === 'number' ? s.click_x : null,
        click_y: typeof s.click_y === 'number' ? s.click_y : null,
        labeled_at: s.labeled_at,
      }});
    }}
  }}
  const payload = {{schema_version: 1, date: manifest.date, generated_at: new Date().toISOString(), labels}};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
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
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--validation-dir", default=None, help="default: output/validation/projection/<date>")
    ap.add_argument("--num-flyovers", type=int, default=5)
    ap.add_argument("--frames-per-flyover", type=int, default=8)
    ap.add_argument("--daylight-utc", default="10:30,23:30")
    ap.add_argument("--min-duration-s", type=float, default=20.0)
    ap.add_argument("--min-span-px", type=float, default=150.0,
                    help="minimum pixel span of the track — filters near-overhead near-stationary pings")
    ap.add_argument("--seconds-per-frame", type=float, default=1.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--overlay-scale", type=float, default=0.5,
                    help="downscale factor applied to the rendered overlay PNGs")
    args = ap.parse_args()

    date = datetime.date.fromisoformat(args.date)
    config = load_config(args.config)
    output_dir = Path(args.output_dir) / date.isoformat()
    projections_path = output_dir / "projections.jsonl"
    ocr_path = output_dir / "ocr.jsonl"
    if not projections_path.exists():
        raise SystemExit(
            f"Missing {projections_path}. Run `uv run concam run --date {args.date}` through the project stage first."
        )
    if not ocr_path.exists():
        raise SystemExit(
            f"Missing {ocr_path}. Need OCR for frame 0 anchor; "
            f"run `uv run concam run --date {args.date} --max-frames 1` at minimum."
        )

    validation_dir = Path(args.validation_dir) if args.validation_dir else Path(args.output_dir) / "validation" / "projection" / date.isoformat()
    overlays_dir = validation_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    projections = _load_projections(projections_path)
    anchor_utc = _load_frame_zero_anchor(ocr_path)
    daylight_start_s, daylight_end_s = args.daylight_utc.split(",")
    daylight_start = _parse_hhmm(daylight_start_s)
    daylight_end = _parse_hhmm(daylight_end_s)

    image_size = tuple(int(v) for v in config.calibration.calibration_resolution)
    candidates = _group_flyovers(
        projections,
        daylight_start,
        daylight_end,
        image_size=image_size,
        min_duration_s=args.min_duration_s,
        min_pixel_span_px=args.min_span_px,
    )
    if not candidates:
        raise SystemExit(
            "No flyovers satisfy daylight + duration + span filters. "
            "Try widening --daylight-utc or lowering --min-duration-s / --min-span-px."
        )
    print(f"Found {len(candidates)} candidate flyovers; picking {args.num_flyovers}.")

    flyovers = _pick_spread_flyovers(candidates, args.num_flyovers, image_size=image_size)
    for fly in flyovers:
        print(
            f"  #{fly.idx:02d} {fly.callsign} ({fly.transponder_id}) "
            f"{fly.pings[0]['wall_time_utc']} .. {fly.pings[-1]['wall_time_utc']}  "
            f"dur={fly.duration_s:.0f}s  span={fly.pixel_span:.0f}px  "
            f"center=({fly.center_px[0]:.0f},{fly.center_px[1]:.0f})"
        )

    # For each flyover, decide which pings become labelled frames.
    per_fly_frames = {
        fly.idx: _sample_frame_indices(fly, anchor_utc, args.seconds_per_frame, args.frames_per_flyover)
        for fly in flyovers
    }
    all_frame_indices = sorted({fr["frame_idx"] for fr_list in per_fly_frames.values() for fr in fr_list})

    video_path = Path(args.video) if args.video else Path(config.video.root) / config.video.timelapse_glob.format(date=date)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    duration_s, total_frames = _video_meta(video_path)

    print(f"Decoding {len(all_frame_indices)} frames from {video_path} ...")
    frames = _decode_frames(video_path, all_frame_indices, total_frames, duration_s)

    manifest_flyovers: list[dict] = []
    first_overlay_per_fly: dict[int, np.ndarray] = {}
    total_frames_written = 0
    for fly in flyovers:
        track_px = [(float(p["pixel_x"]), float(p["pixel_y"])) for p in fly.pings]
        fly_frames_manifest: list[dict] = []
        for fr in per_fly_frames[fly.idx]:
            frame = frames.get(fr["frame_idx"])
            if frame is None:
                print(f"  SKIP flyover #{fly.idx:02d} frame {fr['frame_idx']}: decode failed")
                continue
            overlay = _render_overlay(
                frame,
                full_track_px=track_px,
                current_px=(fr["pixel_x"], fr["pixel_y"]),
                scale=args.overlay_scale,
            )
            overlay_name = f"fly{fly.idx:02d}_f{fr['frame_idx']:06d}.png"
            cv2.imwrite(str(overlays_dir / overlay_name), overlay)
            if fly.idx not in first_overlay_per_fly:
                first_overlay_per_fly[fly.idx] = overlay
            fly_frames_manifest.append({
                "frame_idx": fr["frame_idx"],
                "wall_time_utc": fr["wall_time_utc"],
                "pixel_x": fr["pixel_x"],
                "pixel_y": fr["pixel_y"],
                "overlay_png": f"overlays/{overlay_name}",
            })
            total_frames_written += 1
        manifest_flyovers.append({
            "idx": fly.idx,
            "transponder_id": fly.transponder_id,
            "callsign": fly.callsign,
            "duration_s": fly.duration_s,
            "pixel_span_px": fly.pixel_span,
            "center_px": [fly.center_px[0], fly.center_px[1]],
            "frames": fly_frames_manifest,
        })

    overview = _compose_overview(flyovers, first_overlay_per_fly)
    overview_path = validation_dir / "overview.png"
    cv2.imwrite(str(overview_path), overview)

    manifest = {
        "schema_version": 1,
        "date": date.isoformat(),
        "video": str(video_path),
        "anchor_utc": anchor_utc.isoformat(),
        "daylight_utc": args.daylight_utc,
        "image_size": [image_size[0], image_size[1]],
        "overlay_scale": args.overlay_scale,
        "total_frames": total_frames_written,
        "flyovers": manifest_flyovers,
    }
    manifest_path = validation_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    html_path = validation_dir / "labeller.html"
    _write_labeller_html(html_path, manifest_path.name, len(manifest_flyovers), total_frames_written)

    print()
    print(f"  Overview : {overview_path}")
    print(f"  Manifest : {manifest_path}")
    print(f"  Labeller : {html_path}")
    print(f"  Overlays : {overlays_dir} ({total_frames_written} PNGs)")
    print()
    print(f"Next: open {html_path} in a browser, click the visible aircraft on each frame,")
    print(f"      download labels.json, then")
    print(f"  uv run python scripts/projection_alignment_analyze.py --date {args.date} \\")
    print(f"      --labels {validation_dir}/labels.json")


if __name__ == "__main__":
    main()
