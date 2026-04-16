"""Render short annotated MP4 clips for selected contrail episodes.

For each episode, produce a ±5 s clip (max 60 s total) around the episode
onset with:
  - ADS-B ping dot for every flight visible in the frame (white/grey dots,
    larger amber dot for the episode's own flight)
  - Rotated ROI box drawn around the episode flight's detection window:
      green  = score ≥ threshold (contrail detected)
      amber  = score below threshold (flight present, no detection)
      (The detection line is NOT drawn on top of the contrail image so the
      human reviewer can see the actual sky texture.)
  - Callsign + score + UTC wall-time text burned into the top-left corner

Output: 720p H.264 MP4 files in <output_dir>/<date>/clips/.

Usage::

    uv run python scripts/render_episode_clips.py --date 2026-04-08 --top-n 10

    uv run python scripts/render_episode_clips.py --date 2026-04-08 \\
        --episodes EIN108@2026-04-08T04:27:09+00:00,THY12@2026-04-08T05:26:51+00:00

    # Custom video and output dir
    uv run python scripts/render_episode_clips.py --date 2026-04-08 \\
        --video /net/d16/data/contrail-camera/2026_04_08_0000_2359.mp4 \\
        --output-dir output --top-n 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import load_config
from concam.pipeline import resolve_video_path, stage_paths
from concam.projection import PixelPoint, rotated_polygon

logger = logging.getLogger("render_episode_clips")

DEFAULT_CONFIG = REPO_ROOT / "configs" / "mit_green_building.yaml"

# Clip parameters
PRE_ROLL_S = 5   # seconds before episode onset
POST_ROLL_S = 5  # seconds after episode end
MAX_CLIP_S = 60  # hard cap on clip duration

# Output resolution
OUT_WIDTH = 1920
OUT_HEIGHT = 1080

# Drawing constants
DOT_RADIUS_OTHER = 6    # ADS-B dot for background flights (downscaled coords)
DOT_RADIUS_SELF = 12    # ADS-B dot for the episode flight
DOT_RADIUS_OTHER_4K = DOT_RADIUS_OTHER * 4
DOT_RADIUS_SELF_4K = DOT_RADIUS_SELF * 4

COLOR_DOT_OTHER = (200, 200, 200)  # BGR grey for background flights
COLOR_DOT_SELF = (0, 165, 255)     # BGR amber for the episode flight
COLOR_BOX_DETECTED = (0, 220, 0)   # BGR green: score ≥ threshold
COLOR_BOX_UNDETECTED = (80, 80, 80)  # BGR dim grey: flight present, no detection
COLOR_TEXT = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_4K = 2.4
FONT_THICKNESS_4K = 5


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_ocr_index(ocr_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Return (wall_time_utc → frame_idx, frame_idx → wall_time_utc) maps.

    wall_time_utc strings are truncated to the second (microseconds stripped).
    """
    wt_to_idx: dict[str, int] = {}
    idx_to_wt: dict[int, str] = {}
    for rec in _iter_jsonl(ocr_path):
        # Normalise to second-resolution ISO string.
        dt = datetime.datetime.fromisoformat(rec["wall_time_utc"]).replace(microsecond=0)
        wt = dt.isoformat()
        fi = int(rec["frame_idx"])
        wt_to_idx[wt] = fi
        idx_to_wt[fi] = wt
    return wt_to_idx, idx_to_wt


def _load_projections(proj_path: Path) -> dict[tuple[str, str], dict]:
    """Index projections by (wall_time_utc_second, transponder_id)."""
    index: dict[tuple[str, str], dict] = {}
    for rec in _iter_jsonl(proj_path):
        wt = datetime.datetime.fromisoformat(rec["wall_time_utc"]).replace(
            microsecond=0
        ).isoformat()
        key = (wt, rec["transponder_id"])
        index[key] = rec
    return index


def _load_projections_by_time(proj_path: Path) -> dict[str, list[dict]]:
    """Index all projections by wall_time_utc_second.

    Returns dict mapping wall_time_utc (second-rounded) → list of projection
    records for all flights at that timestamp.
    """
    index: dict[str, list[dict]] = {}
    for rec in _iter_jsonl(proj_path):
        wt = datetime.datetime.fromisoformat(rec["wall_time_utc"]).replace(
            microsecond=0
        ).isoformat()
        index.setdefault(wt, []).append(rec)
    return index


def _load_detections(det_path: Path) -> dict[tuple[str, str], dict]:
    """Index detections by (wall_time_utc_second, transponder_id)."""
    index: dict[tuple[str, str], dict] = {}
    for rec in _iter_jsonl(det_path):
        wt = datetime.datetime.fromisoformat(rec["wall_time_utc"]).replace(
            microsecond=0
        ).isoformat()
        key = (wt, rec["transponder_id"])
        index[key] = rec
    return index


def _load_episodes(ep_path: Path) -> list[dict]:
    return list(_iter_jsonl(ep_path))


# ---------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------

def _pick_episodes(
    episodes: list[dict],
    top_n: int,
    threshold: float,
) -> list[dict]:
    above = [e for e in episodes if e["peak_score"] >= threshold]
    sorted_ep = sorted(above, key=lambda e: -e["peak_score"])
    return sorted_ep[:top_n]


def _find_episodes_by_spec(episodes: list[dict], specs: list[str]) -> list[dict]:
    """Find episodes matching 'CALLSIGN@ONSET_ISO' specs."""
    result = []
    for spec in specs:
        if "@" in spec:
            callsign, onset_str = spec.split("@", 1)
            onset_dt = datetime.datetime.fromisoformat(onset_str)
            for ep in episodes:
                ep_onset = datetime.datetime.fromisoformat(ep["onset"])
                if ep["callsign"] == callsign and abs((ep_onset - onset_dt).total_seconds()) < 2:
                    result.append(ep)
                    break
            else:
                logger.warning("No episode found for spec %r", spec)
        else:
            # Treat as episode_id integer (episode position in list, 1-indexed)
            try:
                idx = int(spec) - 1
                result.append(episodes[idx])
            except (ValueError, IndexError):
                logger.warning("Cannot resolve episode spec %r", spec)
    return result


# ---------------------------------------------------------------------------
# Video decoding
# ---------------------------------------------------------------------------

def _open_container(video_path: Path):
    return av.open(str(video_path))


def _frame_idx_to_video_pts(
    frame_idx: int, total_frames: int, duration_s: float, time_base: float
) -> int:
    target_s = (frame_idx / total_frames) * duration_s
    return int(target_s / time_base)


def _iter_clip_frames(
    video_path: Path,
    start_frame_idx: int,
    end_frame_idx: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, BGR ndarray) for each frame in [start, end] inclusive.

    Seeks to start_frame_idx, then decodes forward until end_frame_idx.
    frame_idx is the OCR-space index (real-second counter, 0-based).
    """
    container = _open_container(video_path)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        total_frames = int(stream.frames) if stream.frames else 86402
        duration_s = float(stream.duration * stream.time_base) if stream.duration else 86402.0

        target_pts = _frame_idx_to_video_pts(start_frame_idx, total_frames, duration_s, time_base)
        container.seek(target_pts, stream=stream, any_frame=False, backward=True)

        current_idx = start_frame_idx - 1  # will be incremented on first frame
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            # Map pts back to frame_idx.
            frame_s = frame.pts * time_base
            current_idx = int(round(frame_s / duration_s * total_frames))
            if current_idx < start_frame_idx:
                continue
            if current_idx > end_frame_idx:
                break
            yield current_idx, frame.to_ndarray(format="bgr24")
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def _downscale_coord(x: float, y: float, src_w: int, src_h: int) -> tuple[int, int]:
    sx = x / src_w * OUT_WIDTH
    sy = y / src_h * OUT_HEIGHT
    return int(round(sx)), int(round(sy))


def _annotate_frame(
    frame: np.ndarray,
    wall_time: str,
    episode: dict,
    det_rec: dict | None,
    all_projs: list[dict],
    self_proj: dict | None,
    threshold: float,
    det_config,
) -> np.ndarray:
    """Annotate a single frame and downscale to OUT_WIDTH × OUT_HEIGHT."""
    h, w = frame.shape[:2]

    # --- Draw on full-resolution frame first ---
    annotated = frame.copy()

    # 1. ADS-B dots for all flights at this timestamp (background, grey).
    for proj in all_projs:
        px, py = int(round(proj["pixel_x"])), int(round(proj["pixel_y"]))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(annotated, (px, py), DOT_RADIUS_OTHER_4K, COLOR_DOT_OTHER, -1)

    # 2. ADS-B dot for this episode's flight (amber, larger).
    if self_proj is not None:
        px, py = int(round(self_proj["pixel_x"])), int(round(self_proj["pixel_y"]))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(annotated, (px, py), DOT_RADIUS_SELF_4K, COLOR_DOT_SELF, -1)

    # 3. Rotated ROI box (coloured by detection status — NOT a line on the contrail).
    if self_proj is not None and "path_dx" in self_proj:
        score = det_rec["score"] if det_rec is not None else 0.0
        box_color = COLOR_BOX_DETECTED if score >= threshold else COLOR_BOX_UNDETECTED
        poly = rotated_polygon(
            PixelPoint(x=self_proj["pixel_x"], y=self_proj["pixel_y"]),
            (float(self_proj["path_dx"]), float(self_proj["path_dy"])),
            det_config,
        ).astype(np.int32)
        cv2.polylines(annotated, [poly.reshape(-1, 1, 2)], isClosed=True, color=box_color, thickness=6)

    # 4. Text overlay: callsign, score, wall time (top-left, with shadow).
    score_str = f"{det_rec['score']:.3f}" if det_rec is not None else "—"
    onset_dt = datetime.datetime.fromisoformat(episode["onset"])
    end_dt = datetime.datetime.fromisoformat(episode["end"])
    try:
        cur_dt = datetime.datetime.fromisoformat(wall_time)
        in_episode = onset_dt <= cur_dt <= end_dt
        ep_marker = " [EPISODE]" if in_episode else ""
    except Exception:
        ep_marker = ""

    lines = [
        f"{episode['callsign']}  score={score_str}{ep_marker}",
        f"onset {onset_dt.strftime('%H:%M:%S')} UTC  dur={episode['frame_count']}s",
        f"t={wall_time}",
    ]
    y_pos = 100
    for line in lines:
        cv2.putText(annotated, line, (60, y_pos), FONT, FONT_SCALE_4K,
                    (0, 0, 0), FONT_THICKNESS_4K + 4, cv2.LINE_AA)
        cv2.putText(annotated, line, (60, y_pos), FONT, FONT_SCALE_4K,
                    COLOR_TEXT, FONT_THICKNESS_4K, cv2.LINE_AA)
        y_pos += 90

    # 5. Downscale to 720p (1920×1080).
    out_frame = cv2.resize(annotated, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_AREA)
    return out_frame


# ---------------------------------------------------------------------------
# MP4 encoder
# ---------------------------------------------------------------------------

def _encode_clip(frames_bgr: list[np.ndarray], out_path: Path, fps: int = 30) -> None:
    """Write a list of BGR frames to a 720p H.264 MP4 using PyAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = OUT_WIDTH
        stream.height = OUT_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "22", "preset": "fast"}

        for bgr in frames_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            av_frame = av_frame.reformat(format="yuv420p")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Per-episode rendering
# ---------------------------------------------------------------------------

def _render_episode(
    episode: dict,
    video_path: Path,
    wt_to_idx: dict[str, int],
    idx_to_wt: dict[int, str],
    projs_by_time: dict[str, list[dict]],
    dets_by_key: dict[tuple[str, str], dict],
    out_path: Path,
    threshold: float,
    det_config,
) -> bool:
    """Render and write a clip for a single episode. Returns True on success."""
    onset_dt = datetime.datetime.fromisoformat(episode["onset"]).replace(microsecond=0)
    end_dt = datetime.datetime.fromisoformat(episode["end"]).replace(microsecond=0)
    tid = episode["transponder_id"]

    # Clip window in wall-time.
    clip_start_dt = onset_dt - datetime.timedelta(seconds=PRE_ROLL_S)
    clip_end_dt = min(
        end_dt + datetime.timedelta(seconds=POST_ROLL_S),
        onset_dt + datetime.timedelta(seconds=MAX_CLIP_S - PRE_ROLL_S),
    )

    # Map to frame indices.
    total_frames = max(wt_to_idx.values()) + 1 if wt_to_idx else 86402
    # Find start frame: walk backward from onset until we hit a recorded index.
    start_fi = None
    for delta in range(PRE_ROLL_S + 5):
        t = (onset_dt - datetime.timedelta(seconds=delta)).isoformat()
        if t in wt_to_idx:
            # Start PRE_ROLL_S before onset if possible.
            start_fi = max(0, wt_to_idx[t] - (PRE_ROLL_S - delta))
            break

    end_fi = None
    for delta in range(POST_ROLL_S + 5):
        t = (end_dt + datetime.timedelta(seconds=delta)).isoformat()
        if t in wt_to_idx:
            end_fi = wt_to_idx[t] + (POST_ROLL_S - delta)
            break

    if start_fi is None or end_fi is None:
        logger.warning(
            "Cannot map episode %s onset/end to frame indices; skipping.",
            episode["callsign"],
        )
        return False

    end_fi = min(end_fi, start_fi + MAX_CLIP_S - 1, total_frames - 1)
    if end_fi <= start_fi:
        logger.warning("Empty clip window for %s; skipping.", episode["callsign"])
        return False

    n_frames = end_fi - start_fi + 1
    logger.info(
        "Rendering %s  frames %d–%d  (%d s)  → %s",
        episode["callsign"], start_fi, end_fi, n_frames, out_path.name,
    )

    frames_out: list[np.ndarray] = []
    for fi, bgr in _iter_clip_frames(video_path, start_fi, end_fi):
        wt = idx_to_wt.get(fi)
        if wt is None:
            # Fall back: approximate from frame index assuming 1-fps.
            wt = None

        all_projs = projs_by_time.get(wt, []) if wt else []
        self_proj = next((p for p in all_projs if p["transponder_id"] == tid), None)
        det_rec = dets_by_key.get((wt, tid)) if wt else None

        annotated = _annotate_frame(
            bgr,
            wt or "unknown",
            episode,
            det_rec,
            all_projs,
            self_proj,
            threshold,
            det_config,
        )
        frames_out.append(annotated)

    if not frames_out:
        logger.warning("No frames decoded for %s; skipping.", episode["callsign"])
        return False

    _encode_clip(frames_out, out_path)
    logger.info("  wrote %d frames → %s", len(frames_out), out_path)
    return True


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render annotated MP4 clips for contrail episodes."
    )
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    parser.add_argument(
        "--top-n", type=int, default=10, metavar="N",
        help="Render the top N episodes by peak_score (default: 10)."
    )
    parser.add_argument(
        "--episodes", default=None, metavar="SPECS",
        help=(
            "Comma-separated episode specs: CALLSIGN@ONSET_ISO or 1-based index. "
            "Overrides --top-n when specified."
        ),
    )
    parser.add_argument(
        "--output-dir", default="output", help="Root output dir (default: output)"
    )
    parser.add_argument(
        "--video", default=None, help="Override video path (default: auto-resolved)"
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Site config YAML"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Output MP4 frame rate (default: 30)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    date = datetime.date.fromisoformat(args.date)
    site_config = load_config(Path(args.config))
    paths = stage_paths(Path(args.output_dir), date)

    # Verify required stage outputs exist.
    for key in ("ocr", "projections", "detections", "episodes"):
        if not paths[key].exists():
            logger.error("Missing %s stage output: %s", key, paths[key])
            sys.exit(1)

    # Resolve video path.
    if args.video:
        video_path = Path(args.video)
    else:
        video_path = resolve_video_path(site_config.video, date)
    logger.info("Video: %s", video_path)

    # Load data.
    logger.info("Loading OCR index …")
    wt_to_idx, idx_to_wt = _load_ocr_index(paths["ocr"])
    logger.info("Loading projections …")
    projs_by_time = _load_projections_by_time(paths["projections"])
    logger.info("Loading detections …")
    dets_by_key = _load_detections(paths["detections"])
    logger.info("Loading episodes …")
    episodes = _load_episodes(paths["episodes"])
    logger.info("%d episodes loaded", len(episodes))

    threshold = site_config.aggregation.detection_threshold

    # Select episodes.
    if args.episodes:
        specs = [s.strip() for s in args.episodes.split(",") if s.strip()]
        selected = _find_episodes_by_spec(episodes, specs)
    else:
        selected = _pick_episodes(episodes, args.top_n, threshold)

    logger.info("Rendering %d episode clip(s) …", len(selected))

    clips_dir = paths["base"] / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    det_config = site_config.detection
    n_ok = 0
    for i, ep in enumerate(selected, 1):
        onset_dt = datetime.datetime.fromisoformat(ep["onset"])
        slug = f"{i:02d}_{_safe_name(ep['callsign'])}_{onset_dt.strftime('%H%M%SZ')}"
        out_path = clips_dir / f"{slug}.mp4"
        ok = _render_episode(
            episode=ep,
            video_path=video_path,
            wt_to_idx=wt_to_idx,
            idx_to_wt=idx_to_wt,
            projs_by_time=projs_by_time,
            dets_by_key=dets_by_key,
            out_path=out_path,
            threshold=threshold,
            det_config=det_config,
        )
        if ok:
            n_ok += 1

    print(f"\nDone. {n_ok}/{len(selected)} clips written to {clips_dir}")
    return 0 if n_ok == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
