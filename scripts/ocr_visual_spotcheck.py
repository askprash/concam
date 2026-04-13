"""OCR visual spot-check (PRD item 3).

Samples N frames evenly from a daily timelapse, runs the
``FixedFormatTimestampReader`` on each, and produces:

  * ``ocr_spotcheck_grid.png`` -- a single image grid with one tile per
    sample showing the cropped timestamp ROI, the parsed text, the
    confidence, and the OCR method.  Tiles are sorted ascending by
    confidence so the weakest reads are at the top-left.
  * a stdout summary table: total frames, exact-parse rate, fallback
    rate, mean/median confidence, and the 5 lowest-confidence parsed
    results.

Default sample is 30 frames, default video is the 2026-04-08 timelapse
on the cluster, default config is ``configs/mit_green_building.yaml``.

Usage::

    uv run python scripts/ocr_visual_spotcheck.py
    uv run python scripts/ocr_visual_spotcheck.py \\
        --video /net/d16/data/contrail-camera/2026_04_09_0000_2359.mp4 \\
        --num-frames 50 \\
        --output-dir output/validation/ocr
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import av
import cv2
import numpy as np

from concam.config import load_config
from concam.ocr import FixedFormatTimestampReader, TimestampRead


def _crop_ocr_roi(frame: np.ndarray, region, position) -> np.ndarray:
    h, w = frame.shape[:2]
    rh, rw = region
    if position == "top_right":
        return frame[0:rh, w - rw : w]
    if position == "top_left":
        return frame[0:rh, 0:rw]
    if position == "bottom_right":
        return frame[h - rh : h, w - rw : w]
    if position == "bottom_left":
        return frame[h - rh : h, 0:rw]
    raise ValueError(f"unsupported position: {position}")


def _video_duration_and_frame_count(video_path: Path) -> tuple[float, int]:
    """Return (duration_seconds, decoded_frame_count) for a video file.

    PyAV's ``stream.frames`` is unreliable for some containers (returns 0),
    so we fall back to duration * average_rate when the metadata is missing.
    """
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if stream.frames:
            return duration_s, int(stream.frames)
        # Fallback: estimate from duration + average frame rate.
        rate = float(stream.average_rate) if stream.average_rate else 30.0
        return duration_s, int(round(duration_s * rate))
    finally:
        container.close()


def _sample_frames(
    video_path: Path,
    sample_indices: list[int],
    total_frames: int,
    duration_s: float,
) -> list[tuple[int, np.ndarray]]:
    """Decode the requested frame indices using PyAV time-based seeks.

    Whole-file decode on a 24h timelapse takes >10 minutes; for 30 samples
    we instead seek to the target time (stream.time_base units), then
    decode forward until we land on a frame at or past the desired
    container-relative position.  For each seek we accept the *first*
    keyframe-aligned frame we get -- the spot check tolerates being off
    by a few seconds, which is below the resolution of the 1-fps overlay
    text we're validating anyway.
    """
    out: list[tuple[int, np.ndarray]] = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for target_idx in sorted(set(sample_indices)):
            target_time_s = (target_idx / total_frames) * duration_s if total_frames else 0.0
            target_pts = int(target_time_s / float(time_base))
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            decoded = None
            for frame in container.decode(stream):
                decoded = frame
                # First decoded frame after a backward seek is usually the
                # nearest keyframe BEFORE the target; advance until pts >= target.
                if frame.pts is not None and frame.pts >= target_pts:
                    break
            if decoded is None:
                continue
            out.append((target_idx, decoded.to_ndarray(format="bgr24")))
    finally:
        container.close()
    return out


def _annotate_tile(
    crop_bgr: np.ndarray,
    parsed_text: str,
    confidence: float,
    method: str,
    status: str,
    frame_idx: int,
) -> np.ndarray:
    """Compose a tile: the crop on top, three lines of text below."""
    # Crop is ~875x80; downscale slightly so the grid fits a reasonable size.
    target_w = 700
    h, w = crop_bgr.shape[:2]
    scale = target_w / w
    crop_resized = cv2.resize(crop_bgr, (target_w, int(round(h * scale))))

    text_h = 110
    crop_h = crop_resized.shape[0]
    tile = np.full((crop_h + text_h, target_w, 3), 32, dtype=np.uint8)
    tile[:crop_h] = crop_resized

    # Color-code by status: green ok, yellow low_confidence, red parse_failed.
    if status == "ok":
        status_color = (140, 220, 140)
    elif status == "low_confidence":
        status_color = (140, 220, 240)
    else:
        status_color = (140, 140, 240)

    line1 = f"frame {frame_idx}  conf {confidence:.3f}  {method}"
    line2 = f"text: {parsed_text}"
    line3 = f"status: {status}"
    cv2.putText(
        tile, line1, (10, crop_h + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        tile, line2, (10, crop_h + 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        tile, line3, (10, crop_h + 92),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA,
    )
    return tile


def _compose_grid(tiles: list[np.ndarray], cols: int = 6) -> np.ndarray:
    """Pack tiles into a grid image."""
    if not tiles:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    pad = 8
    grid_h = rows * th + (rows + 1) * pad
    grid_w = cols * tw + (cols + 1) * pad
    grid = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        grid[y : y + th, x : x + tw] = tile
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video",
        default="/net/d16/data/contrail-camera/2026_04_08_0000_2359.mp4",
        help="Path to the daily timelapse mp4 to sample from.",
    )
    ap.add_argument(
        "--config",
        default="configs/mit_green_building.yaml",
        help="Site config (provides OCR ROI position/size and threshold).",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of frames to sample uniformly across the video.",
    )
    ap.add_argument(
        "--output-dir",
        default="output/validation/ocr",
        help="Directory in which to write the grid PNG (created if absent).",
    )
    ap.add_argument("--cols", type=int, default=5, help="Grid column count.")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    site = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration_s, total_frames = _video_duration_and_frame_count(video_path)
    if total_frames < args.num_frames:
        raise SystemExit(
            f"video reports only {total_frames} frames (< requested {args.num_frames})"
        )
    print(
        f"Video {video_path.name}: ~{duration_s:.0f}s, "
        f"~{total_frames} frames, sampling {args.num_frames} evenly."
    )

    # Uniform indices.  Skip 0 so the first sample isn't the (often unusual)
    # leading keyframe; pick at half-bucket offsets across the file.
    n = args.num_frames
    sample_indices = [
        int(round((i + 0.5) * total_frames / n)) for i in range(n)
    ]
    print(f"Decoding video to extract {len(sample_indices)} frames "
          "(seek-based, should take ~1 frame per second on average)...")
    samples = _sample_frames(video_path, sample_indices, total_frames, duration_s)
    print(f"Got {len(samples)} frames back.")

    reader = FixedFormatTimestampReader(site.ocr)

    rows: list[dict] = []
    for frame_idx, frame in samples:
        crop = _crop_ocr_roi(frame, site.ocr.timestamp_region, site.ocr.timestamp_position)
        result: TimestampRead = reader.read(frame)
        rows.append({
            "frame_idx": frame_idx,
            "crop": crop,
            "parsed_dt": result.parsed_dt,
            "text": result.text,
            "confidence": result.confidence,
            "per_char_confidence": result.per_char_confidence,
            "method": result.method,
            "status": result.status,
        })

    rows.sort(key=lambda r: r["confidence"])

    tiles = [
        _annotate_tile(
            r["crop"],
            r["text"],
            r["confidence"],
            r["method"],
            r["status"],
            r["frame_idx"],
        )
        for r in rows
    ]
    grid = _compose_grid(tiles, cols=args.cols)
    grid_path = out_dir / "ocr_spotcheck_grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"Wrote {grid_path} ({grid.shape[1]}x{grid.shape[0]} px)")

    # --- summary table ---
    n = len(rows)
    parsed = sum(1 for r in rows if r["parsed_dt"] is not None)
    template = sum(1 for r in rows if r["method"] == "template")
    fallback = sum(1 for r in rows if r["method"] == "easyocr_fallback")
    confs = [r["confidence"] for r in rows]
    mean_conf = statistics.fmean(confs) if confs else 0.0
    median_conf = statistics.median(confs) if confs else 0.0

    print()
    print("=" * 64)
    print(f"OCR spot-check summary: {video_path.name}")
    print("=" * 64)
    print(f"frames sampled:          {n}")
    print(f"parsed (any method):     {parsed}/{n}  ({parsed/n:.1%})")
    print(f"template path success:   {template}/{n}  ({template/n:.1%})")
    print(f"easyocr fallback used:   {fallback}/{n}  ({fallback/n:.1%})")
    print(f"mean   confidence:       {mean_conf:.3f}")
    print(f"median confidence:       {median_conf:.3f}")
    print()
    print("5 lowest-confidence reads:")
    for r in rows[:5]:
        dt_str = r["parsed_dt"].isoformat() if r["parsed_dt"] else "<unparsed>"
        print(
            f"  frame {r['frame_idx']:>6d}  conf {r['confidence']:.3f}  "
            f"method={r['method']:<18s}  status={r['status']:<14s}  "
            f"text={r['text']!r}  parsed={dt_str}"
        )
    print()
    print("Review the grid PNG at:", grid_path)


if __name__ == "__main__":
    main()
