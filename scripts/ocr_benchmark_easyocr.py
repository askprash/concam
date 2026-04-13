"""Benchmark template OCR vs EasyOCR on the same set of timelapse frames.

Samples a small number of frames from a video, runs both the
``FixedFormatTimestampReader`` (template path) and a fresh
``easyocr.Reader`` on each one, and reports per-frame and aggregate
timing alongside the parsed text and confidence.

Use this to justify (or reconsider) the template-first design: on the
MIT Green Building overlay the template path should be 1-2 orders of
magnitude faster than EasyOCR while producing essentially identical
parsed timestamps.

Usage::

    uv run python scripts/ocr_benchmark_easyocr.py \\
        --video /net/d16/data/contrail-camera/2026_04_08_0000_2359.mp4 \\
        --num-frames 5

The first EasyOCR call carries the model-load latency (a few seconds);
we report it separately from the per-frame inference time so the
comparison stays meaningful.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import av
import cv2
import numpy as np

from concam.config import load_config
from concam.ocr import FixedFormatTimestampReader
from concam.ocr._fallback_clean import clean_easyocr_output
from concam.ocr.parser import parse_canonical_timestamp


def _video_meta(video_path: Path) -> tuple[float, int]:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if stream.frames:
            return duration_s, int(stream.frames)
        rate = float(stream.average_rate) if stream.average_rate else 30.0
        return duration_s, int(round(duration_s * rate))
    finally:
        container.close()


def _sample(video_path: Path, indices, total_frames, duration_s):
    """Same seek-based sampler as the spot-check script (kept local to avoid a
    cross-script import for a 20-line helper)."""
    out = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for target in sorted(set(indices)):
            t_s = (target / total_frames) * duration_s if total_frames else 0.0
            target_pts = int(t_s / float(time_base))
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            decoded = None
            for frame in container.decode(stream):
                decoded = frame
                if frame.pts is not None and frame.pts >= target_pts:
                    break
            if decoded is not None:
                out.append((target, decoded.to_ndarray(format="bgr24")))
    finally:
        container.close()
    return out


def _crop_roi(frame, region, position):
    h, w = frame.shape[:2]
    rh, rw = region
    if position == "top_right":
        return frame[0:rh, w - rw : w]
    raise ValueError(position)


def _easyocr_read(reader, roi):
    """Run EasyOCR on a timestamp ROI and return (text_or_None, conf, raw)."""
    if roi.ndim == 2:
        rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    else:
        rgb = roi
    mask = cv2.inRange(rgb, (230, 230, 230), (255, 255, 255))
    masked = cv2.bitwise_and(rgb, cv2.merge([mask, mask, mask]))
    result = reader.readtext(
        masked,
        paragraph=False,
        allowlist="0123456789:/-AMPONTUEWDHFRISamp ",
        rotation_info=[0],
    )
    if not result:
        return None, 0.0, result
    stitched = clean_easyocr_output(result).strip()
    if stitched.endswith(" AM") or stitched.endswith(" PM"):
        stitched = stitched[:-3].rstrip()
    confs = [float(item[2]) for item in result if len(item) >= 3]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    try:
        parse_canonical_timestamp(stitched)
        parsed_ok = True
    except ValueError:
        parsed_ok = False
    return stitched if parsed_ok else None, avg_conf, result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video",
        default="/net/d16/data/contrail-camera/2026_04_08_0000_2359.mp4",
    )
    ap.add_argument("--config", default="configs/mit_green_building.yaml")
    ap.add_argument("--num-frames", type=int, default=5)
    ap.add_argument("--start-frac", type=float, default=0.25)
    ap.add_argument("--end-frac", type=float, default=0.83)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    site = load_config(args.config)
    duration_s, total_frames = _video_meta(video_path)
    span = total_frames * (args.end_frac - args.start_frac)
    base = total_frames * args.start_frac
    indices = [
        int(round(base + (i + 0.5) * span / args.num_frames))
        for i in range(args.num_frames)
    ]
    print(f"Video {video_path.name}: ~{total_frames} frames, sampling {len(indices)}.")
    samples = _sample(video_path, indices, total_frames, duration_s)
    print(f"Decoded {len(samples)} frames.\n")

    # Template reader: warm once.
    print("Loading template reader...")
    t0 = time.perf_counter()
    template = FixedFormatTimestampReader(site.ocr)
    template_load_s = time.perf_counter() - t0
    print(f"  template reader ready in {template_load_s*1000:.1f} ms\n")

    print("Loading EasyOCR (CPU; model download may take a moment first time)...")
    import easyocr  # local import so the script is usable without easyocr installed
    t0 = time.perf_counter()
    eocr = easyocr.Reader(["en"], gpu=False, verbose=False)
    eocr_load_s = time.perf_counter() - t0
    print(f"  easyocr ready in {eocr_load_s:.2f} s\n")

    # Warm the easyocr graph with one no-op call so the first measured frame
    # isn't unfairly slow due to lazy initialisation.
    if samples:
        warm_roi = _crop_roi(samples[0][1], site.ocr.timestamp_region, site.ocr.timestamp_position)
        t0 = time.perf_counter()
        _easyocr_read(eocr, warm_roi)
        eocr_warm_s = time.perf_counter() - t0
        print(f"  easyocr first-call (warmup) {eocr_warm_s*1000:.1f} ms\n")

    rows = []
    for frame_idx, frame in samples:
        roi = _crop_roi(frame, site.ocr.timestamp_region, site.ocr.timestamp_position)

        t0 = time.perf_counter()
        tres = template.read(frame)
        t_template_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        e_text, e_conf, _ = _easyocr_read(eocr, roi)
        t_easy_s = time.perf_counter() - t0

        rows.append({
            "frame": frame_idx,
            "template_text": tres.text,
            "template_conf": tres.confidence,
            "template_ms": t_template_s * 1000,
            "easyocr_text": e_text,
            "easyocr_conf": e_conf,
            "easyocr_ms": t_easy_s * 1000,
            "match": (tres.text == e_text),
        })

    # Per-frame table.
    print("=" * 100)
    print(f"{'frame':>7} {'template_ms':>12} {'easy_ms':>10} {'tpl_conf':>9} "
          f"{'easy_conf':>10} {'match':>6}  template_text             easyocr_text")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['frame']:>7d} {r['template_ms']:>12.2f} {r['easyocr_ms']:>10.1f} "
            f"{r['template_conf']:>9.3f} {r['easyocr_conf']:>10.3f} "
            f"{('Y' if r['match'] else 'N'):>6}  "
            f"{r['template_text']:<24s}  {r['easyocr_text'] or '<unparsed>'}"
        )
    print("=" * 100)

    # Aggregate.
    tpl_times = [r["template_ms"] for r in rows]
    eocr_times = [r["easyocr_ms"] for r in rows]
    matches = sum(1 for r in rows if r["match"])
    print()
    print(f"frames compared:           {len(rows)}")
    print(f"text-string match:         {matches}/{len(rows)}")
    print(f"template median latency:   {statistics.median(tpl_times):.2f} ms "
          f"(mean {statistics.fmean(tpl_times):.2f} ms)")
    print(f"easyocr  median latency:   {statistics.median(eocr_times):.1f} ms "
          f"(mean {statistics.fmean(eocr_times):.1f} ms)")
    if tpl_times and eocr_times:
        speedup = statistics.median(eocr_times) / statistics.median(tpl_times)
        print(f"template speedup (median): {speedup:.0f}x")
    print()
    print("(EasyOCR model load:       "
          f"{eocr_load_s:.2f} s, one-time per process)")


if __name__ == "__main__":
    main()
