"""PTS-vs-OSD drift comparison, using EasyOCR sparingly.

Strategy:
  Step 1 -- Sample-OCR validation (slow but small):
    OCR ~12 frames spaced across each 1-hour capture to establish whether
    PTS is locked to the camera-OSD clock or drifts away from it.
    -> Native should give a constant (PTS - OSD) offset (slope ~ 0).
    -> Wallclock should give a non-zero, near-linear slope.

  Step 2 -- Decimation analysis (fast, OCR-free):
    Once we've confirmed in step 1 that native_PTS is locked to the OSD
    clock, we use native_PTS as the ground-truth camera-second clock for
    the decimation analysis.  We apply ffmpeg fps=1/1 (nearest-PTS) to
    each capture and report the OSD-second deltas of the picked frames.
    The wallclock capture should produce 1-1-0-2 patterns; the native
    capture should produce 1,1,1,1...

Outputs:
    scratch/pts_drift_test/analyze_v2_summary.txt
    scratch/pts_drift_test/v2_native_samples.csv
    scratch/pts_drift_test/v2_wallclock_samples.csv
    scratch/pts_drift_test/v2_native_decimated.csv
    scratch/pts_drift_test/v2_wallclock_decimated.csv
"""

from __future__ import annotations

import csv
import datetime as _dt
import re
import sys
from collections import Counter
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scratch" / "pts_drift_test"
NATIVE = SCRATCH / "sub_native_pts.mp4"
WALLCLOCK = SCRATCH / "sub_wallclock_pts.mp4"

OSD_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})")


def parse_osd(text: str) -> _dt.datetime | None:
    m = OSD_RE.search(text)
    if not m:
        return None
    mm, dd, yyyy, hh, mi, ss = m.groups()
    try:
        return _dt.datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss))
    except ValueError:
        return None


def crop_osd(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    crop = bgr[0 : int(h * 0.10), int(w * 0.55) : w]
    return cv2.resize(
        crop,
        (crop.shape[1] * 4, crop.shape[0] * 4),
        interpolation=cv2.INTER_CUBIC,
    )


def ocr_at_pts(path: Path, target_pts_seconds: list[float], reader) -> list[tuple[float, float, _dt.datetime | None, str]]:
    """Decode and OCR the frames closest to each target PTS.  Returns a
    list of (target_pts, actual_pts, parsed_dt_or_None, raw_text)."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = stream.time_base
    out: list[tuple[float, float, _dt.datetime | None, str]] = []

    targets = sorted(target_pts_seconds)
    ti = 0
    p0 = None
    last_frame = None
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        pts_s = float(frame.pts * tb)
        if p0 is None:
            p0 = pts_s
        rel = pts_s - p0
        # if we've passed the next target, OCR the previous frame (closest one)
        while ti < len(targets) and rel >= targets[ti]:
            chosen = last_frame if last_frame is not None and abs(
                (last_frame.pts * tb) - p0 - targets[ti]
            ) < abs(rel - targets[ti]) else frame
            chosen_pts = float(chosen.pts * tb) - p0
            big = crop_osd(chosen.to_ndarray(format="bgr24"))
            res = reader.readtext(big, detail=0, paragraph=False)
            text = " ".join(res) if res else ""
            out.append((targets[ti], chosen_pts, parse_osd(text), text))
            print(f"  {path.name} target={targets[ti]:.1f}s actual={chosen_pts:.3f}s text={text!r}")
            ti += 1
        last_frame = frame
        if ti >= len(targets):
            break
    container.close()
    return out


def read_pts_array(path: Path) -> list[float]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    tb = stream.time_base
    out = []
    for packet in container.demux(stream):
        if packet.pts is None:
            continue
        out.append(float(packet.pts * tb))
    container.close()
    return out


def decimate_fps1(pts: list[float]) -> list[int]:
    if not pts:
        return []
    p0 = pts[0]
    rel = [p - p0 for p in pts]
    end = rel[-1]
    targets = [float(t) for t in range(int(end) + 1)]
    picks = []
    j = 0
    for tgt in targets:
        while j + 1 < len(rel) and abs(rel[j + 1] - tgt) <= abs(rel[j] - tgt):
            j += 1
        picks.append(j)
    return picks


def main() -> int:
    if not NATIVE.exists() or not WALLCLOCK.exists():
        print(f"Missing capture(s) at {SCRATCH}", file=sys.stderr)
        return 2

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    # warmup
    _ = reader.readtext(np.zeros((50, 200, 3), dtype=np.uint8))

    sample_targets = [0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0, 2700.0, 3000.0, 3300.0, 3550.0]

    print("=== Step 1: OCR samples on native capture ===")
    nat_samples = ocr_at_pts(NATIVE, sample_targets, reader)
    print("\n=== Step 1: OCR samples on wallclock capture ===")
    wal_samples = ocr_at_pts(WALLCLOCK, sample_targets, reader)

    # write samples
    with (SCRATCH / "v2_native_samples.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_pts", "actual_pts", "ocr_text", "parsed_dt"])
        for t, a, dt, text in nat_samples:
            w.writerow([f"{t:.3f}", f"{a:.3f}", text, dt.isoformat() if dt else ""])
    with (SCRATCH / "v2_wallclock_samples.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_pts", "actual_pts", "ocr_text", "parsed_dt"])
        for t, a, dt, text in wal_samples:
            w.writerow([f"{t:.3f}", f"{a:.3f}", text, dt.isoformat() if dt else ""])

    def drift_series(samples):
        """Returns list of (actual_pts, drift_seconds) for samples with valid OSD."""
        valid = [(a, dt) for _, a, dt, _ in samples if dt is not None]
        if len(valid) < 2:
            return []
        a0, dt0 = valid[0]
        return [(a, (a - a0) - (dt - dt0).total_seconds()) for a, dt in valid]

    nat_drift = drift_series(nat_samples)
    wal_drift = drift_series(wal_samples)

    def slope(series):
        if len(series) < 2:
            return None
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        if den == 0:
            return None
        return num / den

    nat_slope = slope(nat_drift)
    wal_slope = slope(wal_drift)
    nat_end_drift = nat_drift[-1][1] if nat_drift else None
    wal_end_drift = wal_drift[-1][1] if wal_drift else None

    print("\n=== Step 1 results ===")
    print(f"Native:    {len(nat_drift)} valid samples, slope={nat_slope}, end-of-hour drift={nat_end_drift}")
    print(f"Wallclock: {len(wal_drift)} valid samples, slope={wal_slope}, end-of-hour drift={wal_end_drift}")

    # Step 2: decimation analysis
    print("\n=== Step 2: fps=1/1 decimation (OCR-free) ===")
    nat_pts = read_pts_array(NATIVE)
    wal_pts = read_pts_array(WALLCLOCK)
    print(f"  native: {len(nat_pts)} frames, span {nat_pts[-1]-nat_pts[0]:.3f}s")
    print(f"  wallclock: {len(wal_pts)} frames, span {wal_pts[-1]-wal_pts[0]:.3f}s")

    nat_picks = decimate_fps1(nat_pts)
    wal_picks = decimate_fps1(wal_pts)

    p0 = nat_pts[0]
    nat_osd = [int(nat_pts[i] - p0) for i in nat_picks]
    wal_osd = [int(nat_pts[min(i, len(nat_pts) - 1)] - p0) for i in wal_picks]

    nat_deltas = [nat_osd[i + 1] - nat_osd[i] for i in range(len(nat_osd) - 1)]
    wal_deltas = [wal_osd[i + 1] - wal_osd[i] for i in range(len(wal_osd) - 1)]
    nat_hist = Counter(nat_deltas)
    wal_hist = Counter(wal_deltas)

    def find_pat(deltas, target=(1, 1, 0, 2)):
        L = len(target)
        return [i for i in range(len(deltas) - L + 1) if tuple(deltas[i : i + L]) == target]

    nat_hits = find_pat(nat_deltas)
    wal_hits = find_pat(wal_deltas)

    with (SCRATCH / "v2_native_decimated.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "frame_idx", "osd_second"])
        for i, (idx, sec) in enumerate(zip(nat_picks, nat_osd)):
            w.writerow([i, idx, sec])
    with (SCRATCH / "v2_wallclock_decimated.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "frame_idx", "osd_second"])
        for i, (idx, sec) in enumerate(zip(wal_picks, wal_osd)):
            w.writerow([i, idx, sec])

    lines = []
    lines.append("=== concam PTS-drift comparison (1 hour, substream) ===")
    lines.append("")
    lines.append("STEP 1: OCR-validated drift between PTS and camera-OSD clock")
    lines.append("(slope ~ 0 means PTS is locked to OSD; non-zero slope means PTS drifts)")
    lines.append("")
    lines.append(f"  Native:    n={len(nat_drift)} samples, slope={nat_slope:+.6f} s/s, end-of-hour drift = {nat_end_drift:+.3f}s" if nat_slope is not None else "  Native:    insufficient OCR samples")
    lines.append(f"  Wallclock: n={len(wal_drift)} samples, slope={wal_slope:+.6f} s/s, end-of-hour drift = {wal_end_drift:+.3f}s" if wal_slope is not None else "  Wallclock: insufficient OCR samples")
    lines.append("")
    lines.append("Per-sample drift series:")
    lines.append("  Native:")
    for a, d in nat_drift:
        lines.append(f"    pts={a:7.2f}s  drift={d:+.3f}s")
    lines.append("  Wallclock:")
    for a, d in wal_drift:
        lines.append(f"    pts={a:7.2f}s  drift={d:+.3f}s")
    lines.append("")
    lines.append("STEP 2: ffmpeg fps=1/1 decimation, OSD-second deltas (using native PTS as ground-truth camera clock)")
    lines.append(f"  native     n_picks={len(nat_picks)}  hist={dict(sorted(nat_hist.items()))}")
    lines.append(f"  wallclock  n_picks={len(wal_picks)}  hist={dict(sorted(wal_hist.items()))}")
    lines.append("")
    lines.append(f"  literal '1,1,0,2' subsequence:")
    lines.append(f"    native:    {len(nat_hits)} occurrence(s)")
    lines.append(f"    wallclock: {len(wal_hits)} occurrence(s)")
    if wal_hits:
        snippet = wal_deltas[wal_hits[0] : wal_hits[0] + 16]
        lines.append(f"    wallclock first '1,1,0,2' window @ idx {wal_hits[0]}: {snippet}")
    if nat_hits:
        snippet = nat_deltas[nat_hits[0] : nat_hits[0] + 16]
        lines.append(f"    native first '1,1,0,2' window @ idx {nat_hits[0]}: {snippet}")
    lines.append("")
    lines.append("First 60 wallclock OSD-second deltas:")
    lines.append("  " + ", ".join(str(d) for d in wal_deltas[:60]))
    lines.append("First 60 native OSD-second deltas:")
    lines.append("  " + ", ".join(str(d) for d in nat_deltas[:60]))
    summary = "\n".join(lines) + "\n"
    (SCRATCH / "analyze_v2_summary.txt").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
