"""Scan a single daily timelapse hour-by-hour using PyAV seeking.

For each hour of the day we seek to the keyframe at-or-before the
hour-mark and decode forward N frames, OCR each, and report the same
yield + delta + pattern stats per hour.

This is fast because seeking jumps to the nearest keyframe instead of
decoding from frame 0 every time.

Usage:
    .venv/bin/python scripts/pts_drift_test/scan_one_day.py PATH [N]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from concam.config import load_config
from concam.ocr import FixedFormatTimestampReader

WINDOW = 300  # frames per hour-window, default


def walk_window_fresh(path: Path, start_pts_seconds: float, n: int, reader: FixedFormatTimestampReader):
    """Open the file fresh, seek to start_pts_seconds, decode n frames; OCR each.

    Re-opening the container per window is the simplest way to avoid PyAV
    decoder-state contamination between sequential seeks.  Slower than a
    single-pass walk but reliably correct.
    """
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Container.seek without `stream=` uses microseconds (av.time_base = 1_000_000).
        target_us = int(start_pts_seconds * 1_000_000)
        container.seek(target_us, any_frame=False, backward=True)

        deltas = []
        last = None
        n_ok = 0
        n_seen = 0
        first_dt = None
        last_dt = None

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            f_pts = float(frame.pts * stream.time_base)
            if f_pts < start_pts_seconds:
                continue

            n_seen += 1
            bgr = frame.to_ndarray(format="bgr24")
            r = reader.read(bgr)
            if r.ok and r.parsed_dt is not None:
                n_ok += 1
                if first_dt is None:
                    first_dt = r.parsed_dt
                last_dt = r.parsed_dt
                if last is not None:
                    deltas.append(int((r.parsed_dt - last).total_seconds()))
                last = r.parsed_dt
            else:
                last = None

            if n_seen >= n:
                break

        return n_seen, n_ok, deltas, first_dt, last_dt
    finally:
        container.close()


def find_pat(deltas, target):
    L = len(target)
    return sum(1 for j in range(len(deltas) - L + 1) if tuple(deltas[j : j + L]) == target)


def summarize(label: str, n_seen, n_ok, deltas, first_dt, last_dt) -> str:
    yield_pct = n_ok / max(n_seen, 1)
    clipped = []
    for d in deltas:
        if d in (-2, -1, 0, 1, 2, 3):
            clipped.append(d)
        else:
            clipped.append("other")
    hist = Counter(clipped)
    pat = (
        find_pat(deltas, (0, 1, 1, 2)),
        find_pat(deltas, (1, 1, 0, 2)),
        find_pat(deltas, (1, 1, 2, 0)),
        find_pat(deltas, (1, 2, 0, 1)),
    )
    return (
        f"{label}: yield={yield_pct:6.1%} ({n_ok}/{n_seen})  "
        f"hist={dict(sorted((str(k), v) for k, v in hist.items()))}  "
        f"pat[(0112)|(1102)|(1120)|(1201)]={pat}  "
        f"first/last={first_dt} / {last_dt}"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: scan_one_day.py PATH [N]", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    n_per_window = int(sys.argv[2]) if len(sys.argv) > 2 else WINDOW

    cfg = load_config(ROOT / "configs" / "mit_green_building.yaml")
    reader = FixedFormatTimestampReader(cfg.ocr)

    print(f"Scanning {path}", flush=True)
    print(f"Windows: 24 hours, {n_per_window} frames each", flush=True)
    print("=" * 78, flush=True)

    # Daily timelapse: 30fps playback, frame_idx == real_second_of_day, so
    # PTS_seconds(frame_idx) = frame_idx / 30. Hour H starts at frame H*3600,
    # i.e. PTS_seconds = H * 120.
    for hour in range(24):
        start_s = hour * 120.0
        n_seen, n_ok, deltas, first_dt, last_dt = walk_window_fresh(
            path, start_s, n_per_window, reader
        )
        label = f"{hour:02d}:00"
        print(summarize(label, n_seen, n_ok, deltas, first_dt, last_dt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
