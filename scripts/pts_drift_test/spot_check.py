"""Spot-check the OSD-second delta pattern in any daily timelapse file.

Usage:
    .venv/bin/python scripts/pts_drift_test/spot_check.py PATH [start_frame] [n]
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


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: spot_check.py PATH [start_frame] [n]", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 28800
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 600

    cfg = load_config(ROOT / "configs" / "mit_green_building.yaml")
    reader = FixedFormatTimestampReader(cfg.ocr)

    print(f"file: {path}", flush=True)
    print(f"window: frames {start}..{start+n}", flush=True)

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    deltas = []
    last = None
    first_dt = None
    last_dt = None
    n_ok = 0
    i = 0
    for frame in container.decode(stream):
        if i >= start + n:
            break
        if i >= start:
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
        i += 1
        if i % 5000 == 0:
            print(f"  decoded {i}", flush=True)
    container.close()

    hist = Counter(deltas)

    def find_pat(target):
        L = len(target)
        return sum(1 for j in range(len(deltas) - L + 1) if tuple(deltas[j : j + L]) == target)

    print(f"OCR yield: {n_ok}/{n} = {n_ok/n:.1%}", flush=True)
    print(f"first OSD: {first_dt}  last OSD: {last_dt}", flush=True)
    print(f"delta histogram: {dict(sorted(hist.items()))}", flush=True)
    print(
        f"(0,1,1,2)|(1,1,0,2)|(1,1,2,0)|(1,2,0,1) counts: "
        f"{find_pat((0,1,1,2))}|{find_pat((1,1,0,2))}|{find_pat((1,1,2,0))}|{find_pat((1,2,0,1))}",
        flush=True,
    )
    print(f"first 40 deltas: {deltas[:40]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
