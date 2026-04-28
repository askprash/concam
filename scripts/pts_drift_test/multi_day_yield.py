"""Sweep OCR yield + delta-distribution across many daily timelapses.

For each available daily file we:
  * Walk a 600-frame window starting at noon (frame 43200, real-time 12:00 EDT).
  * Report:
      - yield (fraction of frames with status="ok")
      - delta histogram, clipping wild values to "other"
      - count of literal patterns {(0,1,1,2), (1,1,0,2), (1,1,2,0), (1,2,0,1)}
      - first / last OSD parsed (sanity check for mis-parsed months/years)

Usage:
    .venv/bin/python scripts/pts_drift_test/multi_day_yield.py [n_days]
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

TIMELAPSE_DIR = Path("/net/d16/data/contrail-camera")
START_FRAME = 43200  # 12:00 EDT
WINDOW = 600


def walk_one(path: Path, reader: FixedFormatTimestampReader):
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    deltas: list[int] = []
    last = None
    n_ok = 0
    first_dt = None
    last_dt = None
    n_seen = 0
    i = 0
    for frame in container.decode(stream):
        if i >= START_FRAME + WINDOW:
            break
        if i >= START_FRAME:
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
        i += 1
    container.close()
    return n_seen, n_ok, deltas, first_dt, last_dt


def find_pat(deltas, target):
    L = len(target)
    return sum(1 for j in range(len(deltas) - L + 1) if tuple(deltas[j : j + L]) == target)


def summarize(name: str, n_seen: int, n_ok: int, deltas: list[int], first_dt, last_dt) -> str:
    yield_pct = n_ok / max(n_seen, 1)
    # Clip delta histogram to {-2, -1, 0, 1, 2, 3, "other"}
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
    expected_first = name.replace("_", "-")[:10] + " 12:00:0?"
    fdt = str(first_dt) if first_dt is not None else "None"
    ldt = str(last_dt) if last_dt is not None else "None"
    n_one = hist.get(1, 0)
    n_outliers = hist.get("other", 0)
    return (
        f"{name}: yield={yield_pct:6.1%} "
        f"({n_ok}/{n_seen})  "
        f"delta-1={n_one:3d}  outliers={n_outliers:3d}  "
        f"hist={dict(sorted((str(k), v) for k, v in hist.items()))}  "
        f"pat[(0112)|(1102)|(1120)|(1201)]={pat}  "
        f"first/last={fdt} / {ldt}"
    )


def main() -> int:
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    cfg = load_config(ROOT / "configs" / "mit_green_building.yaml")
    reader = FixedFormatTimestampReader(cfg.ocr)

    files = sorted(TIMELAPSE_DIR.glob("2026_04_*_0000_2359.mp4"))
    files = files[-n_days:]

    print(f"Sampling {len(files)} day(s) at frame {START_FRAME}..{START_FRAME+WINDOW} (12:00 EDT, 10 min)", flush=True)
    print("=" * 78, flush=True)
    for f in files:
        name = f.stem.split("_0000_2359")[0]
        try:
            n_seen, n_ok, deltas, first_dt, last_dt = walk_one(f, reader)
        except Exception as exc:
            print(f"{name}: ERROR {exc}", flush=True)
            continue
        print(summarize(name, n_seen, n_ok, deltas, first_dt, last_dt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
