#!/usr/bin/env python3
"""Summarise what the OCR date fix changed for one processed day.

The headline metric for GitHub #1 is **time coverage**: before the fix,
detection silently stopped at the first confident OCR date misread, because the
detect stage joins frame timestamps against projections on the full ISO
datetime (date included).  Raw detection-row counts are a poor proxy -- one row
is one aircraft scored at one timestamp, and most score 0 -- so this reports
the detected time span and episode counts alongside them.

Compares the ``.pre-ocrfix`` backups written by slurm/reprocess_ocr_fix_array.sh
against the freshly regenerated files, and writes
``<output-dir>/<date>/ocrfix_report.json``.

Usage:
    python scripts/ocrfix_day_report.py --date 2026-04-11 [--output-dir output]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path


def _ocr_date_stats(path: Path, day: datetime.date) -> dict:
    """Count frames whose timestamp date falls outside {day, day+1}.

    A clean day spans the local date plus the post-midnight rollover into the
    next UTC date, so those two are the only legitimate values.
    """
    if not path.exists():
        return {"present": False}
    allowed = {day.isoformat(), (day + datetime.timedelta(days=1)).isoformat()}
    frames = 0
    corrupt = 0
    bad_dates: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            marker = '"wall_time_utc": "'
            i = line.find(marker)
            if i < 0:
                continue
            d = line[i + len(marker):i + len(marker) + 10]
            frames += 1
            if d not in allowed:
                corrupt += 1
                bad_dates[d] = bad_dates.get(d, 0) + 1
    top = sorted(bad_dates.items(), key=lambda kv: -kv[1])[:5]
    return {
        "present": True,
        "frames": frames,
        "out_of_day_frames": corrupt,
        "out_of_day_pct": round(100.0 * corrupt / frames, 2) if frames else 0.0,
        "top_bad_dates": top,
    }


def _detection_stats(path: Path) -> dict:
    """Rows, scored-time span, and how many aircraft ever scored a positive."""
    if not path.exists():
        return {"present": False}
    rows = 0
    first = last = None
    positives = 0
    aircraft: set[str] = set()
    aircraft_positive: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            t = r.get("wall_time_utc")
            if t:
                if first is None or t < first:
                    first = t
                if last is None or t > last:
                    last = t
            key = r.get("transponder_id") or r.get("callsign")
            if key:
                aircraft.add(key)
            if (r.get("score") or 0) > 0:
                positives += 1
                if key:
                    aircraft_positive.add(key)
    return {
        "present": True,
        "rows": rows,
        "first_utc": first,
        "last_utc": last,
        "positive_rows": positives,
        "aircraft_scored": len(aircraft),
        "aircraft_with_positive": len(aircraft_positive),
    }


def _episode_count(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    with path.open() as f:
        return {"present": True, "episodes": sum(1 for line in f if line.strip())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--suffix", default=".pre-ocrfix",
                    help="backup suffix written before the reprocess")
    args = ap.parse_args()

    day = datetime.date.fromisoformat(args.date)
    base = Path(args.output_dir) / args.date
    if not base.is_dir():
        print(f"ERROR: no such day directory: {base}", file=sys.stderr)
        return 2

    def paired(name: str, fn):
        stem, _, ext = name.rpartition(".")
        backup = base / f"{stem}{args.suffix}.{ext}"
        return {"before": fn(backup), "after": fn(base / name)}

    report = {
        "date": args.date,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ocr": {
            "before": _ocr_date_stats(base / f"ocr{args.suffix}.jsonl", day),
            "after": _ocr_date_stats(base / "ocr.jsonl", day),
        },
        "detections": {
            "before": _detection_stats(base / f"detections{args.suffix}.jsonl"),
            "after": _detection_stats(base / "detections.jsonl"),
        },
        "episodes": {
            "before": _episode_count(base / f"episodes{args.suffix}.jsonl"),
            "after": _episode_count(base / "episodes.jsonl"),
        },
    }

    out = base / "ocrfix_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    d_before = report["detections"]["before"]
    d_after = report["detections"]["after"]
    print(f"[report] {args.date}")
    print(f"  OCR out-of-day frames: "
          f"{report['ocr']['before'].get('out_of_day_frames', '?')} -> "
          f"{report['ocr']['after'].get('out_of_day_frames', '?')}")
    print(f"  detection span: {d_before.get('first_utc')}..{d_before.get('last_utc')} -> "
          f"{d_after.get('first_utc')}..{d_after.get('last_utc')}")
    print(f"  episodes: {report['episodes']['before'].get('episodes', '?')} -> "
          f"{report['episodes']['after'].get('episodes', '?')}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
