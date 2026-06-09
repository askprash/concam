#!/usr/bin/env python
"""Report which calendar days the feder ADS-B store has data for.

ADS-B flight data is not available in real time — it lags the live date by a
few days. The daily publish cron uses this to gate on data availability: the
pipeline can only run a day once feder has that day's flights.

Uses the feder Python API (``feder.available_days``), never raw SQLite.

Examples:
    # Latest available calendar day (one line, YYYY-MM-DD):
    python scripts/feder_available_days.py --latest

    # feder-available days within the last N days, newest first (the cron feeds
    # this list through its video-complete / not-yet-published checks):
    python scripts/feder_available_days.py --within 14
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mit_green_building.yaml"
FALLBACK_DATA_DIR = "/home/mcast/data/feder"


def _resolve_data_dir(config: str | None, data_dir: str | None) -> str:
    if data_dir:
        return data_dir
    try:
        from concam.config import load_config

        return load_config(config or str(DEFAULT_CONFIG)).adsb.data_dir
    except Exception:
        return FALLBACK_DATA_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--within", type=int, default=14,
                    help="scan this many days back from the newest day with data")
    ap.add_argument("--latest", action="store_true",
                    help="print only the latest COMPLETE day and exit")
    ap.add_argument("--min-hours", type=float, default=20.0,
                    help="a day counts as available only if its ADS-B coverage "
                         "spans at least this many hours (guards partially-ingested "
                         "latest days; full days are ~24h)")
    ap.add_argument("--config", default=None, help="site YAML (for the feder data dir)")
    ap.add_argument("--data-dir", default=None, help="override the feder data dir")
    args = ap.parse_args()

    os.environ["FEDER_DATA_DIR"] = _resolve_data_dir(args.config, args.data_dir)
    import feder

    ranges = feder.available_days()  # list[(start_date, end_date)] inclusive
    if not ranges:
        print("no feder data available", file=sys.stderr)
        return 1
    newest = max(end for _start, end in ranges)

    def covered_hours(day: datetime.date) -> float:
        """Total hours of ADS-B coverage feder has for `day` (0 if none/error)."""
        try:
            intervals = feder.available_times(day)
        except Exception:
            return 0.0
        return sum((e - s).total_seconds() for s, e in intervals) / 3600.0

    def complete(day: datetime.date) -> bool:
        return covered_hours(day) >= args.min_hours

    # Walk back from the newest day that has ANY data, emitting only days whose
    # coverage is complete enough (newest first). The latest day is frequently
    # mid-ingest (e.g. only the first few UTC hours), so it is excluded until full.
    emitted = []
    for i in range(max(1, args.within)):
        day = newest - datetime.timedelta(days=i)
        if complete(day):
            emitted.append(day)

    if args.latest:
        if emitted:
            print(emitted[0].isoformat())
            return 0
        print("no complete feder day in window", file=sys.stderr)
        return 1

    for day in emitted:
        print(day.isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
