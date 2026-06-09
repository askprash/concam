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
                    help="emit feder-available days within this many days of the "
                         "latest available day (newest first)")
    ap.add_argument("--latest", action="store_true",
                    help="print only the latest available day and exit")
    ap.add_argument("--config", default=None, help="site YAML (for the feder data dir)")
    ap.add_argument("--data-dir", default=None, help="override the feder data dir")
    args = ap.parse_args()

    os.environ["FEDER_DATA_DIR"] = _resolve_data_dir(args.config, args.data_dir)
    import feder

    ranges = feder.available_days()  # list[(start_date, end_date)] inclusive
    if not ranges:
        print("no feder data available", file=sys.stderr)
        return 1

    latest = max(end for _start, end in ranges)
    if args.latest:
        print(latest.isoformat())
        return 0

    def available(day: datetime.date) -> bool:
        return any(start <= day <= end for start, end in ranges)

    for i in range(max(1, args.within)):
        day = latest - datetime.timedelta(days=i)
        if available(day):
            print(day.isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
