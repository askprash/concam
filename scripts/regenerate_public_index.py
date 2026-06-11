#!/usr/bin/env python3
"""Regenerate ~/public_html/concam/dates.json + index.html from the published
date subdirectories.

Called by scripts/publish_public_date.sh after each publish, and runnable
standalone as a finalizer after a batch backfill (where many concurrent
publish jobs would otherwise race on dates.json).

Each dates.json entry carries a ``labelers`` list — the human reviewers with a
committed label export for that date under the repo ``labels/`` directory —
so the labeler calendar can render one colored dot per labeler per day.

Usage: regenerate_public_index.py [<public_root>] [--labels-dir DIR]
       (default public_root: ~/public_html/concam; labels-dir: <repo>/labels)

Runs under the system python3 (3.8+) — publish_public_date.sh invokes it
outside the uv venv — so keep it stdlib-only and 3.8-compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Published dirs: YYYY-MM-DD plus optional `-suffix` for A/B variants
# (e.g. 2026-04-09-tuned alongside 2026-04-09).
_DATE_DIR_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:-[a-z0-9]+)?")
# Label exports: YYYY-MM-DD_<labeler>.json or YYYY-MM-DD_<labeler>_labels.json.
_LABEL_FILE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(.+?)(?:_labels)?\.json")


def scan_labelers(labels_dir: Path) -> dict[str, list[str]]:
    """Map YYYY-MM-DD -> sorted unique labeler ids with a label export.

    Prefers the ``date`` / ``labeler_id`` fields inside each JSON file and
    falls back to parsing the filename for legacy exports without metadata.
    Malformed or non-matching files are skipped.
    """
    out: dict[str, set[str]] = {}
    if not labels_dir.is_dir():
        return {}
    for path in sorted(labels_dir.glob("*.json")):
        m = _LABEL_FILE_RE.fullmatch(path.name)
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        date = data.get("date") or (m.group(1) if m else None)
        labeler = data.get("labeler_id") or (m.group(2) if m else None)
        if not date or not labeler:
            continue
        out.setdefault(date, set()).add(labeler)
    return {date: sorted(ids) for date, ids in out.items()}


def build_dates(
    public_root: Path, labelers_by_date: dict[str, list[str]]
) -> list[dict]:
    """One entry per published date dir with a readable manifest, newest first.

    A/B variant dirs (2026-04-09-tuned) inherit the base date's labelers —
    the human labels apply to the day, not the detector variant.
    """
    dates = []
    for child in sorted(public_root.iterdir()):
        if not child.is_dir():
            continue
        m = _DATE_DIR_RE.fullmatch(child.name)
        if not m:
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            data = json.loads(manifest_path.read_text())
        except Exception:
            continue
        ep_total = len(data.get("episodes", []))
        thr = data.get("detection_threshold", 0.0)
        ep_detected = sum(
            1 for e in data["episodes"]
            if e.get("peak_score", 0.0) >= thr
        )
        dates.append({
            "date": child.name,
            "episodes": ep_total,
            "detected": ep_detected,
            "labelers": labelers_by_date.get(m.group(1), []),
        })

    # Newest first in the dropdown and landing page.
    dates.sort(key=lambda d: d["date"], reverse=True)
    return dates


def render_index_html(dates: list[dict]) -> str:
    rows = "\n".join(
        f'  <li><a href="{d["date"]}/labeler.html">{d["date"]}</a>'
        f' <span class="note">&mdash; {d["episodes"]} flight passes,'
        f' {d["detected"]} above detector threshold'
        + (f' &middot; labeled by {", ".join(d["labelers"])}'
           if d.get("labelers") else "")
        + '</span></li>'
        for d in dates
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ConCam daily review</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.3rem; }}
  li {{ margin: 0.3em 0; }}
  .note {{ color: #666; font-size: 0.9em; }}
  code {{ background: #f3f3f3; padding: 0 0.25em; border-radius: 3px; }}
  .banner {{ border: 1px solid #f0c36d; background: #fdf6e3; border-radius: 6px;
            padding: 0.75rem 1rem; margin: 0 0 1.5rem; font-size: 0.95rem; }}
  .banner strong {{ font-size: 1.05rem; }}
  .banner .bookmark-link {{ display: block; margin-top: 0.5rem; padding: 0.5rem 0.75rem;
            background: #f3f3f3; border-radius: 4px; font-family: ui-monospace,
            SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85rem;
            word-break: break-all; }}
</style>
</head>
<body>
<div class="banner">
  <strong>&#128204; Bookmark this page.</strong><br>
  Asked to log in twice? You opened the address without the trailing slash.
  Use this exact link (replace <code>YOURNAME</code> with your username):
  <a class="bookmark-link" href="index.html">https://hex.mit.edu/~prash/concam/index.html?user=YOURNAME</a>
</div>
<h1>MIT ConCam daily review</h1>
<p>
  Daily sky-camera timelapse from the MIT Green Building, with ADS-B flight
  tracks and automatic contrail detections overlaid. Click a date to open the
  labeler; scrub the video and use the sidebar to jump to each flight pass.
  Yellow tracks crossed the detector threshold; blue tracks did not (but may
  still show a real contrail &mdash; that's what reviewers are checking for).
</p>
<ul>
{rows}
</ul>
<p class="note">
  Tip: video is a 24 h timelapse at 1 frame/second (30 fps playback = 30&times;
  real-time). Use the sidebar to seek.
</p>
</body>
</html>
"""


def regenerate(public_root: Path, labels_dir: Path) -> list[dict]:
    dates = build_dates(public_root, scan_labelers(labels_dir))
    (public_root / "dates.json").write_text(
        json.dumps({"dates": dates}, indent=2) + "\n"
    )
    (public_root / "index.html").write_text(render_index_html(dates))
    return dates


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("public_root", nargs="?",
                    default=os.path.expanduser("~/public_html/concam"))
    ap.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "labels")
    args = ap.parse_args(argv)
    dates = regenerate(Path(args.public_root), args.labels_dir)
    print(f"  wrote dates.json with {len(dates)} dates")


if __name__ == "__main__":
    main(sys.argv[1:])
