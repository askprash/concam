#!/usr/bin/env python3
"""Regenerate ~/public_html/concam/dates.json + index.html from the published
date subdirectories.

This is the same logic embedded in the tail of scripts/publish_public_date.sh,
extracted so it can be run once as a finalizer after a batch backfill (where
many concurrent publish jobs would otherwise race on dates.json).

Usage: regenerate_public_index.py <public_root>
       (default public_root: ~/public_html/concam)
"""
import json
import os
import re
import sys
from pathlib import Path

public_root = Path(
    sys.argv[1] if len(sys.argv) > 1
    else os.path.expanduser("~/public_html/concam")
)

dates = []
for child in sorted(public_root.iterdir()):
    if not child.is_dir():
        continue
    # Accept YYYY-MM-DD plus optional `-suffix` for A/B variants
    # (e.g. 2026-04-09-tuned alongside 2026-04-09).
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:-[a-z0-9]+)?", child.name):
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
    })

# Newest first in the dropdown and landing page.
dates.sort(key=lambda d: d["date"], reverse=True)
(public_root / "dates.json").write_text(json.dumps({"dates": dates}, indent=2) + "\n")

# Landing page.
rows = "\n".join(
    f'  <li><a href="{d["date"]}/labeler.html">{d["date"]}</a>'
    f' <span class="note">&mdash; {d["episodes"]} flight passes,'
    f' {d["detected"]} above detector threshold</span></li>'
    for d in dates
)
html = f"""<!doctype html>
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
(public_root / "index.html").write_text(html)
print(f"  wrote dates.json with {len(dates)} dates")
