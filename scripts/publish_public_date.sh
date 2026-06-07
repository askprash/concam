#!/usr/bin/env bash
#
# Publish a single date's pipeline output to ~/public_html/concam/<date>/
# so external reviewers can access it via hex.mit.edu.
#
# Usage: scripts/publish_public_date.sh YYYY-MM-DD
#
# Preconditions:
#   - `concam run --date <date>` has completed (detections.jsonl, episodes.jsonl,
#     projections.jsonl, pipeline.duckdb under output/<date>/).
#   - Raw video exists at /net/d16/data/contrail-camera/YYYY_MM_DD_0000_2359.mp4.
#
# Actions:
#   1. Runs `concam bundle` to produce output/<date>/bundles/prash/ (the DB-backed
#      bundle with episodes assigned to labeler "prash").
#   2. Runs scripts/build_public_bundle.py to synthesize the all-flight-passes
#      manifest.json, copying labeler.html from the template.
#   3. Symlinks (not copies) the raw video into public_html — avoids burning
#      disk quota since /net/d16 is accessible and Apache's SymLinksIfOwnerMatch
#      follows owner-matching symlinks.
#   4. Regenerates ~/public_html/concam/dates.json and the landing index.html
#      so the dropdown in the labeler and the landing page pick up the new date.

set -euo pipefail

DATE="${1:?ERROR: usage: $0 YYYY-MM-DD}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PUBLIC_ROOT="$HOME/public_html/concam"
PUBLIC_DATE_DIR="$PUBLIC_ROOT/$DATE"
RAW_VIDEO="/net/d16/data/contrail-camera/$(echo "$DATE" | tr '-' '_')_0000_2359.mp4"

cd "$REPO_DIR"

if [[ ! -f "$RAW_VIDEO" ]]; then
    echo "ERROR: raw video not found at $RAW_VIDEO" >&2
    exit 2
fi
if [[ ! -f "output/$DATE/detections.jsonl" ]]; then
    echo "ERROR: output/$DATE/detections.jsonl missing — run the pipeline first" >&2
    exit 2
fi

# 1. DB-backed bundle for labeler "prash" — produces output/<date>/bundles/prash/
#    with manifest.json + labeler.html. Idempotent: re-running regenerates.
echo "[publish] running concam bundle for $DATE"
uv run concam bundle --date "$DATE" --labelers prash --overlap-fraction 0.0

# 2. Synthesize the public bundle from the current projections + detections.
echo "[publish] building public bundle"
mkdir -p "$PUBLIC_DATE_DIR"
uv run python scripts/build_public_bundle.py \
    --date "$DATE" \
    --source-bundle "output/$DATE/bundles/prash" \
    --out-dir "$PUBLIC_DATE_DIR"

# 3. Symlink the raw video (no copy). Overwrites a prior symlink or file.
ln -sfn "$RAW_VIDEO" "$PUBLIC_DATE_DIR/video.mp4"

chmod a+r "$PUBLIC_DATE_DIR/manifest.json" "$PUBLIC_DATE_DIR/labeler.html"

# 4. Rebuild dates.json + landing page from whichever subdirectories exist under
#    $PUBLIC_ROOT. A directory is counted only if it has a readable
#    manifest.json — in-progress publishes don't leak into the dropdown.
echo "[publish] regenerating dates.json + index.html"
python3 - "$PUBLIC_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

public_root = Path(sys.argv[1])
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
PY

chmod a+r "$PUBLIC_ROOT/dates.json" "$PUBLIC_ROOT/index.html"
echo "[publish] done: https://hex.mit.edu/~prash/concam/$DATE/labeler.html"
