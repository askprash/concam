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
#    Delegated to regenerate_public_index.py (stdlib-only) so the logic lives
#    in exactly one place; it also attaches per-date labeler lists from labels/.
echo "[publish] regenerating dates.json + index.html"
python3 "$REPO_DIR/scripts/regenerate_public_index.py" "$PUBLIC_ROOT"

chmod a+r "$PUBLIC_ROOT/dates.json" "$PUBLIC_ROOT/index.html"
echo "[publish] done: https://hex.mit.edu/~prash/concam/$DATE/labeler.html"
