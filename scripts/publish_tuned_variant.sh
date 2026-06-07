#!/usr/bin/env bash
#
# Publish a tuned-detector A/B variant alongside an already-published date.
# Pairs with scripts/run_tuned_detect.py + scripts/build_public_bundle.py.
#
# Usage: scripts/publish_tuned_variant.sh <base-date> <variant-suffix> <tuned-config>
#   e.g. scripts/publish_tuned_variant.sh 2026-04-09 tuned configs/mit_green_building.tuned.yaml
#
# Preconditions:
#   - The base date has already been published (output/<base-date>/ has
#     ocr.jsonl, adsb.json, projections.jsonl; ~/public_html/concam/<base-date>/
#     exists with manifest.json + video.mp4 symlink).
#
# Actions:
#   1. Run detect+aggregate+store with the tuned config into output/<variant>/.
#   2. Generate the labeler bundle (writes manifest.json with variant date).
#   3. Build the public bundle (synthesizes the all-passes manifest).
#   4. Symlink the raw video into ~/public_html/concam/<variant>/.
#   5. Regenerate dates.json + landing page so the dropdown picks the variant up.

set -euo pipefail

BASE_DATE="${1:?Usage: $0 <base-date> <variant-suffix> <tuned-config>}"
SUFFIX="${2:?variant suffix (e.g. 'tuned')}"
TUNED_CONFIG="${3:?tuned config path}"

VARIANT="${BASE_DATE}-${SUFFIX}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PUBLIC_ROOT="$HOME/public_html/concam"
PUBLIC_DATE_DIR="$PUBLIC_ROOT/$VARIANT"
RAW_VIDEO="/net/d16/data/contrail-camera/$(echo "$BASE_DATE" | tr '-' '_')_0000_2359.mp4"
VARIANT_OUT="$REPO_DIR/output/$VARIANT"

cd "$REPO_DIR"

if [[ ! -f "$RAW_VIDEO" ]]; then
    echo "ERROR: raw video not found at $RAW_VIDEO" >&2
    exit 2
fi
if [[ ! -f "output/$BASE_DATE/projections.jsonl" ]]; then
    echo "ERROR: base date pipeline cache missing under output/$BASE_DATE/" >&2
    exit 2
fi

# 1. Tuned detect+aggregate+store into output/<variant>/.
echo "[publish-tuned] running tuned detect for $VARIANT"
uv run python scripts/run_tuned_detect.py \
    --base-date "$BASE_DATE" \
    --variant-dir "$VARIANT_OUT" \
    --config "$TUNED_CONFIG"

# 2. Bundle generation needs ocr.jsonl/adsb.json/projections.jsonl alongside the
#    detect outputs. Symlink the upstream caches into the variant dir so
#    `concam bundle` can read them.
for f in ocr.jsonl adsb.json projections.jsonl; do
    if [[ ! -e "$VARIANT_OUT/$f" ]]; then
        ln -sf "../$BASE_DATE/$f" "$VARIANT_OUT/$f"
    fi
done

# `concam bundle` requires a real ISO date. Pass the base date and override
# --output-dir to point at our variant folder's parent. It writes to
# output/$BASE_DATE/bundles/... which is wrong for the variant. Workaround:
# temporarily rename the variant dir to the base date inside a scratch root.
# Simpler: invoke generate_bundle directly via a one-liner.
echo "[publish-tuned] generating labeler bundle"
uv run python -c "
import datetime
from pathlib import Path
import sys
sys.path.insert(0, '$REPO_DIR')
from concam.bundle import generate_bundles
from concam.config import load_config
from concam.pipeline import resolve_video_path

date = datetime.date.fromisoformat('$BASE_DATE')
site = load_config(Path('$TUNED_CONFIG'))
variant_dir = Path('$VARIANT_OUT')
out_root = variant_dir / 'bundles'
out_root.mkdir(parents=True, exist_ok=True)

result = generate_bundles(
    date=date,
    labelers=['prash'],
    overlap_fraction=0.0,
    db_path=variant_dir / 'pipeline.duckdb',
    projections_path=variant_dir / 'projections.jsonl',
    detections_path=variant_dir / 'detections.jsonl',
    video_path=resolve_video_path(site.video, date),
    image_size=tuple(site.calibration.calibration_resolution),
    output_dir=out_root,
    ocr_path=variant_dir / 'ocr.jsonl' if (variant_dir / 'ocr.jsonl').exists() else None,
    detection_threshold=site.aggregation.detection_threshold,
)
for lbl, bdir in result.items():
    print(f'  {lbl}: {bdir}')
"

SOURCE_BUNDLE="$VARIANT_OUT/bundles/prash"
if [[ ! -f "$SOURCE_BUNDLE/manifest.json" ]]; then
    echo "ERROR: bundle manifest not produced at $SOURCE_BUNDLE/manifest.json" >&2
    exit 2
fi

# 3. Public bundle, with date overridden to the variant string so the dropdown
#    treats it as a separate entry.
echo "[publish-tuned] building public bundle"
mkdir -p "$PUBLIC_DATE_DIR"
uv run python scripts/build_public_bundle.py \
    --date "$BASE_DATE" \
    --source-bundle "$SOURCE_BUNDLE" \
    --projections "$VARIANT_OUT/projections.jsonl" \
    --detections "$VARIANT_OUT/detections.jsonl" \
    --out-dir "$PUBLIC_DATE_DIR" \
    --config "$TUNED_CONFIG"

# Override manifest.date with the variant string so the dropdown shows
# "<base>-<suffix>" and the page comparison is unambiguous.
uv run python -c "
import json
from pathlib import Path
mp = Path('$PUBLIC_DATE_DIR/manifest.json')
m = json.loads(mp.read_text())
m['date'] = '$VARIANT'
mp.write_text(json.dumps(m, indent=2) + '\n')
"

# 4. Video symlink (same raw video as the base date — only the detector output
#    differs).
ln -sfn "$RAW_VIDEO" "$PUBLIC_DATE_DIR/video.mp4"

chmod a+r "$PUBLIC_DATE_DIR/manifest.json" "$PUBLIC_DATE_DIR/labeler.html"

# 5. Regenerate dates.json + index.html (same logic as publish_public_date.sh,
#    inlined here so we don't re-publish the base date as a side-effect).
echo "[publish-tuned] regenerating dates.json + index.html"
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

dates.sort(key=lambda d: d["date"], reverse=True)
(public_root / "dates.json").write_text(json.dumps({"dates": dates}, indent=2) + "\n")

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
</style>
</head>
<body>
<h1>MIT ConCam daily review</h1>
<ul>
{rows}
</ul>
</body>
</html>
"""
(public_root / "index.html").write_text(html)
print(f"  wrote dates.json with {len(dates)} entries")
PY

chmod a+r "$PUBLIC_ROOT/dates.json" "$PUBLIC_ROOT/index.html"
echo "[publish-tuned] done: https://hex.mit.edu/~prash/concam/$VARIANT/labeler.html"
