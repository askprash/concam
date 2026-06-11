#!/bin/bash
#SBATCH --job-name=concam-republish
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/republish-%j.log
#
# Republish every public date: regenerated manifest (mask-filtered detections,
# exclusion_regions, pixel-overlap flags, compact pings) where pipeline
# outputs exist, else labeler.html only; then rebuild dates.json + index.html.
#
# Usage: sbatch slurm/republish_public.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
mkdir -p slurm/logs

for d in $(ls "$HOME/public_html/concam" | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'); do
  if [ -f "output/$d/projections.jsonl" ] && [ -f "output/$d/bundles/prash/manifest.json" ]; then
    uv run python scripts/build_public_bundle.py --date "$d" \
        --source-bundle "output/$d/bundles/prash" \
        --out-dir "$HOME/public_html/concam/$d" >/dev/null \
      && echo "regen  $d" || echo "FAIL   $d"
  else
    cp concam/bundle/templates/labeler.html "$HOME/public_html/concam/$d/labeler.html" \
      && echo "uionly $d"
  fi
  chmod a+r "$HOME/public_html/concam/$d/labeler.html" "$HOME/public_html/concam/$d/manifest.json" 2>/dev/null || true
done
python3 scripts/regenerate_public_index.py
chmod a+r "$HOME/public_html/concam/dates.json" "$HOME/public_html/concam/index.html"
echo "=== done"
