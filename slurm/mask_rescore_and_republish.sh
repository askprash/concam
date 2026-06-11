#!/bin/bash
#SBATCH --job-name=concam-mask-rescore
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/mask-rescore-%j.log
#
# 1. Re-score the labeled episodes of 2026-04-08 / 2026-04-09 with and without
#    the static building mask (scripts/rescore_labeled_episodes.py) to measure
#    the mask's real FP-kill / TP-collateral at the production threshold.
# 2. Republish every public date: regenerated manifest (exclusion_regions +
#    pixel-overlap flags) where pipeline outputs exist, else labeler.html only;
#    then rebuild dates.json + index.html (with per-date labeler dots).
#
# Usage: sbatch slurm/mask_rescore_and_republish.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
mkdir -p slurm/logs output/validation/mask_rescore

for d in 2026-04-09 2026-04-08; do
  echo "=== rescore $d"
  uv run python scripts/rescore_labeled_episodes.py \
      --date "$d" --out "output/validation/mask_rescore/$d.json"
done

echo "=== republish public dates"
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
