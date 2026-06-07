#!/bin/bash
#SBATCH --job-name=concam-hpo-tuneval
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G  # 04-09's merged label set caches ~1.5k padded crops; large
                   # flight-bbox ROIs make these multi-MB each (16G OOM'd at 21G RSS).
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/hpo-tuneval-%j.log
#
# Tune the detector on 2026-04-09 (ALL reviewers merged for scenario diversity)
# and report HELD-OUT performance on 2026-04-19 (never seen by the tuner).
# Does NOT write the production/tuned config and does NOT publish — it stops at
# a report for human review.
#
# Usage: sbatch slurm/hpo_tune_validate.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
mkdir -p slurm/logs output/hpo

TRAIN_DATE=2026-04-09
VAL_DATE=2026-04-19
BASE_CONFIG=configs/mit_green_building.yaml

echo "=== [1/3] tuning sweep on ${TRAIN_DATE} (all reviewers merged) ==="
uv run python scripts/detection_hpo.py \
    --date "${TRAIN_DATE}" \
    --labels labels/${TRAIN_DATE}_lrsand_labels.json \
             labels/${TRAIN_DATE}_thendo.json \
             labels/${TRAIN_DATE}_prash.json \
             labels/${TRAIN_DATE}_reviewer-1.json \
    --manifest "${HOME}/public_html/concam/${TRAIN_DATE}/manifest.json" \
    --config "${BASE_CONFIG}" \
    --out-dir "output/hpo/${TRAIN_DATE}"

echo "=== [2/3] held-out sweep on ${VAL_DATE} (same grid, lrsand labels) ==="
uv run python scripts/detection_hpo.py \
    --date "${VAL_DATE}" \
    --labels labels/${VAL_DATE}_lrsand_labels.json \
    --manifest "${HOME}/public_html/concam/${VAL_DATE}/manifest.json" \
    --config "${BASE_CONFIG}" \
    --out-dir "output/hpo/${VAL_DATE}"

echo "=== [3/3] select winner on ${TRAIN_DATE}, report held-out on ${VAL_DATE} ==="
uv run python scripts/hpo_select_and_validate.py \
    --train-results "output/hpo/${TRAIN_DATE}/sweep_results.json" \
    --val-results   "output/hpo/${VAL_DATE}/sweep_results.json" \
    --base-config   "${BASE_CONFIG}" \
    --out           "output/hpo/holdout_validation.md"

echo "=== done. Review: output/hpo/${TRAIN_DATE}/sweep_report.md,"
echo "             output/hpo/${VAL_DATE}/sweep_report.md,"
echo "             output/hpo/holdout_validation.md ==="
echo "No config was written and nothing was published (review-only run)."
