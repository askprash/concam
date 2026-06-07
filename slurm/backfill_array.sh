#!/bin/bash
#SBATCH --job-name=concam-backfill
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/backfill-%A_%a.log
###############################################################################
# Backfill driver: one Slurm array task per date.
#
# Each task looks up its date by line number (SLURM_ARRAY_TASK_ID) in the date
# list passed as $1, then runs the same two steps as
# slurm/pipeline_and_publish.sh: `concam run` followed by
# scripts/publish_public_date.sh.
#
# Idempotent: a date whose public manifest.json already exists is skipped, so
# the array can be safely resubmitted (e.g. --array of just the failed tasks).
#
# Usage:
#   sbatch --array=1-<N>%<conc> slurm/backfill_array.sh <datelist.txt>
###############################################################################
set -euo pipefail

LIST="${1:?usage: sbatch --array=1-N%C slurm/backfill_array.sh <datelist.txt>}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

DATE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
if [[ -z "${DATE}" ]]; then
  echo "[backfill] no date on line ${SLURM_ARRAY_TASK_ID} of ${LIST}; nothing to do."
  exit 0
fi

PUBLIC_MANIFEST="$HOME/public_html/concam/${DATE}/manifest.json"
if [[ -f "$PUBLIC_MANIFEST" ]]; then
  echo "[backfill] ${DATE} already published (${PUBLIC_MANIFEST} exists); skipping."
  exit 0
fi

echo "=== concam backfill: task ${SLURM_ARRAY_TASK_ID} -> ${DATE} ==="
echo "Node:  $(hostname)"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo ""
echo "=== Step 1: concam run ${DATE} ==="
uv run concam run --date "${DATE}" --output-dir output

echo ""
echo "=== Step 2: publish ${DATE} ==="
scripts/publish_public_date.sh "${DATE}"

echo ""
echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
