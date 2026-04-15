#!/bin/bash
#SBATCH --job-name=concam-regression
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/regression-%j.log

# Re-run the detect stage from the cached OCR/ADS-B/projections for April-8,
# then run the regression baseline script to update metrics.json, report.md,
# and spot-check panels.
#
# Usage: sbatch slurm/regression_rerun.sh [date] [output_dir]
# Default date: 2026-04-08

set -euo pipefail

DATE="${1:-2026-04-08}"
OUTPUT_DIR="${2:-output}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "=== concam regression re-run ==="
echo "Date:       ${DATE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Repo:       ${REPO_DIR}"
echo "Node:       $(hostname)"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${REPO_DIR}/slurm/logs"

cd "${REPO_DIR}"

# Step 1: Re-run detect+aggregate+store with the current config.
# OCR and ADS-B caches are preserved; only the detection output is regenerated.
echo ""
echo "=== Step 1: detect + aggregate + store ==="
uv run concam run --date "${DATE}" --output-dir "${OUTPUT_DIR}" --from-stage detect

# Step 2: Run the regression baseline script to update metrics and panels.
echo ""
echo "=== Step 2: regression baseline script ==="
uv run python scripts/regression_e2e.py --date "${DATE}" --output-dir "${OUTPUT_DIR}" --verbose

echo ""
echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
