#!/bin/bash
#SBATCH --job-name=concam-detect-bundle
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/detect-bundle-%j.log

# Run detect+aggregate+store from cached OCR/ADS-B/projections, then generate
# a labeler bundle.
#
# Usage: sbatch slurm/detect_and_bundle.sh <date> [labeler] [output_dir]
#   date    — YYYY-MM-DD (required)
#   labeler — bundle labeler name (default: prash)
#   output  — output root dir (default: output)

set -euo pipefail

DATE="${1:?Usage: sbatch detect_and_bundle.sh YYYY-MM-DD [labeler] [output_dir]}"
LABELER="${2:-prash}"
OUTPUT_DIR="${3:-output}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "=== concam detect + bundle ==="
echo "Date:       ${DATE}"
echo "Labeler:    ${LABELER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Repo:       ${REPO_DIR}"
echo "Node:       $(hostname)"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${REPO_DIR}/slurm/logs"
cd "${REPO_DIR}"

echo ""
echo "=== Step 1: detect + aggregate + store ==="
uv run concam run --date "${DATE}" --output-dir "${OUTPUT_DIR}" --from-stage detect

echo ""
echo "=== Step 2: generate bundle ==="
uv run concam bundle --date "${DATE}" --output-dir "${OUTPUT_DIR}" --labelers "${LABELER}"

BUNDLE_DIR="${REPO_DIR}/${OUTPUT_DIR}/bundles/${DATE}/${LABELER}"
echo ""
echo "Bundle ready: ${BUNDLE_DIR}/labeler.html"
echo "To view: cd ${BUNDLE_DIR} && python3 -m http.server 8080"
echo ""
echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
