#!/bin/bash
#SBATCH --job-name=concam-pipeline-publish
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/pipeline-publish-%j.log

# Run the concam pipeline (optionally from a given stage) and then publish
# the resulting bundle to ~/public_html/concam/<date>/ via
# scripts/publish_public_date.sh.
#
# Usage:
#   sbatch slurm/pipeline_and_publish.sh <date> [from_stage] [output_dir]
#     date       — YYYY-MM-DD (required)
#     from_stage — one of ocr|adsb|project|detect|aggregate|store (optional;
#                  omit for a full run from OCR)
#     output_dir — output root dir (default: output)

set -euo pipefail

DATE="${1:?Usage: sbatch pipeline_and_publish.sh YYYY-MM-DD [from_stage] [output_dir]}"
FROM_STAGE="${2:-}"
OUTPUT_DIR="${3:-output}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "=== concam pipeline + publish ==="
echo "Date:       ${DATE}"
echo "From stage: ${FROM_STAGE:-<start>}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Repo:       ${REPO_DIR}"
echo "Node:       $(hostname)"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${REPO_DIR}/slurm/logs"
cd "${REPO_DIR}"

echo ""
echo "=== Step 1: concam run ==="
if [ -n "${FROM_STAGE}" ]; then
    uv run concam run --date "${DATE}" --output-dir "${OUTPUT_DIR}" --from-stage "${FROM_STAGE}"
else
    uv run concam run --date "${DATE}" --output-dir "${OUTPUT_DIR}"
fi

echo ""
echo "=== Step 2: publish to public_html ==="
scripts/publish_public_date.sh "${DATE}"

echo ""
echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
