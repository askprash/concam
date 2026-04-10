#!/bin/bash
#SBATCH --job-name=concam-pipeline
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/concam-%j.log

# Usage: sbatch slurm/run_pipeline.sh YYYY-MM-DD [output_dir]
#   e.g. sbatch slurm/run_pipeline.sh 2026-04-08

set -euo pipefail

DATE="${1:?ERROR: first argument must be a date (YYYY-MM-DD)}"
OUTPUT_DIR="${2:-output/${DATE}}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== concam pipeline ==="
echo "Date:       ${DATE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Repo:       ${REPO_DIR}"
echo "Node:       $(hostname)"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${REPO_DIR}/slurm/logs"
mkdir -p "${OUTPUT_DIR}"

# uv manages the virtualenv; no conda activation needed
cd "${REPO_DIR}"
uv run concam run --date "${DATE}" --output-dir "${OUTPUT_DIR}"

echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
