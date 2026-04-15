#!/bin/bash
#SBATCH --job-name=concam-pipeline
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/concam-%j.log

# Usage: sbatch slurm/run_pipeline.sh YYYY-MM-DD [output_dir] [from_stage]
#   e.g. sbatch slurm/run_pipeline.sh 2026-04-08
#        sbatch slurm/run_pipeline.sh 2026-04-08 output detect

set -euo pipefail

DATE="${1:?ERROR: first argument must be a date (YYYY-MM-DD)}"
OUTPUT_DIR="${2:-output}"
FROM_STAGE="${3:-}"
# SLURM_SUBMIT_DIR is set to the directory where sbatch was called, which is
# the repo root when called as `sbatch slurm/run_pipeline.sh ...`.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "=== concam pipeline ==="
echo "Date:       ${DATE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "From stage: ${FROM_STAGE:-<start>}"
echo "Repo:       ${REPO_DIR}"
echo "Node:       $(hostname)"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${REPO_DIR}/slurm/logs"
mkdir -p "${OUTPUT_DIR}"

# uv manages the virtualenv; no conda activation needed
cd "${REPO_DIR}"

if [ -n "${FROM_STAGE}" ]; then
    uv run concam run --date "${DATE}" --output-dir "${OUTPUT_DIR}" --from-stage "${FROM_STAGE}"
else
    uv run concam run --date "${DATE}" --output-dir "${OUTPUT_DIR}"
fi

echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
