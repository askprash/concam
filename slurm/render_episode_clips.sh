#!/bin/bash
#SBATCH --job-name=concam-clips
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/clips-%j.log

# Render annotated MP4 episode clips for one or more dates.
#
# Usage:
#   sbatch slurm/render_episode_clips.sh YYYY-MM-DD [YYYY-MM-DD ...]
#   sbatch slurm/render_episode_clips.sh 2026-04-08 2026-04-09
#
# Optional env overrides (set before sbatch or in the script):
#   CLIPS_TOP_N      — max episodes per date (default: all above threshold)
#   CLIPS_PRE_ROLL   — seconds before onset (default: 10)
#   CLIPS_POST_ROLL  — seconds after episode end (default: 10)
#   CLIPS_MAX_CLIP   — hard cap per clip in seconds (default: 90)
#   OUTPUT_DIR       — pipeline output root (default: output)

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "ERROR: provide at least one date (YYYY-MM-DD)" >&2
    exit 1
fi

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PRE_ROLL="${CLIPS_PRE_ROLL:-10}"
POST_ROLL="${CLIPS_POST_ROLL:-10}"
MAX_CLIP="${CLIPS_MAX_CLIP:-90}"

# If CLIPS_TOP_N is set use --top-n, otherwise render all above threshold
# by passing a very large number.
TOP_N="${CLIPS_TOP_N:-9999}"

echo "=== concam episode clip renderer ==="
echo "Dates:      $*"
echo "Output dir: ${OUTPUT_DIR}"
echo "Pre-roll:   ${PRE_ROLL} s"
echo "Post-roll:  ${POST_ROLL} s"
echo "Max clip:   ${MAX_CLIP} s"
echo "Top-N:      ${TOP_N}"
echo "Repo:       ${REPO_DIR}"
echo "Node:       $(hostname)"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

mkdir -p "${REPO_DIR}/slurm/logs"
cd "${REPO_DIR}"

TOTAL_OK=0
TOTAL_FAIL=0

for DATE in "$@"; do
    echo "--- ${DATE} ---"
    CLIPS_DIR="${OUTPUT_DIR}/${DATE}/clips"

    if ! uv run python scripts/render_episode_clips.py \
            --date "${DATE}" \
            --output-dir "${OUTPUT_DIR}" \
            --top-n "${TOP_N}" \
            --pre-roll "${PRE_ROLL}" \
            --post-roll "${POST_ROLL}" \
            --max-clip "${MAX_CLIP}" \
            --verbose; then
        echo "WARN: render_episode_clips.py exited non-zero for ${DATE}" >&2
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    else
        N=$(ls "${CLIPS_DIR}"/*.mp4 2>/dev/null | wc -l)
        echo "${DATE}: wrote ${N} clips to ${CLIPS_DIR}"
        TOTAL_OK=$((TOTAL_OK + N))
    fi
    echo
done

echo "=== Done: ${TOTAL_OK} clips written, ${TOTAL_FAIL} date(s) failed ==="
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
