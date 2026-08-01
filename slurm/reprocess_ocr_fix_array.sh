#!/bin/bash
#SBATCH --job-name=concam-ocrfix
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/ocrfix-%A_%a.log
###############################################################################
# Archive-wide reprocess after the GitHub #1 fix (merged as PR #2): the OCR
# timestamp date is now derived from the processing day instead of being read
# off the frame, so detection no longer dies at the first confident date
# misread.
#
# A scan of every output/*/ocr.jsonl found 143 of 186 processed days carrying
# dates outside {day, day+1} -- typically 50-80% of frames -- so nearly the
# whole archive lost most of its detection hours, not just the days flagged on
# the issue.
#
# Per date, one array task does:
#   1. back up ocr/detections/episodes as *.pre-ocrfix.jsonl (first run only, so
#      resubmission never overwrites the pre-fix baseline)
#   2. regenerate ONLY the OCR stage with the fix
#   3. concam run --from-stage detect  (detect + aggregate + store, reusing the
#      cached ADS-B projections, which are independent of OCR)
#   4. republish the public labeler bundle for the date
#   5. write output/<date>/ocrfix_report.json comparing before vs after
#
# The backups in step 1 are load-bearing beyond rollback: episode IDs are
# assigned by position (1..N) at store time, so restoring the lost detection
# hours renumbers episodes -- and labels/*.json key on bare episode_id.  The
# pre-fix episodes.jsonl is what lets scripts/remap_labels_after_reprocess.py
# rebuild the label -> episode mapping on its natural key
# (transponder_id, onset).  Do not delete it before labels are remapped.
#
# Usage:
#   sbatch --array=1-<N>%<conc> slurm/reprocess_ocr_fix_array.sh <datelist.txt>
###############################################################################
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"   # uv, for cron/minimal environments

LIST="${1:?usage: sbatch --array=1-N%C slurm/reprocess_ocr_fix_array.sh <datelist.txt>}"
OUTPUT_DIR="${2:-output}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

DATE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
if [[ -z "${DATE}" ]]; then
  echo "[ocrfix] no date on line ${SLURM_ARRAY_TASK_ID} of ${LIST}; nothing to do."
  exit 0
fi

BASE="${OUTPUT_DIR}/${DATE}"

echo "=== concam OCR-fix reprocess: task ${SLURM_ARRAY_TASK_ID} -> ${DATE} ==="
echo "Repo:  ${REPO_DIR} @ $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Base:  ${BASE}"
echo "Node:  $(hostname)"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Refuse to run with pre-fix code: without the fix this would rewrite ocr.jsonl
# with the same corrupt dates and destroy the .pre-ocrfix baseline's meaning.
if ! grep -q "context-derived date" concam/pipeline/stages.py; then
  echo "[ocrfix] ERROR: concam/pipeline/stages.py lacks the OCR date-derivation fix." >&2
  exit 3
fi

if [[ ! -f "${BASE}/projections.jsonl" ]]; then
  echo "[ocrfix] ERROR: ${BASE}/projections.jsonl missing; this date needs the adsb+project stages first." >&2
  exit 2
fi

echo ""
echo "=== Step 1/5: back up pre-fix artifacts ==="
for f in ocr detections episodes; do
  src="${BASE}/${f}.jsonl"
  dst="${BASE}/${f}.pre-ocrfix.jsonl"
  if [[ -f "$src" && ! -f "$dst" ]]; then
    # cp -a (a real copy), never cp -al: the pipeline truncates and rewrites
    # these files in place, which would clobber a hardlinked "backup".
    cp -a "$src" "$dst"
    echo "[ocrfix] backed up $(basename "$src") -> $(basename "$dst")"
  elif [[ -f "$dst" ]]; then
    echo "[ocrfix] baseline $(basename "$dst") already exists; keeping the original pre-fix copy"
  else
    echo "[ocrfix] no $(basename "$src") to back up"
  fi
done

echo ""
echo "=== Step 2/5: regenerate OCR with the fix ==="
uv run python scripts/rerun_ocr_stage.py --date "${DATE}" --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Step 3/5: detect + aggregate + store (cached projections) ==="
uv run concam run --date "${DATE}" --output-dir "${OUTPUT_DIR}" --from-stage detect

echo ""
echo "=== Step 4/5: republish public bundle ==="
scripts/publish_public_date.sh "${DATE}"

echo ""
echo "=== Step 5/5: before/after report ==="
uv run python scripts/ocrfix_day_report.py --date "${DATE}" --output-dir "${OUTPUT_DIR}"

echo ""
echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
