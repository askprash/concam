#!/bin/bash
#SBATCH --job-name=concam-faststart
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/faststart-%A_%a.log
###############################################################################
# Rewrite daily timelapse MP4s in place so the moov box precedes mdat
# ("faststart"), letting a browser start playback without first range-fetching
# the tail of a multi-GB file.
#
# The remux is LOSSLESS: `-map 0 -c copy` re-orders container boxes only, it
# never re-encodes.  The recorder has emitted faststart dailies since the
# `-movflags +faststart` cutover (2026-06-28 onward); this backfills the
# earlier ones.
#
# Written IN PLACE, deliberately: the dailies are the only copy of the footage
# (the recorder deletes the hourly segments after the daily encode), and a
# proxy copy would duplicate ~1.2 TB.  In-place is made safe by never touching
# the original until a full-fidelity temp file has been verified:
#
#   1. probe the source (codec, dimensions, frame count)
#   2. remux to a temp file in the same directory (same filesystem -> the later
#      mv is an atomic rename, not a copy)
#   3. verify the temp file: same codec/dimensions/frame count, and moov now
#      precedes mdat
#   4. only then rename temp over the original, and restore group/permissions
#
# Any failure removes the temp file and leaves the original untouched.
# Idempotent: an already-faststart file is skipped, so the array can be
# resubmitted for failed tasks.
#
# Usage:
#   sbatch --array=1-<N>%<conc> slurm/faststart_remux_array.sh <filelist.txt>
# where filelist.txt holds one absolute .mp4 path per line.
###############################################################################
set -euo pipefail

LIST="${1:?usage: sbatch --array=1-N%C slurm/faststart_remux_array.sh <filelist.txt>}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

SRC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
if [[ -z "${SRC}" ]]; then
  echo "[faststart] no path on line ${SLURM_ARRAY_TASK_ID} of ${LIST}; nothing to do."
  exit 0
fi

echo "=== concam faststart remux: task ${SLURM_ARRAY_TASK_ID} ==="
echo "File:  ${SRC}"
echo "Node:  $(hostname)"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -f "${SRC}" ]]; then
  echo "[faststart] ERROR: source does not exist: ${SRC}" >&2
  exit 1
fi

# ffmpeg/ffprobe live in the recorder's conda environment, not on the default
# PATH (the same env slurm/record_and_timelapse.sh activates).
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate skycam_processing

# Guard against racing the recorder: it writes the daily for a date to a temp
# file and moves it into place, but refuse anything freshly modified anyway.
AGE_SECONDS=$(( $(date +%s) - $(stat -c %Y "${SRC}") ))
if (( AGE_SECONDS < 86400 )); then
  echo "[faststart] SKIP: ${SRC} was modified $((AGE_SECONDS / 3600))h ago; the recorder may still be writing it." >&2
  exit 0
fi

# Already fixed?  Nothing to do (makes resubmission cheap and safe).
if python3 scripts/mp4_faststart_check.py "${SRC}" >/dev/null 2>&1; then
  echo "[faststart] SKIP: ${SRC} already has moov before mdat."
  exit 0
fi

probe() {
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,nb_frames \
    -of default=noprint_wrappers=1:nokey=1 "$1" | tr '\n' ' '
}

SRC_PROBE=$(probe "${SRC}")
SRC_BYTES=$(stat -c %s "${SRC}")
echo "[faststart] source: ${SRC_PROBE} (${SRC_BYTES} bytes)"
if [[ -z "${SRC_PROBE// /}" ]]; then
  echo "[faststart] ERROR: could not probe a video stream in ${SRC}; leaving it alone." >&2
  exit 1
fi

TMP="$(dirname "${SRC}")/.faststart.$(basename "${SRC}").${SLURM_JOB_ID:-$$}.tmp.mp4"
cleanup() { rm -f "${TMP}"; }
trap cleanup EXIT

echo "[faststart] remuxing -> ${TMP}"
ffmpeg -hide_banner -loglevel error -nostdin \
  -i "${SRC}" -map 0 -c copy -movflags +faststart "${TMP}"

DST_PROBE=$(probe "${TMP}")
DST_BYTES=$(stat -c %s "${TMP}")
echo "[faststart] output: ${DST_PROBE} (${DST_BYTES} bytes)"

# --- Verification gates: any failure aborts and leaves the original intact ---
if [[ "${SRC_PROBE}" != "${DST_PROBE}" ]]; then
  echo "[faststart] ERROR: stream mismatch. src='${SRC_PROBE}' dst='${DST_PROBE}'. Original untouched." >&2
  exit 1
fi

# A box re-order changes size only by the moov/free shuffle; a large delta means
# something was dropped.  1% of the source is far more than that shuffle costs.
MIN_BYTES=$(( SRC_BYTES - SRC_BYTES / 100 ))
if (( DST_BYTES < MIN_BYTES )); then
  echo "[faststart] ERROR: output ${DST_BYTES} < 99% of source ${SRC_BYTES}. Original untouched." >&2
  exit 1
fi

if ! python3 scripts/mp4_faststart_check.py "${TMP}"; then
  echo "[faststart] ERROR: remuxed file is still not faststart. Original untouched." >&2
  exit 1
fi

# --- Commit: atomic rename within the same directory/filesystem -------------
# Capture the original mode so a file recorded by another account keeps its
# group-readable permissions after we replace it.
SRC_MODE=$(stat -c %a "${SRC}")
mv -f "${TMP}" "${SRC}"
trap - EXIT
chgrp contrailcam "${SRC}" 2>/dev/null || echo "[faststart] WARN: could not chgrp ${SRC}" >&2
chmod "${SRC_MODE}" "${SRC}" 2>/dev/null || echo "[faststart] WARN: could not chmod ${SRC}" >&2

echo "[faststart] committed: $(python3 scripts/mp4_faststart_check.py "${SRC}")"
echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
