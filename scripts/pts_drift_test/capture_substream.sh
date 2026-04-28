#!/bin/bash
#SBATCH --job-name=concam-pts-capture
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/pts-capture-%x-%j.log

# Capture 1 hour of the camera substream via RTSP, with or without
# -use_wallclock_as_timestamps 1, to verify whether that flag is the
# source of the OSD-vs-PTS drift that produces the 1-1-0-2 pattern in
# the daily timelapses.
#
# Usage: sbatch -J concam-pts-native    scripts/pts_drift_test/capture_substream.sh native
#        sbatch -J concam-pts-wallclock scripts/pts_drift_test/capture_substream.sh wallclock

set -euo pipefail

MODE="${1:?Usage: capture_substream.sh native|wallclock}"
RTSP_URL="${RTSP_URL:-rtsp://admin2:Contrails@10.83.0.139/Preview_01_sub}"
OUT_DIR="${OUT_DIR:-scratch/pts_drift_test}"
DURATION="${DURATION:-3600}"

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/sub_${MODE}_pts.mp4"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate skycam_processing

case "$MODE" in
  native)
    EXTRA_INPUT_FLAGS=()
    ;;
  wallclock)
    EXTRA_INPUT_FLAGS=(-use_wallclock_as_timestamps 1 -avoid_negative_ts make_zero)
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac

echo "=== concam PTS-drift capture ==="
echo "Mode:     $MODE"
echo "URL:      $RTSP_URL"
echo "Duration: ${DURATION}s"
echo "Output:   $OUT"
echo "Extra input flags: ${EXTRA_INPUT_FLAGS[*]:-<none>}"
echo "Started at $(date -u +%FT%TZ) on $(hostname)"
date -u +%s > "$OUT.start_unix_utc"

ffmpeg -hide_banner -loglevel warning \
  -rtsp_transport tcp \
  "${EXTRA_INPUT_FLAGS[@]}" \
  -fflags +genpts+igndts+discardcorrupt \
  -i "$RTSP_URL" \
  -map 0:v:0 -c copy -t "$DURATION" \
  "$OUT"

date -u +%s > "$OUT.end_unix_utc"
echo "Finished at $(date -u +%FT%TZ)"
ls -la "$OUT"
