#!/bin/bash
# Graceful cutover from the legacy LAE_skycam recorder job to the
# in-repo slurm/record_and_timelapse.sh job.
#
# Why a helper:
#   `scancel <jobid>` sends SIGTERM and, after a grace period, SIGKILL.
#   If ffmpeg gets SIGKILL'd mid-segment, the in-progress hourly segment
#   may have an unwritten moov atom and be unplayable. We want SIGTERM
#   only, with a hand-managed grace window long enough for the segment
#   muxer to finalize, and only escalate to SIGKILL if SIGTERM is ignored.
#
# What this script does:
#   1. scancel --signal=TERM <OLD_JOB_ID>            (graceful)
#   2. wait up to GRACE_SECONDS for the job to disappear from squeue
#   3. if it's still alive, scancel --signal=KILL    (hard)
#   4. submit the new job: sbatch slurm/record_and_timelapse.sh
#   5. tail the new job's log so you can watch the first segment land
#
# Usage:
#   scripts/cutover_recorder.sh <OLD_JOB_ID>
#
# Tip: run this within the first few minutes of an hour (e.g. just past
# the top of the hour) so the segment that gets cut short is small.

set -euo pipefail

OLD_JOB="${1:?Usage: cutover_recorder.sh <OLD_JOB_ID>}"
GRACE_SECONDS="${GRACE_SECONDS:-30}"
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "=== concam recorder cutover ==="
echo "Old job:  $OLD_JOB"
echo "Repo:     $REPO_DIR"
echo "Time:     $(date)"
echo

# 0. Sanity-check the old job exists and is one of ours.
if ! squeue -j "$OLD_JOB" -h -u "$USER" >/dev/null 2>&1; then
  echo "ERROR: Job $OLD_JOB not found in your squeue. Aborting." >&2
  exit 1
fi
old_name=$(squeue -j "$OLD_JOB" -h -o "%j" 2>/dev/null || true)
echo "Old job name: $old_name"
echo

# 1. SIGTERM.
echo "[1/4] Sending SIGTERM to job $OLD_JOB ..."
scancel --signal=TERM "$OLD_JOB"

# 2. Poll for graceful exit.
echo "[2/4] Waiting up to ${GRACE_SECONDS}s for graceful shutdown ..."
elapsed=0
while (( elapsed < GRACE_SECONDS )); do
  if ! squeue -j "$OLD_JOB" -h -u "$USER" >/dev/null 2>&1; then
    echo "       job $OLD_JOB exited cleanly after ${elapsed}s"
    break
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

# 3. Force-kill if still alive.
if squeue -j "$OLD_JOB" -h -u "$USER" >/dev/null 2>&1; then
  echo "[3/4] Job $OLD_JOB still alive after ${GRACE_SECONDS}s. Forcing SIGKILL ..."
  scancel "$OLD_JOB"
  sleep 5
  if squeue -j "$OLD_JOB" -h -u "$USER" >/dev/null 2>&1; then
    echo "ERROR: Job $OLD_JOB refused to die. Aborting before submitting new job." >&2
    exit 2
  fi
else
  echo "[3/4] Force-kill not needed."
fi

# 4. Submit new job.
echo "[4/4] Submitting new recorder job ..."
cd "$REPO_DIR"
mkdir -p slurm/logs
submit_out=$(sbatch slurm/record_and_timelapse.sh)
echo "       $submit_out"
NEW_JOB=$(echo "$submit_out" | awk '{print $4}')

echo
echo "=== Done. Old job: $OLD_JOB (terminated). New job: $NEW_JOB (submitted) ==="
echo
echo "Watch the new job pick up the stream:"
echo "  tail -F slurm/logs/record-timelapse-${NEW_JOB}.log"
