#!/bin/bash
# Daily cron wrapper: detect contrails on the most recent finished daily
# timelapse(s), build the labeling bundle, and deploy it to the public reviewer
# site (~/public_html/concam/<date>/).
#
# Designed to run from cron at 01:00 local time. It is intentionally lightweight
# — it only checks the video and submits the heavy work to SLURM
# (slurm/pipeline_and_publish.sh, which runs `concam run` then
# scripts/publish_public_date.sh). Nothing CPU/RAM-heavy runs on the login node.
#
# It is IDEMPOTENT and self-healing:
#   - skips any date already published (manifest.json present) unless --force,
#   - skips a video that is missing, too small, or still being written, and
#   - looks back over the last LOOKBACK_DAYS days so a slow encode (video not
#     ready by 01:00) or a missed run is picked up on the next night.
# In the normal case only "yesterday" is new, so only it gets submitted.
#
# Usage:
#   scripts/daily_publish_cron.sh                # yesterday + lookback
#   scripts/daily_publish_cron.sh 2026-06-07     # one explicit date
#   scripts/daily_publish_cron.sh 2026-06-07 --force   # rebuild even if published
#   LOOKBACK_DAYS=5 scripts/daily_publish_cron.sh
#   DRY_RUN=1 scripts/daily_publish_cron.sh      # log decisions, submit nothing
#
# Crontab line (01:00 daily):
#   0 1 * * * /home/prash/contrails/mit-concam-pipeline/scripts/daily_publish_cron.sh \
#             >> /home/prash/contrails/mit-concam-pipeline/slurm/logs/daily_publish.cron.log 2>&1

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# cron runs with a minimal PATH that lacks uv (~/.local/bin) and the SLURM CLI.
# Put both on PATH so the submitted job (which SLURM seeds from this env) and the
# sbatch call below both resolve. The SLURM script also hardens this itself.
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

SBATCH="${SBATCH:-$(command -v sbatch || echo /usr/local/bin/sbatch)}"
VIDEO_ROOT="${VIDEO_ROOT:-/net/d16/data/contrail-camera}"
PUBLIC_ROOT="${PUBLIC_ROOT:-$HOME/public_html/concam}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-3}"
MIN_VIDEO_BYTES="${MIN_VIDEO_BYTES:-$((100 * 1024 * 1024))}"   # 100 MB floor
STABLE_SECS="${STABLE_SECS:-600}"                              # mtime must be >10 min old
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$REPO_DIR/slurm/logs"
ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] [daily_publish] $*"; }

FORCE=0
DATES=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) DATES+=("$arg") ;;
    *) log "WARN ignoring unrecognized arg: $arg" ;;
  esac
done

# Default set of dates: yesterday back through LOOKBACK_DAYS (most recent first).
if [[ ${#DATES[@]} -eq 0 ]]; then
  for ((i = 1; i <= LOOKBACK_DAYS; i++)); do
    DATES+=("$(date -d "-${i} day" +%Y-%m-%d)")
  done
fi

log "start: dates=[${DATES[*]}] force=$FORCE dry_run=$DRY_RUN"

submitted=0 skipped=0
for DATE in "${DATES[@]}"; do
  und="${DATE//-/_}"
  video="$VIDEO_ROOT/${und}_0000_2359.mp4"
  manifest="$PUBLIC_ROOT/$DATE/manifest.json"

  # 1. Idempotency: already deployed?
  if [[ -f "$manifest" && "$FORCE" -ne 1 ]]; then
    log "$DATE: already published — skip (use --force to rebuild)"; skipped=$((skipped + 1)); continue
  fi

  # 2. Video present?
  if [[ ! -f "$video" ]]; then
    log "$DATE: daily video not found ($video) — not ready, skip"; skipped=$((skipped + 1)); continue
  fi

  # 3. Looks complete: non-trivial size AND not modified recently (atomic-mv
  #    means a finished file is stable; the stability check guards a mid-write).
  size=$(stat -c %s "$video")
  if (( size < MIN_VIDEO_BYTES )); then
    log "$DATE: video only $size bytes (< $MIN_VIDEO_BYTES) — likely incomplete, skip"; skipped=$((skipped + 1)); continue
  fi
  age=$(( $(date +%s) - $(stat -c %Y "$video") ))
  if (( age < STABLE_SECS )); then
    log "$DATE: video modified ${age}s ago (< ${STABLE_SECS}s) — still writing?, defer"; skipped=$((skipped + 1)); continue
  fi

  # 4. Submit the heavy pipeline+publish job to SLURM.
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "$DATE: [dry-run] would submit: $SBATCH slurm/pipeline_and_publish.sh $DATE (video ok, ${size} bytes)"
    submitted=$((submitted + 1)); continue
  fi
  jobid=$("$SBATCH" --parsable slurm/pipeline_and_publish.sh "$DATE")
  log "$DATE: submitted SLURM job $jobid (pipeline + publish)"
  submitted=$((submitted + 1))
done

log "done: submitted=$submitted skipped=$skipped"
