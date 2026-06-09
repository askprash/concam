#!/bin/bash
# Daily cron wrapper: publish the labeling bundle for the most recent day(s)
# whose ADS-B flight data has become available.
#
# ADS-B (feder) data is NOT real-time — it lags the live date by a few days, and
# the pipeline cannot run a day until feder has that day's flights. So the gating
# question is "what is the latest day feder has data for?", NOT "is yesterday's
# video done." This script:
#   1. asks feder for the available days (scripts/feder_available_days.py),
#   2. for each feder-available day in a lookback window (newest first), checks
#      the daily timelapse video is present + complete, and that the date is not
#      already deployed, then
#   3. submits slurm/pipeline_and_publish.sh <date> (concam run -> publish), so
#      the heavy work runs on the cluster, not the login node.
#
# IDEMPOTENT + self-healing: skips dates already published (manifest.json present)
# unless --force, skips videos that are missing/too-small/still-writing, and the
# lookback window means a day is picked up automatically on the first night after
# feder catches up to it. In steady state only the newest just-available day is new.
#
# Usage:
#   scripts/daily_publish_cron.sh                  # feder-available days in window
#   scripts/daily_publish_cron.sh 2026-06-03       # one explicit date
#   scripts/daily_publish_cron.sh 2026-06-03 --force   # rebuild even if published
#   LOOKBACK_DAYS=21 scripts/daily_publish_cron.sh
#   DRY_RUN=1 scripts/daily_publish_cron.sh        # log decisions, submit nothing
#
# Crontab line (01:00 daily):
#   0 1 * * * /home/prash/contrails/mit-concam-pipeline/scripts/daily_publish_cron.sh \
#             >> /home/prash/contrails/mit-concam-pipeline/slurm/logs/daily_publish.cron.log 2>&1

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# cron runs with a minimal PATH that lacks uv (~/.local/bin) and the SLURM CLI.
# Put both on PATH so the feder query, the sbatch call, and the submitted job
# (which SLURM seeds from this env) all resolve. The SLURM script hardens uv too.
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

SBATCH="${SBATCH:-$(command -v sbatch || echo /usr/local/bin/sbatch)}"
VIDEO_ROOT="${VIDEO_ROOT:-/net/d16/data/contrail-camera}"
PUBLIC_ROOT="${PUBLIC_ROOT:-$HOME/public_html/concam}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-14}"
FEDER_MARGIN_DAYS="${FEDER_MARGIN_DAYS:-0}"   # hold back the newest N feder days
                                              # (guard if the latest day's ADS-B
                                              # is only partially ingested)
MIN_VIDEO_BYTES="${MIN_VIDEO_BYTES:-$((100 * 1024 * 1024))}"   # 100 MB floor
STABLE_SECS="${STABLE_SECS:-600}"                              # mtime must be >10 min old
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$REPO_DIR/slurm/logs"
ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] [daily_publish] $*"; }

FORCE=0
EXPLICIT_DATES=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) EXPLICIT_DATES+=("$arg") ;;
    *) log "WARN ignoring unrecognized arg: $arg" ;;
  esac
done

# Candidate dates: explicit args if given, else the feder-available days in the
# lookback window (newest first). The latter is the normal cron path and is the
# whole point — we never consider a day feder has no flights for.
DATES=()
if [[ ${#EXPLICIT_DATES[@]} -gt 0 ]]; then
  DATES=("${EXPLICIT_DATES[@]}")
  log "start: explicit dates=[${DATES[*]}] force=$FORCE dry_run=$DRY_RUN"
else
  latest=$(uv run python scripts/feder_available_days.py --latest 2>/dev/null || true)
  if ! mapfile -t DATES < <(uv run python scripts/feder_available_days.py --within "$LOOKBACK_DAYS" 2>/dev/null) || [[ ${#DATES[@]} -eq 0 ]]; then
    log "ERROR: could not determine feder ADS-B availability — feder store down or empty. Exiting."
    exit 1
  fi
  log "start: latest feder day=${latest:-?}; ${#DATES[@]} feder-available day(s) in ${LOOKBACK_DAYS}d window; margin=${FEDER_MARGIN_DAYS}d; force=$FORCE dry_run=$DRY_RUN"
  # Optional guard: drop the newest FEDER_MARGIN_DAYS days (latest day's ADS-B may
  # still be ingesting). cutoff = latest - margin; keep only days <= cutoff.
  if [[ -n "${latest:-}" && "$FEDER_MARGIN_DAYS" -gt 0 ]]; then
    cutoff=$(date -d "$latest -${FEDER_MARGIN_DAYS} day" +%F)
    kept=()
    for d in "${DATES[@]}"; do [[ "$d" > "$cutoff" ]] && log "$d: within ${FEDER_MARGIN_DAYS}d feder margin — defer" || kept+=("$d"); done
    DATES=("${kept[@]}")
  fi
fi

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
    log "$DATE: daily video not found ($video) — skip"; skipped=$((skipped + 1)); continue
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
    log "$DATE: [dry-run] would submit: $SBATCH slurm/pipeline_and_publish.sh $DATE (feder ok, video ${size} bytes)"
    submitted=$((submitted + 1)); continue
  fi
  jobid=$("$SBATCH" --parsable slurm/pipeline_and_publish.sh "$DATE")
  log "$DATE: submitted SLURM job $jobid (pipeline + publish)"
  submitted=$((submitted + 1))
done

log "done: submitted=$submitted skipped=$skipped"
