#!/bin/bash
#SBATCH --job-name=concam-record-timelapse
#SBATCH --time=30-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/record-timelapse-%j.log

###############################################################################
# Concam RTSP recorder + nightly timelapse stitcher.
#
# This is a port of LAE_skycam/stream_save/slurm_save_stream.sh into the
# mit-concam-pipeline repo, with one substantive change: we no longer pass
# `-use_wallclock_as_timestamps 1` to the recorder ffmpeg. That flag caused
# the daily timelapses to show a non-uniform OSD-second pattern after
# `fps=1/1` decimation (a cyclic permutation of {0, 1, 1, 2}).
#
# Empirical proof and full root-cause writeup: docs/pts_drift_bug.md
# Reproducer (parallel substream captures): scripts/pts_drift_test/
#
# What this script does, structurally:
#   * Runs two child processes:
#       (1) `run_raw_record` — long-lived ffmpeg recording the RTSP main
#           stream into 1-hour H.265 segments, rotating on the host clock
#           (segment_atclocktime + strftime); auto-restarts on ffmpeg exit.
#       (2) `run_transcoder` — once a day at midnight, concats the previous
#           day's hourly segments and re-encodes to AV1 with a 30 fps / 1 fps-
#           sampled timelapse filter; cleans up raw segments on success,
#           quarantines them on failure.
#   * Traps SIGTERM/SIGINT/SIGHUP and signals both children to drain.
#
# Cutover from the legacy LAE_skycam job (566007 yt_stream_timelapse):
#   See scripts/cutover_recorder.sh for the SIGTERM-then-SIGKILL helper.
#   Output paths are intentionally identical to the legacy script so the
#   daily-timelapse stitcher and ConCam frontend pick up segments seamlessly
#   across the cutover.
###############################################################################

set -uo pipefail
umask 0002

###############################################################################
# 0 · Configuration
###############################################################################
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RTSP_URL="${RTSP_URL:-rtsp://admin2:Contrails@10.83.0.139/Preview_01_main}"
WORKDIR="${WORKDIR:-/net/d16/data/contrail-camera}"
RAW_DIR="${RAW_DIR:-/net/d16/data/contrail-camera/raw_segments_clean}"
FAILED_DIR="${FAILED_DIR:-/net/d16/data/contrail-camera/failed_raw_clean}"
LEGACY_RAW_DIRS_CSV="${LEGACY_RAW_DIRS_CSV:-}"
LEGACY_RAW_DIRS=()
if [[ -n "$LEGACY_RAW_DIRS_CSV" ]]; then
  IFS=':' read -r -a LEGACY_RAW_DIRS <<< "$LEGACY_RAW_DIRS_CSV"
fi

# We keep raw segments at 1 hour (3600s) for safety.
# If a file corrupts, you only lose 1 hour instead of 2.
SEG_TIME="${SEG_TIME:-3600}"

# Daily timelapse processing time: Midnight only
PROCESS_HOUR="${PROCESS_HOUR:-00}"

# Limit length of videos to 24 hours (just in case)
MAX_HMS="${MAX_HMS:-23:59:59.900}"

# How long to sleep after processing (5 mins is safe)
POST_PROCESS_SLEEP="${POST_PROCESS_SLEEP:-300}"

# Timelapse terminology:
# - sample period: how often a fresh source frame is kept from the camera feed
# - playback fps:  the frame rate of the final MP4
TIMELAPSE_SAMPLE_PERIOD_SECONDS="${TIMELAPSE_SAMPLE_PERIOD_SECONDS:-1}"
TIMELAPSE_PLAYBACK_FPS="${TIMELAPSE_PLAYBACK_FPS:-30}"
DEFAULT_TIMELAPSE_FILTER="fps=1/${TIMELAPSE_SAMPLE_PERIOD_SECONDS},settb=AVTB,setpts=N/(${TIMELAPSE_PLAYBACK_FPS}*TB),fps=${TIMELAPSE_PLAYBACK_FPS}"
TIMELAPSE_FILTER="${TIMELAPSE_FILTER:-$DEFAULT_TIMELAPSE_FILTER}"
DAILY_VIDEO_CODEC="${DAILY_VIDEO_CODEC:-libsvtav1}"
DAILY_VIDEO_PRESET="${DAILY_VIDEO_PRESET:-12}"
DAILY_VIDEO_CRF="${DAILY_VIDEO_CRF:-35}"
STALE_LOCK_MINUTES="${STALE_LOCK_MINUTES:-360}"

ensure_dir() {
  local dir="$1"
  mkdir -p "$dir"
  chmod g+rws "$dir" 2>/dev/null || true
}

assert_dir_writable() {
  local dir="$1"
  local write_test="$dir/.write_test.$$"

  if ! : > "$write_test" 2>/dev/null; then
    echo "$(date) [INIT] ERROR: Cannot write to $dir" >&2
    return 1
  fi

  rm -f "$write_test"
}

format_duration() {
  local total_seconds="$1"
  awk -v total="$total_seconds" 'BEGIN {
    hours = int(total / 3600)
    minutes = int((total - (hours * 3600)) / 60)
    seconds = total - (hours * 3600) - (minutes * 60)
    printf "%02dh %02dm %05.2fs", hours, minutes, seconds
  }'
}

probe_duration_seconds() {
  ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null || true
}

RAW_SEARCH_DIRS=("$RAW_DIR")
for legacy_dir in "${LEGACY_RAW_DIRS[@]}"; do
  if [[ -d "$legacy_dir" ]]; then
    RAW_SEARCH_DIRS+=("$legacy_dir")
  fi
done

ensure_dir "$WORKDIR"
ensure_dir "$RAW_DIR"
ensure_dir "$FAILED_DIR"

assert_dir_writable "$WORKDIR" || exit 1
assert_dir_writable "$RAW_DIR" || exit 1
assert_dir_writable "$FAILED_DIR" || exit 1

# Activate conda environment with ffmpeg
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate skycam_processing

echo "Starting FFmpeg recording... Job ID: ${SLURM_JOB_ID:-<no-slurm>}"
echo "Saving timelapse segments to: $WORKDIR"
echo "Writing raw camera segments to: $RAW_DIR"
echo "Daily timelapse sample period: 1 frame every ${TIMELAPSE_SAMPLE_PERIOD_SECONDS}s"
echo "Daily timelapse playback rate: ${TIMELAPSE_PLAYBACK_FPS} fps"
echo "Daily timelapse filter: $TIMELAPSE_FILTER"
if [[ ${#RAW_SEARCH_DIRS[@]} -gt 1 ]]; then
  echo "Searching legacy raw directories for catchup: ${RAW_SEARCH_DIRS[*]}"
fi

###############################################################################
# 1 · Define Functions for Each Task
###############################################################################

# --- Function 1: The Raw Recording Process ---
#
# IMPORTANT: do NOT add `-use_wallclock_as_timestamps 1` here. Letting the
# camera's RTSP-side PTS (= camera RTC, which is what renders the burned-in
# OSD overlay) flow through is what keeps the OSD-second deltas at exactly 1
# after the timelapse stitcher applies fps=1/1. See docs/pts_drift_bug.md.
run_raw_record() {
  echo "[RECORDER] Starting raw recording loop."
  local restart_count=0

  while true; do
    echo "[RECORDER] Starting FFmpeg (restart #$restart_count)..."

    ffmpeg \
      -hide_banner -loglevel warning \
      -err_detect ignore_err \
      -rtsp_transport tcp \
      -timeout 10000000 \
      -thread_queue_size 1024 \
      -reorder_queue_size 1024 \
      -fflags +genpts+igndts+discardcorrupt \
      -avoid_negative_ts make_zero \
      -i "$RTSP_URL" \
      -map 0:v:0 -map 0:a:0 \
      -c copy \
      -f segment \
      -segment_time "$SEG_TIME" \
      -segment_atclocktime 1 \
      -reset_timestamps 1 \
      -strftime 1 \
      "$RAW_DIR/%Y-%m-%d_%H-00-00.mp4"

    local exit_code=$?
    restart_count=$((restart_count + 1))
    echo "$(date) [RECORDER] FFmpeg exited with code $exit_code — restarting in 5s… (restart #$restart_count)" >&2
    sleep 5
  done
}

# Helper function to process a full day (00:00 - 23:00)
process_full_day() {
    local process_date="$1"

    local output_file="$WORKDIR/${process_date}_0000_2359.mp4"
    local temp_file="$WORKDIR/${process_date}_0000_2359_temp.mp4"
    local lockfile="$WORKDIR/${process_date}_0000_2359_processing.lock"

    # Skip if the final output already exists.
    if [[ -f "$output_file" ]]; then
        echo "$(date) [TRANSCODER] Timelapse for $process_date already exists, skipping."
        return
    fi

    # Recover from stale lockfiles left behind by crashes or manual job kills.
    if [[ -f "$lockfile" ]]; then
        local lock_age_minutes
        lock_age_minutes=$(awk -v now="$(date +%s)" -v ts="$(stat -c %Y "$lockfile")" 'BEGIN { printf "%.0f", (now - ts) / 60 }')
        if (( lock_age_minutes > STALE_LOCK_MINUTES )); then
            echo "$(date) [TRANSCODER] Removing stale lockfile for $process_date ($lock_age_minutes minutes old)."
            rm -f "$lockfile"
        else
            echo "$(date) [TRANSCODER] Active lockfile found for $process_date, skipping."
            return
        fi
    fi

    echo "$(date) [TRANSCODER] Creating 24h timelapse for $process_date"
    touch "$lockfile"

    # Find all raw files from 00 to 23 hours
    local file_list=()
    local total_valid_seconds="0"
    local hour_pattern=""
    declare -A seen_segment_names=()

    shopt -s nullglob
    for hour in $(seq -w 00 23); do
        hour_pattern="${process_date//_/-}_${hour}-*.mp4"
        for search_dir in "${RAW_SEARCH_DIRS[@]}"; do
            for raw_file in "$search_dir"/$hour_pattern; do
                local segment_name
                local duration

                segment_name=$(basename "$raw_file")
                if [[ -n "${seen_segment_names[$segment_name]+x}" ]]; then
                    continue
                fi
                seen_segment_names["$segment_name"]=1

                # Verify file integrity before adding to list.
                duration=$(probe_duration_seconds "$raw_file")
                if [[ -n "$duration" ]] && [[ $(echo "$duration >= 5" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
                    file_list+=("$raw_file")
                    total_valid_seconds=$(awk -v total="$total_valid_seconds" -v add="$duration" 'BEGIN { printf "%.3f", total + add }')
                else
                    echo "$(date) [TRANSCODER] Skipping corrupted/short file: $raw_file (duration: ${duration:-unknown})"
                fi
            done
        done
    done
    shopt -u nullglob

    if [[ ${#file_list[@]} -eq 0 ]]; then
        echo "$(date) [TRANSCODER] No valid raw files found for $process_date"
        rm "$lockfile"
        return
    fi

    echo "$(date) [TRANSCODER] Found ${#file_list[@]} valid files for $process_date totalling $(format_duration "$total_valid_seconds"). Creating daily AV1 output..."

    # Create a temporary file list for ffmpeg concat
    local filelist_txt="$WORKDIR/${process_date}_filelist.txt"
    > "$filelist_txt"  # Clear the file

    # Sort files by timestamp and add to concat list
    printf "%s\n" "${file_list[@]}" | sort | while read -r file; do
        echo "file '$(realpath "$file")'" >> "$filelist_txt"
    done

    # Create the daily output.
    if ffmpeg -hide_banner -loglevel error \
        -f concat -safe 0 -i "$filelist_txt" \
        -to "$MAX_HMS" \
        -vf "$TIMELAPSE_FILTER" \
        -c:v "$DAILY_VIDEO_CODEC" -preset "$DAILY_VIDEO_PRESET" -crf "$DAILY_VIDEO_CRF" \
        -pix_fmt yuv420p \
        -max_muxing_queue_size 9999 \
        -an \
        -avoid_negative_ts make_zero \
        -movflags +faststart \
        "$temp_file"; then

        # Move to final location
        mv "$temp_file" "$output_file"
        echo "$(date) [TRANSCODER] Successfully created daily AV1 output: $output_file"

        # Clean up raw files after successful processing
        for file in "${file_list[@]}"; do
            rm "$file"
        done
        rm "$filelist_txt"
        rm "$lockfile"

    else
        # --- QUARANTINE LOGIC ---
        echo "$(date) [TRANSCODER] ERROR: Failed to create timelapse for $process_date" >&2
        rm -f "$temp_file"
        rm -f "$filelist_txt"
        rm "$lockfile"

        echo "$(date) [TRANSCODER] Moving failed raw files to quarantine: $FAILED_DIR"
        local quarantine_dir="$FAILED_DIR/${process_date}"
        mkdir -p "$quarantine_dir"

        # Move all the files that were part of this failed attempt
        for file in "${file_list[@]}"; do
            if [[ -f "$file" ]]; then
                mv "$file" "$quarantine_dir/"
            fi
        done
    fi
}

# --- Function 2: The Daily Timelapse Creator with Catchup ---
run_transcoder() {
    echo "[TRANSCODER] Starting 24-hour timelapse creator - will process at 00:00 for the previous day."

    # 1. Startup Catchup: Check if we missed Yesterday's processing
    echo "[TRANSCODER] Checking for missed processing on startup..."
    local yesterday_date=$(date -d "yesterday" +%Y_%m_%d)
    local yesterday_file="$WORKDIR/${yesterday_date}_0000_2359.mp4"

    if [[ ! -f "$yesterday_file" ]]; then
        echo "[TRANSCODER] CATCHUP: Processing missed full day for $yesterday_date"
        process_full_day "$yesterday_date"
    else
        echo "[TRANSCODER] Previous day ($yesterday_date) already processed."
    fi

    echo "[TRANSCODER] Startup catchup complete. Waiting for Midnight..."

    # 2. Main Loop
    while true; do
        local current_hour=$(date +%H)
        local current_minute=$(date +%M)
        local current_time="${current_hour}:${current_minute}"

        echo "$(date) [TRANSCODER] Current time: $current_time, looking for $PROCESS_HOUR:00-29"

        # Check if it is Midnight (00:00 - 00:29)
        if [[ "$current_hour" == "$PROCESS_HOUR" && 10#$current_minute -lt 30 ]]; then

            # At midnight, we process YESTERDAY's data
            local process_date=$(date -d "yesterday" +%Y_%m_%d)

            echo "$(date) [TRANSCODER] Midnight detected. Processing data for: $process_date"
            process_full_day "$process_date"

            # Sleep to avoid re-processing this window
            echo "$(date) [TRANSCODER] Processing complete. Sleeping for $POST_PROCESS_SLEEP seconds..."
            sleep "$POST_PROCESS_SLEEP"

        else
            # Maintenance: Quarantine very old files (older than 2 days)
            # This handles files that might have been skipped or left over from crashes
            for search_dir in "${RAW_SEARCH_DIRS[@]}"; do
                while IFS= read -r stale_file; do
                    local quarantine_subdir="$FAILED_DIR/old_unprocessed"
                    echo "$(date) [TRANSCODER] Quarantining old unprocessed file: $stale_file"
                    mkdir -p "$quarantine_subdir"
                    mv "$stale_file" "$quarantine_subdir/"
                done < <(find "$search_dir" -maxdepth 1 -type f -name "*.mp4" -mtime +2 -print 2>/dev/null)
            done

            # Sleep 5 minutes before checking time again
            sleep 300
        fi
    done
}


###############################################################################
# 2 · Cleanup and Process Management
###############################################################################

pids=()
cleanup() {
  echo "Caught signal. Shutting down all processes..."
  for pid in "${pids[@]}"; do
    pkill -P "$pid"
    kill "$pid" 2>/dev/null
  done
  echo "Cleanup complete. Exiting."
  exit 0
}
trap cleanup SIGINT SIGTERM SIGHUP

###############################################################################
# 3 · Main Execution
###############################################################################

run_raw_record &
pids+=($!)

run_transcoder &
pids+=($!)

echo "All processes started. PIDs: ${pids[*]}"
echo "Script is running. Waiting for signal to terminate..."

# Wait for BOTH processes to exit
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All processes exited. Shutting down."
cleanup
