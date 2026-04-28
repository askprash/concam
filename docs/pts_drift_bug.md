# Daily-timelapse OSD irregularity: root cause and fix

## What the collaborator reported

Looking at `2026_04_01_0000_2359.mp4`, the burned-in timestamp on consecutive
frames advanced as `1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 1, …` instead of the
expected `1, 1, 1, 1, …` (one frame per real-world second).  Spot-checks on
later April days show the same multiset `{0, 1, 1, 2}` of deltas with
varying phase and varying severity.

The bug is **strongly intermittent** — most legacy days look clean for most
hours, with the cyclic pattern manifesting in a few isolated 5–60 minute
windows scattered across the day.  Sample by hour rather than by day; a
single window picked at the wrong time can falsely report "no bug."

## Where the bug lives

`LAE_skycam/stream_save/slurm_save_stream.sh` is the RTSP recorder that
writes the hourly raw segments which the daily-timelapse generator
consumes.  The recorder invokes ffmpeg with **two flags whose interaction
is the bug**:

```
ffmpeg -rtsp_transport tcp \
       -use_wallclock_as_timestamps 1 \
       -fflags +genpts+igndts+discardcorrupt \
       -i $RTSP_URL ...
```

and the timelapse filter chain decimates with `fps=1/1` (nearest-PTS).

`-use_wallclock_as_timestamps 1` overrides each packet's demuxed PTS with
the recording host's wallclock at the moment the packet arrived.  After
that, `fps=1/1` picks the input frame whose **host-wallclock PTS** is
closest to each integer-second target.  But the burned-in OSD is rendered
from the **camera's RTC**, which drifts independently of the host clock.
The two clocks are good to ~100 ms over an hour but they tick on different
sub-second boundaries — so the picked frames' OSD seconds wobble.

## Empirical proof (1-hour controlled test, 2026-04-27)

Two parallel SLURM captures of the same camera substream, started in the
same second, identical except for the wallclock flag:

| Capture                 | flag                                | end-of-hour PTS-vs-OSD drift | OCR-validated slope |
|-------------------------|-------------------------------------|------------------------------|---------------------|
| `sub_native_pts.mp4`    | (none)                              | **+55 ms**                   | +1.0 × 10⁻⁵ s/s     |
| `sub_wallclock_pts.mp4` | `-use_wallclock_as_timestamps 1`    | **−75 ms**                   | −7.4 × 10⁻⁶ s/s     |

Apply ffmpeg `fps=1/1` to each and inspect the OSD-second delta of every
consecutive output-frame pair (using native PTS as ground-truth for
camera-RTC seconds):

```
native     n=3599  histogram = {1: 3599}
wallclock  n=3599  histogram = {0: 900, 1: 1800, 2: 899}

native deltas    : 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
wallclock deltas : 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, ...
```

The wallclock cycle `0, 1, 1, 2` is exactly the same multiset
`{0, 1, 1, 2}` the collaborator observed in the daily timelapse — different
starting phase, same arithmetic.  Phase depends on where the camera-RTC
tick boundary lands at the start of the recording.

Total OSD time covered by the 3599 wallclock-decimated outputs:
0×900 + 1×1800 + 2×899 = 3598 s.  So the picker is approximately right
*on average* (within one second per hour), but every individual second
chooses a frame from the wrong OSD-second neighborhood, producing the
visible jitter.

Why the bug is invisible to the rest of the pipeline:
`concam.ocr.tracker.TrustButVerifyTracker` reanchors after a few
consistent OCR reads and projects through `is_stuck` (delta=0) anomalies,
so the per-flight-track timeline downstream is smoothed.  Detection /
overlay output is correct.  The problem is purely cosmetic — but very
visible to anyone scrubbing the daily timelapse.

## Recommended fix

Drop the wallclock flag from the recorder and trust the camera-side PTS:

```diff
  ffmpeg -rtsp_transport tcp \
-        -use_wallclock_as_timestamps 1 \
         -fflags +genpts+igndts+discardcorrupt \
         -i "$RTSP_URL" ...
```

`-fflags +genpts` already covers the missing-PTS case; the camera Reolink
substream emits well-formed PTS that match the OSD clock to within the
~10 ms RMS jitter we measured in step 1.  After this change, the daily
timelapse's OSD will tick `1, 1, 1, 1, …` for the whole day.

## What was actually deployed (2026-04-27)

The fix was deployed by porting the recorder into this repo at
`slurm/record_and_timelapse.sh` (verbatim copy of
`LAE_skycam/stream_save/slurm_save_stream.sh` with the wallclock flag
removed and the SBATCH job renamed away from the misleading
`yt_stream_timelapse`).  The legacy LAE_skycam SLURM job was cancelled
(graceful SIGTERM) and the new in-repo job submitted via
`scripts/cutover_recorder.sh`.  Cutover happened at ~16:08 EDT, costing
~8 minutes of recording from the day (the new job's ffmpeg truncated the
in-progress `2026-04-27_16-00-00.mp4` segment when it opened the same
filename).

## Verification (2026-04-28, on the 2026-04-27 timelapse)

The 2026-04-27 daily timelapse spans the cutover, so its first half
(00:00–16:00 EDT) comes from the legacy recorder and its second half
(16:08–23:59 EDT) from the in-repo recorder.  An hourly OCR scan via
`scripts/pts_drift_test/scan_one_day.py` (300-frame window per hour at
100 % yield) shows:

| range                | hours covered | bug pattern present in   |
|----------------------|---------------|--------------------------|
| 00:00–15:59 (legacy) | 16 hours      | 00:00 and 04:00 windows; isolated single-frame glitches at 01:00 and 11:00; **rest clean `{1: 299}`** |
| 16:08–23:59 (fixed)  | 8 hours       | **none — every hour `{1: 299}`** |

The 00:00 window histogram from the legacy half: `{0: 74, 1: 151, 2: 74}`
with 27 hits of the `(0,1,1,2)`-family pattern.  The 04:00 window:
`{0: 78, 1: 142, 2: 79}` with 52 pattern hits.  Both are textbook
manifestations of the `fps=1/1` × wallclock-PTS interaction.

A multi-day yield sweep (`scripts/pts_drift_test/multi_day_yield.py`)
across the previous 21 days confirms the bug's intermittent character:
April 11 at noon shows it dramatically (100 % yield, `{0: 164, 1: 270, 2: 165}`,
21 occurrences of `(0,1,1,2)`), while most other days at the same hour
read uniformly `{1: 599}`.  Days with low OCR yield (Apr 18 at 73 %,
Apr 20 at 48 %, Apr 25 at 58 %) tend to show OCR misread artifacts
(`-2` outliers from a single misclassified digit) rather than the genuine
cyclic pattern; treat sub-99 % yield results as unreliable for pattern
detection.

**Hypothesis for the intermittence:** the legacy recorder loop restarts
ffmpeg whenever the RTSP stream blips
(`[RECORDER] FFmpeg exited with code N — restarting in 5s…`), and each
restart re-randomizes the wallclock-vs-camera-RTC phase.  When the new
phase happens to put `fps=1/1`'s integer-second pick boundaries near a
camera-RTC tick edge, the cyclic pattern manifests until the next
restart.  Unconfirmed but consistent with the data.

### Caveat: don't expect zero jitter

Even with native PTS, there's residual jitter on the order of ±15 ms (see
the per-sample drift series in `analyze_v2_summary.txt`).  At fps=1/1
that's far inside the half-second tolerance, so OSD deltas stay at 1.  But
if we ever switch to fps=2/1 or finer, this margin shrinks; we'd need to
re-validate.

### File rotation note

The recorder rotates segments with `-segment_time 3600`.  Without
`-use_wallclock_as_timestamps 1`, segment boundaries align with PTS not
wallclock.  We should make sure the daily-timelapse stitcher is keyed off
the file's wallclock filename (which it already is — `glob` based on
`{date:%Y-%m-%d}_*.mp4`) rather than relying on PTS-zero meaning
midnight.

## Reproducer & artifacts

Code committed to the repo:
- `slurm/record_and_timelapse.sh` — the in-repo recorder (the fix)
- `scripts/cutover_recorder.sh` — graceful SIGTERM-then-resubmit helper used to swap the legacy job for the new one
- `scripts/pts_drift_test/capture_substream.sh` — sbatch recorder for the parallel native-vs-wallclock 1-hour test
- `scripts/pts_drift_test/analyze_v2.py` — OCR-validated drift + decimation analysis on the parallel-substream captures
- `scripts/pts_drift_test/scan_one_day.py` — hour-by-hour OCR scan on a daily timelapse (uses PyAV seek, fast)
- `scripts/pts_drift_test/multi_day_yield.py` — multi-day OCR yield sweep across daily timelapses
- `scripts/pts_drift_test/spot_check.py` — single-window OCR spot check

Run-time outputs (gitignored, in `scratch/pts_drift_test/`):
- `sub_native_pts.mp4`, `sub_wallclock_pts.mp4` — the two parallel captures
- `analyze_v2_summary.txt` — full numeric report
- `v2_*_samples.csv` — OCR'd validation samples
- `v2_*_decimated.csv` — fps=1/1 picked frames with OSD seconds
