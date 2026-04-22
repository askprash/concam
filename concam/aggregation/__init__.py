"""Per-day episode aggregation: group consecutive detections into episodes."""

from __future__ import annotations

import datetime
import statistics
from dataclasses import dataclass, replace

from concam.config import AggregationConfig


@dataclass
class FrameResult:
    """Detection result for one (frame, flight) pair."""

    time: datetime.datetime  # UTC, timezone-aware
    callsign: str
    transponder_id: str
    score: float  # 0-1 detection confidence
    pixel_line: tuple[float, float, float, float] | None  # (x1, y1, x2, y2) or None
    # Contrail length in metres after adaptive ROI growth (None when unavailable).
    contrail_length_m: float | None = None


@dataclass
class Episode:
    """A contiguous run of above-threshold detections for one flight."""

    callsign: str
    transponder_id: str
    onset: datetime.datetime  # UTC
    end: datetime.datetime  # UTC
    peak_score: float
    peak_pixel_line: tuple[float, float, float, float] | None
    frame_count: int
    # Contrail length in metres at the peak-score frame (None when unavailable).
    peak_contrail_length_m: float | None = None


def _rolling_median(values: list[float], window: int) -> list[float]:
    """Centred rolling median. Window=1 → identity. Odd windows centre exactly;
    even windows (at the edges, where half the window clips) use median_low —
    the *lower* of the two middle values rather than their mean — so an
    isolated first-frame score spike paired only with its below-threshold
    right neighbour stays suppressed instead of getting averaged halfway up."""
    if window <= 1 or len(values) <= 1:
        return list(values)
    half = window // 2
    out: list[float] = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(statistics.median_low(values[lo:hi]))
    return out


def aggregate_episodes(
    frame_results: list[FrameResult],
    config: AggregationConfig,
) -> list[Episode]:
    """Group consecutive above-threshold frame results into episodes.

    Frame results are grouped per-flight (by transponder_id). Within each
    flight, scores are rolling-median-smoothed with window
    ``config.smoothing_window_frames`` (set to 1 to disable), then frames
    with smoothed score >= config.detection_threshold are kept and split
    into runs wherever the time gap between adjacent kept frames exceeds
    config.max_gap_seconds. Each run becomes one Episode.

    The peak_score / peak_pixel_line reported on each episode come from the
    **unsmoothed** score at that frame — the median only gates whether a
    frame counts, it does not rewrite what the detector reported.

    Returns episodes sorted by (onset, callsign).
    """
    # Group by flight
    by_flight: dict[str, list[FrameResult]] = {}
    for fr in frame_results:
        by_flight.setdefault(fr.transponder_id, []).append(fr)

    episodes: list[Episode] = []
    window = max(1, int(config.smoothing_window_frames))

    for transponder_id, results in by_flight.items():
        # Sort by time and apply the rolling median over the full per-flight
        # timeline (all frames, not just above-threshold ones — the low scores
        # are exactly what we want to pull down single-frame spikes).
        ordered = sorted(results, key=lambda r: r.time)
        smoothed_scores = _rolling_median([r.score for r in ordered], window)

        # Keep frames whose smoothed score clears threshold. Preserve the
        # original FrameResult (with unsmoothed score) so peak_score reported
        # on the episode is the raw detector output; smoothing is only a gate.
        above = [fr for fr, s in zip(ordered, smoothed_scores)
                 if s >= config.detection_threshold]
        if not above:
            continue

        # Walk through and split on gaps
        run_start = 0
        for i in range(1, len(above)):
            gap = (above[i].time - above[i - 1].time).total_seconds()
            if gap > config.max_gap_seconds:
                episodes.append(_build_episode(above[run_start:i]))
                run_start = i
        # Final run
        episodes.append(_build_episode(above[run_start:]))

    episodes.sort(key=lambda e: (e.onset, e.callsign))
    return episodes


def _build_episode(frames: list[FrameResult]) -> Episode:
    """Build a single Episode from a contiguous run of FrameResults."""
    peak_idx = max(range(len(frames)), key=lambda i: frames[i].score)
    return Episode(
        callsign=frames[0].callsign,
        transponder_id=frames[0].transponder_id,
        onset=frames[0].time,
        end=frames[-1].time,
        peak_score=frames[peak_idx].score,
        peak_pixel_line=frames[peak_idx].pixel_line,
        frame_count=len(frames),
        peak_contrail_length_m=frames[peak_idx].contrail_length_m,
    )
