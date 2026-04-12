"""Per-day episode aggregation: group consecutive detections into episodes."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from concam.config import AggregationConfig


@dataclass
class FrameResult:
    """Detection result for one (frame, flight) pair."""

    time: datetime.datetime  # UTC, timezone-aware
    callsign: str
    transponder_id: str
    score: float  # 0-1 detection confidence
    pixel_line: tuple[float, float, float, float] | None  # (x1, y1, x2, y2) or None


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


def aggregate_episodes(
    frame_results: list[FrameResult],
    config: AggregationConfig,
) -> list[Episode]:
    """Group consecutive above-threshold frame results into episodes.

    Frame results are grouped per-flight (by transponder_id). Within each
    flight, consecutive frames with score >= config.detection_threshold are
    merged into a single episode as long as the time gap between adjacent
    frames is <= config.max_gap_seconds. A gap larger than that splits
    into separate episodes.

    Returns episodes sorted by (onset, callsign).
    """
    # Group by flight
    by_flight: dict[str, list[FrameResult]] = {}
    for fr in frame_results:
        by_flight.setdefault(fr.transponder_id, []).append(fr)

    episodes: list[Episode] = []

    for transponder_id, results in by_flight.items():
        # Filter to above-threshold and sort by time
        above = sorted(
            [r for r in results if r.score >= config.detection_threshold],
            key=lambda r: r.time,
        )
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
    )
