"""Tests for concam.aggregation: episode grouping from frame-level detections."""

from __future__ import annotations

import datetime

import pytest

from concam.aggregation import Episode, FrameResult, aggregate_episodes
from concam.config import AggregationConfig

UTC = datetime.timezone.utc
BASE_TIME = datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=UTC)


def _t(seconds: int) -> datetime.datetime:
    return BASE_TIME + datetime.timedelta(seconds=seconds)


def _fr(
    seconds: int,
    score: float,
    callsign: str = "UAL123",
    transponder_id: str = "A12345",
    pixel_line: tuple[float, float, float, float] | None = (10.0, 20.0, 30.0, 40.0),
) -> FrameResult:
    return FrameResult(
        time=_t(seconds),
        callsign=callsign,
        transponder_id=transponder_id,
        score=score,
        pixel_line=pixel_line,
    )


# Explicit smoothing_window_frames=1 (disabled) so these unit tests exercise the
# threshold logic in isolation without median-smoothing side-effects.
# (The production default is smoothing_window_frames=3, matching the base YAML.)
DEFAULT_CONFIG = AggregationConfig(detection_threshold=0.3, max_gap_seconds=30.0,
                                   smoothing_window_frames=1)


# ---------------------------------------------------------------------------
# Core grouping logic
# ---------------------------------------------------------------------------


def test_three_consecutive_merge_into_one():
    """Three consecutive above-threshold frames merge into one episode."""
    results = [_fr(0, 0.5), _fr(1, 0.7), _fr(2, 0.6)]
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.callsign == "UAL123"
    assert ep.onset == _t(0)
    assert ep.end == _t(2)
    assert ep.frame_count == 3


def test_gap_splits_into_two():
    """A gap larger than max_gap_seconds splits into two episodes."""
    results = [_fr(0, 0.5), _fr(1, 0.7), _fr(60, 0.6)]  # 59s gap > 30s
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert len(episodes) == 2
    assert episodes[0].end == _t(1)
    assert episodes[1].onset == _t(60)


def test_peak_score_is_maximum():
    """Peak score is the maximum over the episode's frames."""
    results = [_fr(0, 0.4), _fr(1, 0.9), _fr(2, 0.5)]
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert len(episodes) == 1
    assert episodes[0].peak_score == 0.9


def test_peak_pixel_line_from_best_frame():
    """Peak pixel line comes from the highest-scoring frame."""
    results = [
        _fr(0, 0.4, pixel_line=(1, 2, 3, 4)),
        _fr(1, 0.9, pixel_line=(10, 20, 30, 40)),
        _fr(2, 0.5, pixel_line=(5, 6, 7, 8)),
    ]
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert episodes[0].peak_pixel_line == (10, 20, 30, 40)


def test_below_threshold_excluded():
    """Frames below threshold are not included in episodes."""
    results = [_fr(0, 0.5), _fr(1, 0.1), _fr(2, 0.6)]
    # Middle frame is below 0.3 threshold, so it's excluded.
    # The remaining frames at t=0 and t=2 have a 2s gap (< 30s), so they merge.
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert len(episodes) == 1
    assert episodes[0].frame_count == 2


def test_all_below_threshold_no_episodes():
    """All frames below threshold produce no episodes."""
    results = [_fr(0, 0.1), _fr(1, 0.2)]
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert episodes == []


def test_empty_input():
    episodes = aggregate_episodes([], DEFAULT_CONFIG)
    assert episodes == []


def test_single_frame_episode():
    """A single above-threshold frame creates a single-frame episode."""
    results = [_fr(0, 0.5)]
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert len(episodes) == 1
    assert episodes[0].frame_count == 1
    assert episodes[0].onset == episodes[0].end


# ---------------------------------------------------------------------------
# Multi-flight grouping
# ---------------------------------------------------------------------------


def test_separate_flights_separate_episodes():
    """Different flights produce separate episodes even if temporally overlapping."""
    results = [
        _fr(0, 0.5, callsign="UAL1", transponder_id="A1"),
        _fr(1, 0.6, callsign="UAL1", transponder_id="A1"),
        _fr(0, 0.7, callsign="DAL2", transponder_id="B2"),
        _fr(1, 0.8, callsign="DAL2", transponder_id="B2"),
    ]
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert len(episodes) == 2
    callsigns = {e.callsign for e in episodes}
    assert callsigns == {"UAL1", "DAL2"}


def test_output_sorted_by_onset_then_callsign():
    """Episodes are sorted by (onset, callsign)."""
    results = [
        _fr(10, 0.5, callsign="ZZZ", transponder_id="Z1"),
        _fr(0, 0.5, callsign="AAA", transponder_id="A1"),
    ]
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert episodes[0].callsign == "AAA"
    assert episodes[1].callsign == "ZZZ"


# ---------------------------------------------------------------------------
# Edge: gap exactly at threshold
# ---------------------------------------------------------------------------


def test_gap_exactly_at_max_does_not_split():
    """A gap equal to max_gap_seconds does NOT split (<=, not <)."""
    results = [_fr(0, 0.5), _fr(30, 0.6)]  # exactly 30s gap == max_gap
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert len(episodes) == 1


def test_gap_just_over_max_splits():
    """A gap one second over max_gap_seconds splits."""
    results = [_fr(0, 0.5), _fr(31, 0.6)]  # 31s > 30s
    episodes = aggregate_episodes(results, DEFAULT_CONFIG)
    assert len(episodes) == 2


# ---------------------------------------------------------------------------
# Temporal smoothing (rolling-median gate)
# ---------------------------------------------------------------------------


def test_single_frame_spike_suppressed_by_smoothing():
    """One above-threshold frame surrounded by below-threshold neighbours
    should NOT open an episode when smoothing_window_frames=3. This is the
    "bright cloud edge briefly aligning with the flight vector" FP case."""
    results = [_fr(0, 0.0), _fr(1, 0.9), _fr(2, 0.0), _fr(3, 0.0)]
    cfg = AggregationConfig(
        detection_threshold=0.3, max_gap_seconds=30.0, smoothing_window_frames=3,
    )
    episodes = aggregate_episodes(results, cfg)
    assert episodes == []


def test_sustained_run_survives_smoothing():
    """Three consecutive above-threshold frames still merge when smoothed —
    the median of three high scores is still high."""
    results = [_fr(0, 0.5), _fr(1, 0.7), _fr(2, 0.6), _fr(3, 0.0)]
    cfg = AggregationConfig(
        detection_threshold=0.3, max_gap_seconds=30.0, smoothing_window_frames=3,
    )
    episodes = aggregate_episodes(results, cfg)
    assert len(episodes) == 1
    # peak_score is the raw (unsmoothed) detector value, not the median.
    assert episodes[0].peak_score == 0.7


def test_smoothing_window_1_is_identity():
    """Window=1 (default today) should exactly match legacy behaviour —
    single-frame spikes still open episodes."""
    results = [_fr(0, 0.0), _fr(1, 0.9), _fr(2, 0.0)]
    cfg = AggregationConfig(
        detection_threshold=0.3, max_gap_seconds=30.0, smoothing_window_frames=1,
    )
    episodes = aggregate_episodes(results, cfg)
    assert len(episodes) == 1
    assert episodes[0].peak_score == 0.9
