"""Property tests for the TrustButVerifyTracker.

These exercise the three behaviours the PRD calls out explicitly:
  - Perfect monotone sequences never re-anchor.
  - A single bad reading among many good ones is projected through.
  - A sustained offset triggers a re-anchor to the new timeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from hypothesis import given, settings, strategies as st

from concam.ocr.tracker import TrustButVerifyTracker


START = datetime(2026, 4, 12, 0, 0, 0)
SECONDS_PER_FRAME = 0.25  # 4 fps raw segments
FRAME_INTERVAL_SECONDS = 0.25


def _make_tracker() -> TrustButVerifyTracker:
    return TrustButVerifyTracker(
        start_local=START,
        frame_interval_seconds=FRAME_INTERVAL_SECONDS,
        seconds_per_frame=SECONDS_PER_FRAME,
    )


# -- Property 1: clean monotone sequence never re-anchors ------------------

@given(n_frames=st.integers(min_value=6, max_value=200))
def test_perfect_monotone_never_reanchors(n_frames: int) -> None:
    tracker = _make_tracker()
    for frame_num in range(n_frames):
        ts = START + timedelta(seconds=frame_num * SECONDS_PER_FRAME)
        out_ts, status = tracker.validate(frame_num, ts, is_valid=True)
        assert out_ts == ts
        assert "RE-ANCHORED" not in status
        assert status != "anomaly_projected"
    # After a long clean run the trusted timeline should be saturated.
    assert len(tracker.trusted_timeline) == tracker.history_size


# -- Property 2: one bad read among good ones is projected ------------------

def test_single_bad_read_is_projected() -> None:
    tracker = _make_tracker()
    # Build up the trusted history.
    for i in range(tracker.history_size):
        ts = START + timedelta(seconds=i * SECONDS_PER_FRAME)
        tracker.validate(i, ts, is_valid=True)

    # One bad frame: OCR claims a backward jump.
    bad_frame = tracker.history_size
    bad_ts = START - timedelta(hours=1)  # clearly backward / impossible
    out_ts, status = tracker.validate(bad_frame, bad_ts, is_valid=True)

    expected = START + timedelta(seconds=bad_frame * SECONDS_PER_FRAME)
    assert out_ts == expected, (
        f"expected projection {expected}, got {out_ts}"
    )
    assert status.startswith("anomaly_projected")

    # Next frame returns to a consistent reading; the tracker accepts it.
    good_frame = bad_frame + 1
    good_ts = START + timedelta(seconds=good_frame * SECONDS_PER_FRAME)
    out_ts, status = tracker.validate(good_frame, good_ts, is_valid=True)
    assert out_ts == good_ts
    assert status == "consistent"


# -- Property 3: a sustained offset triggers a re-anchor --------------------

def test_sustained_offset_triggers_reanchor() -> None:
    tracker = _make_tracker()
    # Good history.
    for i in range(tracker.history_size):
        ts = START + timedelta(seconds=i * SECONDS_PER_FRAME)
        tracker.validate(i, ts, is_valid=True)

    # A coherent new timeline five minutes ahead.  Each frame is consistent
    # with the previous frame but all are offset from the trusted timeline.
    offset = timedelta(minutes=5)
    start_frame = tracker.history_size
    saw_reanchor = False
    for i in range(tracker.reanchor_threshold + 1):
        frame_num = start_frame + i
        ts = START + timedelta(seconds=frame_num * SECONDS_PER_FRAME) + offset
        _, status = tracker.validate(frame_num, ts, is_valid=True)
        if status == "RE-ANCHORED":
            saw_reanchor = True
            break

    assert saw_reanchor, "expected re-anchor after sustained offset"
    # After re-anchor the trusted timeline is reseeded from the (short)
    # contender, so we briefly return to phase 1 until the history refills.
    # The key guarantees we care about: (a) the tracker accepts subsequent
    # on-offset reads as valid, and (b) no "anomaly_projected" status shows
    # up during the rebuild.
    for i in range(1, 6):
        next_frame = start_frame + tracker.reanchor_threshold + i
        next_ts = START + timedelta(seconds=next_frame * SECONDS_PER_FRAME) + offset
        out_ts, status = tracker.validate(next_frame, next_ts, is_valid=True)
        assert out_ts == next_ts
        assert status in {"consistent", "building_history (4/5)", "building_history (5/5)"}


# -- Property 4: invalid OCR before history projects from seed -------------

def test_invalid_before_any_history_projects_forward() -> None:
    tracker = _make_tracker()
    # Frame 0 with no valid OCR: tracker has nothing to anchor to, so it
    # steps the start_local by frame_interval_seconds.
    out, status = tracker.validate(0, None, is_valid=False)
    assert status == "building_history_invalid_ocr"
    assert out == START + timedelta(seconds=FRAME_INTERVAL_SECONDS)


@given(
    n_good=st.integers(min_value=6, max_value=40),
    bad_at=st.integers(min_value=5, max_value=20),
)
@settings(max_examples=25)
def test_bad_read_then_recover(n_good: int, bad_at: int) -> None:
    """After a single failure we rejoin the good timeline without re-anchoring."""
    tracker = _make_tracker()
    bad_at = min(bad_at, n_good - 1)
    for i in range(n_good):
        ts = START + timedelta(seconds=i * SECONDS_PER_FRAME)
        if i == bad_at and len(tracker.trusted_timeline) >= tracker.history_size:
            # Inject one invalid read well after history is built.
            out, status = tracker.validate(i, None, is_valid=False)
            assert out == ts  # projection happens to land exactly on ts
            assert status == "anomaly_projected"
        else:
            out, status = tracker.validate(i, ts, is_valid=True)
            assert out == ts
            assert "RE-ANCHORED" not in status
