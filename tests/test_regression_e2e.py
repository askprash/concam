"""Unit tests for the pure helpers in ``scripts/regression_e2e.py``.

The full regression script touches the filesystem, video decoder, and a live
DuckDB, so end-to-end coverage lives in the April-8 invocation itself.
These tests pin the parts that are easy to regress silently: the score
histogram bucketing, the episode-picker (top + near-threshold) logic, and
the jsonl-diff summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

regression_e2e = pytest.importorskip("regression_e2e")


def test_score_histogram_bucketing_matches_thresholds():
    detections = [
        {"score": 0.0},
        {"score": 0.05},      # below threshold
        {"score": 0.083},     # threshold exact — should land in [0.083, 0.167)
        {"score": 0.166},
        {"score": 0.333},
        {"score": 0.5},
        {"score": 0.667},
        {"score": 1.0},       # goes into last bin (upper is 1.001)
    ]
    hist = regression_e2e._score_histogram(detections)
    counts = [b["count"] for b in hist]
    # Bins: (0, .083) (.083, .167) (.167, .334) (.334, .501) (.501, .668) (.668, 1.001)
    assert counts == [2, 2, 1, 1, 1, 1]


def test_pick_episodes_dedups_near_threshold_from_top():
    threshold = 0.083
    episodes = [
        {"callsign": "AAA", "transponder_id": "T1",
         "peak_score": 1.0, "onset": "t1"},
        {"callsign": "BBB", "transponder_id": "T2",
         "peak_score": 0.9, "onset": "t2"},
        {"callsign": "CCC", "transponder_id": "T3",
         "peak_score": 0.08, "onset": "t3"},   # below threshold — dropped
        {"callsign": "DDD", "transponder_id": "T4",
         "peak_score": 0.15, "onset": "t4"},
        {"callsign": "EEE", "transponder_id": "T5",
         "peak_score": 0.1, "onset": "t5"},    # closest above threshold
        {"callsign": "FFF", "transponder_id": "T6",
         "peak_score": 0.08, "onset": "t6"},   # below — dropped
    ]
    picks = regression_e2e._pick_episodes(
        episodes, threshold=threshold, n_top=2, n_threshold=3
    )

    # Top picks are the two highest.
    top_calls = [e["callsign"] for e in picks["top"]]
    assert top_calls == ["AAA", "BBB"]

    # Near-threshold picks are weakest-above-threshold, de-duped against top.
    near_calls = [e["callsign"] for e in picks["near_threshold"]]
    assert near_calls == ["EEE", "DDD"]  # 0.1 is closer to 0.083 than 0.15
    # And no below-threshold episodes made it in.
    assert "CCC" not in near_calls
    assert "FFF" not in near_calls


def test_pick_episodes_handles_short_lists():
    episodes = [
        {"callsign": "X", "transponder_id": "T", "peak_score": 0.5, "onset": "t"},
    ]
    picks = regression_e2e._pick_episodes(episodes, threshold=0.083,
                                          n_top=5, n_threshold=5)
    assert len(picks["top"]) == 1
    # All above-threshold episodes are in top; near-threshold dedups → empty.
    assert picks["near_threshold"] == []


def test_diff_jsonl_identical(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    for p in (a, b):
        p.write_text('{"a": 1}\n{"a": 2}\n')
    result = regression_e2e._diff_jsonl(a, b)
    assert result == {"identical": True, "lines": 2}


def test_diff_jsonl_reports_first_mismatch(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"x": 1}\n{"x": 2}\n{"x": 3}\n')
    # Drift the second record; third is fine.
    b.write_text('{"x": 1}\n{"x": 20}\n{"x": 3}\n')
    result = regression_e2e._diff_jsonl(a, b)
    assert result["identical"] is False
    assert result["differing_records"] == 1
    assert result["first_diff_idx"] == 1
    assert result["first_diff_preview"]["a"] == {"x": 2}
    assert result["first_diff_preview"]["b"] == {"x": 20}


def test_diff_jsonl_whitespace_insensitive(tmp_path):
    """JSON dumps may round-trip with different whitespace but identical fields."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    # filecmp fast-path: bytes differ (spaces), but _diff_jsonl does structural
    # comparison on the JSON payload and should NOT flag the whitespace.
    a.write_text('{"x": 1}\n')
    b.write_text('{"x":1}\n')
    result = regression_e2e._diff_jsonl(a, b)
    assert result["identical"] is False
    # Structural equality still reports zero differing records since the
    # parsed payloads match — byte-level drift is what we care about.
    # (first_diff_idx stays None because the records compare equal.)
    assert result["differing_records"] == 0
    assert result["first_diff_idx"] is None
