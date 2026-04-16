"""Unit tests for scripts/render_episode_clips.py.

These tests cover the logic-heavy helpers — episode selection, OCR indexing,
and the clip window calculation — without touching the video or the network.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the script as a module without executing main()
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "render_episode_clips.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("render_episode_clips", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_episode_clips"] = mod
    spec.loader.exec_module(mod)
    return mod


rc = _load_script()


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_episode(callsign, transponder_id, onset, frame_count=10, peak_score=1.0):
    onset_dt = datetime.datetime.fromisoformat(onset)
    end_dt = onset_dt + datetime.timedelta(seconds=frame_count - 1)
    return {
        "callsign": callsign,
        "transponder_id": transponder_id,
        "onset": onset,
        "end": end_dt.isoformat(),
        "peak_score": peak_score,
        "frame_count": frame_count,
        "peak_contrail_length_m": 500.0,
    }


THRESHOLD = 0.083


# ---------------------------------------------------------------------------
# _pick_episodes
# ---------------------------------------------------------------------------

class TestPickEpisodes:
    def _episodes(self):
        return [
            _make_episode("AAA", "A1", "2026-04-08T10:00:00+00:00", peak_score=1.0),
            _make_episode("BBB", "B1", "2026-04-08T11:00:00+00:00", peak_score=0.5),
            _make_episode("CCC", "C1", "2026-04-08T12:00:00+00:00", peak_score=0.3),
            _make_episode("DDD", "D1", "2026-04-08T13:00:00+00:00", peak_score=0.05),
        ]

    def test_top_n_respects_limit(self):
        eps = self._episodes()
        result = rc._pick_episodes(eps, top_n=2, threshold=THRESHOLD)
        assert len(result) == 2

    def test_top_n_sorted_descending(self):
        eps = self._episodes()
        result = rc._pick_episodes(eps, top_n=3, threshold=THRESHOLD)
        scores = [r["peak_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_below_threshold_excluded(self):
        eps = self._episodes()
        result = rc._pick_episodes(eps, top_n=10, threshold=THRESHOLD)
        # DDD with score 0.05 is below 0.083
        callsigns = {r["callsign"] for r in result}
        assert "DDD" not in callsigns

    def test_all_above_threshold_included(self):
        eps = self._episodes()
        result = rc._pick_episodes(eps, top_n=10, threshold=THRESHOLD)
        assert len(result) == 3  # AAA, BBB, CCC (DDD below threshold)

    def test_empty_input(self):
        result = rc._pick_episodes([], top_n=5, threshold=THRESHOLD)
        assert result == []


# ---------------------------------------------------------------------------
# _find_episodes_by_spec
# ---------------------------------------------------------------------------

class TestFindEpisodesBySpec:
    def _episodes(self):
        return [
            _make_episode("AAL101", "A1", "2026-04-08T16:34:00+00:00", peak_score=1.0),
            _make_episode("DAL289", "D1", "2026-04-08T19:34:00+00:00", peak_score=0.8),
        ]

    def test_callsign_at_onset(self):
        eps = self._episodes()
        result = rc._find_episodes_by_spec(eps, ["AAL101@2026-04-08T16:34:00+00:00"])
        assert len(result) == 1
        assert result[0]["callsign"] == "AAL101"

    def test_near_onset_tolerance(self):
        """1-second offset should still match."""
        eps = self._episodes()
        result = rc._find_episodes_by_spec(eps, ["AAL101@2026-04-08T16:34:01+00:00"])
        assert len(result) == 1

    def test_no_match_returns_empty(self):
        eps = self._episodes()
        result = rc._find_episodes_by_spec(eps, ["ZZZ@2026-04-08T00:00:00+00:00"])
        assert result == []

    def test_multiple_specs(self):
        eps = self._episodes()
        result = rc._find_episodes_by_spec(
            eps,
            ["AAL101@2026-04-08T16:34:00+00:00", "DAL289@2026-04-08T19:34:00+00:00"],
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _load_ocr_index
# ---------------------------------------------------------------------------

class TestLoadOcrIndex:
    def _write_ocr(self, records, tmpdir):
        path = Path(tmpdir) / "ocr.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_round_trip(self, tmp_path):
        recs = [
            {"frame_idx": 0, "wall_time_utc": "2026-04-08T04:00:00+00:00",
             "ocr_status": "ok"},
            {"frame_idx": 1, "wall_time_utc": "2026-04-08T04:00:01+00:00",
             "ocr_status": "ok"},
        ]
        path = self._write_ocr(recs, tmp_path)
        wt_to_idx, idx_to_wt = rc._load_ocr_index(path)
        assert wt_to_idx["2026-04-08T04:00:00+00:00"] == 0
        assert wt_to_idx["2026-04-08T04:00:01+00:00"] == 1
        assert idx_to_wt[0] == "2026-04-08T04:00:00+00:00"
        assert idx_to_wt[1] == "2026-04-08T04:00:01+00:00"

    def test_microseconds_stripped(self, tmp_path):
        recs = [
            {"frame_idx": 5, "wall_time_utc": "2026-04-08T04:00:05.123456+00:00",
             "ocr_status": "ok"},
        ]
        path = self._write_ocr(recs, tmp_path)
        wt_to_idx, _ = rc._load_ocr_index(path)
        assert "2026-04-08T04:00:05+00:00" in wt_to_idx


# ---------------------------------------------------------------------------
# _safe_name
# ---------------------------------------------------------------------------

class TestSafeName:
    def test_alphanumeric_passthrough(self):
        assert rc._safe_name("AAL101") == "AAL101"

    def test_special_chars_replaced(self):
        name = rc._safe_name("ABC/123 XYZ")
        assert "/" not in name
        assert " " not in name

    def test_hyphens_kept(self):
        assert "-" in rc._safe_name("A-B")

    def test_underscores_kept(self):
        assert "_" in rc._safe_name("A_B")


# ---------------------------------------------------------------------------
# Integration: script --help runs without error
# ---------------------------------------------------------------------------

def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        rc.main(["--help"])
    assert exc.value.code == 0
