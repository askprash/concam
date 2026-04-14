"""Unit tests for the projection alignment validation helpers (PRD item 13)."""

from __future__ import annotations

import datetime
import importlib.util
import math
import sys
from pathlib import Path


def _load(name: str):
    path = Path(__file__).parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extract = _load("projection_alignment_extract")
analyze = _load("projection_alignment_analyze")


def _ping(tid: str, callsign: str, sec: int, x: float, y: float) -> dict:
    t = datetime.datetime(2026, 4, 9, 15, 0, sec, tzinfo=datetime.timezone.utc)
    return {
        "wall_time_utc": t.isoformat(),
        "transponder_id": tid,
        "callsign": callsign,
        "pixel_x": x,
        "pixel_y": y,
        "path_dx": 1.0,
        "path_dy": 0.0,
        "roi": {"x": int(x), "y": int(y), "w": 80, "h": 80},
    }


class TestGroupFlyovers:
    def test_filters_out_out_of_frame_pings(self):
        rows = [
            _ping("A", "AAA1", i, -10, 100) for i in range(30)
        ]  # x < 0, outside frame
        flyovers = extract._group_flyovers(
            rows,
            daylight_start=datetime.time(0, 0),
            daylight_end=datetime.time(23, 59),
            image_size=(3840, 2160),
            min_duration_s=5.0,
            min_pixel_span_px=50.0,
        )
        assert flyovers == []

    def test_enforces_duration_and_span(self):
        too_short = [_ping("A", "AAA", i, 100 + i * 10, 500) for i in range(3)]  # 2 s span
        long_enough = [_ping("B", "BBB", i, 100 + i * 10, 500) for i in range(30)]
        flyovers = extract._group_flyovers(
            too_short + long_enough,
            daylight_start=datetime.time(0, 0),
            daylight_end=datetime.time(23, 59),
            image_size=(3840, 2160),
            min_duration_s=10.0,
            min_pixel_span_px=100.0,
        )
        assert len(flyovers) == 1
        assert flyovers[0].transponder_id == "B"
        assert flyovers[0].pixel_span >= 100.0
        assert flyovers[0].duration_s >= 10.0

    def test_splits_reentry_on_time_gap(self):
        # Same transponder appearing twice with a 600 s gap → two flyovers.
        first = [_ping("A", "AAA", i, 100 + i * 10, 500) for i in range(30)]
        t_late = datetime.datetime(2026, 4, 9, 16, 0, 0, tzinfo=datetime.timezone.utc)
        second = []
        for i in range(30):
            p = _ping("A", "AAA", 0, 2000 + i * 10, 500)
            p["wall_time_utc"] = (t_late + datetime.timedelta(seconds=i)).isoformat()
            second.append(p)
        flyovers = extract._group_flyovers(
            first + second,
            daylight_start=datetime.time(0, 0),
            daylight_end=datetime.time(23, 59),
            image_size=(3840, 2160),
            min_duration_s=10.0,
            min_pixel_span_px=100.0,
        )
        assert len(flyovers) == 2
        # Sorted implicitly by insertion; both should be flagged as separate crossings.
        centers = sorted(f.center_px[0] for f in flyovers)
        assert centers[0] < 500 < centers[1]

    def test_respects_daylight_window(self):
        # All pings at 02:00 UTC — outside a 10:30-23:30 window.
        rows = []
        for i in range(30):
            t = datetime.datetime(2026, 4, 9, 2, 0, i, tzinfo=datetime.timezone.utc)
            p = _ping("A", "AAA", i, 100 + i * 10, 500)
            p["wall_time_utc"] = t.isoformat()
            rows.append(p)
        flyovers = extract._group_flyovers(
            rows,
            daylight_start=datetime.time(10, 30),
            daylight_end=datetime.time(23, 30),
            image_size=(3840, 2160),
            min_duration_s=5.0,
            min_pixel_span_px=50.0,
        )
        assert flyovers == []


class TestPickSpreadFlyovers:
    def _make(self, tid: str, center_x: float, center_y: float) -> extract.Flyover:
        pings = []
        for i in range(15):
            p = _ping(tid, tid, i, center_x + i * 2, center_y)
            pings.append(p)
        return extract.Flyover(idx=-1, transponder_id=tid, callsign=tid, pings=pings)

    def test_picks_far_apart_flyovers(self):
        # Four candidates: one near center, three far apart in the FOV corners.
        candidates = [
            self._make("CENTER", 1920, 1080),
            self._make("TL", 200, 200),
            self._make("TR", 3600, 200),
            self._make("BR", 3600, 1900),
        ]
        picked = extract._pick_spread_flyovers(candidates, n=3, image_size=(3840, 2160))
        picked_tids = {f.transponder_id for f in picked}
        assert len(picked) == 3
        # At least two of the three corners should make the cut over CENTER.
        corners = {"TL", "TR", "BR"}
        assert len(picked_tids & corners) >= 2

    def test_returns_all_if_fewer_than_n(self):
        candidates = [self._make("A", 500, 500), self._make("B", 2500, 1500)]
        picked = extract._pick_spread_flyovers(candidates, n=5, image_size=(3840, 2160))
        assert len(picked) == 2
        assert [f.idx for f in picked] == [0, 1]


class TestSampleFrameIndices:
    def test_evenly_spaced_samples(self):
        anchor = datetime.datetime(2026, 4, 9, 4, 0, 0, tzinfo=datetime.timezone.utc)
        pings = [_ping("A", "AAA", i, 100 + i, 500) for i in range(10)]
        # Shift all wall_times to start at 04:10:00 so frame_idx = 600 + sec
        for i, p in enumerate(pings):
            t = anchor + datetime.timedelta(seconds=600 + i)
            p["wall_time_utc"] = t.isoformat()
        fly = extract.Flyover(idx=0, transponder_id="A", callsign="AAA", pings=pings)

        chosen = extract._sample_frame_indices(fly, anchor, seconds_per_frame=1.0, frames_per_flyover=4)
        assert len(chosen) == 4
        # First and last pings always represented.
        assert chosen[0]["frame_idx"] == 600
        assert chosen[-1]["frame_idx"] == 609
        # Monotone.
        frame_idxs = [c["frame_idx"] for c in chosen]
        assert frame_idxs == sorted(frame_idxs)


class TestOffsetSummary:
    def _lab(self, fly_idx: int, px: float, py: float, cx: float, cy: float, visible=True) -> dict:
        return {
            "flyover_idx": fly_idx,
            "callsign": f"F{fly_idx}",
            "transponder_id": f"T{fly_idx}",
            "frame_idx": 0,
            "wall_time_utc": "2026-04-09T15:00:00+00:00",
            "projected_pixel_x": px,
            "projected_pixel_y": py,
            "click_x": cx if visible else None,
            "click_y": cy if visible else None,
            "visible": visible,
        }

    def test_zero_offset_gives_go(self):
        labels = [self._lab(0, 100, 100, 100, 100) for _ in range(3)]
        summary = analyze.summarise(labels)
        assert summary["overall"]["verdict"] == "GO"
        assert summary["overall"]["median_offset_px"] == 0.0

    def test_large_offset_fails(self):
        labels = [self._lab(0, 1000, 1000, 1200, 1200) for _ in range(3)]  # offset ≈ 283 px
        summary = analyze.summarise(labels)
        assert summary["overall"]["verdict"] == "NO-GO"
        assert summary["overall"]["median_offset_px"] > analyze.GO_THRESHOLD_PX

    def test_notvisible_labels_skipped(self):
        labels = [
            self._lab(0, 100, 100, 100, 100),
            self._lab(0, 200, 200, None, None, visible=False),
            self._lab(0, 300, 300, 310, 310),
        ]
        summary = analyze.summarise(labels)
        assert summary["overall"]["n_visible_labelled"] == 2
        assert summary["overall"]["n_not_visible"] == 1
        offsets = [r["offset_px"] for r in summary["rows"]]
        # 0 and hypot(10, 10) ≈ 14.14
        assert math.isclose(min(offsets), 0.0)
        assert math.isclose(max(offsets), math.hypot(10, 10), abs_tol=1e-9)

    def test_offset_sign_convention(self):
        # projected is 50 px right and 30 px below click -> dx=+50, dy=+30
        labels = [self._lab(0, 150, 130, 100, 100)]
        summary = analyze.summarise(labels)
        r = summary["rows"][0]
        assert math.isclose(r["offset_dx"], 50.0)
        assert math.isclose(r["offset_dy"], 30.0)

    def test_per_flyover_breakdown(self):
        labels = [
            self._lab(0, 100, 100, 100, 100),
            self._lab(0, 200, 200, 205, 205),
            self._lab(1, 300, 300, 400, 400),  # ≈ 141 px offset
        ]
        summary = analyze.summarise(labels)
        assert len(summary["per_flyover"]) == 2
        by_idx = {f["flyover_idx"]: f for f in summary["per_flyover"]}
        assert by_idx[0]["n_frames"] == 2
        assert by_idx[1]["n_frames"] == 1
        assert by_idx[0]["median_offset_px"] < by_idx[1]["median_offset_px"]

    def test_empty_labels_is_inconclusive(self):
        summary = analyze.summarise([])
        assert summary["overall"]["verdict"] == "inconclusive"
        assert math.isnan(summary["overall"]["median_offset_px"])
