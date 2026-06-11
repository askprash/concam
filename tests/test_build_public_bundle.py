"""Tests for scripts/build_public_bundle.py exclusion-region plumbing."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from concam.detection.static_mask import save_static_mask

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

_spec = importlib.util.spec_from_file_location(
    "build_public_bundle", SCRIPTS_DIR / "build_public_bundle.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["build_public_bundle"] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)

exclusion_regions_block = _module.exclusion_regions_block


@dataclass
class _DetCfg:
    static_mask_path: str | None = None
    timestamp_exclusion_region: list | None = None


def test_none_when_nothing_configured():
    assert exclusion_regions_block(_DetCfg()) is None


def test_timestamp_only():
    block = exclusion_regions_block(
        _DetCfg(timestamp_exclusion_region=[0, 95, 2950, 3840])
    )
    assert block == {"polygons": [], "timestamp_region": [0, 95, 2950, 3840]}


def test_mask_polygons_included(tmp_path):
    mask = np.zeros((200, 300), dtype=bool)
    mask[120:190, 40:260] = True  # big "building" blob > min_area
    p = tmp_path / "mask.npz"
    save_static_mask(mask, p)
    block = exclusion_regions_block(_DetCfg(static_mask_path=str(p)))
    assert block is not None
    assert len(block["polygons"]) == 1
    xs = [v[0] for v in block["polygons"][0]]
    ys = [v[1] for v in block["polygons"][0]]
    assert min(xs) >= 39 and max(xs) <= 260
    assert min(ys) >= 119 and max(ys) <= 190


def test_missing_mask_file_ignored(tmp_path):
    block = exclusion_regions_block(
        _DetCfg(static_mask_path=str(tmp_path / "absent.npz"),
                timestamp_exclusion_region=[0, 95, 2950, 3840])
    )
    assert block is not None and block["polygons"] == []


# ---------------------------------------------------------------------------
# Pixel-space sustained-overlap flagging
# ---------------------------------------------------------------------------

sustained_overlap_ids = _module.sustained_overlap_ids


def _ep(eid, tid, t0, t1):
    return {"episode_id": eid, "transponder_id": tid,
            "onset": f"2026-04-09T{t0}+00:00", "end": f"2026-04-09T{t1}+00:00"}


def _track(pings):
    return {"pings": [
        {"wall_time_utc": f"2026-04-09T{t}+00:00", "pixel_x": x, "pixel_y": y}
        for t, x, y in pings
    ]}


def test_sustained_parallel_tracks_flagged():
    eps = [_ep(1, "A", "12:00:00", "12:02:00"), _ep(2, "B", "12:00:00", "12:02:00")]
    tracks = {
        # Two flights 100 px apart for the whole window.
        "A": _track([("12:00:00", 1000, 500), ("12:02:00", 1600, 500)]),
        "B": _track([("12:00:00", 1000, 600), ("12:02:00", 1600, 600)]),
    }
    assert sustained_overlap_ids(eps, tracks, sep_px=200) == {1, 2}


def test_distant_tracks_not_flagged():
    eps = [_ep(1, "A", "12:00:00", "12:02:00"), _ep(2, "B", "12:00:00", "12:02:00")]
    tracks = {
        "A": _track([("12:00:00", 1000, 500), ("12:02:00", 1600, 500)]),
        "B": _track([("12:00:00", 1000, 1500), ("12:02:00", 1600, 1500)]),
    }
    assert sustained_overlap_ids(eps, tracks, sep_px=200) == set()


def test_transient_crossing_not_flagged():
    # Perpendicular crossing: close only for an instant, median far apart.
    eps = [_ep(1, "A", "12:00:00", "12:02:00"), _ep(2, "B", "12:00:00", "12:02:00")]
    tracks = {
        "A": _track([("12:00:00", 0, 1000), ("12:02:00", 2000, 1000)]),
        "B": _track([("12:00:00", 1000, 0), ("12:02:00", 1000, 2000)]),
    }
    assert sustained_overlap_ids(eps, tracks, sep_px=200) == set()


def test_no_time_overlap_not_flagged():
    eps = [_ep(1, "A", "12:00:00", "12:01:00"), _ep(2, "B", "13:00:00", "13:01:00")]
    tracks = {
        "A": _track([("12:00:00", 1000, 500), ("12:01:00", 1600, 500)]),
        "B": _track([("13:00:00", 1000, 520), ("13:01:00", 1600, 520)]),
    }
    assert sustained_overlap_ids(eps, tracks, sep_px=200) == set()


def test_same_flight_multiple_passes_not_self_compared():
    eps = [_ep(1, "A", "12:00:00", "12:01:00"), _ep(2, "A", "12:00:00", "12:01:00")]
    tracks = {"A": _track([("12:00:00", 1000, 500), ("12:01:00", 1600, 500)])}
    assert sustained_overlap_ids(eps, tracks, sep_px=200) == set()


# ---------------------------------------------------------------------------
# Manifest size: dead-frame filtering + compact pings
# ---------------------------------------------------------------------------

build_manifest = _module.build_manifest
compact_pings = _module.compact_pings


def test_compact_pings_schema_and_rounding():
    pings = [
        {"wall_time_utc": "2026-04-09T05:23:00+00:00",
         "pixel_x": 3160.6542367450793, "pixel_y": 1471.2205221015925,
         "alt_m": 9717.98, "alt_baro_m": 9717.98, "dist_km": 198.26},
        {"wall_time_utc": "2026-04-09T05:23:01+00:00",
         "pixel_x": 3158.76, "pixel_y": 1471.32,
         "alt_m": 9718.0, "alt_baro_m": None, "dist_km": None},
    ]
    out = compact_pings(pings, step_s=1)
    assert out[0] == {"t": 1775712180000, "x": 3160.7, "y": 1471.2,
                      "alt_baro_m": 9718, "dist_km": 198.26}
    # alt_m kept only as fallback when barometric is missing.
    assert out[1]["alt_m"] == 9718
    assert "alt_baro_m" not in out[1]
    assert "dist_km" not in out[1]


def test_compact_pings_thinning_keeps_endpoints():
    pings = [
        {"wall_time_utc": f"2026-04-09T05:23:{s:02d}+00:00",
         "pixel_x": 100.0 + s, "pixel_y": 50.0, "alt_m": None,
         "alt_baro_m": None, "dist_km": None}
        for s in range(11)
    ]
    out = compact_pings(pings, step_s=2)
    # 0,2,4,6,8,10 — every 2s, last ping always kept.
    assert [p["x"] for p in out] == [100, 102, 104, 106, 108, 110]


# ---------------------------------------------------------------------------
# Static-mask detection filtering (ADR-0002 applied at manifest-build time)
# ---------------------------------------------------------------------------

line_masked_fraction = _module.line_masked_fraction
apply_static_mask_filter = _module.apply_static_mask_filter


def _left_half_mask(h: int = 100, w: int = 200) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :100] = True  # columns 0..99 masked
    return mask


def test_line_masked_fraction_fully_inside():
    assert line_masked_fraction([10, 50, 90, 50], _left_half_mask()) == 1.0


def test_line_masked_fraction_fully_outside():
    assert line_masked_fraction([110, 50, 190, 50], _left_half_mask()) == 0.0


def test_line_masked_fraction_straddles_boundary():
    frac = line_masked_fraction([50, 50, 150, 50], _left_half_mask())
    assert 0.4 <= frac <= 0.6


def test_line_masked_fraction_out_of_bounds_counts_unmasked():
    mask = np.ones((100, 100), dtype=bool)
    # Entirely outside the frame: out-of-bounds samples count as unmasked.
    assert line_masked_fraction([200, 200, 400, 400], mask) == 0.0
    # Half the segment hangs off the right edge of an all-True mask.
    frac = line_masked_fraction([50, 50, 150, 50], mask)
    assert 0.4 <= frac <= 0.6


def _signal_episode():
    return {
        "episode_id": 1,
        "peak_score": 9.0,
        "peak_pixel_line": [10.0, 50.0, 90.0, 50.0],  # fully masked
        "frames": [
            {"wall_time_utc": "2026-04-09T12:00:00+00:00",
             "score": 9.0, "pixel_line": [10.0, 50.0, 90.0, 50.0]},
            {"wall_time_utc": "2026-04-09T12:00:05+00:00",
             "score": 4.0, "pixel_line": [110.0, 50.0, 190.0, 50.0]},
        ],
    }


def test_masked_frames_suppressed_and_peak_recomputed():
    ep = _signal_episode()
    apply_static_mask_filter([ep], _left_half_mask())
    assert ep["peak_score"] == 4.0
    assert ep["peak_pixel_line"] == [110.0, 50.0, 190.0, 50.0]
    # The suppressed frame (score 0, line None) carries no signal and is
    # dropped, mirroring build_manifest's signal-only frames filter.
    assert ep["frames"] == [
        {"wall_time_utc": "2026-04-09T12:00:05+00:00",
         "score": 4.0, "pixel_line": [110.0, 50.0, 190.0, 50.0]},
    ]


def test_all_frames_masked_zeroes_episode():
    ep = _signal_episode()
    ep["frames"][1]["pixel_line"] = [20.0, 80.0, 80.0, 80.0]  # also masked
    apply_static_mask_filter([ep], _left_half_mask())
    assert ep["peak_score"] == 0.0
    assert ep["peak_pixel_line"] is None
    assert ep["frames"] == []


def test_unmasked_episode_untouched():
    import copy

    ep = _signal_episode()
    ep["peak_pixel_line"] = [110.0, 50.0, 190.0, 50.0]
    ep["frames"][0]["pixel_line"] = [120.0, 40.0, 180.0, 40.0]
    before = copy.deepcopy(ep)
    apply_static_mask_filter([ep], _left_half_mask())
    assert ep == before


def test_majority_threshold_minority_masked_survives():
    # ~40% of the line is masked: below the >0.5 majority rule, untouched.
    ep = _signal_episode()
    ep["peak_pixel_line"] = [60.0, 50.0, 160.0, 50.0]
    ep["frames"] = [
        {"wall_time_utc": "2026-04-09T12:00:00+00:00",
         "score": 9.0, "pixel_line": [60.0, 50.0, 160.0, 50.0]},
    ]
    apply_static_mask_filter([ep], _left_half_mask())
    assert ep["peak_score"] == 9.0
    assert ep["peak_pixel_line"] == [60.0, 50.0, 160.0, 50.0]


def test_lineless_scored_frame_kept():
    # A frame with score > 0 but no pixel_line cannot be mask-tested; it stays.
    ep = {
        "episode_id": 1,
        "peak_score": 9.0,
        "peak_pixel_line": [10.0, 50.0, 90.0, 50.0],
        "frames": [
            {"wall_time_utc": "2026-04-09T12:00:00+00:00",
             "score": 9.0, "pixel_line": [10.0, 50.0, 90.0, 50.0]},
            {"wall_time_utc": "2026-04-09T12:00:05+00:00",
             "score": 2.0, "pixel_line": None},
        ],
    }
    apply_static_mask_filter([ep], _left_half_mask())
    assert ep["peak_score"] == 2.0
    assert ep["peak_pixel_line"] is None
    assert len(ep["frames"]) == 1


def test_filter_returns_suppressed_frame_count():
    ep = _signal_episode()
    assert apply_static_mask_filter([ep], _left_half_mask()) == 1
