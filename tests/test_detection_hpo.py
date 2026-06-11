"""Tests for scripts/detection_hpo.py — reliable-label loading, daylight
filtering, and the preprocessing-variant grid (June-2026 reliable retune)."""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from concam.config import DetectionConfig

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

_spec = importlib.util.spec_from_file_location(
    "detection_hpo", SCRIPTS_DIR / "detection_hpo.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["detection_hpo"] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)

build_config_for_combo = _module.build_config_for_combo
filter_daylight = _module.filter_daylight
load_reliable_labels = _module.load_reliable_labels
parse_daylight_window = _module.parse_daylight_window
variant_id_for_config = _module.variant_id_for_config
PREPROC_VARIANTS = _module.PREPROC_VARIANTS


# ---------------------------------------------------------------------------
# Reliable-label loading
# ---------------------------------------------------------------------------

def _write_reliable(tmp_path: Path) -> Path:
    payload = {
        "generated_at": "2026-06-11T00:00:00+00:00",
        "source": "test",
        "labels": {
            "2026-04-08": {
                "1": {"label": "contrail", "labelers": ["thendo"], "votes": 1},
                "2": {"label": "no_contrail", "labelers": ["thendo", "lrsand"],
                      "votes": 2},
                "3": {"label": "unsure", "labelers": ["thendo"], "votes": 1},
            },
            "2026-04-09": {
                "7": {"label": "contrail", "labelers": ["lrsand"], "votes": 1},
            },
        },
        "conflicts": {},
    }
    p = tmp_path / "reliable_labels.json"
    p.write_text(json.dumps(payload))
    return p


class TestLoadReliableLabels:
    def test_selects_date_and_coerces_int_ids(self, tmp_path):
        labels, sources = load_reliable_labels(_write_reliable(tmp_path), "2026-04-08")
        assert labels == {1: "contrail", 2: "no_contrail"}
        assert sources == {1: ["thendo"], 2: ["thendo", "lrsand"]}

    def test_non_definite_labels_dropped(self, tmp_path):
        labels, _ = load_reliable_labels(_write_reliable(tmp_path), "2026-04-08")
        assert 3 not in labels  # "unsure" is not a tuning label

    def test_other_date_isolated(self, tmp_path):
        labels, _ = load_reliable_labels(_write_reliable(tmp_path), "2026-04-09")
        assert labels == {7: "contrail"}

    def test_missing_date_aborts(self, tmp_path):
        with pytest.raises(SystemExit):
            load_reliable_labels(_write_reliable(tmp_path), "2026-04-19")


# ---------------------------------------------------------------------------
# Daylight window
# ---------------------------------------------------------------------------

def _manifest_with_onsets(onsets: dict[int, str]) -> dict:
    return {"episodes": [
        {"episode_id": eid, "onset": onset} for eid, onset in onsets.items()
    ]}


class TestParseDaylightWindow:
    def test_parses_hhmm_pair(self):
        win = parse_daylight_window("11:00,22:30")
        assert win == (datetime.time(11, 0), datetime.time(22, 30))

    def test_all_disables(self):
        assert parse_daylight_window("all") is None
        assert parse_daylight_window("ALL") is None
        assert parse_daylight_window("") is None


class TestFilterDaylight:
    WINDOW = (datetime.time(11, 0), datetime.time(22, 30))

    def test_keeps_only_onsets_inside_window(self):
        manifest = _manifest_with_onsets({
            1: "2026-04-08T04:04:26+00:00",   # night — drop
            2: "2026-04-08T11:00:00+00:00",   # boundary — keep (inclusive)
            3: "2026-04-08T15:30:00+00:00",   # day — keep
            4: "2026-04-08T22:30:00+00:00",   # boundary — keep (inclusive)
            5: "2026-04-08T23:10:00+00:00",   # dusk — drop
        })
        labels = {i: "contrail" for i in range(1, 6)}
        kept = filter_daylight(labels, manifest, self.WINDOW)
        assert sorted(kept) == [2, 3, 4]

    def test_none_window_passes_through(self):
        manifest = _manifest_with_onsets({1: "2026-04-08T04:04:26+00:00"})
        labels = {1: "no_contrail"}
        assert filter_daylight(labels, manifest, None) == labels

    def test_labeled_episode_missing_from_manifest_dropped(self):
        manifest = _manifest_with_onsets({1: "2026-04-08T12:00:00+00:00"})
        labels = {1: "contrail", 99: "contrail"}
        assert sorted(filter_daylight(labels, manifest, self.WINDOW)) == [1]

    def test_compares_in_utc(self):
        # 08:00-04:00 == 12:00 UTC — inside the window despite the local hour.
        manifest = _manifest_with_onsets({1: "2026-04-08T08:00:00-04:00"})
        labels = {1: "contrail"}
        assert filter_daylight(labels, manifest, self.WINDOW) == labels


# ---------------------------------------------------------------------------
# Preprocessing-variant grid
# ---------------------------------------------------------------------------

class TestVariantGrid:
    def test_variant_ids_match_built_configs(self):
        # The hardcoded variant ids must agree with the canonical formatter —
        # hpo_select_and_validate matches train/holdout combos on this string.
        base = DetectionConfig()
        for variant, overrides in PREPROC_VARIANTS:
            cfg = build_config_for_combo(base, overrides, 99.5, 180)
            assert variant_id_for_config(cfg) == variant

    def test_covers_every_supported_preprocessing_mode(self):
        modes = {ov["preprocessing"] for _, ov in PREPROC_VARIANTS}
        assert modes == {"none", "local_contrast", "cross_grad"}

    def test_combo_overrides_applied(self):
        base = DetectionConfig()
        cfg = build_config_for_combo(
            base, {"preprocessing": "cross_grad", "cross_grad_gain": 0.75},
            99.3, 240)
        assert cfg.preprocessing == "cross_grad"
        assert cfg.cross_grad_gain == 0.75
        assert cfg.canny_percentile_high == 99.3
        assert cfg.roi_along_px == 240

    def test_static_mask_stays_on(self):
        # The sweep MUST keep the static building mask exactly as configured
        # in the base YAML (ADR-0002): combos may not silently drop it.
        base = DetectionConfig(static_mask_path="/some/site/mask.npz")
        for _, overrides in PREPROC_VARIANTS:
            cfg = build_config_for_combo(base, overrides, 99.0, 120)
            assert cfg.static_mask_path == "/some/site/mask.npz"

    def test_unswept_knobs_keep_base_values(self):
        base = DetectionConfig(angle_tolerance_deg=9.0, long_line_min_px=33.0,
                               blur_kernel=5, score_length_norm_px=170.0)
        cfg = build_config_for_combo(
            base, {"preprocessing": "none"}, 99.7, 120)
        assert cfg.angle_tolerance_deg == 9.0
        assert cfg.long_line_min_px == 33.0
        assert cfg.blur_kernel == 5
        assert cfg.score_length_norm_px == 170.0


# ---------------------------------------------------------------------------
# End-to-end smoke: reliable labels -> daylight filter -> sweep -> report
# ---------------------------------------------------------------------------

def _build_synthetic_video(path: Path, n_frames: int, streak: bool) -> None:
    """1 fps 640x360 noise video with a horizontal streak burned into every
    frame (the 'contrail' the positive episode's ROI points at)."""
    from fractions import Fraction

    import av
    import cv2
    import numpy as np

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=1)
    stream.width, stream.height = 640, 360
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "0", "preset": "ultrafast"}
    rng = np.random.default_rng(7)
    for i in range(n_frames):
        arr = np.clip(rng.normal(70, 10, (360, 640, 3)), 0, 255).astype(np.uint8)
        if streak:
            cv2.line(arr, (170, 180), (470, 180), (230, 230, 230), 3)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = i
        frame.time_base = Fraction(1, 1)
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def test_sweep_end_to_end_smoke(tmp_path, monkeypatch, capsys):
    """Full main() pass on synthetic inputs: the night episode is filtered
    out by the daylight window, the streak episode separates from the empty
    one, and the results/report artifacts land with the variant grid."""
    import numpy as np

    from concam.detection.static_mask import save_static_mask

    start = "2026-04-08T12:00:00+00:00"

    def times(t0: int, n: int) -> list[str]:
        return [f"2026-04-08T12:00:{t0 + i:02d}+00:00" for i in range(n)]

    pos_times, neg_times = times(5, 12), times(8, 10)
    manifest = {
        "video": {"start_utc": start, "seconds_per_frame": 1.0},
        "episodes": [
            {"episode_id": 1, "transponder_id": "AAA111",
             "onset": pos_times[0],
             "frames": [{"wall_time_utc": t} for t in pos_times]},
            {"episode_id": 2, "transponder_id": "CCC333",
             "onset": neg_times[0],
             "frames": [{"wall_time_utc": t} for t in neg_times]},
            {"episode_id": 3, "transponder_id": "BBB222",
             "onset": "2026-04-08T04:00:00+00:00",  # night — filtered out
             "frames": [{"wall_time_utc": "2026-04-08T04:00:00+00:00"}]},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    rows = []
    for t in pos_times:  # ROI over the streak
        rows.append({"transponder_id": "AAA111", "wall_time_utc": t,
                     "pixel_x": 320.0, "pixel_y": 180.0,
                     "path_dx": 1.0, "path_dy": 0.0,
                     "roi": {"x": 120, "y": 80, "w": 400, "h": 200}})
    for t in neg_times:  # ROI over empty sky
        rows.append({"transponder_id": "CCC333", "wall_time_utc": t,
                     "pixel_x": 160.0, "pixel_y": 40.0,
                     "path_dx": 1.0, "path_dy": 0.0,
                     "roi": {"x": 60, "y": 10, "w": 200, "h": 60}})
    (tmp_path / "projections.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    (tmp_path / "reliable_labels.json").write_text(json.dumps({
        "labels": {"2026-04-08": {
            "1": {"label": "contrail", "labelers": ["t"], "votes": 1},
            "2": {"label": "no_contrail", "labelers": ["t"], "votes": 1},
            "3": {"label": "contrail", "labelers": ["t"], "votes": 1},
        }}}))

    # Site YAML: defaults + an (empty) static mask so the mask code path runs.
    save_static_mask(np.zeros((360, 640), dtype=bool), tmp_path / "mask.npz")
    (tmp_path / "site.yaml").write_text(
        "name: synthetic\n"
        "detection:\n"
        "  timestamp_exclusion_region: null\n"
        f"  static_mask_path: {tmp_path / 'mask.npz'}\n")

    _build_synthetic_video(tmp_path / "video.mp4", n_frames=40, streak=True)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "detection_hpo.py",
        "--date", "2026-04-08",
        "--reliable-labels", str(tmp_path / "reliable_labels.json"),
        "--manifest", str(tmp_path / "manifest.json"),
        "--projections", str(tmp_path / "projections.jsonl"),
        "--video", str(tmp_path / "video.mp4"),
        "--config", str(tmp_path / "site.yaml"),
        "--out-dir", str(out_dir),
        "--frames-per-episode", "3",
    ])
    assert _module.main() == 0

    data = json.loads((out_dir / "sweep_results.json").read_text())
    assert data["n_labeled_total"] == 3
    assert data["n_pos"] == 1 and data["n_neg"] == 1  # night episode dropped
    assert data["daylight_utc"] == "11:00,22:30"
    n_combos = (len(PREPROC_VARIANTS)
                * len(_module.CANNY_PCT_HIGH) * len(_module.ROI_ALONG_PX))
    assert len(data["results"]) == n_combos
    assert {r["variant"] for r in data["results"]} == {v for v, _ in PREPROC_VARIANTS}
    assert "variant" in data["baseline"]
    # The streak episode must outrank the empty one for the best combo.
    best = data["results"][0]
    assert best["pos_min"] > best["neg_max"]
    assert (out_dir / "sweep_report.md").read_text().startswith("# HPO sweep")
