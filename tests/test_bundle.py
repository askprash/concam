"""Tests for the labeler bundle generator."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from concam.aggregation import Episode
from concam.bundle import (
    Assignment,
    _symlink_video,
    assign_episodes,
    generate_bundles,
)
from concam.bundle import calibration_block
from concam.cli import main as cli_main
from concam.storage import Database

# ---------------------------------------------------------------------------
# Import build_public_bundle script helpers
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_public_bundle import (  # noqa: E402
    _build_flight_tracks_with_altitude,
    build_manifest,
)

# Camera site defaults (mirrors AdsbConfig defaults)
_SITE_LAT = 42.360444
_SITE_LON = -71.089238


# ---------- Assignment ----------


def test_assign_episodes_deterministic() -> None:
    ids = list(range(1, 21))  # 20 episodes
    a1 = assign_episodes(ids, ["alice", "bob"], overlap_fraction=0.2)
    a2 = assign_episodes(ids, ["alice", "bob"], overlap_fraction=0.2)
    assert a1 == a2


def test_assign_episodes_coverage_and_overlap() -> None:
    ids = list(range(1, 101))
    a = assign_episodes(ids, ["alice", "bob"], overlap_fraction=0.2)
    alice = set(a["alice"].episode_ids)
    bob = set(a["bob"].episode_ids)
    overlap = alice & bob
    # Overlap should be ~20 episodes (round(100*0.2) = 20).
    assert len(overlap) == 20
    # Every episode is labeled at least once.
    assert alice | bob == set(ids)
    # Both Assignments expose the same overlap set.
    assert set(a["alice"].overlap_episode_ids) == overlap
    assert set(a["bob"].overlap_episode_ids) == overlap


def test_assign_episodes_zero_overlap() -> None:
    ids = list(range(1, 11))
    a = assign_episodes(ids, ["alice", "bob"], overlap_fraction=0.0)
    alice = set(a["alice"].episode_ids)
    bob = set(a["bob"].episode_ids)
    assert alice & bob == set()
    assert alice | bob == set(ids)


def test_assign_episodes_full_overlap() -> None:
    ids = [1, 2, 3]
    a = assign_episodes(ids, ["alice", "bob"], overlap_fraction=1.0)
    assert set(a["alice"].episode_ids) == {1, 2, 3}
    assert set(a["bob"].episode_ids) == {1, 2, 3}


def test_assign_episodes_empty() -> None:
    a = assign_episodes([], ["alice", "bob"], overlap_fraction=0.2)
    assert a["alice"].episode_ids == ()
    assert a["bob"].episode_ids == ()


def test_assign_episodes_three_labelers() -> None:
    ids = list(range(1, 31))
    a = assign_episodes(ids, ["a", "b", "c"], overlap_fraction=0.3)
    union = set()
    for asgn in a.values():
        union |= set(asgn.episode_ids)
    assert union == set(ids)
    # The overlap set is shared.
    overlap_sets = [set(a[lbl].overlap_episode_ids) for lbl in "abc"]
    assert overlap_sets[0] == overlap_sets[1] == overlap_sets[2]


def test_assign_episodes_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        assign_episodes([1, 2], ["a"], overlap_fraction=1.5)
    with pytest.raises(ValueError):
        assign_episodes([1, 2], [], overlap_fraction=0.2)


# ---------- Symlink video ----------


def test_symlink_video(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundles" / "alice"
    bundle_dir.mkdir(parents=True)
    video = tmp_path / "videos" / "day.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    link_name = _symlink_video(video, bundle_dir)
    # Returned name is just the filename (no directory component).
    assert link_name == "video.mp4"
    # A symlink with that name exists in the bundle dir.
    link = bundle_dir / link_name
    assert link.is_symlink()
    assert link.resolve() == video.resolve()


# ---------- End-to-end bundle generation ----------


def _populate_duckdb(
    db_path: Path, date: datetime.date, episodes: list[Episode]
) -> None:
    with Database(db_path) as db:
        db.create_schema()
        db.insert_episodes(episodes, date=date, start_id=1)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@pytest.fixture
def synthetic_pipeline_outputs(tmp_path: Path) -> dict:
    """Build a minimal but complete set of pipeline outputs on disk."""
    date = datetime.date(2026, 4, 8)
    base_ts = datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)
    # OCR file: frame 0 starts a few hours before the first episode, matching a
    # realistic timelapse that covers a full local day (EDT -> UTC+4h shift).
    ocr_start = datetime.datetime(2026, 4, 8, 4, 0, 0, tzinfo=datetime.timezone.utc)

    # 5 episodes from 3 flights (2 flights have multiple episodes over the day).
    episodes: list[Episode] = []
    for i in range(5):
        onset = base_ts + datetime.timedelta(minutes=10 * i)
        end = onset + datetime.timedelta(seconds=5)
        episodes.append(
            Episode(
                callsign=f"FL{i}",
                transponder_id=f"TID{i % 3}",  # 3 distinct flights
                onset=onset,
                end=end,
                peak_score=0.5 + 0.05 * i,
                peak_pixel_line=(10.0 * i, 20.0, 100.0 + 10.0 * i, 120.0),
                frame_count=5,
            )
        )

    db_path = tmp_path / "pipeline.duckdb"
    _populate_duckdb(db_path, date, episodes)

    # Assign per-flight lat/lon so that distances are meaningfully different:
    #   TID0 (i=0,3): near ~30 km north of camera site
    #   TID1 (i=1,4): far ~189 km west of camera site
    #   TID2 (i=2):   medium ~14 km northwest of camera site
    # Camera site: lat=42.360444, lon=-71.089238
    _flight_lat = {0: 42.630444, 1: 42.360444, 2: 42.460444, 3: 42.630444, 4: 42.360444}
    _flight_lon = {0: -71.089238, 1: -73.389238, 2: -71.189238, 3: -71.089238, 4: -73.389238}

    # Projections: 10 pings per flight across the day.
    projections: list[dict] = []
    for i in range(5):
        for k in range(10):
            t = base_ts + datetime.timedelta(minutes=10 * i, seconds=k)
            projections.append(
                {
                    "wall_time_utc": t.isoformat(),
                    "callsign": f"FL{i}",
                    "transponder_id": f"TID{i % 3}",
                    "pixel_x": 500.0 + k,
                    "pixel_y": 600.0 + k,
                    "path_dx": 1.0,
                    "path_dy": 0.0,
                    "roi": {"x": 400, "y": 500, "w": 200, "h": 200},
                    "lat": _flight_lat[i],
                    "lon": _flight_lon[i],
                }
            )
    proj_path = tmp_path / "projections.jsonl"
    _write_jsonl(proj_path, projections)

    # Detections: 5 per episode at onset..end.
    detections: list[dict] = []
    for ep in episodes:
        for k in range(5):
            t = ep.onset + datetime.timedelta(seconds=k)
            detections.append(
                {
                    "wall_time_utc": t.isoformat(),
                    "callsign": ep.callsign,
                    "transponder_id": ep.transponder_id,
                    "score": ep.peak_score,
                    "pixel_line": list(ep.peak_pixel_line),
                    "method": "hough_canny",
                }
            )
    det_path = tmp_path / "detections.jsonl"
    _write_jsonl(det_path, detections)

    video_path = tmp_path / "day.mp4"
    video_path.write_bytes(b"fake-video-bytes")

    # OCR jsonl: one record per frame; frame 0 anchors video start.
    ocr_records = [
        {
            "frame_idx": 0,
            "wall_time_utc": ocr_start.isoformat(),
            "confidence": 0.9,
            "method": "template",
            "ocr_status": "ok",
            "tracker_status": "ok",
        }
    ]
    ocr_path = tmp_path / "ocr.jsonl"
    _write_jsonl(ocr_path, ocr_records)

    return {
        "date": date,
        "db": db_path,
        "projections": proj_path,
        "detections": det_path,
        "video": video_path,
        "episodes": episodes,
        "ocr": ocr_path,
        "video_start_utc": ocr_start.isoformat(),
    }


def test_generate_bundles_creates_manifest_and_html(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    out = tmp_path / "bundles"
    result = generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice", "bob"],
        overlap_fraction=0.2,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
    )
    assert set(result.keys()) == {"alice", "bob"}
    for lbl in ("alice", "bob"):
        bundle_dir = result[lbl]
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "labeler.html").exists()


def test_manifest_covers_all_episodes_together(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    out = tmp_path / "bundles"
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice", "bob"],
        overlap_fraction=0.4,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
    )
    with open(out / "alice" / "manifest.json") as f:
        alice = json.load(f)
    with open(out / "bob" / "manifest.json") as f:
        bob = json.load(f)

    alice_ids = {ep["episode_id"] for ep in alice["episodes"]}
    bob_ids = {ep["episode_id"] for ep in bob["episodes"]}
    assert alice_ids | bob_ids == {1, 2, 3, 4, 5}
    # With 5 episodes and 0.4 fraction: round(5*0.4) = 2 overlap.
    assert len(alice_ids & bob_ids) == 2
    assert set(alice["overlap_episode_ids"]) == alice_ids & bob_ids


def test_manifest_episode_has_frames_and_peak_line(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    out = tmp_path / "bundles"
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice"],
        overlap_fraction=0.0,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
    )
    with open(out / "alice" / "manifest.json") as f:
        m = json.load(f)
    assert m["schema_version"] == 1
    assert m["date"] == "2026-04-08"
    assert m["labeler_id"] == "alice"
    assert len(m["episodes"]) == 5
    for ep in m["episodes"]:
        assert "frames" in ep and len(ep["frames"]) == 5
        assert ep["peak_pixel_line"] is not None
        assert len(ep["peak_pixel_line"]) == 4
    # Flight tracks include all three flight ids.
    assert set(m["flight_tracks"].keys()) == {"TID0", "TID1", "TID2"}


def test_manifest_video_path_is_relative(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    out = tmp_path / "bundles"
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice"],
        overlap_fraction=0.0,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
    )
    with open(out / "alice" / "manifest.json") as f:
        m = json.load(f)
    # Video is symlinked into the bundle dir as "video.mp4" (no path traversal
    # needed — the symlink sits alongside manifest.json and labeler.html).
    assert m["video"]["path"] == "video.mp4"
    # The symlink must exist in the bundle dir.
    assert (out / "alice" / "video.mp4").is_symlink()


def test_generate_bundles_is_deterministic(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    kwargs = dict(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice", "bob"],
        overlap_fraction=0.2,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
    )
    generate_bundles(output_dir=out1, **kwargs)
    generate_bundles(output_dir=out2, **kwargs)
    for lbl in ("alice", "bob"):
        m1 = json.loads((out1 / lbl / "manifest.json").read_text())
        m2 = json.loads((out2 / lbl / "manifest.json").read_text())
        # generated_at differs across runs, and the video paths differ because
        # the bundle dirs differ. Strip both before comparing.
        for m in (m1, m2):
            m.pop("generated_at", None)
            m.pop("video", None)
        assert m1 == m2


# ---------- CLI ----------


def test_manifest_includes_video_start_utc(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    out = tmp_path / "bundles"
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice"],
        overlap_fraction=0.0,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
        ocr_path=synthetic_pipeline_outputs["ocr"],
        detection_threshold=0.45,
    )
    with open(out / "alice" / "manifest.json") as f:
        m = json.load(f)
    assert m["video"]["start_utc"] == synthetic_pipeline_outputs["video_start_utc"]
    assert m["detection_threshold"] == 0.45


def test_manifest_omits_start_utc_when_ocr_missing(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    out = tmp_path / "bundles"
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice"],
        overlap_fraction=0.0,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
    )
    with open(out / "alice" / "manifest.json") as f:
        m = json.load(f)
    assert "start_utc" not in m["video"]
    # Default threshold still serialized.
    assert "detection_threshold" in m


# ---------- Calibration block ----------


def _synthetic_calibration() -> SimpleNamespace:
    """Build the smallest object that satisfies calibration_block().

    calibration_block() reads exactly six attributes:
      camera_matrix            – ndarray (3, 3)
      distortion_coefficients  – ndarray with .flatten() → 1-D iterable
      rotation                 – ndarray (3, 3), .tolist()
      translation              – ndarray with .flatten() → 1-D iterable
      camera_gps               – indexable; [2] is the camera altitude in metres
      calibration_resolution   – iterable of two ints (width, height)

    We use SimpleNamespace so the real Calibration.__init__ (which calls
    cv2.Rodrigues and sets up pyproj transforms) is never triggered, keeping
    the test fast and dependency-free.
    """
    cam_matrix = np.array(
        [[2000.0, 0.0, 1920.0],
         [0.0,    2000.0, 1080.0],
         [0.0,    0.0,    1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.array([0.1, -0.05, 0.0, 0.0, 0.01], dtype=np.float64).reshape(-1, 1)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.array([[0.5], [-0.3], [10.0]], dtype=np.float64)
    camera_gps = np.array([42.36, -71.09, 75.0])  # lat, lon, alt_m

    return SimpleNamespace(
        camera_matrix=cam_matrix,
        distortion_coefficients=dist_coeffs,
        rotation=rotation,
        translation=translation,
        camera_gps=camera_gps,
        calibration_resolution=(3840, 2160),
    )


def test_calibration_block_keys_and_types() -> None:
    """calibration_block() must return a dict with the expected keys and types."""
    calib = _synthetic_calibration()
    block = calibration_block(calib)

    # Exact key set.
    assert set(block.keys()) == {
        "camera_matrix",
        "distortion_coefficients",
        "rotation",
        "translation",
        "camera_alt_m",
        "calibration_resolution",
    }

    # camera_matrix: 3×3 nested list of floats.
    assert isinstance(block["camera_matrix"], list)
    assert len(block["camera_matrix"]) == 3
    assert all(len(row) == 3 for row in block["camera_matrix"])
    assert block["camera_matrix"][0][0] == pytest.approx(2000.0)
    assert block["camera_matrix"][0][2] == pytest.approx(1920.0)

    # distortion_coefficients: flat list (flatten() was called).
    assert isinstance(block["distortion_coefficients"], list)
    assert block["distortion_coefficients"][0] == pytest.approx(0.1)

    # rotation: 3×3 nested list; identity in our synthetic fixture.
    assert block["rotation"] == [[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]]

    # translation: flat list (flatten() was called).
    assert isinstance(block["translation"], list)
    assert block["translation"][2] == pytest.approx(10.0)

    # camera_alt_m: float, taken from camera_gps[2].
    assert isinstance(block["camera_alt_m"], float)
    assert block["camera_alt_m"] == pytest.approx(75.0)

    # calibration_resolution: list of two ints.
    assert block["calibration_resolution"] == [3840, 2160]


def test_generate_bundles_injects_calibration_block(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    """generate_bundles(..., calibration=<calib>) must write a 'calibration'
    key into every labeler's manifest.json, with the keys calibration_block
    produces."""
    out = tmp_path / "bundles"
    calib = _synthetic_calibration()
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice"],
        overlap_fraction=0.0,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
        calibration=calib,
    )
    with open(out / "alice" / "manifest.json") as f:
        m = json.load(f)

    assert "calibration" in m
    cb = m["calibration"]
    assert set(cb.keys()) == {
        "camera_matrix",
        "distortion_coefficients",
        "rotation",
        "translation",
        "camera_alt_m",
        "calibration_resolution",
    }
    assert cb["camera_alt_m"] == pytest.approx(75.0)
    assert cb["calibration_resolution"] == [3840, 2160]


def test_generate_bundles_omits_calibration_when_not_provided(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    """Omitting calibration= must leave the manifest without a 'calibration'
    key (None-safe; the default is None)."""
    out = tmp_path / "bundles"
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice"],
        overlap_fraction=0.0,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
    )
    with open(out / "alice" / "manifest.json") as f:
        m = json.load(f)
    assert "calibration" not in m


# ---------- Labeler HTML structure ----------


def test_labeler_html_has_overlay_controls_and_logic(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    """The copied labeler.html must contain the canvas-overlay wiring that the
    student sees — toggles, manifest fields, per-frame draw loop, time mapping.
    Verified statically because we have no headless browser in the test env."""
    out = tmp_path / "bundles"
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice"],
        overlap_fraction=0.0,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
        ocr_path=synthetic_pipeline_outputs["ocr"],
    )
    html = (out / "alice" / "labeler.html").read_text()

    # Toggle controls (item 12 step 5).
    assert 'id="toggle-tracks"' in html
    assert 'id="toggle-detections"' in html

    # Canvas overlay and video element.
    assert 'id="overlay"' in html
    assert 'id="video"' in html

    # Time mapping and key manifest fields.
    assert "manifest.video.start_utc" in html
    assert "seconds_per_frame" in html
    assert "detection_threshold" in html

    # Draw loop synced via requestVideoFrameCallback (RVFC) for frame-accurate
    # overlay. The labeler falls back to rAF when RVFC is unavailable, and
    # still redraws on explicit seek/pause events so the overlay is correct
    # when the video is not playing.
    assert "requestVideoFrameCallback" in html
    assert 'addEventListener("play"' in html
    assert 'addEventListener("seeked"' in html
    assert 'addEventListener("pause"' in html
    assert 'addEventListener("loadeddata"' in html

    # Track and detection drawing routines.
    assert "drawTrack" in html
    assert "drawDetections" in html


def test_labeler_html_has_label_controls_and_export(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    """Item 13: verify label controls, localStorage persistence, and export
    are wired into the template. Static assertions because we have no
    headless browser in CI — see item 12 tests for the same pattern."""
    out = tmp_path / "bundles"
    generate_bundles(
        date=synthetic_pipeline_outputs["date"],
        labelers=["alice"],
        overlap_fraction=0.0,
        db_path=synthetic_pipeline_outputs["db"],
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        video_path=synthetic_pipeline_outputs["video"],
        image_size=(3840, 2160),
        output_dir=out,
    )
    html = (out / "alice" / "labeler.html").read_text()

    # Three-way label values come from a VALID_LABELS array that is injected
    # into radio `value=` attributes at render time.
    assert '"contrail"' in html
    assert '"no_contrail"' in html
    assert '"unsure"' in html
    assert "VALID_LABELS" in html
    assert 'type="radio"' in html

    # Persistence slider.
    assert 'type="range"' in html
    assert 'min="1"' in html and 'max="5"' in html

    # Notes textarea (constructed dynamically; test the hooks instead).
    assert 'createElement("textarea")' in html
    assert "label_notes" in html

    # localStorage persistence: per (date, labeler) key + save/load helpers.
    assert "localStorage" in html
    assert "concam-labels:" in html
    assert "loadLabels" in html and "saveLabels" in html

    # Export button and JSON download path matching insert_labels schema.
    assert 'id="export-btn"' in html
    assert "labelsForExport" in html
    assert "episode_id" in html and "labeler_id" in html
    assert "persistence_rating" in html and "label_notes" in html
    assert "label_timestamp" in html

    # Manifest+error handling already covered by item 12 test; re-check the
    # user-facing error panel so a regression that removes it is caught here.
    assert 'id="error-panel"' in html


def test_cli_bundle_missing_outputs(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "bundle",
            "--date",
            "2026-04-08",
            "--labelers",
            "alice",
            "--labelers",
            "bob",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "missing" in result.output.lower()


def test_cli_bundle_end_to_end(
    synthetic_pipeline_outputs: dict, tmp_path: Path
) -> None:
    # Lay out the synthetic outputs where stage_paths expects them.
    output_root = tmp_path / "out"
    date_dir = output_root / "2026-04-08"
    date_dir.mkdir(parents=True)
    (date_dir / "pipeline.duckdb").write_bytes(
        synthetic_pipeline_outputs["db"].read_bytes()
    )
    (date_dir / "projections.jsonl").write_text(
        synthetic_pipeline_outputs["projections"].read_text()
    )
    (date_dir / "detections.jsonl").write_text(
        synthetic_pipeline_outputs["detections"].read_text()
    )
    # Also stage the OCR file so the CLI picks up video_start_utc.
    (date_dir / "ocr.jsonl").write_text(
        synthetic_pipeline_outputs["ocr"].read_text()
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "bundle",
            "--date",
            "2026-04-08",
            "--labelers",
            "alice",
            "--labelers",
            "bob",
            "--output-dir",
            str(output_root),
            "--video",
            str(synthetic_pipeline_outputs["video"]),
        ],
    )
    assert result.exit_code == 0, result.output
    bundles_root = date_dir / "bundles"
    assert (bundles_root / "alice" / "manifest.json").exists()
    assert (bundles_root / "bob" / "manifest.json").exists()
    assert (bundles_root / "alice" / "labeler.html").exists()
    # CLI should propagate video_start_utc and detection_threshold from config.
    with open(bundles_root / "alice" / "manifest.json") as f:
        m = json.load(f)
    assert m["video"]["start_utc"] == synthetic_pipeline_outputs["video_start_utc"]
    assert "detection_threshold" in m


# ---------- Public bundle distance fields ----------


def test_flight_tracks_pings_have_dist_km(
    synthetic_pipeline_outputs: dict,
) -> None:
    """Every ping in flight_tracks must have a numeric dist_km >= 0."""
    tracks = _build_flight_tracks_with_altitude(
        synthetic_pipeline_outputs["projections"],
        site_lat=_SITE_LAT,
        site_lon=_SITE_LON,
    )
    for tid, track in tracks.items():
        for ping in track["pings"]:
            assert "dist_km" in ping, f"TID {tid}: ping missing dist_km"
            assert ping["dist_km"] is not None, f"TID {tid}: dist_km is None"
            assert isinstance(ping["dist_km"], float), f"TID {tid}: dist_km not float"
            assert ping["dist_km"] >= 0.0, f"TID {tid}: dist_km < 0"


def test_flight_tracks_dist_km_values(
    synthetic_pipeline_outputs: dict,
) -> None:
    """TID0 (near ~30 km) and TID1 (far ~189 km) have clearly different dist_km."""
    tracks = _build_flight_tracks_with_altitude(
        synthetic_pipeline_outputs["projections"],
        site_lat=_SITE_LAT,
        site_lon=_SITE_LON,
    )
    tid0_dists = [p["dist_km"] for p in tracks["TID0"]["pings"]]
    tid1_dists = [p["dist_km"] for p in tracks["TID1"]["pings"]]
    # All TID0 pings share the same lat/lon, so all distances are equal.
    assert all(d == tid0_dists[0] for d in tid0_dists)
    assert all(d == tid1_dists[0] for d in tid1_dists)
    # Near flight clearly closer than far flight.
    assert tid0_dists[0] < 50.0, f"TID0 dist unexpectedly large: {tid0_dists[0]}"
    assert tid1_dists[0] > 100.0, f"TID1 dist unexpectedly small: {tid1_dists[0]}"
    # Exact rounded values from fixture coords (verified by hand).
    assert tid0_dists[0] == pytest.approx(30.02), f"TID0 dist_km={tid0_dists[0]}"
    assert tid1_dists[0] == pytest.approx(188.97), f"TID1 dist_km={tid1_dists[0]}"


def test_episodes_have_closest_approach_km(
    synthetic_pipeline_outputs: dict,
) -> None:
    """Every episode in the public bundle must have closest_approach_km (float or None)."""
    source_manifest = {
        "schema_version": 1,
        "date": "2026-04-08",
        "video": {"path": "video.mp4"},
        "image_size": [3840, 2160],
        "flight_tracks": {},
    }
    manifest = build_manifest(
        source_manifest=source_manifest,
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        max_gap_seconds=30.0,
        detection_threshold=0.45,
        site_lat=_SITE_LAT,
        site_lon=_SITE_LON,
    )
    for ep in manifest["episodes"]:
        assert "closest_approach_km" in ep, f"episode {ep['episode_id']} missing field"
        val = ep["closest_approach_km"]
        assert val is None or isinstance(val, float), (
            f"episode {ep['episode_id']} closest_approach_km has wrong type: {type(val)}"
        )
        if val is not None:
            assert val >= 0.0


def test_episode_closest_approach_km_value(
    synthetic_pipeline_outputs: dict,
) -> None:
    """Episode for TID0 (near flight) must have closest_approach_km == 30.02.

    Episode i=0 has transponder_id=TID0, onset=base_ts, end=base_ts+5s.
    TID0 pings for i=0 span base_ts+0s..base_ts+9s; those with t in [onset,end]
    are k=0..5 (6 pings), all at the same lat/lon -> dist_km=30.02 for all.
    So min == 30.02.
    """
    source_manifest = {
        "schema_version": 1,
        "date": "2026-04-08",
        "video": {"path": "video.mp4"},
        "image_size": [3840, 2160],
        "flight_tracks": {},
    }
    manifest = build_manifest(
        source_manifest=source_manifest,
        projections_path=synthetic_pipeline_outputs["projections"],
        detections_path=synthetic_pipeline_outputs["detections"],
        max_gap_seconds=30.0,
        detection_threshold=0.45,
        site_lat=_SITE_LAT,
        site_lon=_SITE_LON,
    )
    # Find TID0 episodes (there may be multiple runs if pings are gapped).
    tid0_eps = [ep for ep in manifest["episodes"] if ep["transponder_id"] == "TID0"]
    assert len(tid0_eps) > 0, "No TID0 episodes found"
    # All TID0 pings are at the same near location, so every TID0 episode
    # that has in-window pings gets closest_approach_km == 30.02.
    for ep in tid0_eps:
        assert ep["closest_approach_km"] == pytest.approx(30.02), (
            f"TID0 episode {ep['episode_id']} closest_approach_km="
            f"{ep['closest_approach_km']!r}"
        )

    # TID1 episodes must be at the far distance.
    tid1_eps = [ep for ep in manifest["episodes"] if ep["transponder_id"] == "TID1"]
    for ep in tid1_eps:
        assert ep["closest_approach_km"] == pytest.approx(188.97), (
            f"TID1 episode {ep['episode_id']} closest_approach_km="
            f"{ep['closest_approach_km']!r}"
        )


def test_dist_km_none_for_missing_lat_lon(tmp_path: Path) -> None:
    """Pings with missing lat/lon must have dist_km=None (older data robustness)."""
    proj_path = tmp_path / "proj.jsonl"
    base_ts = datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)
    records = [
        {
            "wall_time_utc": (base_ts + datetime.timedelta(seconds=k)).isoformat(),
            "callsign": "FL0",
            "transponder_id": "TID0",
            "pixel_x": 500.0,
            "pixel_y": 600.0,
            # No lat/lon fields
        }
        for k in range(3)
    ]
    with open(proj_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    tracks = _build_flight_tracks_with_altitude(
        proj_path, site_lat=_SITE_LAT, site_lon=_SITE_LON
    )
    for ping in tracks["TID0"]["pings"]:
        assert ping["dist_km"] is None, "Expected None for ping without lat/lon"
