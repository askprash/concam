"""Tests for the labeler bundle generator."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from concam.aggregation import Episode
from concam.bundle import (
    Assignment,
    _relative_video_path,
    assign_episodes,
    generate_bundles,
)
from concam.cli import main as cli_main
from concam.storage import Database


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


# ---------- Relative video path ----------


def test_relative_video_path(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundles" / "alice"
    bundle_dir.mkdir(parents=True)
    video = tmp_path / "videos" / "day.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    rel = _relative_video_path(video, bundle_dir)
    # Expected two levels up then into videos/
    assert rel == str(Path("..") / ".." / "videos" / "day.mp4")


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
    # Video is at tmp_path/day.mp4; bundle is at tmp_path/bundles/alice/
    # So the relative path should climb two levels and land on day.mp4.
    assert m["video"]["path"].endswith("day.mp4")
    assert m["video"]["path"].startswith("..")
    # Not an absolute path.
    assert not Path(m["video"]["path"]).is_absolute()


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

    # Draw loop synced to timeupdate + seeked.
    assert 'addEventListener("timeupdate"' in html
    assert 'addEventListener("seeked"' in html

    # Track and detection drawing routines.
    assert "drawTrack" in html
    assert "drawDetections" in html


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
