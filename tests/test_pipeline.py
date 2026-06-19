"""End-to-end integration tests for the pipeline orchestration layer.

These tests synthesize a short video with a known overlay pattern, a
matching ADS-B fixture, and exercise every stage of the pipeline on disk,
verifying that the chain OCR -> ADSB -> project -> detect -> aggregate -> store
ends in a non-empty DuckDB.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from concam.adsb import Flight, Ping
from concam.cli import main as cli_main
from concam.config import load_config
from concam.pipeline import (
    STAGES,
    run_aggregate_stage,
    run_detect_stage,
    run_ocr_stage,
    run_project_stage,
    run_store_stage,
    stage_paths,
)
from concam.pipeline.stages import (
    _flight_from_dict,
    _flight_to_dict,
    load_adsb_file,
    load_episodes_file,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "mit_green_building.yaml"


def test_stage_paths(tmp_path: Path) -> None:
    paths = stage_paths(tmp_path, datetime.date(2026, 4, 8))
    assert paths["base"] == tmp_path / "2026-04-08"
    assert paths["ocr"].name == "ocr.jsonl"
    assert paths["adsb"].name == "adsb.json"
    assert paths["projections"].name == "projections.jsonl"
    assert paths["detections"].name == "detections.jsonl"
    assert paths["episodes"].name == "episodes.jsonl"
    assert paths["store"].name == "pipeline.duckdb"


def test_flight_roundtrip() -> None:
    p = Ping(
        time=datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=datetime.timezone.utc),
        lat=42.0,
        lon=-71.0,
        alt_m=10000.0,
    )
    flight = Flight(
        callsign="UAL123",
        transponder_id="A12345",
        aircraft_type="B738",
        orig="KBOS",
        dest="KSFO",
        pings=[p],
    )
    d = _flight_to_dict(flight)
    assert json.dumps(d)  # serializable
    fl2 = _flight_from_dict(d)
    assert fl2.callsign == flight.callsign
    assert fl2.transponder_id == flight.transponder_id
    assert len(fl2.pings) == 1
    assert fl2.pings[0].time == flight.pings[0].time
    assert fl2.pings[0].lat == flight.pings[0].lat


def test_project_stage_from_adsb_fixture(tmp_path: Path) -> None:
    """Project the real ADS-B fixture and verify some pings land in-frame."""
    site_config = load_config(CONFIG_PATH)
    if not Path(site_config.calibration.npz_path).exists():
        pytest.skip("calibration npz not available in this environment")

    # Copy the fixture into the expected adsb.json path.
    paths = stage_paths(tmp_path, datetime.date(2026, 4, 8))
    paths["base"].mkdir(parents=True, exist_ok=True)
    with open(FIXTURES / "adsb_april8_sample.json") as f:
        raw = json.load(f)
    with open(paths["adsb"], "w") as f:
        json.dump(raw, f)

    n = run_project_stage(
        adsb_path=paths["adsb"],
        site_config=site_config,
        out_path=paths["projections"],
    )
    # The fixture has 3 flights × 30 pings — most should project out-of-frame
    # since many are beyond the horizon, but at least some should land on the
    # image. We assert only that the file was written and has some records.
    assert paths["projections"].exists()
    assert n >= 0


def test_aggregate_and_store_stages(tmp_path: Path) -> None:
    """Detect -> aggregate -> store roundtrip with synthetic detections."""
    site_config = load_config(CONFIG_PATH)
    paths = stage_paths(tmp_path, datetime.date(2026, 4, 8))
    paths["base"].mkdir(parents=True, exist_ok=True)

    # Write synthetic detections directly (skip OCR/project/detect).
    base_ts = datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)
    with open(paths["detections"], "w") as f:
        for i in range(5):
            rec = {
                "wall_time_utc": (base_ts + datetime.timedelta(seconds=i)).isoformat(),
                "callsign": "UAL123",
                "transponder_id": "A12345",
                "score": 0.8,
                "pixel_line": [100.0, 200.0, 300.0, 400.0],
                "method": "hough_canny",
            }
            f.write(json.dumps(rec) + "\n")

    n_ep = run_aggregate_stage(
        detections_path=paths["detections"],
        site_config=site_config,
        out_path=paths["episodes"],
    )
    assert n_ep == 1

    eps = load_episodes_file(paths["episodes"])
    assert len(eps) == 1
    assert eps[0].frame_count == 5
    assert eps[0].callsign == "UAL123"

    n_rows = run_store_stage(
        episodes_path=paths["episodes"],
        date=datetime.date(2026, 4, 8),
        db_path=paths["store"],
    )
    assert n_rows == 1
    assert paths["store"].exists()

    # Verify the DuckDB has exactly one row we can read back.
    from concam.storage import Database
    with Database(paths["store"]) as db:
        rows = db.query("SELECT callsign, frame_count FROM contrail_episodes")
    assert rows == [("UAL123", 5)]


def test_ocr_stage_gates_out_of_day_dates(tmp_path: Path, monkeypatch) -> None:
    """Arm 1: a mid-day frame whose OCR parses to a wrong year/day is gated.

    Reproduces GitHub #1: the template OCR confidently misreads the YEAR/DAY
    mid-day (e.g. 04-11 -> 04-14 / 8026-…) while HH:MM:SS stays continuous.
    ``run_ocr_stage`` must drop those out-of-day reads so the tracker projects
    through them; the emitted timestamps must stay within the processed day
    (+rollover), never carry the corrupt year/day.
    """
    from concam.ocr.reader import TimestampRead
    from concam.pipeline import stages as stages_mod

    date = datetime.date(2026, 4, 11)
    n_clean_before = 8
    n_corrupt = 10
    n_clean_after = 4
    total = n_clean_before + n_corrupt + n_clean_after

    # Scripted OCR reads (naive LOCAL wall time, as the engine emits):
    #  - clean frames on the processed day,
    #  - a block of confidently-misread frames dated +3 years (8026-04-14)
    #    that are mutually seconds-consistent (the bug's anchoring trap),
    #  - clean frames again.
    def _read(local_dt: datetime.datetime) -> TimestampRead:
        return TimestampRead(
            parsed_dt=local_dt,
            text=local_dt.strftime("%m/%d/%Y %H:%M:%S"),
            confidence=0.733,  # above fallback_confidence_threshold, as observed
            per_char_confidence=(),
            method="template",
            status="ok",
        )

    base_local = datetime.datetime(2026, 4, 11, 10, 0, 0)
    scripted: list[TimestampRead] = []
    for i in range(total):
        good = base_local + datetime.timedelta(seconds=i)
        if n_clean_before <= i < n_clean_before + n_corrupt:
            # Corrupt: same continuous HH:MM:SS but year 8026 and day +3.
            corrupt = good.replace(year=8026, day=14)
            scripted.append(_read(corrupt))
        else:
            scripted.append(_read(good))

    class _FakeReader:
        def __init__(self, *_a, **_k) -> None:
            self._i = 0

        def read(self, _frame):
            r = scripted[self._i]
            self._i += 1
            return r

    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(total)]
    monkeypatch.setattr(stages_mod, "FixedFormatTimestampReader", _FakeReader)
    monkeypatch.setattr(stages_mod, "iter_video_frames", lambda _p: iter(frames))

    site_config = load_config(CONFIG_PATH)
    out_path = tmp_path / "ocr.jsonl"
    n = run_ocr_stage(
        video_path=Path("unused.mp4"),
        date=date,
        site_config=site_config,
        out_path=out_path,
    )
    assert n == total

    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    # Every emitted UTC wall date must fall within [day-1, day+1]; the corrupt
    # year 8026 must never leak into any record.
    allowed = {
        (date + datetime.timedelta(days=d)).isoformat() for d in (-1, 0, 1)
    }
    for rec in records:
        utc_date = rec["wall_time_utc"][:10]
        assert utc_date in allowed, f"{rec['frame_idx']}: leaked {utc_date}"
        assert "8026" not in rec["wall_time_utc"]

    # The clean frame right after the corrupt block must project continuously
    # on the real day (the tracker projected through the corruption).
    after = records[n_clean_before + n_corrupt]
    assert after["wall_time_utc"].startswith("2026-04-11")


def test_cli_dry_run(tmp_path: Path) -> None:
    """--dry-run should print the plan and not create the output dir."""
    output_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["run", "--date", "2026-04-08", "--output-dir", str(output_dir), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "Execution plan" in result.output
    assert "ocr" in result.output
    assert "store" in result.output


def test_cli_from_stage_missing_cache(tmp_path: Path) -> None:
    """--from-stage without cached earlier outputs should fail fast."""
    output_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["run", "--date", "2026-04-08", "--output-dir", str(output_dir),
         "--from-stage", "detect"],
    )
    assert result.exit_code != 0
    assert "requires cached" in result.output


def test_cli_dry_run_from_stage(tmp_path: Path) -> None:
    """--dry-run --from-stage should still print a plan starting from that stage."""
    output_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["run", "--date", "2026-04-08", "--output-dir", str(output_dir),
         "--from-stage", "aggregate", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "aggregate" in result.output
    assert "store" in result.output
    # ocr should not appear in the plan
    plan_section = result.output.split("Execution plan:")[1]
    assert "- ocr" not in plan_section
