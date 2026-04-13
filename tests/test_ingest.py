"""Tests for concam.ingest: label JSON validation and DuckDB ingestion."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from concam.aggregation import Episode
from concam.cli import main as cli_main
from concam.ingest import (
    LabelValidationError,
    ingest_label_files,
    load_label_file,
    validate_payload,
)
from concam.storage import Database

UTC = datetime.timezone.utc
TEST_DATE = datetime.date(2026, 4, 8)
BASE_TIME = datetime.datetime(2026, 4, 8, 14, 30, 0, tzinfo=UTC)


def _episode(callsign: str, transponder_id: str, offset: int = 0) -> Episode:
    return Episode(
        callsign=callsign,
        transponder_id=transponder_id,
        onset=BASE_TIME + datetime.timedelta(seconds=offset),
        end=BASE_TIME + datetime.timedelta(seconds=offset + 10),
        peak_score=0.8,
        peak_pixel_line=(1.0, 2.0, 3.0, 4.0),
        frame_count=11,
    )


def _payload(
    *,
    labeler_id: str = "alice",
    labels: list[dict] | None = None,
    date: str = "2026-04-08",
    schema_version: int = 1,
) -> dict:
    return {
        "schema_version": schema_version,
        "date": date,
        "labeler_id": labeler_id,
        "exported_at": "2026-04-13T12:00:00.000Z",
        "labels": labels if labels is not None else [
            {
                "episode_id": 1,
                "label": "contrail",
                "labeler_id": labeler_id,
                "persistence_rating": 4,
                "label_timestamp": "2026-04-13T11:59:00.000Z",
                "label_notes": "clear trail",
            },
        ],
    }


def _seed_db(path: Path, n_episodes: int = 2) -> None:
    with Database(path) as db:
        db.create_schema()
        episodes = [_episode(f"FL{i}", f"T{i}", offset=i * 100) for i in range(n_episodes)]
        db.insert_episodes(episodes, TEST_DATE)


# ---------------------------------------------------------------------------
# validate_payload — schema contract
# ---------------------------------------------------------------------------


def test_validate_happy_path():
    out = validate_payload(_payload(), expected_date=TEST_DATE)
    assert out["labeler_id"] == "alice"
    assert out["date"] == TEST_DATE
    assert len(out["labels"]) == 1
    lbl = out["labels"][0]
    assert lbl["episode_id"] == 1
    assert lbl["label"] == "contrail"
    assert lbl["labeler_id"] == "alice"
    assert lbl["persistence_rating"] == 4
    assert lbl["label_notes"] == "clear trail"
    # Timestamp coerced to tz-aware UTC datetime
    assert isinstance(lbl["label_timestamp"], datetime.datetime)
    assert lbl["label_timestamp"].tzinfo is not None
    assert lbl["label_timestamp"] == datetime.datetime(2026, 4, 13, 11, 59, 0, tzinfo=UTC)


def test_validate_rejects_wrong_schema_version():
    with pytest.raises(LabelValidationError, match="schema_version"):
        validate_payload(_payload(schema_version=2), expected_date=TEST_DATE)


def test_validate_rejects_date_mismatch():
    with pytest.raises(LabelValidationError, match="expected 2026-04-08"):
        validate_payload(_payload(date="2026-04-09"), expected_date=TEST_DATE)


def test_validate_rejects_unknown_label():
    p = _payload(labels=[{"episode_id": 1, "label": "maybe", "labeler_id": "alice"}])
    with pytest.raises(LabelValidationError, match="label"):
        validate_payload(p, expected_date=TEST_DATE)


def test_validate_rejects_inner_labeler_mismatch():
    p = _payload(labels=[{"episode_id": 1, "label": "contrail", "labeler_id": "mallory"}])
    with pytest.raises(LabelValidationError, match="labeler_id"):
        validate_payload(p, expected_date=TEST_DATE)


def test_validate_rejects_bad_persistence_rating():
    p = _payload(labels=[{
        "episode_id": 1, "label": "contrail", "labeler_id": "alice",
        "persistence_rating": 7,
    }])
    with pytest.raises(LabelValidationError, match="persistence_rating"):
        validate_payload(p, expected_date=TEST_DATE)


def test_validate_rejects_duplicate_episode_ids():
    p = _payload(labels=[
        {"episode_id": 1, "label": "contrail", "labeler_id": "alice"},
        {"episode_id": 1, "label": "no_contrail", "labeler_id": "alice"},
    ])
    with pytest.raises(LabelValidationError, match="duplicate episode_id"):
        validate_payload(p, expected_date=TEST_DATE)


def test_validate_rejects_naive_timestamp():
    p = _payload(labels=[{
        "episode_id": 1, "label": "contrail", "labeler_id": "alice",
        "label_timestamp": "2026-04-13T11:59:00",  # no tz
    }])
    with pytest.raises(LabelValidationError, match="tz-aware"):
        validate_payload(p, expected_date=TEST_DATE)


def test_validate_accepts_missing_optional_fields():
    p = _payload(labels=[{
        "episode_id": 1, "label": "unsure", "labeler_id": "alice",
    }])
    out = validate_payload(p, expected_date=TEST_DATE)
    lbl = out["labels"][0]
    assert "persistence_rating" not in lbl
    assert "label_timestamp" not in lbl
    assert "label_notes" not in lbl


def test_validate_drops_empty_notes():
    p = _payload(labels=[{
        "episode_id": 1, "label": "contrail", "labeler_id": "alice",
        "label_notes": "",
    }])
    out = validate_payload(p, expected_date=TEST_DATE)
    assert "label_notes" not in out["labels"][0]


# ---------------------------------------------------------------------------
# load_label_file — file-level errors
# ---------------------------------------------------------------------------


def test_load_invalid_json(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{not valid json")
    with pytest.raises(LabelValidationError, match="invalid JSON"):
        load_label_file(p, expected_date=TEST_DATE)


def test_load_reports_file_path_in_error(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(_payload(schema_version=99)))
    with pytest.raises(LabelValidationError, match=str(p)):
        load_label_file(p, expected_date=TEST_DATE)


# ---------------------------------------------------------------------------
# ingest_label_files — end-to-end with DuckDB
# ---------------------------------------------------------------------------


def test_ingest_single_file_updates_existing_row(tmp_path: Path):
    db_path = tmp_path / "pipeline.duckdb"
    _seed_db(db_path)

    label_file = tmp_path / "alice_labels.json"
    label_file.write_text(json.dumps(_payload()))

    counts = ingest_label_files(db_path, [label_file], expected_date=TEST_DATE)
    assert counts == {"alice": 1}

    with Database(db_path) as db:
        rows = db.query(
            "SELECT episode_id, label, labeler_id, persistence_rating, label_notes "
            "FROM contrail_episodes WHERE episode_id = 1"
        )
    assert len(rows) == 1
    assert rows[0] == (1, "contrail", "alice", 4, "clear trail")


def test_ingest_two_labelers_overlap_produces_two_rows(tmp_path: Path):
    """Both labelers labeling episode 1 → two rows; non-overlap each update in place."""
    db_path = tmp_path / "pipeline.duckdb"
    _seed_db(db_path, n_episodes=2)

    alice_file = tmp_path / "alice.json"
    alice_file.write_text(json.dumps(_payload(
        labeler_id="alice",
        labels=[
            {"episode_id": 1, "label": "contrail", "labeler_id": "alice"},
            {"episode_id": 2, "label": "no_contrail", "labeler_id": "alice"},
        ],
    )))
    bob_file = tmp_path / "bob.json"
    bob_file.write_text(json.dumps(_payload(
        labeler_id="bob",
        labels=[
            {"episode_id": 1, "label": "unsure", "labeler_id": "bob"},
        ],
    )))

    counts = ingest_label_files(db_path, [alice_file, bob_file], expected_date=TEST_DATE)
    assert counts == {"alice": 2, "bob": 1}

    with Database(db_path) as db:
        # Episode 1 has two rows (overlap)
        rows = db.query(
            "SELECT labeler_id, label FROM contrail_episodes "
            "WHERE episode_id = 1 ORDER BY labeler_id"
        )
        assert rows == [("alice", "contrail"), ("bob", "unsure")]
        # Episode 2 has exactly one row, alice's
        rows2 = db.query(
            "SELECT labeler_id, label FROM contrail_episodes WHERE episode_id = 2"
        )
        assert rows2 == [("alice", "no_contrail")]


def test_ingest_aborts_on_bad_file_before_writing(tmp_path: Path):
    """If any input file is malformed, the DB should be untouched."""
    db_path = tmp_path / "pipeline.duckdb"
    _seed_db(db_path)

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_payload()))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_payload(schema_version=99)))

    with pytest.raises(LabelValidationError):
        ingest_label_files(db_path, [good, bad], expected_date=TEST_DATE)

    with Database(db_path) as db:
        rows = db.query("SELECT label FROM contrail_episodes WHERE episode_id = 1")
    assert rows == [(None,)]


def test_ingest_persists_label_timestamp(tmp_path: Path):
    db_path = tmp_path / "pipeline.duckdb"
    _seed_db(db_path)

    f = tmp_path / "alice.json"
    f.write_text(json.dumps(_payload()))
    ingest_label_files(db_path, [f], expected_date=TEST_DATE)

    with Database(db_path) as db:
        rows = db.query(
            "SELECT label_timestamp FROM contrail_episodes WHERE episode_id = 1"
        )
    assert len(rows) == 1
    ts = rows[0][0]
    assert ts is not None
    # DuckDB returns tz-aware datetime; compare against the UTC time we supplied.
    # Normalise both sides to UTC for robust comparison across DuckDB tz quirks.
    assert ts.astimezone(UTC) == datetime.datetime(2026, 4, 13, 11, 59, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_ingest_labels_happy_path(tmp_path: Path):
    """`concam ingest-labels` resolves DB via --output-dir and writes labels."""
    date_dir = tmp_path / "output" / TEST_DATE.isoformat()
    date_dir.mkdir(parents=True)
    db_path = date_dir / "pipeline.duckdb"
    _seed_db(db_path)

    label_file = tmp_path / "alice.json"
    label_file.write_text(json.dumps(_payload()))

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "ingest-labels",
        "--date", TEST_DATE.isoformat(),
        "--output-dir", str(tmp_path / "output"),
        "--labels", str(label_file),
    ])
    assert result.exit_code == 0, result.output
    assert "alice: 1 labels" in result.output

    with Database(db_path) as db:
        rows = db.query(
            "SELECT label, labeler_id FROM contrail_episodes WHERE episode_id = 1"
        )
    assert rows == [("contrail", "alice")]


def test_cli_ingest_labels_missing_db(tmp_path: Path):
    label_file = tmp_path / "alice.json"
    label_file.write_text(json.dumps(_payload()))

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "ingest-labels",
        "--date", TEST_DATE.isoformat(),
        "--output-dir", str(tmp_path / "output"),
        "--labels", str(label_file),
    ])
    assert result.exit_code != 0
    assert "DuckDB not found" in result.output


def test_cli_ingest_labels_malformed_file(tmp_path: Path):
    date_dir = tmp_path / "output" / TEST_DATE.isoformat()
    date_dir.mkdir(parents=True)
    db_path = date_dir / "pipeline.duckdb"
    _seed_db(db_path)

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "ingest-labels",
        "--date", TEST_DATE.isoformat(),
        "--output-dir", str(tmp_path / "output"),
        "--labels", str(bad),
    ])
    assert result.exit_code != 0
    assert "invalid JSON" in result.output
