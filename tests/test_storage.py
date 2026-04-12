"""Tests for concam.storage: DuckDB episode and label storage."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from concam.aggregation import Episode
from concam.storage import Database

UTC = datetime.timezone.utc
BASE_TIME = datetime.datetime(2026, 4, 8, 14, 30, 0, tzinfo=UTC)
TEST_DATE = datetime.date(2026, 4, 8)


def _episode(
    callsign: str = "UAL123",
    transponder_id: str = "A12345",
    onset_offset: int = 0,
    end_offset: int = 10,
    peak_score: float = 0.85,
    peak_pixel_line: tuple[float, float, float, float] | None = (100.0, 200.0, 300.0, 400.0),
    frame_count: int = 11,
) -> Episode:
    return Episode(
        callsign=callsign,
        transponder_id=transponder_id,
        onset=BASE_TIME + datetime.timedelta(seconds=onset_offset),
        end=BASE_TIME + datetime.timedelta(seconds=end_offset),
        peak_score=peak_score,
        peak_pixel_line=peak_pixel_line,
        frame_count=frame_count,
    )


@pytest.fixture
def db(tmp_path: Path) -> Database:
    """Create a fresh in-tmpdir database for each test."""
    d = Database(tmp_path / "test.duckdb")
    d.create_schema()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


def test_create_schema_idempotent(db: Database):
    """create_schema can be called twice without error."""
    db.create_schema()  # second call
    rows = db.query("SELECT COUNT(*) FROM contrail_episodes")
    assert rows[0][0] == 0


# ---------------------------------------------------------------------------
# Episode insert and round-trip
# ---------------------------------------------------------------------------


def test_insert_and_query_episode(db: Database):
    """Insert one episode and query it back with identical values."""
    ep = _episode()
    db.insert_episodes([ep], TEST_DATE)

    rows = db.query(
        "SELECT episode_id, date, callsign, transponder_id, "
        "onset, end_time, frame_count, peak_score, "
        "peak_line_x1, peak_line_y1, peak_line_x2, peak_line_y2 "
        "FROM contrail_episodes WHERE episode_id = 1"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == 1  # episode_id
    assert row[1] == TEST_DATE  # date
    assert row[2] == "UAL123"
    assert row[3] == "A12345"
    # DuckDB returns tz-aware timestamps
    assert row[4].replace(tzinfo=UTC) == ep.onset or row[4] == ep.onset
    assert row[6] == 11  # frame_count
    assert row[7] == pytest.approx(0.85)
    assert row[8] == pytest.approx(100.0)
    assert row[9] == pytest.approx(200.0)
    assert row[10] == pytest.approx(300.0)
    assert row[11] == pytest.approx(400.0)


def test_insert_episode_null_pixel_line(db: Database):
    """Episode with no pixel line stores NULLs for line columns."""
    ep = _episode(peak_pixel_line=None)
    db.insert_episodes([ep], TEST_DATE)

    rows = db.query(
        "SELECT peak_line_x1, peak_line_y1, peak_line_x2, peak_line_y2 "
        "FROM contrail_episodes WHERE episode_id = 1"
    )
    assert rows[0] == (None, None, None, None)


def test_insert_multiple_episodes(db: Database):
    """Multiple episodes get sequential IDs."""
    eps = [
        _episode(callsign="UAL1", transponder_id="A1"),
        _episode(callsign="DAL2", transponder_id="B2", onset_offset=100, end_offset=110),
    ]
    count = db.insert_episodes(eps, TEST_DATE)
    assert count == 2

    rows = db.query("SELECT episode_id, callsign FROM contrail_episodes ORDER BY episode_id")
    assert len(rows) == 2
    assert rows[0][0] == 1
    assert rows[0][1] == "UAL1"
    assert rows[1][0] == 2
    assert rows[1][1] == "DAL2"


# ---------------------------------------------------------------------------
# Label insertion
# ---------------------------------------------------------------------------


def test_insert_label_updates_episode(db: Database):
    """Inserting a label fills the nullable label fields on the episode row."""
    db.insert_episodes([_episode()], TEST_DATE)

    now = datetime.datetime(2026, 4, 10, 9, 0, 0, tzinfo=UTC)
    db.insert_labels([{
        "episode_id": 1,
        "label": "contrail",
        "labeler_id": "alice",
        "persistence_rating": 4,
        "label_timestamp": now,
        "label_notes": "clear trail",
    }])

    rows = db.query(
        "SELECT label, persistence_rating, labeler_id, label_notes "
        "FROM contrail_episodes WHERE episode_id = 1"
    )
    assert len(rows) == 1
    assert rows[0][0] == "contrail"
    assert rows[0][1] == 4
    assert rows[0][2] == "alice"
    assert rows[0][3] == "clear trail"


def test_overlap_two_labelers(db: Database):
    """Two labelers labeling the same episode produce two rows."""
    db.insert_episodes([_episode()], TEST_DATE)

    now = datetime.datetime(2026, 4, 10, 9, 0, 0, tzinfo=UTC)
    db.insert_labels([{
        "episode_id": 1,
        "label": "contrail",
        "labeler_id": "alice",
        "persistence_rating": 4,
        "label_timestamp": now,
    }])
    db.insert_labels([{
        "episode_id": 1,
        "label": "no_contrail",
        "labeler_id": "bob",
        "persistence_rating": 2,
        "label_timestamp": now,
    }])

    rows = db.query(
        "SELECT labeler_id, label FROM contrail_episodes "
        "WHERE episode_id = 1 ORDER BY labeler_id"
    )
    assert len(rows) == 2
    assert rows[0] == ("alice", "contrail")
    assert rows[1] == ("bob", "no_contrail")


def test_relabel_same_labeler_updates_in_place(db: Database):
    """Same labeler re-labeling updates rather than creating a duplicate."""
    db.insert_episodes([_episode()], TEST_DATE)

    now = datetime.datetime(2026, 4, 10, 9, 0, 0, tzinfo=UTC)
    db.insert_labels([{
        "episode_id": 1,
        "label": "contrail",
        "labeler_id": "alice",
        "label_timestamp": now,
    }])
    db.insert_labels([{
        "episode_id": 1,
        "label": "unsure",
        "labeler_id": "alice",
        "label_timestamp": now,
    }])

    rows = db.query(
        "SELECT label FROM contrail_episodes WHERE episode_id = 1"
    )
    assert len(rows) == 1
    assert rows[0][0] == "unsure"


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------


def test_query_with_params(db: Database):
    db.insert_episodes([_episode()], TEST_DATE)
    rows = db.query(
        "SELECT callsign FROM contrail_episodes WHERE peak_score > ?",
        [0.5],
    )
    assert len(rows) == 1
    assert rows[0][0] == "UAL123"
