"""Tests for concam.agreement: inter-rater agreement on overlap labels."""

from __future__ import annotations

import datetime
import math
import random
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from concam.agreement import (
    LABEL_CLASSES,
    cohen_kappa,
    compute_agreement,
    confusion_matrix,
    load_overlap_labels,
    pairs_from_overlap,
    percent_agreement,
)
from concam.aggregation import Episode
from concam.cli import main as cli_main
from concam.storage import Database

UTC = datetime.timezone.utc
DATE = datetime.date(2026, 4, 8)
BASE_TIME = datetime.datetime(2026, 4, 8, 12, 0, 0, tzinfo=UTC)


def _episode(callsign: str, transponder: str, score: float = 0.8) -> Episode:
    return Episode(
        callsign=callsign,
        transponder_id=transponder,
        onset=BASE_TIME,
        end=BASE_TIME + datetime.timedelta(seconds=10),
        peak_score=score,
        peak_pixel_line=(0.0, 0.0, 1.0, 1.0),
        frame_count=11,
    )


def _make_db(tmp_path: Path, n_episodes: int = 5) -> Path:
    db_path = tmp_path / "test.duckdb"
    with Database(db_path) as db:
        db.create_schema()
        episodes = [_episode(f"FL{i:03d}", f"T{i:05d}") for i in range(n_episodes)]
        db.insert_episodes(episodes, date=DATE)
    return db_path


def _ingest(db_path: Path, labels: list[dict]) -> None:
    with Database(db_path) as db:
        db.insert_labels(labels)


# ---------------------------------------------------------------------------
# Cohen's kappa formula
# ---------------------------------------------------------------------------

def test_kappa_perfect_agreement_is_one():
    # 10 pairs all on the diagonal across two classes
    m = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 0]])
    assert cohen_kappa(m) == pytest.approx(1.0)


def test_kappa_random_labels_near_zero():
    rng = random.Random(0)
    pairs = [
        ("alice", rng.choice(LABEL_CLASSES), "bob", rng.choice(LABEL_CLASSES))
        for _ in range(2000)
    ]
    m = confusion_matrix(pairs)
    k = cohen_kappa(m)
    # Two independent uniform raters → expected kappa = 0; variance is ~0.02 on n=2000.
    assert abs(k) < 0.1


def test_kappa_systematic_disagreement_is_negative():
    # alice always 'contrail', bob always 'no_contrail' → po=0, pe=0, kappa undefined → nan
    m = np.array([[0, 5, 0], [0, 0, 0], [0, 0, 0]])
    # po=0, row marg=(1,0,0), col marg=(0,1,0), pe=0, kappa = (0-0)/(1-0) = 0
    assert cohen_kappa(m) == pytest.approx(0.0)


def test_kappa_one_disagreement_in_ten():
    m = np.array([[5, 1, 0], [0, 4, 0], [0, 0, 0]])
    po = 9 / 10
    # row marg = (6/10, 4/10, 0), col marg = (5/10, 5/10, 0)
    pe = 6 / 10 * 5 / 10 + 4 / 10 * 5 / 10
    expected = (po - pe) / (1 - pe)
    assert cohen_kappa(m) == pytest.approx(expected)


def test_percent_agreement_diagonal_fraction():
    m = np.array([[3, 1, 0], [1, 4, 1], [0, 0, 0]])
    assert percent_agreement(m) == pytest.approx(7 / 10)


def test_percent_agreement_empty_matrix_is_nan():
    m = np.zeros((3, 3), dtype=int)
    assert math.isnan(percent_agreement(m))


# ---------------------------------------------------------------------------
# Confusion matrix construction
# ---------------------------------------------------------------------------

def test_confusion_matrix_counts():
    pairs = [
        ("alice", "contrail", "bob", "contrail"),
        ("alice", "contrail", "bob", "no_contrail"),
        ("alice", "no_contrail", "bob", "no_contrail"),
        ("alice", "unsure", "bob", "contrail"),
    ]
    m = confusion_matrix(pairs)
    # rows = alice, cols = bob, order: contrail, no_contrail, unsure
    assert m[0, 0] == 1  # alice contrail, bob contrail
    assert m[0, 1] == 1  # alice contrail, bob no_contrail
    assert m[1, 1] == 1
    assert m[2, 0] == 1
    assert m.sum() == 4


def test_confusion_matrix_rejects_unknown_label():
    pairs = [("alice", "maybe", "bob", "contrail")]
    with pytest.raises(ValueError):
        confusion_matrix(pairs)


# ---------------------------------------------------------------------------
# Overlap-set extraction from DuckDB
# ---------------------------------------------------------------------------

def test_load_overlap_labels_only_returns_multilabeler_episodes(tmp_path: Path):
    db_path = _make_db(tmp_path, n_episodes=4)
    # ep 1: only alice  → not overlap
    # ep 2: alice + bob  → overlap
    # ep 3: bob only     → not overlap
    # ep 4: alice + bob  → overlap
    _ingest(db_path, [
        {"episode_id": 1, "labeler_id": "alice", "label": "contrail"},
        {"episode_id": 2, "labeler_id": "alice", "label": "contrail"},
        {"episode_id": 2, "labeler_id": "bob", "label": "no_contrail"},
        {"episode_id": 3, "labeler_id": "bob", "label": "unsure"},
        {"episode_id": 4, "labeler_id": "alice", "label": "contrail"},
        {"episode_id": 4, "labeler_id": "bob", "label": "contrail"},
    ])

    overlap = load_overlap_labels(db_path, DATE)
    assert set(overlap.keys()) == {2, 4}
    assert overlap[2] == {"alice": "contrail", "bob": "no_contrail"}
    assert overlap[4] == {"alice": "contrail", "bob": "contrail"}


def test_load_overlap_filters_by_date(tmp_path: Path):
    db_path = _make_db(tmp_path, n_episodes=2)
    _ingest(db_path, [
        {"episode_id": 1, "labeler_id": "alice", "label": "contrail"},
        {"episode_id": 1, "labeler_id": "bob", "label": "contrail"},
    ])
    overlap = load_overlap_labels(db_path, DATE + datetime.timedelta(days=1))
    assert overlap == {}


def test_pairs_from_overlap_unordered_lex_pairs():
    overlap = {
        7: {"alice": "contrail", "bob": "no_contrail", "carol": "contrail"},
    }
    pairs = pairs_from_overlap(overlap)
    # 3 labelers → 3 pairs, lex-ordered
    assert len(pairs) == 3
    labelers = {(p[0], p[2]) for p in pairs}
    assert labelers == {("alice", "bob"), ("alice", "carol"), ("bob", "carol")}


# ---------------------------------------------------------------------------
# End-to-end through compute_agreement
# ---------------------------------------------------------------------------

def test_compute_agreement_perfect(tmp_path: Path):
    db_path = _make_db(tmp_path, n_episodes=5)
    _ingest(db_path, [
        {"episode_id": eid, "labeler_id": labeler, "label": "contrail"}
        for eid in range(1, 6) for labeler in ("alice", "bob")
    ])
    report = compute_agreement(db_path, DATE)
    assert report.n_episodes == 5
    assert report.n_pairs == 5
    assert report.percent_agreement == pytest.approx(1.0)
    # All on one class → kappa undefined; we return 1.0 for trivial perfect agreement
    assert report.cohen_kappa == pytest.approx(1.0)
    assert report.labelers == ("alice", "bob")


def test_compute_agreement_mixed(tmp_path: Path):
    db_path = _make_db(tmp_path, n_episodes=4)
    _ingest(db_path, [
        # ep 1: agree contrail
        {"episode_id": 1, "labeler_id": "alice", "label": "contrail"},
        {"episode_id": 1, "labeler_id": "bob", "label": "contrail"},
        # ep 2: agree no_contrail
        {"episode_id": 2, "labeler_id": "alice", "label": "no_contrail"},
        {"episode_id": 2, "labeler_id": "bob", "label": "no_contrail"},
        # ep 3: disagree contrail / no_contrail
        {"episode_id": 3, "labeler_id": "alice", "label": "contrail"},
        {"episode_id": 3, "labeler_id": "bob", "label": "no_contrail"},
        # ep 4: alice unsure, bob contrail
        {"episode_id": 4, "labeler_id": "alice", "label": "unsure"},
        {"episode_id": 4, "labeler_id": "bob", "label": "contrail"},
    ])
    report = compute_agreement(db_path, DATE)
    assert report.n_pairs == 4
    assert report.percent_agreement == pytest.approx(0.5)
    assert report.confusion[0, 0] == 1  # contrail/contrail
    assert report.confusion[1, 1] == 1  # no_contrail/no_contrail
    assert report.confusion[0, 1] == 1  # alice contrail, bob no_contrail
    assert report.confusion[2, 0] == 1  # alice unsure, bob contrail


def test_compute_agreement_empty_when_no_overlap(tmp_path: Path):
    db_path = _make_db(tmp_path, n_episodes=2)
    _ingest(db_path, [
        {"episode_id": 1, "labeler_id": "alice", "label": "contrail"},
    ])
    report = compute_agreement(db_path, DATE)
    assert report.n_pairs == 0
    assert report.n_episodes == 0
    assert math.isnan(report.percent_agreement)
    assert math.isnan(report.cohen_kappa)
    assert "No overlap-set pairs" in report.format()


def test_report_format_contains_metrics(tmp_path: Path):
    db_path = _make_db(tmp_path, n_episodes=2)
    _ingest(db_path, [
        {"episode_id": 1, "labeler_id": "alice", "label": "contrail"},
        {"episode_id": 1, "labeler_id": "bob", "label": "contrail"},
        {"episode_id": 2, "labeler_id": "alice", "label": "no_contrail"},
        {"episode_id": 2, "labeler_id": "bob", "label": "contrail"},
    ])
    text = compute_agreement(db_path, DATE).format()
    assert "Cohen's kappa" in text
    assert "Percent agreement" in text
    assert "contrail" in text
    assert "no_contrail" in text


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

def test_cli_agreement_happy_path(tmp_path: Path):
    out_dir = tmp_path / "output"
    date_dir = out_dir / DATE.isoformat()
    date_dir.mkdir(parents=True)
    db_path = date_dir / "pipeline.duckdb"
    with Database(db_path) as db:
        db.create_schema()
        db.insert_episodes([_episode("FL000", "T00000")], date=DATE)
        db.insert_labels([
            {"episode_id": 1, "labeler_id": "alice", "label": "contrail"},
            {"episode_id": 1, "labeler_id": "bob", "label": "contrail"},
        ])

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["agreement", "--date", DATE.isoformat(), "--output-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "Cohen's kappa" in result.output
    assert "1.000" in result.output


def test_cli_agreement_missing_db(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["agreement", "--date", DATE.isoformat(), "--output-dir", str(tmp_path / "nope")],
    )
    assert result.exit_code != 0
    assert "DuckDB not found" in result.output
