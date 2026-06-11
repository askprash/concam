"""Label ingestion: parse and validate labeler export JSON, write to DuckDB.

The labeler HTML (``concam/bundle/templates/labeler.html``) exports a JSON
payload of the shape::

    {
      "schema_version": 1,
      "date": "2026-04-08",
      "labeler_id": "alice",
      "exported_at": "2026-04-13T12:34:56.000Z",
      "labels": [
        {
          "episode_id": 1,
          "label": "contrail",
          "labeler_id": "alice",
          "persistence_rating": 4,          # optional, 1..5
          "label_timestamp": "2026-04-13T12:34:50.000Z",  # optional, ISO 8601
          "label_notes": "clear trail"       # optional
        },
        ...
      ]
    }

``ingest_label_files`` validates each file, coerces ``label_timestamp`` strings
to tz-aware datetimes, and calls :meth:`Database.insert_labels` which handles
the overlap case (two labelers on the same episode ⇒ two rows).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from concam.storage import Database

SUPPORTED_SCHEMA_VERSION = 1
VALID_LABELS = frozenset({"contrail", "no_contrail", "unsure"})
# Categorical persistence judgement (replaced the 1-5 slider; the numeric
# persistence_rating is still accepted from legacy exports).
VALID_PERSISTENCE = frozenset({"short", "potentially_persistent"})


class LabelValidationError(ValueError):
    """Raised when a label JSON payload does not match the expected schema."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise LabelValidationError(msg)


def _parse_timestamp(value: str, field: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp string into a tz-aware UTC datetime.

    Accepts the trailing ``Z`` that JavaScript's ``Date.toISOString()`` produces.
    """
    _require(isinstance(value, str), f"{field}: expected ISO string, got {type(value).__name__}")
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(s)
    except ValueError as e:
        raise LabelValidationError(f"{field}: invalid ISO timestamp {value!r}: {e}")
    if dt.tzinfo is None:
        # Labeler always emits UTC; be explicit rather than accept naive.
        raise LabelValidationError(
            f"{field}: expected tz-aware ISO timestamp (ends in 'Z' or offset), got {value!r}"
        )
    return dt.astimezone(datetime.timezone.utc)


def validate_payload(payload: dict, *, expected_date: datetime.date) -> dict:
    """Validate an export payload.

    Returns a normalised dict with `label_timestamp` coerced to datetime.
    Raises LabelValidationError on any schema violation.
    """
    _require(isinstance(payload, dict), "top level must be a JSON object")

    schema_version = payload.get("schema_version")
    _require(
        schema_version == SUPPORTED_SCHEMA_VERSION,
        f"schema_version: expected {SUPPORTED_SCHEMA_VERSION}, got {schema_version!r}",
    )

    date_str = payload.get("date")
    _require(isinstance(date_str, str), "date: expected string, got " + type(date_str).__name__)
    try:
        file_date = datetime.date.fromisoformat(date_str)
    except ValueError as e:
        raise LabelValidationError(f"date: invalid ISO date {date_str!r}: {e}")
    _require(
        file_date == expected_date,
        f"date: file is for {file_date}, expected {expected_date}",
    )

    labeler_id = payload.get("labeler_id")
    _require(
        isinstance(labeler_id, str) and labeler_id,
        f"labeler_id: expected non-empty string, got {labeler_id!r}",
    )

    labels = payload.get("labels")
    _require(isinstance(labels, list), "labels: expected a list, got " + type(labels).__name__)

    seen_episode_ids: set[int] = set()
    normalised_labels: list[dict] = []
    for i, record in enumerate(labels):
        ctx = f"labels[{i}]"
        _require(isinstance(record, dict), f"{ctx}: expected object, got {type(record).__name__}")

        eid = record.get("episode_id")
        _require(
            isinstance(eid, int) and not isinstance(eid, bool),
            f"{ctx}.episode_id: expected int, got {eid!r}",
        )
        _require(
            eid not in seen_episode_ids,
            f"{ctx}.episode_id: duplicate episode_id {eid} within the same file",
        )
        seen_episode_ids.add(eid)

        label = record.get("label")
        _require(
            label in VALID_LABELS,
            f"{ctx}.label: expected one of {sorted(VALID_LABELS)}, got {label!r}",
        )

        rec_labeler = record.get("labeler_id", labeler_id)
        _require(
            rec_labeler == labeler_id,
            f"{ctx}.labeler_id: {rec_labeler!r} does not match top-level labeler_id {labeler_id!r}",
        )

        normalised: dict = {
            "episode_id": eid,
            "label": label,
            "labeler_id": labeler_id,
        }

        rating = record.get("persistence_rating")
        if rating is not None:
            _require(
                isinstance(rating, int) and not isinstance(rating, bool) and 1 <= rating <= 5,
                f"{ctx}.persistence_rating: expected int in [1,5], got {rating!r}",
            )
            normalised["persistence_rating"] = rating

        persistence = record.get("persistence")
        if persistence is not None:
            _require(
                persistence in VALID_PERSISTENCE,
                f"{ctx}.persistence: expected one of {sorted(VALID_PERSISTENCE)},"
                f" got {persistence!r}",
            )
            normalised["persistence"] = persistence

        measurement = record.get("measurement_km")
        if measurement is not None:
            _require(
                isinstance(measurement, (int, float)) and not isinstance(measurement, bool),
                f"{ctx}.measurement_km: expected number, got {measurement!r}",
            )
            normalised["measurement_km"] = float(measurement)

        ts = record.get("label_timestamp")
        if ts is not None:
            normalised["label_timestamp"] = _parse_timestamp(ts, f"{ctx}.label_timestamp")

        notes = record.get("label_notes")
        if notes is not None:
            _require(
                isinstance(notes, str),
                f"{ctx}.label_notes: expected string, got {type(notes).__name__}",
            )
            if notes:
                normalised["label_notes"] = notes

        normalised_labels.append(normalised)

    return {
        "schema_version": schema_version,
        "date": file_date,
        "labeler_id": labeler_id,
        "labels": normalised_labels,
    }


def load_label_file(path: Path, *, expected_date: datetime.date) -> dict:
    """Read a label JSON file and return a validated, normalised payload."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as e:
        raise LabelValidationError(f"{path}: invalid JSON: {e}")
    try:
        return validate_payload(payload, expected_date=expected_date)
    except LabelValidationError as e:
        raise LabelValidationError(f"{path}: {e}")


def ingest_label_files(
    db_path: Path,
    label_paths: list[Path],
    *,
    expected_date: datetime.date,
) -> dict[str, int]:
    """Validate and ingest one or more label JSON files into DuckDB.

    Returns a mapping from labeler_id to number of labels ingested.
    Validation is performed for ALL files before any write, so a malformed
    file aborts ingestion without partial state.
    """
    validated: list[dict] = [
        load_label_file(p, expected_date=expected_date) for p in label_paths
    ]

    counts: dict[str, int] = {}
    with Database(db_path) as db:
        db.create_schema()
        for payload in validated:
            n = db.insert_labels(payload["labels"])
            counts[payload["labeler_id"]] = counts.get(payload["labeler_id"], 0) + n
    return counts
