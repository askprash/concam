"""DuckDB-backed storage for contrail episodes and labels."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb

from concam.aggregation import Episode

_SCHEMA_SQL = """
CREATE SEQUENCE IF NOT EXISTS seq_row_id START 1;
CREATE TABLE IF NOT EXISTS contrail_episodes (
    -- Surrogate key (allows two rows per episode for overlap labeling)
    row_id           INTEGER DEFAULT nextval('seq_row_id') PRIMARY KEY,

    -- Episode identity
    episode_id       INTEGER NOT NULL,
    date             DATE NOT NULL,
    callsign         VARCHAR NOT NULL,
    transponder_id   VARCHAR NOT NULL,

    -- Temporal extent
    onset            TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time         TIMESTAMP WITH TIME ZONE NOT NULL,
    frame_count      INTEGER NOT NULL,

    -- Detection
    peak_score       DOUBLE NOT NULL,
    peak_line_x1     DOUBLE,
    peak_line_y1     DOUBLE,
    peak_line_x2     DOUBLE,
    peak_line_y2     DOUBLE,

    -- Label fields (nullable — filled by ingest-labels)
    label            VARCHAR,           -- 'contrail' | 'no_contrail' | 'unsure'
    persistence_rating INTEGER,         -- 1-5
    labeler_id       VARCHAR,
    label_timestamp  TIMESTAMP WITH TIME ZONE,
    label_notes      VARCHAR
);
"""

_INSERT_EPISODE_SQL = """
INSERT INTO contrail_episodes (
    episode_id, date, callsign, transponder_id,
    onset, end_time, frame_count,
    peak_score, peak_line_x1, peak_line_y1, peak_line_x2, peak_line_y2
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_UPDATE_LABEL_SQL = """
UPDATE contrail_episodes
SET label = ?, persistence_rating = ?, labeler_id = ?, label_timestamp = ?, label_notes = ?
WHERE episode_id = ? AND (labeler_id IS NULL OR labeler_id = ?);
"""

_INSERT_LABEL_ROW_SQL = """
INSERT INTO contrail_episodes (
    episode_id, date, callsign, transponder_id,
    onset, end_time, frame_count,
    peak_score, peak_line_x1, peak_line_y1, peak_line_x2, peak_line_y2,
    label, persistence_rating, labeler_id, label_timestamp, label_notes
)
SELECT
    episode_id, date, callsign, transponder_id,
    onset, end_time, frame_count,
    peak_score, peak_line_x1, peak_line_y1, peak_line_x2, peak_line_y2,
    ?, ?, ?, ?, ?
FROM contrail_episodes
WHERE episode_id = ?
LIMIT 1;
"""


class Database:
    """DuckDB wrapper for contrail episode and label storage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.con = duckdb.connect(str(self.path))

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def create_schema(self) -> None:
        """Create the contrail_episodes table if it doesn't exist."""
        self.con.execute(_SCHEMA_SQL)

    def insert_episodes(
        self,
        episodes: list[Episode],
        date: datetime.date,
        start_id: int = 1,
    ) -> int:
        """Insert episodes into the database.

        Args:
            episodes: Episodes to insert.
            date: The processing date (stored as a column for partitioning).
            start_id: First episode_id to assign (auto-incremented from here).

        Returns:
            Number of episodes inserted.
        """
        for i, ep in enumerate(episodes):
            eid = start_id + i
            line = ep.peak_pixel_line
            self.con.execute(
                _INSERT_EPISODE_SQL,
                [
                    eid,
                    date,
                    ep.callsign,
                    ep.transponder_id,
                    ep.onset,
                    ep.end,
                    ep.frame_count,
                    ep.peak_score,
                    line[0] if line else None,
                    line[1] if line else None,
                    line[2] if line else None,
                    line[3] if line else None,
                ],
            )
        return len(episodes)

    def insert_labels(self, labels: list[dict]) -> int:
        """Insert label records into the database.

        Each label dict must have: episode_id, label, labeler_id.
        Optional: persistence_rating, label_timestamp, label_notes.

        If the episode's row is unlabeled or already labeled by the same labeler,
        the row is updated in place. If a different labeler already claimed the
        row, a second row is inserted (overlap set).
        """
        count = 0
        for lbl in labels:
            eid = lbl["episode_id"]
            label = lbl["label"]
            labeler = lbl["labeler_id"]
            rating = lbl.get("persistence_rating")
            ts = lbl.get("label_timestamp")
            notes = lbl.get("label_notes")

            # Check existing labels for this episode
            rows = self.con.execute(
                "SELECT labeler_id FROM contrail_episodes WHERE episode_id = ?",
                [eid],
            ).fetchall()

            labelers = {r[0] for r in rows if r[0] is not None}

            if labeler in labelers:
                # Re-label by same labeler: update their row
                self.con.execute(
                    _UPDATE_LABEL_SQL,
                    [label, rating, labeler, ts, notes, eid, labeler],
                )
            elif any(r[0] is None for r in rows):
                # Unlabeled row exists: fill it in
                self.con.execute(
                    _UPDATE_LABEL_SQL,
                    [label, rating, labeler, ts, notes, eid, labeler],
                )
            else:
                # All rows already labeled by other labelers: insert new row
                self.con.execute(
                    _INSERT_LABEL_ROW_SQL,
                    [label, rating, labeler, ts, notes, eid],
                )
            count += 1
        return count

    def query(self, sql: str, params: list | None = None) -> list[tuple]:
        """Run an arbitrary SQL query and return all rows as tuples."""
        if params:
            return self.con.execute(sql, params).fetchall()
        return self.con.execute(sql).fetchall()
