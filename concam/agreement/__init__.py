"""Inter-rater agreement on overlap-set labels.

Joins the rows two labelers wrote for the same episode_id and reports
percent agreement, Cohen's kappa, and the 3x3 confusion matrix over the
``contrail / no_contrail / unsure`` label space.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from concam.storage import Database

LABEL_CLASSES: tuple[str, str, str] = ("contrail", "no_contrail", "unsure")


@dataclass(frozen=True)
class AgreementReport:
    """Result of inter-rater agreement computation."""

    n_pairs: int
    n_episodes: int
    labelers: tuple[str, ...]
    classes: tuple[str, ...]
    confusion: np.ndarray  # rows = labeler_a, cols = labeler_b, classes order
    percent_agreement: float
    cohen_kappa: float

    def format(self) -> str:
        """Render a human-readable text summary."""
        if self.n_pairs == 0:
            return "No overlap-set pairs found."
        lines = [
            f"Labelers: {', '.join(self.labelers)}",
            f"Overlap episodes: {self.n_episodes}",
            f"Pairs compared:   {self.n_pairs}",
            f"Percent agreement: {self.percent_agreement:.3f}",
            f"Cohen's kappa:     {self.cohen_kappa:.3f}",
            "",
            "Confusion matrix (rows = first labeler, cols = second labeler):",
        ]
        col_w = max(len(c) for c in self.classes) + 2
        header = " " * col_w + "".join(f"{c:>{col_w}}" for c in self.classes)
        lines.append(header)
        for i, c in enumerate(self.classes):
            row = f"{c:>{col_w - 1}} " + "".join(
                f"{int(self.confusion[i, j]):>{col_w}}" for j in range(len(self.classes))
            )
            lines.append(row)
        return "\n".join(lines)


def load_overlap_labels(
    db_path: str | Path,
    date: datetime.date,
) -> dict[int, dict[str, str]]:
    """Return ``{episode_id: {labeler_id: label}}`` for episodes with >=2 labelers.

    Only episodes labeled by two or more distinct, non-null labelers are
    included — by definition the overlap set.
    """
    with Database(db_path) as db:
        rows = db.query(
            """
            SELECT episode_id, labeler_id, label
            FROM contrail_episodes
            WHERE date = ? AND labeler_id IS NOT NULL AND label IS NOT NULL
            """,
            [date],
        )

    by_episode: dict[int, dict[str, str]] = {}
    for episode_id, labeler_id, label in rows:
        by_episode.setdefault(episode_id, {})[labeler_id] = label
    return {eid: lbls for eid, lbls in by_episode.items() if len(lbls) >= 2}


def pairs_from_overlap(
    overlap: dict[int, dict[str, str]],
) -> list[tuple[str, str, str, str]]:
    """Expand each multi-labeler episode into all unordered labeler pairs.

    Returns tuples ``(labeler_a, label_a, labeler_b, label_b)`` with
    ``labeler_a < labeler_b`` (lexicographic) so the orientation is stable.
    """
    pairs: list[tuple[str, str, str, str]] = []
    for episode_id in sorted(overlap.keys()):
        labels = overlap[episode_id]
        for la, lb in combinations(sorted(labels.keys()), 2):
            pairs.append((la, labels[la], lb, labels[lb]))
    return pairs


def confusion_matrix(
    pairs: list[tuple[str, str, str, str]],
    classes: tuple[str, ...] = LABEL_CLASSES,
) -> np.ndarray:
    """3x3 confusion matrix over ``classes`` for the given labeler-pair tuples."""
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for _la, label_a, _lb, label_b in pairs:
        if label_a not in idx or label_b not in idx:
            raise ValueError(f"label outside class set: {label_a!r}, {label_b!r}")
        m[idx[label_a], idx[label_b]] += 1
    return m


def percent_agreement(matrix: np.ndarray) -> float:
    """Fraction of pairs on the diagonal of the confusion matrix."""
    total = matrix.sum()
    if total == 0:
        return float("nan")
    return float(np.trace(matrix)) / float(total)


def cohen_kappa(matrix: np.ndarray) -> float:
    """Cohen's kappa from a square confusion matrix.

    Edge case: when both raters used a single class, kappa is undefined;
    we return 1.0 if the off-diagonals are zero (perfect trivial agreement)
    and ``nan`` otherwise.
    """
    total = matrix.sum()
    if total == 0:
        return float("nan")
    po = float(np.trace(matrix)) / float(total)
    row_marg = matrix.sum(axis=1) / float(total)
    col_marg = matrix.sum(axis=0) / float(total)
    pe = float(np.dot(row_marg, col_marg))
    if pe == 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return (po - pe) / (1.0 - pe)


def compute_agreement(
    db_path: str | Path,
    date: datetime.date,
    classes: tuple[str, ...] = LABEL_CLASSES,
) -> AgreementReport:
    """End-to-end: pull overlap labels from DuckDB and compute the report."""
    overlap = load_overlap_labels(db_path, date)
    pairs = pairs_from_overlap(overlap)
    labelers = sorted({la for la, _, _, _ in pairs} | {lb for _, _, lb, _ in pairs})
    if not pairs:
        return AgreementReport(
            n_pairs=0,
            n_episodes=0,
            labelers=tuple(labelers),
            classes=classes,
            confusion=np.zeros((len(classes), len(classes)), dtype=int),
            percent_agreement=float("nan"),
            cohen_kappa=float("nan"),
        )
    m = confusion_matrix(pairs, classes)
    return AgreementReport(
        n_pairs=len(pairs),
        n_episodes=len(overlap),
        labelers=tuple(labelers),
        classes=classes,
        confusion=m,
        percent_agreement=percent_agreement(m),
        cohen_kappa=cohen_kappa(m),
    )
