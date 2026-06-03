"""Canonical detector-evaluation metrics.

Three public functions cover all evaluation needs across the pipeline:

- :func:`mann_whitney_auc`   — ROC-AUC via rank statistics.
- :func:`youden_threshold`   — Optimal operating point (MIDPOINT candidates).
- :func:`youden_at`          — Youden's J at a fixed threshold (point evaluation).

Predict-positive convention throughout: a sample is classified as positive when
its detection score is *greater than or equal to* the threshold.

Scores are expected to be in [0, 1] (the pipeline's normalised detection score
range), but the math is correct for any real-valued scores.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def rank_metric(value: float, *, worst: float = float("-inf")) -> float:
    """Coerce an undefined (``nan``) metric to a sentinel for deterministic ranking.

    The pure metrics return ``nan`` when undefined (a class has no members).
    Python sorts by ``nan`` non-deterministically (``nan`` compares ``False`` to
    everything), so any ranking that must be reproducible should map undefined
    values to a worst-case sentinel via this helper.  Default ``worst=-inf``
    sinks degenerate rows to the bottom under both ``reverse=True`` sorts and
    negated-ascending sorts.
    """
    return value if not math.isnan(value) else worst


def mann_whitney_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Mann-Whitney rank-based ROC-AUC.

    Returns the probability that a randomly chosen positive scores strictly
    higher than a randomly chosen negative, with ties counting as half a win.

    Args:
        pos: Detection scores for ground-truth positive samples.
        neg: Detection scores for ground-truth negative samples.

    Returns:
        AUC in [0, 1], or ``float('nan')`` if either sequence is empty.

    Note:
        ``nan`` is returned (not 0.5) for empty inputs because the statistic is
        undefined when one class has no members — callers that want a safe
        fallback should check explicitly.
    """
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def youden_threshold(pos: Sequence[float], neg: Sequence[float]) -> tuple[float, float]:
    """Find the threshold that maximises Youden's J using midpoint candidates.

    Candidate thresholds are the midpoints between every pair of adjacent
    distinct values in ``sorted(set(pos) | set(neg))``, plus one sentinel
    below the global minimum and one above the global maximum.  Using midpoints
    instead of exact observed scores avoids tie ambiguity at boundaries and is
    the standard ROC operating-point choice.

    Predict-positive convention: a sample is classified positive iff its score
    is *>= threshold*.  Youden's J = sensitivity + specificity − 1 = TPR − FPR.

    On a tie (multiple thresholds sharing the same J) the lowest threshold is
    returned (maximises recall at equal J).

    Args:
        pos: Detection scores for ground-truth positive samples.
        neg: Detection scores for ground-truth negative samples.

    Returns:
        ``(threshold, youden_j)`` — the optimal threshold and the corresponding
        Youden's J value.  When either sequence is empty, returns
        ``(float('nan'), float('nan'))``.
    """
    if not pos or not neg:
        return float("nan"), float("nan")

    all_vals = sorted(set(list(pos) + list(neg)))

    if len(all_vals) == 1:
        # All scores identical — any threshold gives J=0; put it just below.
        return all_vals[0] - 1e-6, 0.0

    # Sentinels: one below global min, one above global max.
    sentinels_below = [all_vals[0] - 1e-6]
    sentinels_above = [all_vals[-1] + 1e-6]
    midpoints = [(a + b) / 2.0 for a, b in zip(all_vals[:-1], all_vals[1:])]
    candidates = sentinels_below + midpoints + sentinels_above

    best_t = candidates[0]
    best_j = -2.0  # below any achievable J so the first candidate always wins
    for t in candidates:
        tpr = sum(1 for p in pos if p >= t) / len(pos)
        fpr = sum(1 for n in neg if n >= t) / len(neg)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return best_t, best_j


def youden_at(pos: Sequence[float], neg: Sequence[float], threshold: float) -> float:
    """Youden's J at a specific fixed threshold (point evaluation).

    Unlike :func:`youden_threshold` this does *not* search for an optimum; it
    evaluates the operating-point quality of a given threshold.

    Predict-positive convention: a sample is classified positive iff its score
    is *>= threshold*.

    Args:
        pos: Detection scores for ground-truth positive samples.
        neg: Detection scores for ground-truth negative samples.
        threshold: The fixed detection threshold to evaluate.

    Returns:
        Youden's J = sensitivity + specificity − 1 in [−1, 1], or
        ``float('nan')`` if either sequence is empty.
    """
    if not pos or not neg:
        return float("nan")
    sens = sum(1 for s in pos if s >= threshold) / len(pos)
    spec = sum(1 for s in neg if s < threshold) / len(neg)
    return sens + spec - 1.0
