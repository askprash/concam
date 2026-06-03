"""Unit tests for the detection-evaluation metric helpers (PRD item 6).

These helpers now live in ``concam.detection.metrics`` — the importlib hack that
loaded them from the sweep script is no longer needed.

The sweep itself is exercised by running the script against the real manifest.
"""

from __future__ import annotations

import math

from concam.detection.metrics import mann_whitney_auc, youden_threshold


class TestMannWhitneyAUC:
    def test_perfect_separation(self):
        assert mann_whitney_auc([0.7, 0.8, 0.9], [0.1, 0.2, 0.3]) == 1.0

    def test_perfect_inversion(self):
        # All negatives beat all positives -> AUC 0.
        assert mann_whitney_auc([0.1, 0.2], [0.8, 0.9, 1.0]) == 0.0

    def test_all_ties_is_half(self):
        # Every pair tied -> 0.5.
        assert mann_whitney_auc([0.5, 0.5], [0.5, 0.5, 0.5]) == 0.5

    def test_partial_separation(self):
        # 2 pos (0.6, 0.7) vs 2 neg (0.5, 0.8).
        # Pairs: (0.6,0.5)=win, (0.6,0.8)=lose, (0.7,0.5)=win, (0.7,0.8)=lose.
        # 2 wins / 4 = 0.5.
        assert mann_whitney_auc([0.6, 0.7], [0.5, 0.8]) == 0.5

    def test_empty_inputs_return_nan(self):
        # Canonical module returns nan for empty inputs (undefined statistic).
        assert math.isnan(mann_whitney_auc([], [0.1]))
        assert math.isnan(mann_whitney_auc([0.5], []))

    def test_ties_count_as_half_wins(self):
        # 1 pos at 0.5, 1 neg at 0.5 -> 0.5 * 1 / 1 = 0.5.
        assert mann_whitney_auc([0.5], [0.5]) == 0.5


class TestYoudenThreshold:
    def test_perfect_separation_picks_midpoint(self):
        pos = [0.7, 0.8, 0.9]
        neg = [0.1, 0.2, 0.3]
        t, j = youden_threshold(pos, neg)
        # Youden's J at t in (0.3, 0.7) is TPR=1.0 - FPR=0.0 = 1.0.
        assert 0.3 < t < 0.7
        assert math.isclose(j, 1.0)

    def test_no_separation_youden_zero(self):
        _, j = youden_threshold([0.5], [0.5])
        assert j == 0.0

    def test_partial_separation(self):
        # 2 pos at 0.6, 0.7; 2 neg at 0.5, 0.8.
        # Best threshold sits around 0.55: TPR=1.0, FPR=0.5 -> J=0.5.
        t, j = youden_threshold([0.6, 0.7], [0.5, 0.8])
        assert math.isclose(j, 0.5)
        assert 0.5 < t < 0.6

    def test_empty_inputs_return_nan(self):
        # Canonical module returns (nan, nan) for empty inputs.
        t, j = youden_threshold([], [0.1])
        assert math.isnan(t)
        assert math.isnan(j)
