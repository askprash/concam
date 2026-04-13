"""Unit tests for the detection validation sweep helpers (PRD item 6).

We only test the numerically-tricky helpers: AUC and the best-threshold picker.
The sweep itself is exercised by running the script against the real manifest.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def _load_module():
    """Load the sweep script as a module (it lives in scripts/, not on the package path)."""
    import sys
    path = Path(__file__).parent.parent / "scripts" / "detection_validation_sweep.py"
    spec = importlib.util.spec_from_file_location("detection_validation_sweep", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module by name (py3.14+).
    sys.modules["detection_validation_sweep"] = module
    spec.loader.exec_module(module)
    return module


sweep = _load_module()


class TestMannWhitneyAUC:
    def test_perfect_separation(self):
        assert sweep._mann_whitney_auc([0.7, 0.8, 0.9], [0.1, 0.2, 0.3]) == 1.0

    def test_perfect_inversion(self):
        # All negatives beat all positives -> AUC 0.
        assert sweep._mann_whitney_auc([0.1, 0.2], [0.8, 0.9, 1.0]) == 0.0

    def test_all_ties_is_half(self):
        # Every pair tied -> 0.5.
        assert sweep._mann_whitney_auc([0.5, 0.5], [0.5, 0.5, 0.5]) == 0.5

    def test_partial_separation(self):
        # 2 pos (0.6, 0.7) vs 2 neg (0.5, 0.8).
        # Pairs: (0.6,0.5)=win, (0.6,0.8)=lose, (0.7,0.5)=win, (0.7,0.8)=lose.
        # 2 wins / 4 = 0.5.
        assert sweep._mann_whitney_auc([0.6, 0.7], [0.5, 0.8]) == 0.5

    def test_empty_inputs_degenerate_to_half(self):
        assert sweep._mann_whitney_auc([], [0.1]) == 0.5
        assert sweep._mann_whitney_auc([0.5], []) == 0.5

    def test_ties_count_as_half_wins(self):
        # 1 pos at 0.5, 1 neg at 0.5 -> 0.5 * 1 / 1 = 0.5.
        assert sweep._mann_whitney_auc([0.5], [0.5]) == 0.5


class TestBestThreshold:
    def test_perfect_separation_picks_midpoint(self):
        pos = [0.7, 0.8, 0.9]
        neg = [0.1, 0.2, 0.3]
        t, j = sweep._best_threshold(pos, neg)
        # Youden's J at t in (0.3, 0.7) is TPR=1.0 - FPR=0.0 = 1.0.
        assert 0.3 < t < 0.7
        assert math.isclose(j, 1.0)

    def test_no_separation_youden_zero(self):
        _, j = sweep._best_threshold([0.5], [0.5])
        assert j == 0.0

    def test_partial_separation(self):
        # 2 pos at 0.6, 0.7; 2 neg at 0.5, 0.8.
        # Best threshold sits around 0.55: TPR=1.0, FPR=0.5 -> J=0.5.
        t, j = sweep._best_threshold([0.6, 0.7], [0.5, 0.8])
        assert math.isclose(j, 0.5)
        assert 0.5 < t < 0.6

    def test_empty_inputs_return_default(self):
        t, j = sweep._best_threshold([], [0.1])
        assert j == 0.0
        assert t == 0.5
