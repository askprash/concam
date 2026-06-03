"""Property and metamorphic tests for concam.detection.metrics.

Uses hypothesis for property-based testing. All strategies use bounded ranges
(scores in [-10, 10] for numerical stability) and small list sizes to keep
test runs fast.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings, strategies as st

from concam.detection.metrics import (
    mann_whitney_auc,
    rank_metric,
    youden_at,
    youden_threshold,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Scores bounded to a sane range. We use [-10, 10] rather than [0, 1] to
# verify the math works for any real values, not just fractions.
# allow_subnormal=False avoids values so small that arithmetic collapses them.
_score = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_nonempty_scores = st.lists(_score, min_size=1, max_size=20)
_pos_neg = st.tuples(_nonempty_scores, _nonempty_scores)


# ---------------------------------------------------------------------------
# mann_whitney_auc
# ---------------------------------------------------------------------------


class TestMannWhitneyAUCProperties:
    @given(_pos_neg)
    def test_auc_in_unit_interval(self, pos_neg):
        pos, neg = pos_neg
        auc = mann_whitney_auc(pos, neg)
        assert 0.0 <= auc <= 1.0, f"AUC {auc} not in [0,1] for pos={pos}, neg={neg}"

    @given(st.lists(_score, min_size=1, max_size=20))
    def test_perfect_separation_auc_one(self, pos):
        # All pos > all neg: shift neg well below the min pos.
        min_pos = min(pos)
        neg = [min_pos - 1.0 - i * 0.01 for i in range(len(pos))]
        auc = mann_whitney_auc(pos, neg)
        assert math.isclose(auc, 1.0), f"Expected AUC=1.0, got {auc}"

    @given(st.lists(_score, min_size=1, max_size=20))
    def test_reversed_separation_auc_zero(self, neg):
        # All neg > all pos.
        max_neg = max(neg)
        pos = [max_neg + 1.0 + i * 0.01 for i in range(len(neg))]
        # Swap: now actual pos are the high ones — we call with args reversed.
        auc = mann_whitney_auc(neg, pos)  # neg passed as "pos", pos passed as "neg"
        assert math.isclose(auc, 0.0), f"Expected AUC=0.0, got {auc}"

    @given(_nonempty_scores)
    def test_identical_multisets_auc_half(self, x):
        auc = mann_whitney_auc(x, x)
        assert math.isclose(auc, 0.5, abs_tol=1e-9), (
            f"Expected AUC=0.5 for identical lists, got {auc}"
        )

    @given(_pos_neg)
    def test_anti_symmetry(self, pos_neg):
        pos, neg = pos_neg
        auc_pn = mann_whitney_auc(pos, neg)
        auc_np = mann_whitney_auc(neg, pos)
        assert math.isclose(auc_pn + auc_np, 1.0, abs_tol=1e-9), (
            f"Anti-symmetry violated: AUC(p,n)={auc_pn}, AUC(n,p)={auc_np}"
        )

    @given(_pos_neg)
    def test_empty_returns_nan(self, pos_neg):
        pos, neg = pos_neg
        assert math.isnan(mann_whitney_auc([], neg))
        assert math.isnan(mann_whitney_auc(pos, []))

    @given(
        _pos_neg,
        st.floats(
            min_value=0.01, max_value=5.0, allow_nan=False,
            allow_infinity=False, allow_subnormal=False,
        ),
    )
    def test_strictly_increasing_transform_preserves_auc(self, pos_neg, scale):
        """Metamorphic: applying a strictly increasing function leaves AUC unchanged.

        We skip cases where floating-point arithmetic collapses two previously
        distinct values to the same float (unavoidable with finite precision;
        testing the property under such collapse is not meaningful).
        """
        pos, neg = pos_neg
        # f(v) = scale * v + 1 is strictly increasing for scale > 0.
        pos2 = [scale * v + 1.0 for v in pos]
        neg2 = [scale * v + 1.0 for v in neg]

        # Guard: skip if the transform collapsed any (p, n) pair that was
        # previously distinct, since AUC will legitimately change.
        for p_orig, p2 in zip(pos, pos2):
            for n_orig, n2 in zip(neg, neg2):
                orig_cmp = (p_orig > n_orig) - (p_orig < n_orig)
                new_cmp = (p2 > n2) - (p2 < n2)
                assume(orig_cmp == new_cmp)

        auc_orig = mann_whitney_auc(pos, neg)
        auc_transformed = mann_whitney_auc(pos2, neg2)
        assert math.isclose(auc_orig, auc_transformed, abs_tol=1e-9), (
            f"Monotonic-transform invariance failed: {auc_orig} vs {auc_transformed}"
        )


# ---------------------------------------------------------------------------
# youden_threshold
# ---------------------------------------------------------------------------


class TestYoudenThresholdProperties:
    @given(
        _pos_neg,
        st.floats(
            min_value=-5.0, max_value=5.0, allow_nan=False,
            allow_infinity=False, allow_subnormal=False,
        ),
    )
    def test_translation_equivariance(self, pos_neg, shift):
        """Shifting all scores by c leaves Youden's J unchanged and shifts the threshold by c.

        We skip (assume away) cases where adding ``shift`` collapses two
        previously distinct float values together — that would legitimately
        change the ROC curve, not a bug in our implementation.
        """
        pos, neg = pos_neg
        all_orig = list(pos) + list(neg)

        # Guard: no two distinct values in (pos ∪ neg) collapse after the shift.
        for i in range(len(all_orig)):
            for j in range(i + 1, len(all_orig)):
                if all_orig[i] != all_orig[j]:
                    assume((all_orig[i] + shift) != (all_orig[j] + shift))

        pos2 = [v + shift for v in pos]
        neg2 = [v + shift for v in neg]
        t_orig, j_orig = youden_threshold(pos, neg)
        t_shifted, j_shifted = youden_threshold(pos2, neg2)

        # J must be unchanged.
        assert math.isclose(j_orig, j_shifted, abs_tol=1e-9), (
            f"Translation broke J: {j_orig} vs {j_shifted}"
        )

        # Threshold shifts by c (use generous tolerance for sentinel edge cases).
        assert math.isclose(t_orig + shift, t_shifted, abs_tol=1e-4, rel_tol=1e-4), (
            f"Translation equivariance failed: t+shift={t_orig + shift}, "
            f"t_shifted={t_shifted}"
        )

    @given(
        _pos_neg,
        st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False,
            allow_infinity=False, allow_subnormal=False,
        ),
    )
    def test_positive_scale_equivariance(self, pos_neg, scale):
        """Scaling all scores by positive c leaves Youden's J unchanged and scales the threshold.

        J-invariance holds for all inputs where the ordering of (pos, neg) pairs
        is preserved by the scaling (which it is for any scale > 0).  Threshold
        scaling holds for non-degenerate inputs (skip when all values are equal,
        since the sentinel does not scale).
        """
        pos, neg = pos_neg
        t_orig, j_orig = youden_threshold(pos, neg)
        pos2 = [scale * v for v in pos]
        neg2 = [scale * v for v in neg]
        t_scaled, j_scaled = youden_threshold(pos2, neg2)

        # J must be unchanged.
        assert math.isclose(j_orig, j_scaled, abs_tol=1e-9), (
            f"Scale broke J: {j_orig} vs {j_scaled}"
        )

        # Skip the threshold check when all values are equal (sentinel artifact).
        all_vals = sorted(set(list(pos) + list(neg)))
        if len(all_vals) == 1:
            return

        assert math.isclose(scale * t_orig, t_scaled, abs_tol=1e-4, rel_tol=1e-4), (
            f"Scale equivariance failed: scale*t={scale * t_orig}, t_scaled={t_scaled}"
        )

    @given(_pos_neg)
    def test_empty_returns_nan(self, pos_neg):
        pos, neg = pos_neg
        t, j = youden_threshold([], neg)
        assert math.isnan(t) and math.isnan(j)
        t, j = youden_threshold(pos, [])
        assert math.isnan(t) and math.isnan(j)

    @given(_pos_neg)
    def test_youden_j_in_valid_range(self, pos_neg):
        """youden_threshold's returned J must be in [-1, 1]."""
        pos, neg = pos_neg
        _, j = youden_threshold(pos, neg)
        assert -1.0 <= j <= 1.0, f"Youden's J={j} out of range"


# ---------------------------------------------------------------------------
# Consistency: youden_at(pos, neg, youden_threshold(pos, neg)) == max J
# ---------------------------------------------------------------------------


class TestRankMetric:
    """rank_metric guarantees reproducible ranking when a metric is undefined.

    The pure metrics return nan for a degenerate (single-class) input; sorting
    by nan is non-deterministic, so the sweep ranking routes auc/youden through
    rank_metric.  This pins the reproducibility contract those rankings rely on.
    """

    def test_finite_passthrough(self):
        assert rank_metric(0.73) == 0.73
        assert rank_metric(-1.0) == -1.0

    def test_nan_sinks_in_descending_sort(self):
        rows = [{"auc": 0.4}, {"auc": float("nan")}, {"auc": 0.9}]
        rows.sort(key=lambda r: rank_metric(r["auc"]), reverse=True)
        assert [r["auc"] for r in rows[:2]] == [0.9, 0.4]
        assert math.isnan(rows[-1]["auc"])  # undefined ranks last, deterministically


class TestConsistency:
    @given(_pos_neg)
    def test_youden_at_equals_threshold_j(self, pos_neg):
        """youden_at evaluated at the threshold from youden_threshold must match."""
        pos, neg = pos_neg
        t, j_from_thresh = youden_threshold(pos, neg)
        j_at = youden_at(pos, neg, t)
        assert math.isclose(j_from_thresh, j_at, abs_tol=1e-9), (
            f"Consistency failed: youden_threshold J={j_from_thresh}, "
            f"youden_at J={j_at} at threshold={t}"
        )

    @given(_pos_neg)
    def test_youden_threshold_is_global_max(self, pos_neg):
        """No midpoint candidate achieves higher J than the one returned."""
        pos, neg = pos_neg
        t_best, j_best = youden_threshold(pos, neg)

        all_vals = sorted(set(list(pos) + list(neg)))
        if len(all_vals) <= 1:
            return  # degenerate; covered by no_separation test

        midpoints = [(a + b) / 2.0 for a, b in zip(all_vals[:-1], all_vals[1:])]
        sentinels = [all_vals[0] - 1e-6, all_vals[-1] + 1e-6]
        candidates = sentinels + midpoints

        for cand in candidates:
            j_cand = youden_at(pos, neg, cand)
            assert j_cand <= j_best + 1e-9, (
                f"Found higher J={j_cand} at t={cand} than best J={j_best} at t={t_best}"
            )
