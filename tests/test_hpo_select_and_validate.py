"""Tests for scripts/hpo_select_and_validate.py — combo matching across the
train/holdout sweeps, on both the new preprocessing-variant rows and legacy
(cross_grad-only) results files."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

_spec = importlib.util.spec_from_file_location(
    "hpo_select_and_validate", SCRIPTS_DIR / "hpo_select_and_validate.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["hpo_select_and_validate"] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)

_same_combo = _module._same_combo
_row_for_combo = _module._row_for_combo
_variant = _module._variant


def _row(variant: str | None, pct: float, roi: int, *, gain: float = 1.0,
         auc: float = 0.9, yj: float = 0.8) -> dict:
    row = {
        "cross_grad_gain": gain,
        "canny_percentile_high": pct,
        "roi_along_px": roi,
        "auc": auc,
        "best_threshold": 0.12,
        "per_threshold": [
            {"threshold": thr, "tp": 8, "fp": 2, "fn": 2, "tn": 8,
             "youden_j": yj}
            for thr in (0.083, 0.12)
        ],
    }
    if variant is not None:
        row["variant"] = variant
    return row


class TestVariantMatching:
    def test_new_rows_match_on_variant_string(self):
        a = _row("local_contrast(s=25)", 99.5, 180)
        b = _row("local_contrast(s=25)", 99.5, 180)
        c = _row("local_contrast(s=40)", 99.5, 180)
        assert _same_combo(a, b)
        assert not _same_combo(a, c)

    def test_legacy_rows_fall_back_to_gain(self):
        # Pre-June-2026 results files carry no "variant" key; the gain was
        # the only preprocessing knob swept, so it identifies the combo.
        a = _row(None, 99.5, 180, gain=0.75)
        b = _row(None, 99.5, 180, gain=0.75)
        c = _row(None, 99.5, 180, gain=1.5)
        assert _same_combo(a, b)
        assert not _same_combo(a, c)
        assert _variant(a) == "cross_grad(g=0.75)"

    def test_legacy_and_new_interoperate(self):
        # A new-format cross_grad row must find its twin in a legacy file.
        new = _row("cross_grad(g=1.5)", 99.5, 180, gain=1.5)
        legacy = _row(None, 99.5, 180, gain=1.5)
        assert _same_combo(new, legacy)

    def test_missing_combo_aborts(self):
        combo = _row("cross_grad(g=2)", 99.0, 120, gain=2.0)
        others = [_row("none", 99.0, 120), _row("cross_grad(g=2)", 99.3, 120, gain=2.0)]
        with pytest.raises(SystemExit):
            _row_for_combo(others, combo)


def test_end_to_end_report(tmp_path, capsys):
    """main() writes a holdout report whose chosen-config line carries the
    winning variant string."""
    winner = _row("cross_grad(g=1.25)", 99.3, 240, gain=1.25, auc=0.95, yj=0.9)
    other = _row("none", 99.3, 240, auc=0.7, yj=0.5)
    baseline = _row("cross_grad(g=1)", 99.5, 180, gain=1.0, auc=0.8, yj=0.6)

    def payload(date: str) -> dict:
        return {"date": date, "n_pos": 10, "n_neg": 10,
                "labels_merged_from": ["labels/derived/reliable_labels.json#" + date],
                "baseline": baseline,
                "results": [winner, other]}

    train, val = tmp_path / "train.json", tmp_path / "val.json"
    train.write_text(json.dumps(payload("2026-04-08")))
    val.write_text(json.dumps(payload("2026-04-09")))
    cfg = tmp_path / "site.yaml"
    cfg.write_text("name: synthetic\naggregation:\n  detection_threshold: 0.083\n")

    out = tmp_path / "holdout.md"
    old_argv = sys.argv
    sys.argv = ["hpo_select_and_validate.py",
                "--train-results", str(train), "--val-results", str(val),
                "--base-config", str(cfg), "--out", str(out)]
    try:
        assert _module.main() == 0
    finally:
        sys.argv = old_argv

    text = out.read_text()
    assert "cross_grad(g=1.25)" in text
    assert "Generalisation verdict" in text
