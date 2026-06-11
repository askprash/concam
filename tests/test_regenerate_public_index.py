"""Tests for scripts/regenerate_public_index.py (labeler dots on the calendar).

The index generator scans the repo ``labels/`` directory and attaches a
``labelers`` list to each dates.json entry so the labeler calendar can render
one colored dot per human who labeled that day.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

_spec = importlib.util.spec_from_file_location(
    "regenerate_public_index",
    SCRIPTS_DIR / "regenerate_public_index.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["regenerate_public_index"] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)

scan_labelers = _module.scan_labelers
build_dates = _module.build_dates
regenerate = _module.regenerate


def _write_label_file(labels_dir: Path, name: str, payload: dict) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / name).write_text(json.dumps(payload))


def _write_manifest(public_root: Path, date: str, episodes: list[dict],
                    threshold: float = 0.3) -> None:
    d = public_root / date
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "date": date,
        "detection_threshold": threshold,
        "episodes": episodes,
    }))


class TestScanLabelers:
    def test_reads_date_and_labeler_id_from_json(self, tmp_path):
        _write_label_file(tmp_path, "2026-04-09_prash.json", {
            "date": "2026-04-09", "labeler_id": "prash", "labels": []})
        _write_label_file(tmp_path, "2026-04-09_thendo.json", {
            "date": "2026-04-09", "labeler_id": "thendo", "labels": []})
        assert scan_labelers(tmp_path) == {"2026-04-09": ["prash", "thendo"]}

    def test_falls_back_to_filename_when_fields_missing(self, tmp_path):
        # Legacy export naming: DATE_labeler_labels.json with no metadata block.
        _write_label_file(tmp_path, "2026-03-29_thendo_labels.json", {"labels": []})
        assert scan_labelers(tmp_path) == {"2026-03-29": ["thendo"]}

    def test_dedupes_and_sorts_labelers(self, tmp_path):
        _write_label_file(tmp_path, "2026-04-09_thendo.json", {
            "date": "2026-04-09", "labeler_id": "thendo"})
        _write_label_file(tmp_path, "2026-04-09_thendo_labels.json", {
            "date": "2026-04-09", "labeler_id": "thendo"})
        _write_label_file(tmp_path, "2026-04-09_lrsand_labels.json", {
            "date": "2026-04-09", "labeler_id": "lrsand"})
        assert scan_labelers(tmp_path) == {"2026-04-09": ["lrsand", "thendo"]}

    def test_ignores_malformed_and_non_label_files(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "broken.json").write_text("{not json")
        (tmp_path / "README.md").write_text("notes")
        _write_label_file(tmp_path, "2026-04-15_reviewer-1.json", {
            "date": "2026-04-15", "labeler_id": "reviewer-1"})
        assert scan_labelers(tmp_path) == {"2026-04-15": ["reviewer-1"]}

    def test_missing_dir_yields_empty(self, tmp_path):
        assert scan_labelers(tmp_path / "nope") == {}


class TestBuildDates:
    def test_attaches_labelers_to_matching_dates(self, tmp_path):
        _write_manifest(tmp_path, "2026-04-09",
                        [{"peak_score": 0.9}, {"peak_score": 0.1}])
        _write_manifest(tmp_path, "2026-04-10", [{"peak_score": 0.0}])
        dates = build_dates(tmp_path, {"2026-04-09": ["prash", "thendo"]})
        by_date = {d["date"]: d for d in dates}
        assert by_date["2026-04-09"]["labelers"] == ["prash", "thendo"]
        assert by_date["2026-04-10"]["labelers"] == []
        assert by_date["2026-04-09"]["episodes"] == 2
        assert by_date["2026-04-09"]["detected"] == 1

    def test_variant_dirs_inherit_base_date_labelers(self, tmp_path):
        # 2026-04-09-tuned is an A/B variant of 2026-04-09; same human labels.
        _write_manifest(tmp_path, "2026-04-09-tuned", [{"peak_score": 0.5}])
        dates = build_dates(tmp_path, {"2026-04-09": ["prash"]})
        assert dates[0]["labelers"] == ["prash"]

    def test_newest_first(self, tmp_path):
        _write_manifest(tmp_path, "2026-04-09", [])
        _write_manifest(tmp_path, "2026-04-10", [])
        dates = build_dates(tmp_path, {})
        assert [d["date"] for d in dates] == ["2026-04-10", "2026-04-09"]


class TestRegenerate:
    def test_writes_dates_json_and_index(self, tmp_path):
        public_root = tmp_path / "public"
        labels_dir = tmp_path / "labels"
        _write_manifest(public_root, "2026-04-09", [{"peak_score": 0.9}])
        _write_label_file(labels_dir, "2026-04-09_prash.json", {
            "date": "2026-04-09", "labeler_id": "prash"})
        regenerate(public_root, labels_dir)
        payload = json.loads((public_root / "dates.json").read_text())
        assert payload["dates"][0]["labelers"] == ["prash"]
        assert (public_root / "index.html").exists()
