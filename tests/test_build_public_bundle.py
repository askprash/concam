"""Tests for scripts/build_public_bundle.py exclusion-region plumbing."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from concam.detection.static_mask import save_static_mask

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

_spec = importlib.util.spec_from_file_location(
    "build_public_bundle", SCRIPTS_DIR / "build_public_bundle.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["build_public_bundle"] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)

exclusion_regions_block = _module.exclusion_regions_block


@dataclass
class _DetCfg:
    static_mask_path: str | None = None
    timestamp_exclusion_region: list | None = None


def test_none_when_nothing_configured():
    assert exclusion_regions_block(_DetCfg()) is None


def test_timestamp_only():
    block = exclusion_regions_block(
        _DetCfg(timestamp_exclusion_region=[0, 95, 2950, 3840])
    )
    assert block == {"polygons": [], "timestamp_region": [0, 95, 2950, 3840]}


def test_mask_polygons_included(tmp_path):
    mask = np.zeros((200, 300), dtype=bool)
    mask[120:190, 40:260] = True  # big "building" blob > min_area
    p = tmp_path / "mask.npz"
    save_static_mask(mask, p)
    block = exclusion_regions_block(_DetCfg(static_mask_path=str(p)))
    assert block is not None
    assert len(block["polygons"]) == 1
    xs = [v[0] for v in block["polygons"][0]]
    ys = [v[1] for v in block["polygons"][0]]
    assert min(xs) >= 39 and max(xs) <= 260
    assert min(ys) >= 119 and max(ys) <= 190


def test_missing_mask_file_ignored(tmp_path):
    block = exclusion_regions_block(
        _DetCfg(static_mask_path=str(tmp_path / "absent.npz"),
                timestamp_exclusion_region=[0, 95, 2950, 3840])
    )
    assert block is not None and block["polygons"] == []
