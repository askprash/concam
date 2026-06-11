"""Tests for concam.detection.static_mask — persistent-edge scene masking.

The foreground building produces most detector false positives: its
high-contrast edges are present in every frame, so they survive Canny and
emit Hough lines that happen to align with some flight track. The mask is
derived from edge *persistence* across frames sampled over a day: building
edges are static (present in nearly all frames), while sky features (clouds,
contrails) move between samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from concam.detection.static_mask import (
    compute_static_mask,
    load_static_mask,
    mask_to_polygons,
    save_static_mask,
)


def _frames_with_static_box_and_moving_line(
    n: int = 12, h: int = 120, w: int = 160
) -> list[np.ndarray]:
    """Gray sky + fixed bright 'building' box + a line that moves every frame."""
    rng = np.random.default_rng(7)
    frames = []
    for i in range(n):
        f = np.full((h, w), 120, dtype=np.uint8)
        f += rng.integers(0, 3, size=(h, w), dtype=np.uint8)  # sensor noise
        # Static building: bright box bottom-left, hard edges every frame.
        f[70:115, 10:60] = 220
        # Moving contrail-like line: shifts right each frame.
        x = 70 + i * 6
        f[20:24, x : x + 30] = 230
        frames.append(f)
    return frames


class TestComputeStaticMask:
    def test_masks_static_building_edges(self):
        frames = _frames_with_static_box_and_moving_line()
        mask = compute_static_mask(frames, dilate_px=3)
        assert mask.dtype == bool
        assert mask.shape == frames[0].shape
        # The building's top edge (y=70, x in 10..60) must be masked.
        assert mask[68:73, 15:55].any()

    def test_does_not_mask_moving_features(self):
        frames = _frames_with_static_box_and_moving_line()
        mask = compute_static_mask(frames, dilate_px=3)
        # The moving line's band (y≈20-24) should stay unmasked: each position
        # is occupied in only one of 12 samples, far below the persistence
        # threshold.
        assert not mask[15:30, :].any()

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            compute_static_mask([])

    def test_dilation_grows_mask(self):
        frames = _frames_with_static_box_and_moving_line()
        thin = compute_static_mask(frames, dilate_px=1)
        fat = compute_static_mask(frames, dilate_px=9)
        assert fat.sum() > thin.sum()
        assert (fat | thin).sum() == fat.sum()  # superset


class TestPolygons:
    def test_polygons_cover_masked_region(self):
        frames = _frames_with_static_box_and_moving_line()
        mask = compute_static_mask(frames, dilate_px=3)
        polys = mask_to_polygons(mask)
        assert len(polys) >= 1
        # Polygons are [[x, y], ...] vertex lists in full-frame pixel coords.
        xs = [p[0] for poly in polys for p in poly]
        ys = [p[1] for poly in polys for p in poly]
        assert min(xs) >= 0 and max(xs) < mask.shape[1]
        assert min(ys) >= 0 and max(ys) < mask.shape[0]
        # Building box corner must fall inside some polygon's bbox.
        assert any(
            min(p[0] for p in poly) <= 12 and max(p[0] for p in poly) >= 55
            and min(p[1] for p in poly) <= 75 and max(p[1] for p in poly) >= 110
            for poly in polys
        )

    def test_empty_mask_yields_no_polygons(self):
        assert mask_to_polygons(np.zeros((20, 30), dtype=bool)) == []


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        frames = _frames_with_static_box_and_moving_line()
        mask = compute_static_mask(frames, dilate_px=3)
        path = tmp_path / "static_mask.npz"
        save_static_mask(mask, path)
        loaded = load_static_mask(path)
        assert loaded.dtype == bool
        np.testing.assert_array_equal(loaded, mask)

    def test_load_is_cached(self, tmp_path):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        path = tmp_path / "m.npz"
        save_static_mask(mask, path)
        a = load_static_mask(path)
        b = load_static_mask(path)
        assert a is b  # same object — per-path cache, hit once per process
