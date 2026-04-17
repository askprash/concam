"""Unit tests for concam.detection.transforms.

Goal: every transform in TRANSFORMS returns a gray uint8 image of matching
spatial shape, chains compose without crashing, and the ridge-sensitive
transforms (cross_grad, frangi, tophat, local_contrast, dog, tophat_oriented)
actually light up a painted bright line against a dark background.
"""

from __future__ import annotations

import numpy as np
import pytest

from concam.detection.transforms import (
    DISPLAY_COLORMAPS,
    DISPLAY_LABELS,
    TRANSFORMS,
    TRANSFORMS_BY_NAME,
    apply_chain,
)


@pytest.fixture
def bright_line_bgr() -> np.ndarray:
    """A 200x120 BGR patch with a horizontal 3-pixel-wide bright line in the middle."""
    img = np.full((120, 200, 3), 30, dtype=np.uint8)  # dark background
    img[58:61, 20:180] = (240, 240, 240)  # bright line (near-white)
    return img


@pytest.fixture
def prev_bgr_shifted(bright_line_bgr: np.ndarray) -> np.ndarray:
    """Previous-frame analog of bright_line_bgr with the line slightly shifted."""
    img = np.full_like(bright_line_bgr, 30)
    img[55:58, 20:180] = (240, 240, 240)  # line 3 px up
    return img


@pytest.fixture
def horizontal_path_vec() -> tuple[float, float]:
    return (1.0, 0.0)


# ---------------------------------------------------------------------------
# Shape / dtype contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,fn,needs_pv,needs_prev", TRANSFORMS)
def test_transform_shape_and_dtype(
    name: str,
    fn,
    needs_pv: bool,
    needs_prev: bool,
    bright_line_bgr: np.ndarray,
    prev_bgr_shifted: np.ndarray,
    horizontal_path_vec: tuple[float, float],
) -> None:
    kwargs = {}
    if needs_pv:
        kwargs["path_vec"] = horizontal_path_vec
    if needs_prev:
        kwargs["prev_bgr"] = prev_bgr_shifted
    out = fn(bright_line_bgr, **kwargs)
    assert out.dtype == np.uint8, f"{name} returned dtype {out.dtype}"
    assert out.ndim == 2, f"{name} returned {out.ndim}-D array, expected 2-D gray"
    assert out.shape == bright_line_bgr.shape[:2], (
        f"{name} shape {out.shape} != input {bright_line_bgr.shape[:2]}"
    )


# ---------------------------------------------------------------------------
# Semantic: ridge-sensitive transforms brighten the painted line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["cross_grad", "frangi", "tophat", "local_contrast", "dog", "tophat_oriented"],
)
def test_ridge_sensitive_transforms_highlight_line(
    name: str,
    bright_line_bgr: np.ndarray,
    horizontal_path_vec: tuple[float, float],
) -> None:
    spec = TRANSFORMS_BY_NAME[name]
    fn = spec[1]
    kwargs = {}
    if spec[2]:  # needs_pv
        kwargs["path_vec"] = horizontal_path_vec
    out = fn(bright_line_bgr, **kwargs)
    line_rows = out[57:62, 20:180]
    bg_rows = out[10:30, 20:180]
    # Line region mean should be meaningfully above background mean.
    assert float(line_rows.mean()) > float(bg_rows.mean()) + 20.0, (
        f"{name}: line mean {line_rows.mean():.1f} vs bg mean {bg_rows.mean():.1f}"
    )


def test_temporal_diff_highlights_moved_line(
    bright_line_bgr: np.ndarray,
    prev_bgr_shifted: np.ndarray,
) -> None:
    """temporal_diff should light up where the line moved between frames."""
    out = TRANSFORMS_BY_NAME["temporal_diff"][1](
        bright_line_bgr, prev_bgr=prev_bgr_shifted
    )
    assert out.dtype == np.uint8
    # The two line positions differ in rows 55-60; that band should be bright.
    moved = out[55:61, 20:180]
    quiet = out[10:30, 20:180]
    assert float(moved.mean()) > float(quiet.mean()) + 20.0


def test_temporal_diff_with_no_prev_returns_flat_mid_gray(
    bright_line_bgr: np.ndarray,
) -> None:
    out = TRANSFORMS_BY_NAME["temporal_diff"][1](bright_line_bgr, prev_bgr=None)
    assert out.shape == bright_line_bgr.shape[:2]
    assert int(out.min()) == 128 and int(out.max()) == 128


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------

def test_apply_chain_composes_two_transforms(
    bright_line_bgr: np.ndarray,
    horizontal_path_vec: tuple[float, float],
) -> None:
    out = apply_chain(
        bright_line_bgr,
        ["cross_grad", "local_contrast"],
        path_vec=horizontal_path_vec,
    )
    assert out.dtype == np.uint8 and out.ndim == 2


def test_apply_chain_skips_none_and_unknown(bright_line_bgr: np.ndarray) -> None:
    out = apply_chain(bright_line_bgr, ["none", "does_not_exist", "clahe"])
    assert out.dtype == np.uint8 and out.ndim == 2


def test_apply_chain_empty_returns_bgr_unchanged(bright_line_bgr: np.ndarray) -> None:
    out = apply_chain(bright_line_bgr, [])
    # Empty chain = no transforms applied; returns whatever was passed in.
    assert out is bright_line_bgr


def test_apply_chain_color_transform_after_gray_does_not_crash(
    bright_line_bgr: np.ndarray,
) -> None:
    """NRBR after cross_grad: cross_grad emits gray, NRBR coerces back to BGR.
    Semantically degenerate but must not raise."""
    out = apply_chain(
        bright_line_bgr,
        ["cross_grad", "nrbr"],
        path_vec=(1.0, 0.0),
    )
    assert out.dtype == np.uint8 and out.ndim == 2


def test_apply_chain_frangi_with_custom_params(
    bright_line_bgr: np.ndarray,
) -> None:
    out = apply_chain(
        bright_line_bgr,
        ["frangi"],
        transform_params={
            "frangi": {"frangi_sigma_min": 1.0, "frangi_sigma_max": 3.0},
        },
    )
    assert out.dtype == np.uint8 and out.ndim == 2
    # The painted line has cross-section ~3 px so sigmas 1-3 should fire on it.
    assert float(out[57:62, 20:180].mean()) > float(out[10:30, 20:180].mean()) + 20.0


# ---------------------------------------------------------------------------
# Registry hygiene
# ---------------------------------------------------------------------------

def test_every_transform_has_label_and_colormap() -> None:
    for name, _, _, _ in TRANSFORMS:
        assert name in DISPLAY_LABELS, f"{name} missing from DISPLAY_LABELS"
        assert name in DISPLAY_COLORMAPS, f"{name} missing from DISPLAY_COLORMAPS"


def test_transforms_registry_is_unique() -> None:
    names = [t[0] for t in TRANSFORMS]
    assert len(names) == len(set(names)), "TRANSFORMS has duplicate chain keys"
