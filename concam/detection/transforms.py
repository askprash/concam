"""Per-pixel image transforms for the contrail-detector playground.

Each transform takes a BGR uint8 patch (or a 2D gray patch — see below) and
returns a single-channel **uint8 gray** image of the same spatial shape,
chainable for ``notebooks/filter_playground.ipynb``.  Returning gray means
``concam.detection.detect`` receives an already-single-channel input and its
internal ``_to_gray`` becomes a no-op — the colormap-rendered BGR path that
would have thrown away per-pixel sign information via ``cv2.cvtColor`` is
avoided.

Chaining
--------
    >>> chain = ["cross_grad", "local_contrast", "frangi"]
    >>> gray = apply_chain(bgr, chain, path_vec=path_vec, prev_bgr=prev_bgr)
    >>> result = detect(gray, rect, cfg, polygon=poly, path_vec=path_vec)

The first transform in the chain receives the original BGR patch; each
subsequent transform gets the previous transform's gray output.  Transforms
that need color (``nrbr``, ``lab_b``, ``hsv_sat_inv``, ``lab_L``,
``grey_excess``) call ``_ensure_bgr`` to coerce a 2D input back into a
3-channel replicate so chaining never crashes; the transform output is
degenerate in that configuration (there is no meaningful NRBR on a greyscale
input), which is acceptable — colour transforms belong at the head of the
chain.

Visualisation
-------------
``to_display_bgr(gray, name)`` maps a gray output through the canonical
matplotlib colormap each transform was originally rendered with in
``scripts/channel_matrix_spotcheck.py`` (RdBu_r for signed, hot for
magnitude, inferno for grey-excess, greyscale for luminance).  The notebook
uses this for its 4-panel previews; the detector never sees the colormap.

skimage
-------
Where scikit-image offers a cleaner / better-documented primitive the
transform wraps it: ``frangi`` (ridge filter), ``difference_of_gaussians``,
``white_tophat``, ``equalize_adapthist``.  Basic ops (Sobel, GaussianBlur,
colour-space conversion) continue to use cv2 — they're fast and already in
the dep graph.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _ensure_bgr(arr: np.ndarray) -> np.ndarray:
    """Coerce a 2D gray patch to a 3-channel BGR replicate; pass BGR through."""
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


def _ensure_gray(arr: np.ndarray) -> np.ndarray:
    """Coerce a BGR patch to 2D gray; pass gray through."""
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip a float array to [0, 255] and cast to uint8."""
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Return a uint8 gray image; float inputs are rescaled via min-max."""
    if arr.dtype == np.uint8:
        return arr
    f = arr.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros_like(f, dtype=np.uint8)
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Individual transforms — signature (bgr_or_gray, **kwargs) -> gray uint8
# ---------------------------------------------------------------------------

def tf_bgr(bgr: np.ndarray, **_: Any) -> np.ndarray:
    """Identity in greyscale — the canonical reference channel."""
    return _ensure_gray(bgr)


def tf_nrbr(bgr: np.ndarray, **_: Any) -> np.ndarray:
    """(R-B)/(R+B). Rayleigh scattering maps clear sky negative, contrail 0.

    Range [-1, 1] is shifted to 128-centred uint8 (clip to [-1, 1], map
    -1 -> 0, 0 -> 128, +1 -> 255) so downstream Canny sees contrails (near
    the 0-point) as a bright streak against a dark sky background.
    """
    bgr_in = _ensure_bgr(bgr)
    f = bgr_in.astype(np.float32)
    r, b = f[:, :, 2], f[:, :, 0]
    nrbr = (r - b) / (r + b + 1e-6)
    # Contrail near 0, sky negative. Flip sign so contrail > sky for Canny.
    return _clip_uint8(128.0 - 255.0 * np.clip(nrbr, -1.0, 1.0))


def tf_hsv_sat_inv(bgr: np.ndarray, **_: Any) -> np.ndarray:
    """Inverted HSV saturation: low-saturation objects (contrails) are bright."""
    bgr_in = _ensure_bgr(bgr)
    hsv = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2HSV)
    return 255 - hsv[:, :, 1]


def tf_lab_L(bgr: np.ndarray, **_: Any) -> np.ndarray:
    """CIELAB L* channel — perceptual luminance."""
    bgr_in = _ensure_bgr(bgr)
    lab = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0]


def tf_lab_b(bgr: np.ndarray, **_: Any) -> np.ndarray:
    """CIELAB b* channel — yellow-blue axis, perceptual analog of NRBR.

    Shifted so contrail (near 128) maps to mid-gray and sky (negative b*)
    maps to dark.  Sign is flipped vs the raw axis so contrail > sky.
    """
    bgr_in = _ensure_bgr(bgr)
    lab = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2LAB)
    b_star = lab[:, :, 2].astype(np.float32) - 128.0  # 0-centred in [-128, 127]
    # Sky is very negative (blue); contrail near 0. Map so 0 -> 255, -128 -> 0.
    return _clip_uint8(255.0 + b_star * 2.0)


def tf_grey_excess(bgr: np.ndarray, **_: Any) -> np.ndarray:
    """min(R,G,B) / (max(R,G,B)+1). Neutral grey/white high, saturated low."""
    bgr_in = _ensure_bgr(bgr)
    f = bgr_in.astype(np.float32)
    grey_ex = np.min(f, axis=2) / (np.max(f, axis=2) + 1.0)
    return _clip_uint8(grey_ex * 255.0)


def tf_local_contrast(
    bgr: np.ndarray,
    local_contrast_sigma: float = 25.0,
    **_: Any,
) -> np.ndarray:
    """gray - GaussianBlur(gray, sigma). Removes slow cloud background.

    Output is 128-centred so downstream Canny sees the contrail as a bright
    streak against a mid-gray background rather than losing negative values
    to an implicit clip at 0.
    """
    gray = _ensure_gray(bgr).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), float(local_contrast_sigma))
    lc = gray - bg
    return _clip_uint8(128.0 + lc)


def tf_dog(
    bgr: np.ndarray,
    dog_sigma_low: float = 2.0,
    dog_sigma_high: float = 15.0,
    **_: Any,
) -> np.ndarray:
    """Difference of Gaussians. Band-pass tuned to the contrail cross-section.

    Uses ``skimage.filters.difference_of_gaussians`` so sigma semantics and
    boundary handling match scikit-image's documented behaviour.
    """
    from skimage.filters import difference_of_gaussians

    gray = _ensure_gray(bgr).astype(np.float32) / 255.0
    dog = difference_of_gaussians(
        gray, float(dog_sigma_low), float(dog_sigma_high)
    )
    # dog is 0-centred floating point. Amplify and shift to mid-gray for Canny.
    return _clip_uint8(128.0 + dog * 255.0 * 3.0)


def tf_tophat(
    bgr: np.ndarray,
    tophat_kernel_len: int = 80,
    tophat_kernel_width: int = 12,
    **_: Any,
) -> np.ndarray:
    """White-tophat with a horizontal ellipse kernel. Bright elongated features.

    Kernel size is expressed in pixels of the detection patch.  Unchanged
    default (80x12) matches ``channel_matrix_spotcheck.py``.
    """
    gray = _ensure_gray(bgr)
    klen = max(3, int(tophat_kernel_len))
    kw = max(3, int(tophat_kernel_width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (klen, kw))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def tf_tophat_oriented(
    bgr: np.ndarray,
    path_vec: tuple[float, float] | None = None,
    tophat_oriented_length: int = 81,
    tophat_oriented_width: int = 5,
    **_: Any,
) -> np.ndarray:
    """White-tophat with a line kernel rotated to the flight-path direction.

    More discriminating than ``tophat`` when the contrail is diagonal, since
    the kernel fits the along-track geometry exactly.
    """
    gray = _ensure_gray(bgr)
    if path_vec is not None and (path_vec[0] ** 2 + path_vec[1] ** 2) > 0.01:
        angle_deg = float(np.degrees(np.arctan2(float(path_vec[1]), float(path_vec[0]))))
    else:
        angle_deg = 0.0
    klen = max(5, int(tophat_oriented_length)) | 1  # ensure odd
    kw = max(1, int(tophat_oriented_width))
    base = np.zeros((klen, klen), dtype=np.uint8)
    cv2.line(base, (0, klen // 2), (klen - 1, klen // 2), 1, kw)
    M = cv2.getRotationMatrix2D((klen // 2, klen // 2), angle_deg, 1)
    kernel_rot = cv2.warpAffine(base, M, (klen, klen))
    kernel_rot = (kernel_rot > 0).astype(np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_rot)


def tf_clahe(
    bgr: np.ndarray,
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid: int = 8,
    **_: Any,
) -> np.ndarray:
    """CLAHE on the grey channel. Local contrast equalisation.

    cv2.createCLAHE is kept here (vs ``skimage.exposure.equalize_adapthist``)
    because it's 5-10x faster on the 4K-style crops used for contrail
    detection and the output is interchangeable at the parameter level.
    """
    gray = _ensure_gray(bgr)
    tile = max(2, int(clahe_tile_grid))
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip_limit),
        tileGridSize=(tile, tile),
    )
    return clahe.apply(gray)


def tf_cross_grad(
    bgr: np.ndarray,
    path_vec: tuple[float, float] | None = None,
    cross_grad_gain: float = 2.0,
    **_: Any,
) -> np.ndarray:
    """|Sobel . perp(path_vec)|. Gradient magnitude perpendicular to flight."""
    gray = _ensure_gray(bgr).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    if path_vec is not None and (path_vec[0] ** 2 + path_vec[1] ** 2) > 0.01:
        px, py = float(path_vec[0]), float(path_vec[1])
        perp_x, perp_y = -py, px
    else:
        perp_x, perp_y = 0.0, 1.0
    cross = np.abs(sobel_x * perp_x + sobel_y * perp_y)
    return _clip_uint8(cross * float(cross_grad_gain))


def tf_temporal_diff(
    bgr: np.ndarray,
    prev_bgr: np.ndarray | None = None,
    temporal_diff_gain: float = 4.0,
    **_: Any,
) -> np.ndarray:
    """|current - previous|. Static cloud cancels; new contrail remains.

    Returns a uniform 128-gray patch when no previous frame is available, so
    downstream Canny produces no spurious edges.
    """
    if prev_bgr is None:
        return np.full(_ensure_gray(bgr).shape, 128, dtype=np.uint8)
    cur = _ensure_gray(bgr).astype(np.float32)
    prev = _ensure_gray(prev_bgr).astype(np.float32)
    if prev.shape != cur.shape:
        return np.full(cur.shape, 128, dtype=np.uint8)
    diff = np.abs(cur - prev)
    return _clip_uint8(diff * float(temporal_diff_gain))


def tf_frangi(
    bgr: np.ndarray,
    frangi_sigma_min: float = 1.0,
    frangi_sigma_max: float = 4.0,
    frangi_sigma_step: float = 1.0,
    frangi_beta: float = 0.5,
    frangi_gamma: float | None = None,
    **_: Any,
) -> np.ndarray:
    """skimage Frangi ridge response, tuned for bright contrails on dark sky.

    Sigmas sweep ridge widths between ``frangi_sigma_min`` and
    ``frangi_sigma_max`` inclusive, in steps of ``frangi_sigma_step``.
    Response is normalised by its 95th percentile so the output uses its full
    uint8 dynamic range regardless of image noise level.
    """
    from skimage.filters import frangi as skimage_frangi

    gray = _ensure_gray(bgr).astype(np.float32) / 255.0
    sigmas = np.arange(
        max(0.5, float(frangi_sigma_min)),
        max(float(frangi_sigma_min) + 0.5, float(frangi_sigma_max)) + 1e-6,
        max(0.5, float(frangi_sigma_step)),
    )
    kwargs: dict[str, Any] = {
        "sigmas": sigmas,
        "black_ridges": False,
        "beta": float(frangi_beta),
    }
    if frangi_gamma is not None:
        kwargs["gamma"] = float(frangi_gamma)
    response = skimage_frangi(gray, **kwargs)
    p95 = float(np.percentile(response, 95)) if response.size else 0.0
    if p95 <= 0.0:
        p95 = 1e-6
    normed = np.clip(response / p95, 0.0, 1.0)
    return _clip_uint8(normed * 255.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TransformSpec = tuple[str, Callable[..., np.ndarray], bool, bool]

# Ordered to match scripts/channel_matrix_spotcheck.py (rows 1-13) with Frangi
# appended as #14.  Each tuple is (chain_key, fn, needs_path_vec, needs_prev).
TRANSFORMS: list[TransformSpec] = [
    ("bgr", tf_bgr, False, False),
    ("nrbr", tf_nrbr, False, False),
    ("hsv_sat_inv", tf_hsv_sat_inv, False, False),
    ("lab_L", tf_lab_L, False, False),
    ("lab_b", tf_lab_b, False, False),
    ("grey_excess", tf_grey_excess, False, False),
    ("local_contrast", tf_local_contrast, False, False),
    ("dog", tf_dog, False, False),
    ("tophat", tf_tophat, False, False),
    ("tophat_oriented", tf_tophat_oriented, True, False),
    ("clahe", tf_clahe, False, False),
    ("cross_grad", tf_cross_grad, True, False),
    ("temporal_diff", tf_temporal_diff, False, True),
    ("frangi", tf_frangi, False, False),
]

TRANSFORMS_BY_NAME: dict[str, TransformSpec] = {t[0]: t for t in TRANSFORMS}

# Human-readable labels used by the channel-matrix renderer.  Kept here so
# the future channel_matrix_spotcheck refactor can import both the
# transforms and their canonical labels in one go.
DISPLAY_LABELS: dict[str, str] = {
    "bgr": "BGR (ref)",
    "nrbr": "NRBR\n(R-B)/(R+B)",
    "hsv_sat_inv": "HSV Sat\u207b\u00b9\n(white=bright)",
    "lab_L": "LAB-L*\n(luminance)",
    "lab_b": "LAB-b*\n(blue-yellow axis)",
    "grey_excess": "Grey-excess\nmin/max",
    "local_contrast": "Local-contrast\ngray - blur(\u03c325)",
    "dog": "DoG \u03c3(2-15)\nband-pass",
    "tophat": "White-tophat\nhoriz.kernel 80\u00d712",
    "tophat_oriented": "Tophat-oriented\nalong flight path",
    "clahe": "CLAHE\nlocal equalisation",
    "cross_grad": "Cross-path grad\n\u22a5 to flight dir",
    "temporal_diff": "Temporal-diff\n|frame - prev|",
    "frangi": "Frangi\nridge filter",
}

# Canonical display colormaps per-transform (for the notebook's 4-panel
# preview and any visualisation helper that wants to reproduce the
# channel-matrix aesthetic without re-implementing the colormap logic).
DISPLAY_COLORMAPS: dict[str, str] = {
    "bgr": "gray",
    "nrbr": "RdBu_r",
    "hsv_sat_inv": "gray",
    "lab_L": "gray",
    "lab_b": "RdBu_r",
    "grey_excess": "inferno",
    "local_contrast": "RdBu_r",
    "dog": "RdBu_r",
    "tophat": "gray",
    "tophat_oriented": "gray",
    "clahe": "gray",
    "cross_grad": "hot",
    "temporal_diff": "hot",
    "frangi": "hot",
}


def apply_chain(
    bgr: np.ndarray,
    chain: Iterable[str],
    *,
    path_vec: tuple[float, float] | None = None,
    prev_bgr: np.ndarray | None = None,
    transform_params: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Apply a sequence of transforms by name, chaining BGR/gray -> gray.

    Unknown names and the literal "none" are skipped, which lets dropdowns in
    the playground notebook pass a fixed-length chain where some slots are
    disabled.  ``transform_params`` optionally maps a transform name to its
    keyword arguments (used by Frangi / DoG / local_contrast / morphology).
    """
    out = bgr
    params = transform_params or {}
    for name in chain:
        if not name or name == "none":
            continue
        spec = TRANSFORMS_BY_NAME.get(name)
        if spec is None:
            continue
        _, fn, needs_pv, needs_prev = spec
        kwargs: dict[str, Any] = {}
        if needs_pv:
            kwargs["path_vec"] = path_vec
        if needs_prev:
            kwargs["prev_bgr"] = prev_bgr
        kwargs.update(params.get(name, {}))
        out = fn(out, **kwargs)
    # Always return gray so downstream code (detect, matplotlib imshow with
    # cmap='gray') can treat apply_chain's output uniformly regardless of
    # whether any transforms actually ran.
    return _ensure_gray(out)
