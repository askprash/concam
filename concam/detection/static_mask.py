"""Static-scene mask: persistent-edge detection of foreground structures.

The camera scene contains fixed structures — most prominently the tall
building in the foreground — whose high-contrast edges survive Canny in every
frame and emit Hough lines. When a flight track happens to align with one of
those edges, the detector reports a false positive; review showed the building
is the dominant FP source.

Buildings are *static*: their edges sit at the same pixels in (nearly) every
frame of a day, while sky features (clouds, contrails, aircraft) move between
samples taken minutes apart. So the mask is computed from **edge persistence**:
run a fixed Canny over frames sampled across a day, accumulate the per-pixel
edge frequency, and mask pixels that are edges in more than
``persistence_threshold`` of the samples, dilated by a safety margin.

The saved mask is a site artifact (like the calibration ``.npz``): build it
once with ``scripts/build_static_mask.py``, point
``DetectionConfig.static_mask_path`` at it, and the detection kernel zeroes
masked pixels before Canny — same mechanism as the timestamp exclusion.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Per-path cache: the kernel resolves the mask on every detection pass, so the
# file must only be read once per process. Keyed by resolved path string.
_MASK_CACHE: dict[str, np.ndarray] = {}


def compute_static_mask(
    frames,
    *,
    persistence_threshold: float = 0.5,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: int = 5,
    dilate_px: int = 12,
) -> np.ndarray:
    """Boolean (H, W) mask of persistently-edgy (static structure) pixels.

    Args:
        frames: iterable of (H, W) gray or (H, W, 3) BGR uint8 frames sampled
            across a day. A few dozen, spread over hours, is enough — what
            matters is that moving sky features don't repeat positions.
        persistence_threshold: fraction of frames in which a pixel must be a
            Canny edge to count as static. 0.5 tolerates night frames where
            unlit building edges drop out, while still rejecting any feature
            that moves between samples.
        canny_low / canny_high: fixed Canny thresholds. Deliberately *not* the
            detector's adaptive thresholds — persistence, not sensitivity, is
            what separates structure from sky here.
        blur_kernel: pre-Canny Gaussian kernel (odd; 0/1 disables).
        dilate_px: safety margin grown around persistent edges so that slight
            camera shake / rendering halo stays inside the mask.
    """
    acc: np.ndarray | None = None
    n = 0
    for frame in frames:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        if blur_kernel and blur_kernel > 1:
            k = int(blur_kernel) | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(gray, canny_low, canny_high)
        if acc is None:
            acc = np.zeros(edges.shape, dtype=np.int32)
        acc += edges > 0
        n += 1
    if acc is None or n == 0:
        raise ValueError("compute_static_mask: no frames supplied")

    static = (acc / n) >= persistence_threshold
    if dilate_px and dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        static = cv2.dilate(static.astype(np.uint8), kernel) > 0
    return static.astype(bool)


def mask_to_polygons(
    mask: np.ndarray,
    *,
    epsilon_px: float = 4.0,
    min_area_px: float = 64.0,
) -> list[list[list[int]]]:
    """Approximate the mask's connected regions as polygons.

    Returns ``[[[x, y], ...], ...]`` vertex lists in full-frame pixel
    coordinates — the shape shipped in the bundle manifest so the labeler can
    hatch the ignored regions. ``epsilon_px`` is the Douglas-Peucker tolerance;
    regions smaller than ``min_area_px`` are dropped as visual noise.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys: list[list[list[int]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_px, True)
        if len(approx) >= 3:
            polys.append([[int(p[0][0]), int(p[0][1])] for p in approx])
    return polys


def parse_svg_polygons(svg_text: str) -> tuple[list[np.ndarray], tuple[float, float]]:
    """Extract filled polygon outlines from a hand-drawn SVG mask.

    Supports the straight-line path subset Inkscape emits for polygon tools
    (``m/M l/L h/H v/V z/Z``; bare coordinate pairs are implicit linetos).
    Curves are rejected — a mask outline must be redrawn with straight
    segments. Returns ``(polygons, (svg_width, svg_height))`` where each
    polygon is an (N, 2) float array in SVG user units.
    """
    import re

    size_m = re.search(r'<svg[^>]*\bwidth="([\d.]+)"[^>]*\bheight="([\d.]+)"',
                       svg_text, re.S)
    if size_m is None:
        raise ValueError("SVG width/height attributes not found")
    size = (float(size_m.group(1)), float(size_m.group(2)))

    polygons: list[np.ndarray] = []
    for d in re.findall(r'\bd="([^"]+)"', svg_text):
        if not re.match(r"\s*[mM]", d):
            continue
        tokens = re.findall(r"[a-zA-Z]|-?\d*\.?\d+(?:e-?\d+)?", d)
        pts: list[np.ndarray] = []
        cur = np.zeros(2)
        cmd = None
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.isalpha():
                cmd = t
                i += 1
                if cmd in "zZ":
                    if len(pts) >= 3:
                        polygons.append(np.array(pts))
                    pts = []
                continue
            if cmd in ("m", "l"):
                delta = np.array([float(t), float(tokens[i + 1])])
                cur = delta if (cmd == "m" and not pts) else cur + delta
                pts.append(cur.copy())
                i += 2
                cmd = "l" if cmd == "m" else cmd
            elif cmd in ("M", "L"):
                cur = np.array([float(t), float(tokens[i + 1])])
                pts.append(cur.copy())
                i += 2
                cmd = "L" if cmd == "M" else cmd
            elif cmd == "h":
                cur = cur + [float(t), 0.0]; pts.append(cur.copy()); i += 1
            elif cmd == "H":
                cur = np.array([float(t), cur[1]]); pts.append(cur.copy()); i += 1
            elif cmd == "v":
                cur = cur + [0.0, float(t)]; pts.append(cur.copy()); i += 1
            elif cmd == "V":
                cur = np.array([cur[0], float(t)]); pts.append(cur.copy()); i += 1
            else:
                raise ValueError(
                    f"unsupported SVG path command {cmd!r} — redraw the mask "
                    "outline with straight segments (no curves)"
                )
        if len(pts) >= 3:  # unclosed trailing subpath
            polygons.append(np.array(pts))
    return polygons, size


def svg_to_mask(svg_text: str, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a hand-drawn SVG mask onto a (H, W) boolean array.

    Polygon coordinates are scaled from the SVG canvas to ``shape`` (the SVG
    is typically drawn over a downscaled screenshot of the camera frame).
    """
    polygons, (svg_w, svg_h) = parse_svg_polygons(svg_text)
    h, w = shape
    sx, sy = w / svg_w, h / svg_h
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        scaled = np.round(poly * [sx, sy]).astype(np.int32)
        cv2.fillPoly(mask, [scaled], 255)
    return mask > 0


def save_static_mask(mask: np.ndarray, path: str | Path) -> None:
    """Write the mask as a compressed npz (key: ``mask``, bool array)."""
    np.savez_compressed(path, mask=mask.astype(bool))


def load_static_mask(path: str | Path) -> np.ndarray:
    """Load (and cache per-path) a mask written by :func:`save_static_mask`."""
    key = str(Path(path).resolve())
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    with np.load(key) as data:
        mask = data["mask"].astype(bool)
    mask.setflags(write=False)
    _MASK_CACHE[key] = mask
    return mask
