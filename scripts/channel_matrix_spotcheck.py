"""Channel-transform matrix for contrail visibility research.

Renders a grid of different image transformations applied to the same 10
spot-check episodes (top-5 by peak_score + 5 near-threshold) from the April-8
regression baseline.

Layout
------
  Rows  = 12 image transforms
  Cols  = 10 episodes (labeled callsign + score)

Each cell shows an 800×500 px context patch extracted from the full frame
around the projected aircraft position, scaled to a 240×150 display cell.

The transforms span three groups:
  * Color-channel transforms — exploit sky/ice scattering physics
      BGR, NRBR, HSV-Saturation(inv), LAB-L*, LAB-b*, Grey-excess
  * Spatial / structural transforms — cloud-background suppression
      Local-contrast, DoG(2-15), White-tophat, CLAHE
  * Orientation-aware
      Cross-path-gradient, Temporal-diff

Output
------
  output/validation/channel_transforms/<date>/channel_matrix.png
  output/validation/channel_transforms/<date>/summary.json

Usage::

    uv run python scripts/channel_matrix_spotcheck.py --date 2026-04-08
    uv run python scripts/channel_matrix_spotcheck.py --date 2026-04-08 \\
        --video /path/to/video.mp4  # override auto-resolved path
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    sys.exit("matplotlib is required: uv pip install matplotlib")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import load_config
from concam.pipeline import resolve_video_path

DEFAULT_CONFIG = REPO_ROOT / "configs" / "mit_green_building.yaml"

# Context patch extracted from the full frame (pixels)
PATCH_W = 800
PATCH_H = 500

# Display cell size per transform×episode cell
CELL_W = 240
CELL_H = 150

# Left label column width and top label row height (px)
LABEL_W = 180
LABEL_H = 60


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------

def _decode_frames(
    video_path: Path,
    frame_indices: list[int],
) -> dict[int, np.ndarray]:
    """Seek-and-decode each requested frame index. Returns {idx: BGR array}."""
    if not frame_indices:
        return {}
    out: dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        duration_s = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else 0.0
        )
        total = int(stream.frames) if stream.frames else max(1, int(
            round(duration_s * float(stream.average_rate or 30))
        ))
        for target_idx in sorted(frame_indices):
            target_s = (target_idx / total) * duration_s
            target_pts = int(target_s / time_base)
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            decoded = None
            for frame in container.decode(stream):
                decoded = frame
                if frame.pts is not None and frame.pts >= target_pts:
                    break
            if decoded is not None:
                out[target_idx] = decoded.to_ndarray(format="bgr24")
    finally:
        container.close()
    return out


# ---------------------------------------------------------------------------
# Context patch extraction
# ---------------------------------------------------------------------------

def _extract_patch(
    frame: np.ndarray,
    cx: float,
    cy: float,
    patch_w: int = PATCH_W,
    patch_h: int = PATCH_H,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Extract an (patch_h × patch_w) crop centred on (cx, cy), clamped to frame.

    Returns (patch, (x0, y0)) where (x0, y0) is the top-left corner in frame
    coords.
    """
    fh, fw = frame.shape[:2]
    x0 = max(0, min(int(cx - patch_w // 2), fw - patch_w))
    y0 = max(0, min(int(cy - patch_h // 2), fh - patch_h))
    x1 = x0 + patch_w
    y1 = y0 + patch_h
    patch = frame[y0:y1, x0:x1].copy()
    # Pad if near edge
    if patch.shape[:2] != (patch_h, patch_w):
        pad = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
        ph, pw = patch.shape[:2]
        pad[:ph, :pw] = patch
        patch = pad
    return patch, (x0, y0)


# ---------------------------------------------------------------------------
# Individual transforms — all operate on a BGR patch, return BGR uint8 image
# ---------------------------------------------------------------------------

def _apply_colormap(arr: np.ndarray, vmin: float, vmax: float, cmap: str) -> np.ndarray:
    """Map float array → BGR uint8 via matplotlib colormap."""
    normed = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
    mapper = cm.get_cmap(cmap)
    rgba = (mapper(normed) * 255).astype(np.uint8)  # (H, W, 4) RGBA
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


def tf_bgr(bgr: np.ndarray, **_) -> np.ndarray:
    return bgr.copy()


def tf_nrbr(bgr: np.ndarray, **_) -> np.ndarray:
    """(R-B)/(R+B) — Rayleigh scattering: sky → negative (blue), contrail/cloud → 0 (white)."""
    f = bgr.astype(np.float32)
    r, b = f[:, :, 2], f[:, :, 0]
    nrbr = (r - b) / (r + b + 1e-6)
    return _apply_colormap(nrbr, -0.5, 0.5, "RdBu_r")


def tf_hsv_sat_inv(bgr: np.ndarray, **_) -> np.ndarray:
    """Inverted HSV saturation: white/grey objects (low saturation) appear bright.
    Blue sky has moderate saturation; contrails and clouds are near-white = bright here."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_inv = 255 - hsv[:, :, 1]
    return cv2.cvtColor(s_inv, cv2.COLOR_GRAY2BGR)


def tf_lab_L(bgr: np.ndarray, **_) -> np.ndarray:
    """CIELAB L* — perceptual luminance.  High L* = bright (clouds, contrails)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return cv2.cvtColor(lab[:, :, 0], cv2.COLOR_GRAY2BGR)


def tf_lab_b(bgr: np.ndarray, **_) -> np.ndarray:
    """CIELAB b* — yellow-blue axis: sky = very negative (blue), contrail/cloud ≈ 0.
    Perceptually uniform analog of NRBR — less sensitive to exposure variation."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b_star = lab[:, :, 2].astype(np.float32) - 128.0  # shift to 0-centred
    return _apply_colormap(b_star, -80, 80, "RdBu_r")


def tf_grey_excess(bgr: np.ndarray, **_) -> np.ndarray:
    """Grey-excess = min(R,G,B) / (max(R,G,B)+1).  High = neutral grey/white
    (contrails, clouds); low = saturated (blue sky, vegetation)."""
    f = bgr.astype(np.float32)
    grey_ex = np.min(f, axis=2) / (np.max(f, axis=2) + 1.0)
    return _apply_colormap(grey_ex, 0.0, 1.0, "inferno")


def tf_local_contrast(bgr: np.ndarray, **_) -> np.ndarray:
    """Subtract large Gaussian blur (σ=25): removes slow cloud background,
    contrail appears as local bright streak regardless of cloud brightness."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    lc = gray - bg
    return _apply_colormap(lc, -40, 40, "RdBu_r")


def tf_dog(bgr: np.ndarray, **_) -> np.ndarray:
    """Difference of Gaussians σ=(2, 15): band-pass tuned to contrail cross-section
    width (~5-15 px in the 4K image). Suppresses both low-frequency cloud gradients
    and high-frequency pixel noise."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_fine = cv2.GaussianBlur(gray, (0, 0), 2)
    g_coarse = cv2.GaussianBlur(gray, (0, 0), 15)
    dog = g_fine - g_coarse
    return _apply_colormap(dog, -40, 40, "RdBu_r")


def tf_tophat(bgr: np.ndarray, **_) -> np.ndarray:
    """Morphological white-tophat with an elongated elliptical kernel (80×12 px):
    highlights bright, elongated features smaller than the kernel.  Contrails
    appear as bright stripes even against a bright cloud background."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Elongated kernel oriented horizontally — we rotate it below based on path_vec
    # but use a horizontal default here.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 12))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def tf_tophat_oriented(bgr: np.ndarray, path_vec: tuple[float, float] | None = None, **_) -> np.ndarray:
    """Morphological white-tophat with a kernel rotated to the flight-path direction.
    More discriminating than the horizontal version when the contrail is diagonal."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if path_vec is not None and (path_vec[0] ** 2 + path_vec[1] ** 2) > 0.01:
        angle_deg = np.degrees(np.arctan2(float(path_vec[1]), float(path_vec[0])))
    else:
        angle_deg = 0.0
    # Build a thin line kernel, rotate it
    klen, kw = 81, 5
    base = np.zeros((klen, klen), dtype=np.uint8)
    cv2.line(base, (0, klen // 2), (klen - 1, klen // 2), 1, kw)
    M = cv2.getRotationMatrix2D((klen // 2, klen // 2), angle_deg, 1)
    kernel_rot = cv2.warpAffine(base, M, (klen, klen))
    kernel_rot = (kernel_rot > 0).astype(np.uint8)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_rot)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def tf_clahe(bgr: np.ndarray, **_) -> np.ndarray:
    """CLAHE on the L* channel of LAB: locally equalises contrast without the
    colour shift of straight histogram equalisation.  Reveals faint contrails
    embedded in a high-dynamic-range scene (bright cloud + dark sky)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def tf_cross_grad(bgr: np.ndarray, path_vec: tuple[float, float] | None = None, **_) -> np.ndarray:
    """Gradient magnitude in the direction perpendicular to the flight path.
    A contrail is brightest at its centre and fades to background; the
    cross-track gradient fires on both contrail edges while ignoring
    along-track gradients (which correspond to the cloud boundary)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    if path_vec is not None and (path_vec[0] ** 2 + path_vec[1] ** 2) > 0.01:
        px, py = float(path_vec[0]), float(path_vec[1])
        # Perpendicular = (-py, px)
        perp_x, perp_y = -py, px
    else:
        perp_x, perp_y = 0.0, 1.0
    cross = np.abs(sobel_x * perp_x + sobel_y * perp_y)
    return _apply_colormap(cross, 0, 60, "hot")


def tf_temporal_diff(
    bgr: np.ndarray,
    prev_bgr: np.ndarray | None = None,
    **_,
) -> np.ndarray:
    """Absolute difference from the previous frame.  Static cloud and sky
    background cancel; a newly formed contrail stripe remains.  Falls back
    to a uniform grey image when no previous frame is available."""
    if prev_bgr is None or prev_bgr.shape != bgr.shape:
        return np.full_like(bgr, 128)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(gray - prev_gray)
    return _apply_colormap(diff, 0, 60, "hot")


# Registry: (label, function, needs_path_vec, needs_prev_frame)
TRANSFORMS: list[tuple[str, Any, bool, bool]] = [
    ("BGR (ref)", tf_bgr, False, False),
    ("NRBR\n(R-B)/(R+B)", tf_nrbr, False, False),
    ("HSV Sat⁻¹\n(white=bright)", tf_hsv_sat_inv, False, False),
    ("LAB-L*\n(luminance)", tf_lab_L, False, False),
    ("LAB-b*\n(blue-yellow axis)", tf_lab_b, False, False),
    ("Grey-excess\nmin/max", tf_grey_excess, False, False),
    ("Local-contrast\ngray - blur(σ25)", tf_local_contrast, False, False),
    ("DoG σ(2-15)\nband-pass", tf_dog, False, False),
    ("White-tophat\nhoriz.kernel 80×12", tf_tophat, False, False),
    ("Tophat-oriented\nalong flight path", tf_tophat_oriented, True, False),
    ("CLAHE\nlocal equalisation", tf_clahe, False, False),
    ("Cross-path grad\n⊥ to flight dir", tf_cross_grad, True, False),
    ("Temporal-diff\n|frame - prev|", tf_temporal_diff, False, True),
]


# ---------------------------------------------------------------------------
# Grid rendering helpers
# ---------------------------------------------------------------------------

def _cell_label_img(text: str, width: int, height: int, font_scale: float = 0.45) -> np.ndarray:
    img = np.full((height, width, 3), 50, dtype=np.uint8)
    lines = text.split("\n")
    total_h = len(lines) * 18
    y0 = max(10, (height - total_h) // 2 + 12)
    for i, line in enumerate(lines):
        y = y0 + i * 18
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x = max(4, (width - tw) // 2)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return img


def _col_header(text: str, width: int, height: int, score: float) -> np.ndarray:
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    lines = text.split("\n")
    y0 = 16
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        x = max(2, (width - tw) // 2)
        cv2.putText(img, line, (x, y0 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
    # Score bar at bottom
    bar_w = int(width * min(1.0, score))
    color = (0, 200, 0) if score >= 0.5 else (0, 160, 255)
    cv2.rectangle(img, (0, height - 6), (bar_w, height), color, -1)
    return img


def _draw_crosshair(img: np.ndarray, cx: int, cy: int, r: int = 12) -> None:
    """Draw a small crosshair on img in-place."""
    h, w = img.shape[:2]
    color = (0, 255, 255)
    if 0 <= cy < h and 0 <= cx < w:
        cv2.line(img, (max(0, cx - r), cy), (min(w - 1, cx + r), cy), color, 1)
        cv2.line(img, (cx, max(0, cy - r)), (cx, min(h - 1, cy + r)), color, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2026-04-08",
                    help="date in YYYY-MM-DD (used to resolve pipeline artefact paths)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--output-dir", default="output/validation/channel_transforms",
                    help="root dir for outputs; a <date>/ subdirectory is created")
    ap.add_argument("--video", default=None,
                    help="override video path (auto-resolved from config if omitted)")
    ap.add_argument("--patch-w", type=int, default=PATCH_W,
                    help="context patch width in source-frame pixels")
    ap.add_argument("--patch-h", type=int, default=PATCH_H,
                    help="context patch height in source-frame pixels")
    ap.add_argument("--cell-w", type=int, default=CELL_W,
                    help="display cell width in output pixels")
    ap.add_argument("--cell-h", type=int, default=CELL_H,
                    help="display cell height in output pixels")
    args = ap.parse_args()

    date = datetime.date.fromisoformat(args.date)
    config = load_config(args.config)

    out_dir = Path(args.output_dir) / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_dir = REPO_ROOT / "output" / args.date
    metrics_path = REPO_ROOT / "output" / "validation" / "regression" / args.date / "metrics.json"
    projections_path = pipeline_dir / "projections.jsonl"

    if not metrics_path.exists():
        sys.exit(f"metrics.json not found at {metrics_path}. Run regression_e2e.py first.")
    if not projections_path.exists():
        sys.exit(f"projections.jsonl not found at {projections_path}. Run the pipeline first.")

    # --- Load episode metadata from metrics.json ---
    metrics = json.load(open(metrics_path))
    panels = metrics["spot_check"]["panels"]
    print(f"Found {len(panels)} spot-check panels.")

    # --- Build projection lookup: (transponder_id, wall_time_second) -> record ---
    proj_lookup: dict[tuple[str, str], dict] = {}
    with open(projections_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["transponder_id"], r["wall_time_utc"][:19])
            proj_lookup[key] = r

    # --- Resolve video ---
    if args.video:
        video_path = Path(args.video)
    else:
        video_path = resolve_video_path(config.video, date)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    print(f"Video: {video_path}")

    # Collect unique frame indices needed (peak + prev for temporal diff)
    frame_indices_needed: set[int] = set()
    for panel in panels:
        fidx = panel["frame_idx"]
        frame_indices_needed.add(fidx)
        if fidx > 0:
            frame_indices_needed.add(fidx - 1)

    print(f"Decoding {len(frame_indices_needed)} frames …")
    frames = _decode_frames(video_path, sorted(frame_indices_needed))
    print(f"Decoded {len(frames)} frames.")

    # --- Render grid ---
    n_rows = len(TRANSFORMS)
    n_cols = len(panels)
    cw, ch = args.cell_w, args.cell_h
    lw, lh = LABEL_W, LABEL_H

    # Total canvas size
    total_w = lw + n_cols * cw
    total_h = lh + n_rows * ch
    canvas = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

    # Column headers
    for col_idx, panel in enumerate(panels):
        x0 = lw + col_idx * cw
        label_text = f"{panel['callsign']}\nscore={panel['peak_score']:.3f}"
        hdr = _col_header(label_text, cw, lh, panel["peak_score"])
        canvas[:lh, x0:x0 + cw] = hdr

    # Row labels
    for row_idx, (name, fn, needs_pv, needs_prev) in enumerate(TRANSFORMS):
        y0 = lh + row_idx * ch
        lbl = _cell_label_img(name, lw, ch)
        canvas[y0:y0 + ch, :lw] = lbl

    # Cells
    n_cells = n_rows * n_cols
    done = 0
    for col_idx, panel in enumerate(panels):
        fidx = panel["frame_idx"]
        bgr_frame = frames.get(fidx)
        if bgr_frame is None:
            print(f"  SKIP {panel['callsign']} frame {fidx}: not decoded")
            continue

        prev_bgr = frames.get(fidx - 1)

        # Look up projection for path_vec
        key = (panel["transponder_id"], panel["peak_wall_time"][:19])
        proj = proj_lookup.get(key)
        px = float(proj["pixel_x"]) if proj else bgr_frame.shape[1] / 2
        py = float(proj["pixel_y"]) if proj else bgr_frame.shape[0] / 2
        path_vec = (float(proj["path_dx"]), float(proj["path_dy"])) if proj else None

        # Extract context patches from full frame (current + prev)
        patch, (patch_x0, patch_y0) = _extract_patch(
            bgr_frame, px, py, args.patch_w, args.patch_h
        )
        prev_patch: np.ndarray | None = None
        if prev_bgr is not None:
            prev_patch, _ = _extract_patch(
                prev_bgr, px, py, args.patch_w, args.patch_h
            )

        # Aircraft crosshair position in patch coords
        cross_cx = int(px) - patch_x0
        cross_cy = int(py) - patch_y0

        for row_idx, (name, fn, needs_pv, needs_prev) in enumerate(TRANSFORMS):
            # Build kwargs
            kwargs: dict = {}
            if needs_pv:
                kwargs["path_vec"] = path_vec
            if needs_prev:
                kwargs["prev_bgr"] = prev_patch

            transformed = fn(patch, **kwargs)

            # Resize to display cell
            cell = cv2.resize(transformed, (cw, ch), interpolation=cv2.INTER_AREA)

            # Draw scaled crosshair on the cell
            cell_cx = int(cross_cx * cw / args.patch_w)
            cell_cy = int(cross_cy * ch / args.patch_h)
            _draw_crosshair(cell, cell_cx, cell_cy)

            # Place in canvas
            x0 = lw + col_idx * cw
            y0 = lh + row_idx * ch
            canvas[y0:y0 + ch, x0:x0 + cw] = cell

            done += 1
            if done % 20 == 0:
                print(f"  {done}/{n_cells} cells done")

    # Grid lines
    for col_idx in range(n_cols + 1):
        x = lw + col_idx * cw
        cv2.line(canvas, (x, 0), (x, total_h), (80, 80, 80), 1)
    for row_idx in range(n_rows + 1):
        y = lh + row_idx * ch
        cv2.line(canvas, (0, y), (total_w, y), (80, 80, 80), 1)

    # Title bar
    title = f"Channel-transform matrix — {args.date} — {n_cols} episodes × {n_rows} transforms"
    cv2.putText(canvas, title, (lw + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Explanatory legend strip at the bottom ---
    legend_h = 110
    legend = np.full((legend_h, total_w, 3), 25, dtype=np.uint8)
    legend_lines = [
        "COLOR-CHANNEL transforms (rows 1-6):  exploit sky/ice scattering physics",
        "  NRBR=(R-B)/(R+B): sky→blue, contrail→white  |  HSV-Sat-inv: white objects appear bright",
        "  LAB-b*: perceptually-uniform blue-yellow axis  |  Grey-excess: saturated color→dark",
        "SPATIAL transforms (rows 7-12):  cloud-background suppression",
        "  Local-contrast: subtracts slow cloud blob  |  DoG: band-pass for contrail cross-section width",
        "  Tophat: bright streaks on any background  |  Tophat-oriented: rotated to flight direction",
        "  CLAHE: local equalisation  |  Cross-grad: edges ⊥ to flight path  |  Temporal-diff: static bg cancels",
    ]
    for i, line in enumerate(legend_lines):
        cv2.putText(legend, line, (10, 16 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    canvas_with_legend = np.vstack([canvas, legend])

    out_path = out_dir / "channel_matrix.png"
    cv2.imwrite(str(out_path), canvas_with_legend)
    print(f"\nSaved: {out_path}  ({canvas_with_legend.shape[1]}×{canvas_with_legend.shape[0]} px)")

    # --- Per-episode 2-column comparison (bigger cells) for close inspection ---
    # For each episode: one row per transform, 2 cols: BGR ref | transform
    _render_per_episode_sheets(
        panels, frames, proj_lookup, out_dir, args
    )

    # --- Summary JSON ---
    summary = {
        "date": args.date,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "n_transforms": n_rows,
        "n_episodes": n_cols,
        "transforms": [t[0].replace("\n", " ") for t in TRANSFORMS],
        "episodes": [
            {
                "callsign": p["callsign"],
                "label": p["label"],
                "peak_score": p["peak_score"],
                "frame_idx": p["frame_idx"],
            }
            for p in panels
        ],
        "output": str(out_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")


def _render_per_episode_sheets(
    panels: list[dict],
    frames: dict[int, np.ndarray],
    proj_lookup: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Render one tall PNG per episode showing all transforms side-by-side."""
    BIG_W = 480
    BIG_H = 300

    for panel in panels:
        fidx = panel["frame_idx"]
        bgr_frame = frames.get(fidx)
        if bgr_frame is None:
            continue
        prev_bgr = frames.get(fidx - 1)

        key = (panel["transponder_id"], panel["peak_wall_time"][:19])
        proj = proj_lookup.get(key)
        px = float(proj["pixel_x"]) if proj else bgr_frame.shape[1] / 2
        py = float(proj["pixel_y"]) if proj else bgr_frame.shape[0] / 2
        path_vec = (float(proj["path_dx"]), float(proj["path_dy"])) if proj else None

        patch, (patch_x0, patch_y0) = _extract_patch(
            bgr_frame, px, py, args.patch_w, args.patch_h
        )
        prev_patch: np.ndarray | None = None
        if prev_bgr is not None:
            prev_patch, _ = _extract_patch(prev_bgr, px, py, args.patch_w, args.patch_h)

        cross_cx = int(px) - patch_x0
        cross_cy = int(py) - patch_y0

        n_rows = len(TRANSFORMS)
        sheet_w = LABEL_W + 2 * BIG_W  # label | BGR ref | transform
        sheet_h = 50 + n_rows * BIG_H
        sheet = np.full((sheet_h, sheet_w, 3), 30, dtype=np.uint8)

        # Header
        hdr_text = (
            f"{panel['label'].upper()} #{panel['index']}  {panel['callsign']}  "
            f"score={panel['peak_score']:.3f}  frame={fidx}  "
            f"t={panel['peak_wall_time'][11:19]}"
        )
        cv2.putText(sheet, hdr_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(sheet, "BGR (reference)", (LABEL_W + BIG_W // 2 - 60, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(sheet, "Transform", (LABEL_W + BIG_W + BIG_W // 2 - 40, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        bgr_cell = cv2.resize(patch, (BIG_W, BIG_H), interpolation=cv2.INTER_AREA)
        _draw_crosshair(bgr_cell,
                        int(cross_cx * BIG_W / args.patch_w),
                        int(cross_cy * BIG_H / args.patch_h))

        for row_idx, (name, fn, needs_pv, needs_prev) in enumerate(TRANSFORMS):
            y0 = 50 + row_idx * BIG_H

            # Row label
            lbl = _cell_label_img(name, LABEL_W, BIG_H, font_scale=0.5)
            sheet[y0:y0 + BIG_H, :LABEL_W] = lbl

            # BGR reference
            sheet[y0:y0 + BIG_H, LABEL_W:LABEL_W + BIG_W] = bgr_cell

            # Transform
            kwargs: dict = {}
            if needs_pv:
                kwargs["path_vec"] = path_vec
            if needs_prev:
                kwargs["prev_bgr"] = prev_patch
            transformed = fn(patch, **kwargs)
            t_cell = cv2.resize(transformed, (BIG_W, BIG_H), interpolation=cv2.INTER_AREA)
            _draw_crosshair(t_cell,
                            int(cross_cx * BIG_W / args.patch_w),
                            int(cross_cy * BIG_H / args.patch_h))
            sheet[y0:y0 + BIG_H, LABEL_W + BIG_W:LABEL_W + 2 * BIG_W] = t_cell

            # Grid line
            cv2.line(sheet, (0, y0), (sheet_w, y0), (70, 70, 70), 1)

        slug = f"{panel['label']}_{panel['index']:02d}_{panel['callsign']}"
        ep_path = out_dir / f"{slug}_transforms.png"
        cv2.imwrite(str(ep_path), sheet)

    print(f"Per-episode sheets saved to {out_dir}/")


if __name__ == "__main__":
    main()
