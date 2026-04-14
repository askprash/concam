"""NRBR (normalized red-blue ratio) visual spot-check.

For each episode the regression script picked (top-5 + 5-near-threshold),
decode the peak frame from the video and render a 2x2 comparison panel:

    (original BGR crop | NRBR colormap crop)
    (BGR + detection overlay | NRBR + detection overlay)

The detector is run on **both** inputs with identical config so the overlays
are directly comparable. For NRBR we remap the ratio to uint8 grayscale and
tile to 3 channels so the detector's existing BGR code path works without
modification.

NRBR = (R - B) / (R + B + eps). Rayleigh scattering makes clear sky
blue-dominant (NRBR very negative), while the broadband scattering off
ice / water particles in contrails and clouds gives NRBR near 0. The
sky→contrail gradient in NRBR is often cleaner than in raw BGR.

Usage::

    uv run python scripts/nrbr_spotcheck.py --date 2026-04-08
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import load_config
from concam.detection import detect
from concam.pipeline import resolve_video_path
from concam.projection import PixelPoint, Rect, rotated_polygon

DEFAULT_CONFIG = REPO_ROOT / "configs" / "mit_green_building.yaml"

# NRBR display / detection remap range. Clear sky at this camera sits around
# NRBR ≈ -0.3; contrails ≈ 0. A symmetric ±0.5 window gives good contrast
# without clipping interesting structure.
NRBR_DISPLAY_LO = -0.5
NRBR_DISPLAY_HI = 0.5


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _decode_single_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
        total = int(stream.frames) if stream.frames else int(
            round(duration_s * float(stream.average_rate or 30))
        )
        if total == 0:
            return None
        target_time_s = (frame_idx / total) * duration_s
        target_pts = int(target_time_s / float(time_base))
        container.seek(target_pts, stream=stream, any_frame=False, backward=True)
        decoded = None
        for frame in container.decode(stream):
            decoded = frame
            if frame.pts is not None and frame.pts >= target_pts:
                break
        if decoded is None:
            return None
        return decoded.to_ndarray(format="bgr24")
    finally:
        container.close()


def compute_nrbr(frame_bgr: np.ndarray) -> np.ndarray:
    """Return NRBR as float32 in ≈[-1, 1] with the same (H, W) shape as input."""
    frame = frame_bgr.astype(np.float32)
    b = frame[:, :, 0]
    r = frame[:, :, 2]
    denom = r + b + 1e-6
    return (r - b) / denom


def nrbr_to_detector_input(nrbr: np.ndarray) -> np.ndarray:
    """Remap NRBR [-0.5, 0.5] → uint8 BGR so the existing detector works unchanged.

    Contrails (NRBR ≈ 0) become bright pixels (~127–180); clear sky (NRBR ≈ -0.3)
    becomes dark (~50–80). Canny then picks up the gradient at the contrail
    boundary the same way it does in raw BGR.
    """
    scaled = np.clip(
        (nrbr - NRBR_DISPLAY_LO) / (NRBR_DISPLAY_HI - NRBR_DISPLAY_LO),
        0.0, 1.0,
    )
    gray = (scaled * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _draw_overlay(
    rgb_crop: np.ndarray, crop_origin: tuple[int, int],
    polygon: np.ndarray | None, detection_line: tuple[float, ...] | None,
) -> np.ndarray:
    out = rgb_crop.copy()
    x0, y0 = crop_origin
    if polygon is not None:
        pts = polygon.copy()
        pts[:, 0] -= x0
        pts[:, 1] -= y0
        cv2.polylines(out, [pts.astype(np.int32).reshape(-1, 1, 2)],
                      isClosed=True, color=(255, 200, 0), thickness=2)
    if detection_line is not None:
        px1, py1, px2, py2 = detection_line
        cv2.line(out, (int(px1) - x0, int(py1) - y0),
                 (int(px2) - x0, int(py2) - y0),
                 (0, 255, 0), thickness=3)
    return out


def _render_panel(
    episode_meta: dict,
    bgr_frame: np.ndarray,
    nrbr: np.ndarray,
    projection: dict | None,
    det_config,
    out_path: Path,
) -> dict:
    h, w = bgr_frame.shape[:2]

    if projection is None:
        return {"skipped": True, "reason": "no projection"}

    cx = float(projection["pixel_x"])
    cy = float(projection["pixel_y"])
    path_vec = (float(projection["path_dx"]), float(projection["path_dy"]))
    polygon = rotated_polygon(PixelPoint(x=cx, y=cy), path_vec, det_config)

    roi = Rect(
        x=projection["roi"]["x"], y=projection["roi"]["y"],
        w=projection["roi"]["w"], h=projection["roi"]["h"],
    )

    # Re-run detect() on BGR (for parity with stored detection) and on NRBR.
    # prev_frame=None on both sides so the comparison isolates the input
    # representation rather than the temporal-diff pre-processing.
    bgr_result = detect(bgr_frame, roi, det_config,
                        polygon=polygon, path_vec=path_vec, prev_frame=None)
    nrbr_det_input = nrbr_to_detector_input(nrbr)
    nrbr_result = detect(nrbr_det_input, roi, det_config,
                         polygon=polygon, path_vec=path_vec, prev_frame=None)

    pad = 250
    x0 = max(0, int(cx - pad))
    y0 = max(0, int(cy - pad))
    x1 = min(w, int(cx + pad))
    y1 = min(h, int(cy + pad))
    bgr_crop = bgr_frame[y0:y1, x0:x1]
    nrbr_crop = nrbr[y0:y1, x0:x1]

    bgr_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    bgr_overlay = _draw_overlay(bgr_rgb, (x0, y0), polygon, bgr_result.pixel_line)

    # For the NRBR display we render the raw float through matplotlib's RdBu_r
    # colormap so negative (blue-sky) is blue, positive (contrail / cloud / sunlit)
    # is red. That matches scientific convention and makes the contrail "glow".
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(bgr_rgb)
    axes[0, 0].set_title(
        f"BGR — {episode_meta['callsign']} peak={episode_meta['peak_score']:.3f}\n"
        f"{episode_meta['label']} #{episode_meta['index']}"
    )
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(nrbr_crop, cmap="RdBu_r",
                           vmin=NRBR_DISPLAY_LO, vmax=NRBR_DISPLAY_HI)
    axes[0, 1].set_title(f"NRBR  [vmin={NRBR_DISPLAY_LO}, vmax={NRBR_DISPLAY_HI}]")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(bgr_overlay)
    axes[1, 0].set_title(
        f"BGR detect: score={bgr_result.score:.3f} "
        f"(aligned={bgr_result.aligned_lines}, long={bgr_result.num_long_lines})"
    )
    axes[1, 0].axis("off")

    # For the NRBR overlay panel, show the NRBR uint8 conversion that the
    # detector actually saw — not the RdBu_r colormap — so the overlays line
    # up on the exact pixel values Canny operated on.
    nrbr_det_crop_gray = nrbr_to_detector_input(nrbr)[y0:y1, x0:x1]
    nrbr_det_rgb = cv2.cvtColor(nrbr_det_crop_gray, cv2.COLOR_BGR2RGB)
    nrbr_overlay = _draw_overlay(nrbr_det_rgb, (x0, y0), polygon,
                                 nrbr_result.pixel_line)
    axes[1, 1].imshow(nrbr_overlay)
    axes[1, 1].set_title(
        f"NRBR detect: score={nrbr_result.score:.3f} "
        f"(aligned={nrbr_result.aligned_lines}, long={nrbr_result.num_long_lines})"
    )
    axes[1, 1].axis("off")

    fig.suptitle(
        f"{episode_meta['callsign']} @ {episode_meta['peak_wall_time']} "
        f"(frame {episode_meta['frame_idx']})",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return {
        "panel": str(out_path),
        "bgr_score": bgr_result.score,
        "nrbr_score": nrbr_result.score,
        "bgr_aligned": bgr_result.aligned_lines,
        "nrbr_aligned": nrbr_result.aligned_lines,
        "bgr_long": bgr_result.num_long_lines,
        "nrbr_long": nrbr_result.num_long_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NRBR spot-check panels.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--regression-dir", default="output/validation/regression")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    date = datetime.date.fromisoformat(args.date)
    site_config = load_config(args.config)
    source_dir = Path(args.output_dir) / args.date
    reg_dir = Path(args.regression_dir) / args.date
    panels_dir = reg_dir / "nrbr_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((reg_dir / "metrics.json").read_text())
    panel_records = metrics["spot_check"]["panels"]

    video_path = Path(args.video) if args.video else resolve_video_path(
        site_config.video, date
    )

    # Index projections by (callsign, transponder_id, wall_time_utc) for quick lookup.
    projs = {}
    for p in _iter_jsonl(source_dir / "projections.jsonl"):
        key = (p["callsign"], p["transponder_id"], p["wall_time_utc"])
        projs[key] = p

    # Build per-episode lookup from the earlier regression metadata.
    episodes_by_wall = {
        (r["callsign"], r["transponder_id"], r["peak_wall_time"]): r
        for r in panel_records
    }

    summary_rows = []
    for rec in panel_records:
        frame = _decode_single_frame(video_path, rec["frame_idx"])
        if frame is None:
            print(f"[nrbr] decode failed for frame {rec['frame_idx']}")
            continue
        nrbr = compute_nrbr(frame)
        proj = projs.get((rec["callsign"], rec["transponder_id"], rec["peak_wall_time"]))
        episode_meta = {
            "callsign": rec["callsign"],
            "peak_score": rec["peak_score"],
            "peak_wall_time": rec["peak_wall_time"],
            "frame_idx": rec["frame_idx"],
            "label": rec["label"],
            "index": rec["index"],
        }
        out_path = (panels_dir /
                    f"{rec['label']}_{rec['index']:02d}_{rec['callsign']}_"
                    f"{rec['frame_idx']}.png")
        result = _render_panel(
            episode_meta, frame, nrbr, proj, site_config.detection, out_path
        )
        if result.get("skipped"):
            print(f"[nrbr] {out_path.name}: {result['reason']}")
            continue
        summary_rows.append({**episode_meta, **result})
        delta = result["nrbr_score"] - result["bgr_score"]
        print(f"[nrbr] {out_path.name}  bgr={result['bgr_score']:.3f} "
              f"nrbr={result['nrbr_score']:.3f}  delta={delta:+.3f}")

    summary_path = panels_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, default=str))
    print(f"[nrbr] wrote {len(summary_rows)} panels + summary -> {panels_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
