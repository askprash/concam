#!/usr/bin/env python3
"""Build the static-scene mask (buildings) from a daily timelapse video.

Samples frames spread across the daylight window, accumulates Canny edge
persistence (see concam.detection.static_mask), and writes:

  * the boolean mask npz (point ``DetectionConfig.static_mask_path`` at it),
  * an overlay PNG (sample frame with the mask hatched red) for human review.

Persistence separates structure from sky: building edges sit at the same
pixels in every sample; clouds and contrails do not repeat positions across
samples taken ~15 min apart.

Usage:
    uv run python scripts/build_static_mask.py \\
        --video /net/d16/data/contrail-camera/2026_06_08_0000_2359.mp4 \\
        --out configs/static_mask_mit_green_building.npz \\
        --overlay output/static_mask_overlay.png

Multiple --video arguments may be given (recommended: a clear day + an
overcast day) — persistence is accumulated over all sampled frames, so only
structure visible on every day survives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from concam.detection.static_mask import (
    compute_static_mask,
    mask_to_polygons,
    save_static_mask,
    svg_to_mask,
)
from concam.video import decode_frames

# Daily timelapse: 1 frame per real second, 86400 frames nominal.
FRAMES_PER_DAY = 86_400


def sample_indices(n_samples: int, start_hour: float, end_hour: float) -> list[int]:
    """Evenly spaced frame indices inside [start_hour, end_hour) UTC."""
    lo = int(start_hour * 3600)
    hi = int(end_hour * 3600)
    return [int(round(i)) for i in np.linspace(lo, hi - 1, n_samples)]


def render_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sample frame with the masked region tinted + hatched red for review."""
    out = frame.copy()
    tint = out.copy()
    tint[mask] = (0.45 * tint[mask] + 0.55 * np.array([40, 40, 220])).astype(np.uint8)
    out = tint
    # Diagonal hatch lines clipped to the mask, every 40 px.
    hatch = np.zeros(mask.shape, dtype=np.uint8)
    h, w = mask.shape
    for c in range(-h, w, 40):
        cv2.line(hatch, (c, 0), (c + h, h), 255, 2)
    hatched = (hatch > 0) & mask
    out[hatched] = (60, 60, 255)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, action="append", required=True,
                    help="daily timelapse mp4 (repeatable)")
    ap.add_argument("--out", type=Path, required=True, help="output mask npz")
    ap.add_argument("--overlay", type=Path, default=None,
                    help="optional review overlay PNG")
    ap.add_argument("--polygons-out", type=Path, default=None,
                    help="optional JSON file with mask polygons (manifest format)")
    ap.add_argument("--samples-per-video", type=int, default=40)
    # Daylight window (UTC). 12:00–22:00 UTC ≈ 08:00–18:00 Boston: building
    # edges are lit and sky features move; night frames would dilute
    # persistence (unlit facade) without adding information.
    ap.add_argument("--start-hour", type=float, default=12.0)
    ap.add_argument("--end-hour", type=float, default=22.0)
    ap.add_argument("--persistence-threshold", type=float, default=0.5)
    ap.add_argument("--dilate-px", type=int, default=12)
    ap.add_argument("--svg", type=Path, default=None,
                    help="hand-drawn SVG mask outline (straight-segment paths,"
                         " drawn over a screenshot at any resolution); union'd"
                         " with the persistent-edge mask. The manual outline"
                         " captures full building volumes (weak-edged glass"
                         " facades); edge persistence still catches thin"
                         " structures above it (cranes, antennas).")
    args = ap.parse_args()

    frames: list[np.ndarray] = []
    sample_frame = None
    for video in args.video:
        idx = sample_indices(args.samples_per_video, args.start_hour, args.end_hour)
        decoded = decode_frames(video, idx)
        print(f"[mask] {video}: decoded {len(decoded)}/{len(idx)} samples")
        for i in sorted(decoded):
            frames.append(decoded[i])
            if sample_frame is None:
                sample_frame = decoded[i]

    mask = compute_static_mask(
        frames,
        persistence_threshold=args.persistence_threshold,
        dilate_px=args.dilate_px,
    )
    if args.svg is not None:
        manual = svg_to_mask(args.svg.read_text(), mask.shape)
        print(f"[mask] manual SVG outline covers {100.0 * manual.mean():.2f}%; "
              f"union with persistent-edge mask")
        mask = mask | manual
    coverage = 100.0 * mask.mean()
    print(f"[mask] static mask covers {coverage:.2f}% of the frame "
          f"({mask.sum()} px of {mask.size})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_static_mask(mask, args.out)
    print(f"[mask] wrote {args.out}")

    if args.polygons_out is not None:
        import json
        polys = mask_to_polygons(mask)
        args.polygons_out.write_text(json.dumps({"polygons": polys}))
        print(f"[mask] wrote {len(polys)} polygons to {args.polygons_out}")

    if args.overlay is not None and sample_frame is not None:
        overlay = render_overlay(sample_frame, mask)
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.overlay), overlay)
        print(f"[mask] wrote review overlay {args.overlay}")


if __name__ == "__main__":
    main()
