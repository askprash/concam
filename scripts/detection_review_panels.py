"""Render per-candidate 4-panel detection review figures.

Takes an extraction manifest (from ``detection_validation_extract.py``), runs
``concam.detection.detect`` with the site config's tuned parameters on every
candidate, and saves a 4-panel diagnostic PNG per candidate showing:

  1. The crop with the rotated polygon + flight-path vector overlaid.
  2. The post-floor masked crop (what Canny actually sees) with the resolved
     Canny low/high annotated.
  3. The Canny edge map.
  4. The original crop with Hough lines overlaid (aligned=green, rejected=red,
     picked pixel_line thick green) and the live detector score in the title.

Also writes a score-sorted ``index.md`` summary and an ``index.png`` grid so
the full batch can be eyeballed in one shot.

Usage::

    uv run python scripts/detection_review_panels.py \\
        --manifest output/validation/detection/2026-04-08/manifest.json \\
        --labels output/validation/detection/2026-04-08/labels.json \\
        --out-dir output/validation/detection/panels/2026-04-08

    uv run python scripts/detection_review_panels.py \\
        --manifest output/validation/detection/2026-04-08-batch2/manifest.json \\
        --out-dir output/validation/detection/panels/2026-04-08-batch2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np

from concam.config import DetectionConfig, load_config
from concam.detection import detect
from concam.projection import PixelPoint, Rect, rotated_polygon

EXTRACT_PAD = 20


def _decode_frames(
    video_path: Path, frame_indices: list[int],
    upscale_to: tuple[int, int] | None = None,
) -> dict[int, np.ndarray]:
    """Seek to each requested frame_idx and return a {frame_idx: BGR ndarray} map.

    Mirrors the seek/decode loop in ``detection_validation_extract.py`` so the
    prev_frame crops we feed into ``detect()`` are byte-identical to what the
    extraction would have produced for that same frame. If ``upscale_to`` is set
    the decoded frame is bilinearly upscaled to that ``(w, h)`` — used when the
    archive video is lower-res than the calibration (Oct 2025 = 720p, calibration
    = 4K).
    """
    out: dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
        total_frames = (
            int(stream.frames) if stream.frames
            else int(round(duration_s * float(stream.average_rate or 30)))
        )
        for target_idx in sorted(set(i for i in frame_indices if i >= 0)):
            target_time_s = (target_idx / total_frames) * duration_s if total_frames else 0.0
            target_pts = int(target_time_s / float(time_base))
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            decoded = None
            for frame in container.decode(stream):
                decoded = frame
                if frame.pts is not None and frame.pts >= target_pts:
                    break
            if decoded is not None:
                arr = decoded.to_ndarray(format="bgr24")
                if upscale_to is not None and (arr.shape[1], arr.shape[0]) != upscale_to:
                    arr = cv2.resize(arr, upscale_to, interpolation=cv2.INTER_LINEAR)
                out[target_idx] = arr
    finally:
        container.close()
    return out


def _crop_padded(frame: np.ndarray, roi: dict, pad: int = EXTRACT_PAD) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(roi["x"]) - pad)
    y1 = max(0, int(roi["y"]) - pad)
    x2 = min(w, int(roi["x"]) + int(roi["w"]) + pad)
    y2 = min(h, int(roi["y"]) + int(roi["h"]) + pad)
    return frame[y1:y2, x1:x2].copy()


def _reconstruct(cand: dict, crop_shape: tuple[int, int], cfg: DetectionConfig):
    ch, cw = crop_shape
    roi = cand["roi"]
    tlx = max(0, int(roi["x"]) - EXTRACT_PAD)
    tly = max(0, int(roi["y"]) - EXTRACT_PAD)
    center = PixelPoint(
        x=float(cand["pixel_x"]) - tlx,
        y=float(cand["pixel_y"]) - tly,
    )
    path_vec = (float(cand["path_dx"]), float(cand["path_dy"]))
    poly = rotated_polygon(center, path_vec, cfg)
    rect = Rect(x=0, y=0, w=cw, h=ch)
    return rect, poly, path_vec, center


def _compute_panels(
    crop: np.ndarray,
    poly: np.ndarray,
    cfg: DetectionConfig,
    prev_crop: np.ndarray | None = None,
):
    """Reproduce the detector's pre-Canny pipeline so we can render what it saw.

    If ``prev_crop`` is provided and shape-matches, we apply the same
    ``cv2.absdiff`` step ``concam.detection.detect`` does so the rendered
    Canny / Hough panels reflect the diff path the live pipeline runs.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    base = gray
    if prev_crop is not None:
        prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY) if prev_crop.ndim == 3 else prev_crop
        if prev_gray.shape == gray.shape:
            base = cv2.absdiff(gray, prev_gray)
    if cfg.blur_kernel and cfg.blur_kernel > 1:
        k = int(cfg.blur_kernel) | 1
        base = cv2.GaussianBlur(base, (k, k), 0)
    mask = None
    if cfg.use_rotated_mask:
        mask = np.zeros(base.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        masked_values = base[mask > 0]
    else:
        masked_values = base.reshape(-1)
    if cfg.use_adaptive_canny and masked_values.size:
        p_hi = float(np.percentile(masked_values, cfg.canny_percentile_high))
        p_lo = float(np.percentile(masked_values, cfg.canny_percentile_low))
        canny_high = max(int(round(p_hi)), int(cfg.canny_min_high))
        canny_low = max(1, int(round(canny_high * cfg.canny_low_ratio)))
        floor = int(round(p_lo))
    else:
        canny_high = int(cfg.canny_high)
        canny_low = int(cfg.canny_low)
        floor = 0
    crop_for_canny = base.copy()
    if mask is not None:
        crop_for_canny = cv2.bitwise_and(crop_for_canny, crop_for_canny, mask=mask)
    if floor > 0:
        _, crop_for_canny = cv2.threshold(crop_for_canny, floor, 255, cv2.THRESH_TOZERO)
    edges = cv2.Canny(crop_for_canny, canny_low, canny_high)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180.0,
        threshold=int(cfg.hough_threshold),
        minLineLength=int(cfg.hough_min_line_length),
        maxLineGap=int(cfg.hough_max_line_gap),
    )
    raw_lines = [] if raw is None else [tuple(int(v) for v in ln[0]) for ln in raw]
    return {
        "crop_for_canny": crop_for_canny,
        "edges": edges,
        "raw_lines": raw_lines,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "floor": floor,
    }


def _render_panel(
    cand: dict,
    crop: np.ndarray,
    cfg: DetectionConfig,
    label: str | None,
    out_path: Path,
    prev_crop: np.ndarray | None = None,
) -> dict:
    rect, poly, path_vec, _center = _reconstruct(cand, crop.shape[:2], cfg)
    panels = _compute_panels(crop, poly, cfg, prev_crop=prev_crop)
    result = detect(
        crop, rect, cfg,
        polygon=poly, path_vec=path_vec,
        prev_frame=prev_crop,
    )

    path_angle = math.degrees(math.atan2(path_vec[1], path_vec[0])) % 180.0
    tol = float(cfg.angle_tolerance_deg)

    def _aligned(x1, y1, x2, y2):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        return abs(((a - path_angle + 90.0) % 180.0) - 90.0) <= tol

    vis = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    poly_closed = np.vstack([poly, poly[:1]])
    cx = float(cand["pixel_x"]) - max(0, int(cand["roi"]["x"]) - EXTRACT_PAD)
    cy = float(cand["pixel_y"]) - max(0, int(cand["roi"]["y"]) - EXTRACT_PAD)
    L = 40.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axes[0, 0].imshow(vis)
    axes[0, 0].plot(poly_closed[:, 0], poly_closed[:, 1], color="#ffb000", lw=1.5)
    axes[0, 0].plot(
        [cx - L * path_vec[0], cx + L * path_vec[0]],
        [cy - L * path_vec[1], cy + L * path_vec[1]],
        color="#ff6030", lw=1.0, alpha=0.8,
    )
    axes[0, 0].scatter([cx], [cy], c="#ff6030", s=14)
    tag = f"  [{label}]" if label else ""
    axes[0, 0].set_title(
        f"#{cand['idx']:02d}  {cand['callsign']}{tag}\n"
        f"{cand['wall_time_utc']}   path={path_angle:5.1f}°"
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(panels["crop_for_canny"], cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(
        f"post-floor  floor={panels['floor']}   "
        f"canny={panels['canny_low']}/{panels['canny_high']}"
    )
    axes[0, 1].axis("off")

    axes[1, 0].imshow(panels["edges"], cmap="gray")
    axes[1, 0].set_title(f"Canny edges  ({int(panels['edges'].sum() / 255)} edge px)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(vis)
    axes[1, 1].plot(poly_closed[:, 0], poly_closed[:, 1], color="#ffb000", lw=1.0)
    n_aligned = n_rejected = 0
    for (x1, y1, x2, y2) in panels["raw_lines"]:
        if _aligned(x1, y1, x2, y2):
            axes[1, 1].plot([x1, x2], [y1, y2], color="#50e050", lw=1.0, alpha=0.9)
            n_aligned += 1
        else:
            axes[1, 1].plot([x1, x2], [y1, y2], color="#e04040", lw=0.8, alpha=0.5)
            n_rejected += 1
    if result.pixel_line is not None:
        x1, y1, x2, y2 = result.pixel_line
        axes[1, 1].plot([x1, x2], [y1, y2], color="#30ff30", lw=2.5)
    axes[1, 1].set_title(
        f"score={result.score:.3f}  aligned={n_aligned}  "
        f"rejected={n_rejected}  long={result.num_long_lines}  ({result.method})"
    )
    axes[1, 1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return {
        "idx": cand["idx"],
        "callsign": cand["callsign"],
        "wall_time_utc": cand["wall_time_utc"],
        "label": label,
        "score": float(result.score),
        "method": result.method,
        "num_long_lines": int(result.num_long_lines),
        "aligned_lines": int(result.aligned_lines),
        "pixel_line": result.pixel_line,
        "panel_png": out_path.name,
    }


def _write_index(records: list[dict], out_dir: Path, threshold: float) -> None:
    # Sorted worst-positive-first / best-negative-first for quick eyeball.
    def _key(r):
        has_label = r["label"] in ("positive", "negative")
        if has_label:
            # positives at bottom of score: suspicious misses
            # negatives at top of score: suspicious false positives
            if r["label"] == "positive":
                return (0, r["score"])
            return (1, -r["score"])
        return (2, -r["score"])

    rows = sorted(records, key=_key)
    md = ["# Detection review panels", ""]
    md.append(f"Threshold (from config): **{threshold:.3f}**  ")
    n_pos = sum(1 for r in records if r["label"] == "positive")
    n_neg = sum(1 for r in records if r["label"] == "negative")
    n_un = sum(1 for r in records if r["label"] not in ("positive", "negative"))
    tp = sum(1 for r in records if r["label"] == "positive" and r["score"] >= threshold)
    fp = sum(1 for r in records if r["label"] == "negative" and r["score"] >= threshold)
    md.append(
        f"Labeled: positives={n_pos} (TP={tp}), negatives={n_neg} (FP={fp}), unlabeled={n_un}"
    )
    md.append("")
    md.append("| idx | callsign | time | label | score | long | aligned | decision | panel |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        decision = "POS" if r["score"] >= threshold else "neg"
        correct = ""
        if r["label"] == "positive":
            correct = " ✓" if decision == "POS" else " ✗MISS"
        elif r["label"] == "negative":
            correct = " ✓" if decision == "neg" else " ✗FP"
        md.append(
            f"| {r['idx']:02d} | {r['callsign']} | {r['wall_time_utc']} | "
            f"{r['label'] or '-'} | {r['score']:.3f} | {r['num_long_lines']} | "
            f"{r['aligned_lines']} | {decision}{correct} | [panel]({r['panel_png']}) |"
        )
    (out_dir / "index.md").write_text("\n".join(md) + "\n")


def _write_grid(records: list[dict], out_dir: Path, cols: int = 4) -> None:
    imgs = []
    for r in records:
        p = out_dir / r["panel_png"]
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
    if not imgs:
        return
    th = max(i.shape[0] for i in imgs)
    tw = max(i.shape[1] for i in imgs)
    padded = []
    for img in imgs:
        canvas = np.full((th, tw, 3), 20, dtype=np.uint8)
        h, w = img.shape[:2]
        canvas[:h, :w] = img
        padded.append(canvas)
    rows = (len(padded) + cols - 1) // cols
    grid = np.full((rows * th, cols * tw, 3), 12, dtype=np.uint8)
    for i, t in enumerate(padded):
        r, c = i // cols, i % cols
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(str(out_dir / "index.png"), grid)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None, help="Optional labels.json")
    p.add_argument("--config", type=Path, default=Path("configs/mit_green_building.yaml"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--use-prev-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open the manifest video, seek to frame_idx-1 for each candidate, and "
            "feed the matching prev-frame crop to detect() so the panels reflect the "
            "absdiff path the live pipeline runs. Default: on. "
            "Pass --no-use-prev-frame to fall back to single-frame mode."
        ),
    )
    args = p.parse_args()

    site_cfg = load_config(args.config)
    det_cfg = site_cfg.detection
    threshold = site_cfg.aggregation.detection_threshold

    manifest = json.loads(args.manifest.read_text())
    manifest_dir = args.manifest.parent
    label_by_idx: dict[int, str] = {}
    if args.labels and args.labels.exists():
        labels_blob = json.loads(args.labels.read_text())
        label_by_idx = {e["idx"]: e["label"] for e in labels_blob["labels"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # If requested, decode prev frames once up front and crop the same padded AABB.
    prev_crops: dict[int, np.ndarray] = {}
    if args.use_prev_frame:
        video_path = Path(manifest["video"])
        if not video_path.exists():
            raise SystemExit(
                f"--use-prev-frame: video {video_path} from manifest does not exist."
            )
        # Match the extract script's upscale-to-calibration behaviour automatically:
        # if the video is smaller than the calibration, upscale prev frames so the
        # padded AABB crop has the same shape as the saved roi crop.
        calib_w, calib_h = (
            int(site_cfg.calibration.calibration_resolution[0]),
            int(site_cfg.calibration.calibration_resolution[1]),
        )
        upscale_to: tuple[int, int] | None = None
        with av.open(str(video_path)) as _probe:
            _vs = _probe.streams.video[0]
            vid_w, vid_h = int(_vs.codec_context.width), int(_vs.codec_context.height)
        if (vid_w, vid_h) != (calib_w, calib_h):
            print(
                f"Video is {vid_w}x{vid_h}, calibration is {calib_w}x{calib_h} — "
                f"upscaling prev frames to match."
            )
            upscale_to = (calib_w, calib_h)

        prev_indices = [
            int(c["frame_idx"]) - 1 for c in manifest["candidates"]
            if int(c["frame_idx"]) > 0
        ]
        print(
            f"Decoding {len(prev_indices)} prev frames from {video_path.name} "
            f"for diff-mode rendering..."
        )
        prev_full = _decode_frames(video_path, prev_indices, upscale_to=upscale_to)
        for cand in manifest["candidates"]:
            prev_idx = int(cand["frame_idx"]) - 1
            full = prev_full.get(prev_idx)
            if full is None:
                continue
            prev_crops[int(cand["idx"])] = _crop_padded(full, cand["roi"])

    records: list[dict] = []
    for cand in manifest["candidates"]:
        crop_path = manifest_dir / cand["roi_png"]
        crop = cv2.imread(str(crop_path))
        if crop is None:
            print(f"  WARN: missing {crop_path}")
            continue
        out_png = args.out_dir / f"panel_{cand['idx']:02d}.png"
        prev = prev_crops.get(int(cand["idx"]))
        rec = _render_panel(
            cand, crop, det_cfg, label_by_idx.get(cand["idx"]), out_png,
            prev_crop=prev,
        )
        records.append(rec)
        mark = ""
        if rec["label"] == "positive" and rec["score"] < threshold:
            mark = "  <-- MISS"
        elif rec["label"] == "negative" and rec["score"] >= threshold:
            mark = "  <-- FP"
        print(
            f"  #{rec['idx']:02d} {rec['callsign']:<10} {rec['label'] or '-':<8} "
            f"score={rec['score']:.3f} long={rec['num_long_lines']}{mark}"
        )

    _write_index(records, args.out_dir, threshold)
    _write_grid(records, args.out_dir)

    with (args.out_dir / "summary.json").open("w") as f:
        json.dump({"threshold": threshold, "records": records}, f, indent=2, default=str)

    print(f"\nWrote {len(records)} panels to {args.out_dir}")
    print(f"  {args.out_dir / 'index.md'}")
    print(f"  {args.out_dir / 'index.png'}")


if __name__ == "__main__":
    main()
