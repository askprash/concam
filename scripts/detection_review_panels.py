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
from concam.detection import detect, explain
from concam.detection.geometry import candidate_geometry
from concam.detection.viz import compose_grid, render_detection_panels
from concam.video import decode_frames

EXTRACT_PAD = 20




def _crop_padded(frame: np.ndarray, roi: dict, pad: int = EXTRACT_PAD) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(roi["x"]) - pad)
    y1 = max(0, int(roi["y"]) - pad)
    x2 = min(w, int(roi["x"]) + int(roi["w"]) + pad)
    y2 = min(h, int(roi["y"]) + int(roi["h"]) + pad)
    return frame[y1:y2, x1:x2].copy()




def _render_panel(
    cand: dict,
    crop: np.ndarray,
    cfg: DetectionConfig,
    label: str | None,
    out_path: Path,
    prev_crop: np.ndarray | None = None,
) -> dict:
    g = candidate_geometry(
        cand, crop.shape[:2],
        roi_along_px=cfg.roi_along_px,
        roi_cross_px=cfg.roi_cross_px,
        extract_pad=EXTRACT_PAD,
    )
    # Obtain the exact DetectionPass the detector used — panels derived from
    # this are guaranteed to show what detect() actually saw, including any
    # cross_grad / local_contrast preprocessing that _prepare_base applies.
    passed = explain(
        crop, g.rect, cfg,
        polygon=g.polygon, path_vec=g.path_vec,
        prev_frame=prev_crop,
        frame_origin=g.frame_origin,
    )
    result = detect(
        crop, g.rect, cfg,
        polygon=g.polygon, path_vec=g.path_vec,
        prev_frame=prev_crop,
        frame_origin=g.frame_origin,
    )
    panels_named = render_detection_panels(passed, labels=True)
    panel_dict = {name: img for name, img in panels_named}

    path_angle = math.degrees(math.atan2(g.path_vec[1], g.path_vec[0])) % 180.0

    vis = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    poly_closed = np.vstack([g.polygon, g.polygon[:1]])
    cx = g.center.x
    cy = g.center.y
    L = 40.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axes[0, 0].imshow(vis)
    axes[0, 0].plot(poly_closed[:, 0], poly_closed[:, 1], color="#ffb000", lw=1.5)
    axes[0, 0].plot(
        [cx - L * g.path_vec[0], cx + L * g.path_vec[0]],
        [cy - L * g.path_vec[1], cy + L * g.path_vec[1]],
        color="#ff6030", lw=1.0, alpha=0.8,
    )
    axes[0, 0].scatter([cx], [cy], c="#ff6030", s=14)
    tag = f"  [{label}]" if label else ""
    axes[0, 0].set_title(
        f"#{cand['idx']:02d}  {cand['callsign']}{tag}\n"
        f"{cand['wall_time_utc']}   path={path_angle:5.1f}°"
    )
    axes[0, 0].axis("off")

    # Panel 2: preprocessed base (what Canny saw, post-floor) from the DetectionPass.
    axes[0, 1].imshow(cv2.cvtColor(panel_dict["base"], cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    axes[0, 1].set_title(
        f"post-floor  floor={passed.floor}   "
        f"canny={passed.canny_low}/{passed.canny_high}"
    )
    axes[0, 1].axis("off")

    # Panel 3: Canny edges — exactly what the detector used.
    axes[1, 0].imshow(passed.edges, cmap="gray")
    axes[1, 0].set_title(f"Canny edges  ({int(passed.edges.sum() // 255)} edge px)")
    axes[1, 0].axis("off")

    # Panel 4: overlay with aligned (green) and rejected (red) lines + pixel_line.
    axes[1, 1].imshow(vis)
    axes[1, 1].plot(poly_closed[:, 0], poly_closed[:, 1], color="#ffb000", lw=1.0)
    # raw_lines are full-frame; crop is the full input here so crop_origin offsets apply.
    x0, y0 = passed.crop_origin
    aligned_set = set(id(ln) for ln in passed.aligned)
    n_aligned = len(passed.aligned)
    n_rejected = len(passed.raw_lines) - n_aligned
    for ln in passed.raw_lines:
        fx1, fy1, fx2, fy2, _ = ln
        # Translate full-frame -> crop-local for display on the crop image.
        cx1, cy1 = fx1 - x0, fy1 - y0
        cx2, cy2 = fx2 - x0, fy2 - y0
        if id(ln) in aligned_set:
            axes[1, 1].plot([cx1, cx2], [cy1, cy2], color="#50e050", lw=1.0, alpha=0.9)
        else:
            axes[1, 1].plot([cx1, cx2], [cy1, cy2], color="#e04040", lw=0.8, alpha=0.5)
    if result.pixel_line is not None:
        px1, py1, px2, py2 = result.pixel_line
        axes[1, 1].plot([px1 - x0, px2 - x0], [py1 - y0, py2 - y0], color="#30ff30", lw=2.5)
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
    grid = compose_grid(imgs, cols, bg=12)
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
        prev_full = decode_frames(video_path, prev_indices, upscale_to=upscale_to)
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
