"""Projection alignment validation: offset analysis + go/no-go (PRD item 13).

Consumes ``labels.json`` produced by the labeller HTML from
``projection_alignment_extract.py`` and produces:

  * ``summary.md``  — per-flyover median/max offset, total pixel count with
    a visible aircraft, and the go/no-go verdict against the 100 px
    threshold in PRD item 13.
  * ``offsets.csv`` — one row per labelled frame with projected pixel,
    clicked pixel, offset_dx/dy, offset_px.
  * ``offsets_scatter.png`` — scatter of offset vectors placed at the
    projected pixel, colour-coded by flyover. Systematic distortion (e.g.
    radial error) shows up as a directional trend across the FOV.

Usage::

    uv run python scripts/projection_alignment_analyze.py \\
        --date 2026-04-09 --labels output/validation/projection/2026-04-09/labels.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GO_THRESHOLD_PX = 100.0


def load_labels(path: Path) -> dict:
    with path.open() as f:
        payload = json.load(f)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unexpected schema_version in {path}: {payload.get('schema_version')}")
    return payload


def _offset(label: dict) -> tuple[float, float, float]:
    """Return (dx, dy, magnitude) where offset = projected - clicked.

    Positive dx means the overlay is to the RIGHT of the visible aircraft;
    positive dy means the overlay is BELOW it. This convention matches the
    PRD wording (offset vector = track_pixel - aircraft_pixel).
    """
    dx = float(label["projected_pixel_x"]) - float(label["click_x"])
    dy = float(label["projected_pixel_y"]) - float(label["click_y"])
    return dx, dy, math.hypot(dx, dy)


def summarise(labels: list[dict]) -> dict:
    """Compute per-flyover + overall offset statistics.

    Only labels with visible=True and non-null click coords contribute; the
    count of not-visible and unlabelled-within-visible cases is reported
    alongside so a labeller who skipped half the frames doesn't accidentally
    produce an over-confident pass.
    """
    per_fly: dict[int, list[dict]] = {}
    n_visible = 0
    n_notvisible = 0
    n_missing = 0
    for lab in labels:
        if not lab.get("visible", True):
            n_notvisible += 1
            continue
        if lab.get("click_x") is None or lab.get("click_y") is None:
            n_missing += 1
            continue
        dx, dy, mag = _offset(lab)
        per_fly.setdefault(lab["flyover_idx"], []).append({
            **lab,
            "offset_dx": dx,
            "offset_dy": dy,
            "offset_px": mag,
        })
        n_visible += 1

    fly_summaries: list[dict] = []
    for fly_idx in sorted(per_fly):
        offs = [r["offset_px"] for r in per_fly[fly_idx]]
        fly_summaries.append({
            "flyover_idx": fly_idx,
            "callsign": per_fly[fly_idx][0]["callsign"],
            "transponder_id": per_fly[fly_idx][0]["transponder_id"],
            "n_frames": len(offs),
            "median_offset_px": float(np.median(offs)),
            "max_offset_px": float(np.max(offs)),
            "mean_offset_px": float(np.mean(offs)),
        })

    all_offs = [r["offset_px"] for rows in per_fly.values() for r in rows]
    overall = {
        "n_visible_labelled": n_visible,
        "n_not_visible": n_notvisible,
        "n_visible_missing_click": n_missing,
        "median_offset_px": float(np.median(all_offs)) if all_offs else float("nan"),
        "max_offset_px": float(np.max(all_offs)) if all_offs else float("nan"),
        "mean_offset_px": float(np.mean(all_offs)) if all_offs else float("nan"),
        "go_threshold_px": GO_THRESHOLD_PX,
        "verdict": _verdict(all_offs),
    }
    return {
        "per_flyover": fly_summaries,
        "overall": overall,
        "rows": [r for rows in per_fly.values() for r in rows],
    }


def _verdict(offs: list[float]) -> str:
    if not offs:
        return "inconclusive"
    if float(np.median(offs)) <= GO_THRESHOLD_PX:
        return "GO"
    return "NO-GO"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("flyover_idx,frame_idx,projected_pixel_x,projected_pixel_y,click_x,click_y,offset_dx,offset_dy,offset_px\n")
        return
    fields = [
        "flyover_idx", "callsign", "transponder_id", "frame_idx", "wall_time_utc",
        "projected_pixel_x", "projected_pixel_y", "click_x", "click_y",
        "offset_dx", "offset_dy", "offset_px",
    ]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_scatter(path: Path, rows: list[dict], image_size: tuple[int, int]) -> None:
    """Plot offset vectors as short arrows at each projected pixel.

    A radial distortion error shows up as arrows that all point toward (or
    away from) image center; a translation error shows up as parallel
    arrows; a rotation error shows up as a tangential pattern. The labeller
    HTML gives us the click coordinates directly so we can draw both the
    tip (click) and the tail (projected) for each labelled frame.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)  # image-Y increases downward
    ax.set_aspect("equal")
    ax.set_title("Projection offset: projected → clicked aircraft")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")

    cmap = plt.get_cmap("tab10")
    by_fly: dict[int, list[dict]] = {}
    for r in rows:
        by_fly.setdefault(r["flyover_idx"], []).append(r)
    for fly_idx, fly_rows in by_fly.items():
        color = cmap(fly_idx % 10)
        xs = [r["projected_pixel_x"] for r in fly_rows]
        ys = [r["projected_pixel_y"] for r in fly_rows]
        us = [-r["offset_dx"] for r in fly_rows]  # arrow points FROM projected TO clicked
        vs = [-r["offset_dy"] for r in fly_rows]
        ax.quiver(
            xs, ys, us, vs,
            angles="xy", scale_units="xy", scale=1.0,
            color=color, width=0.004, label=f"flyover {fly_idx} ({fly_rows[0]['callsign']})",
        )
        ax.scatter(xs, ys, s=25, color=color, edgecolors="k", linewidths=0.5)

    ax.axhline(image_size[1] / 2, color="#888", linestyle=":", linewidth=0.5)
    ax.axvline(image_size[0] / 2, color="#888", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_markdown(path: Path, summary: dict, date: str, manifest: dict) -> None:
    overall = summary["overall"]
    lines = [
        "# Projection alignment — ADS-B to visible-aircraft offset check",
        "",
        f"Date: **{date}**  ·  Video: `{manifest.get('video', '?')}`",
        f"Daylight window: `{manifest.get('daylight_utc', '?')}` UTC  ·  Image size: {manifest.get('image_size')}",
        "",
        f"## Verdict: **{overall['verdict']}**  (threshold {overall['go_threshold_px']:.0f} px median)",
        "",
        f"- Labelled frames with visible aircraft clicked: **{overall['n_visible_labelled']}**",
        f"- Frames marked not visible: {overall['n_not_visible']}",
        f"- Frames marked visible without a click (data entry error): {overall['n_visible_missing_click']}",
        f"- Overall median offset: **{overall['median_offset_px']:.1f} px**  ·  mean {overall['mean_offset_px']:.1f} px  ·  max {overall['max_offset_px']:.1f} px",
        "",
        "## Per-flyover",
        "",
        "| # | callsign | transponder | n | median px | mean px | max px |",
        "|--:|---------|-------------|--:|----------:|--------:|-------:|",
    ]
    for fly in summary["per_flyover"]:
        lines.append(
            f"| {fly['flyover_idx']:>2} | {fly['callsign']} | {fly['transponder_id']} | "
            f"{fly['n_frames']} | {fly['median_offset_px']:.1f} | {fly['mean_offset_px']:.1f} | {fly['max_offset_px']:.1f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- GO threshold is {overall['go_threshold_px']:.0f} px median — roughly one small-plane silhouette at 4K.",
        "- If the scatter plot shows **parallel** offset arrows the camera extrinsic translation is off.",
        "- If it shows a **radial** pattern (arrows aimed at or away from FOV center) the distortion coefficients are off.",
        "- If it shows a **rotational** / tangential pattern the camera heading (yaw) is off.",
        "- Random, uncorrelated offsets below threshold are acceptable — click noise and aircraft-size uncertainty dominate.",
        "",
        "## Files",
        "",
        f"- `offsets.csv` — one row per clicked frame (projected, click, offset_dx/dy, offset_px)",
        f"- `offsets_scatter.png` — offset vector field across the FOV",
        f"- `manifest.json` — original flyover + frame list (produced by extract script)",
        f"- `labels.json` — labeller output",
        "",
        f"_generated at {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}_",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--manifest", default=None, help="default: <labels>/../manifest.json")
    ap.add_argument("--output-dir", default=None, help="default: dir containing --labels")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"Missing {labels_path}. Label the flyovers via labeller.html first.")
    manifest_path = Path(args.manifest) if args.manifest else labels_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}.")
    out_dir = Path(args.output_dir) if args.output_dir else labels_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_labels(labels_path)
    manifest = json.loads(manifest_path.read_text())
    summary = summarise(payload.get("labels", []))

    write_csv(out_dir / "offsets.csv", summary["rows"])
    plot_scatter(out_dir / "offsets_scatter.png", summary["rows"], tuple(manifest.get("image_size", [3840, 2160])))
    write_markdown(out_dir / "summary.md", summary, payload.get("date", args.date), manifest)

    ov = summary["overall"]
    print(f"Verdict: {ov['verdict']}  (median {ov['median_offset_px']:.1f} px, threshold {ov['go_threshold_px']:.0f} px)")
    print(f"Labelled {ov['n_visible_labelled']} visible frames across {len(summary['per_flyover'])} flyovers.")
    print(f"Files written to: {out_dir}")


if __name__ == "__main__":
    main()
