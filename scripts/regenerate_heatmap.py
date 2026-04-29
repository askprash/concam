"""Regenerate the labeler's flight-level × time heatmap from a manifest.

Mirrors the client-side heatmap rendered in ``concam/bundle/templates/labeler.html``
so that if a reviewer never clicks "Export labels" / "PNG", we can still
reconstruct the same picture from persisted data.

Inputs:
  * ``--manifest``: path to a public bundle's ``manifest.json`` (must carry
    ``alt_baro_m`` / ``alt_m`` in flight_tracks pings — the public bundle does;
    the per-labeler ``concam bundle`` output currently does not).
  * ``--labels`` (optional): a labeler's exported labels JSON, of the same
    shape produced by the labeler's "Export labels" button. When present,
    human labels override auto-detection in the heatmap classification.
    Without it, classification is auto-only (peak_score >= detection_threshold).

Outputs:
  * ``--out-png`` and/or ``--out-svg`` (at least one required).

Usage::

    # Auto-only heatmap from a public bundle:
    uv run python scripts/regenerate_heatmap.py \\
        --manifest ~/public_html/concam/2026-04-26/manifest.json \\
        --out-png /tmp/2026-04-26_auto_heatmap.png

    # With human labels overriding auto-detections:
    uv run python scripts/regenerate_heatmap.py \\
        --manifest ~/public_html/concam/2026-04-26/manifest.json \\
        --labels labels/2026-04-26_prash.json \\
        --out-png /tmp/2026-04-26_prash_heatmap.png \\
        --out-svg /tmp/2026-04-26_prash_heatmap.svg
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Defaults match window.heatmapSpec in labeler.html. The manifest does not
# currently carry these, so we keep them as a single source of truth here.
FL_MIN = 250
FL_MAX = 450
FL_STEP = 10
BIN_MINUTES = 10
DETECTION_THRESHOLD_FALLBACK = 0.083

VALID_LABELS = {"contrail", "no_contrail", "unsure"}

# Colours match the SVG output (Tailwind red-500 / blue-500).
COLOR_NO_OBS = "#1a1a1a"
COLOR_CONTRAIL = (239 / 255, 68 / 255, 68 / 255)
COLOR_CLEAR = (59 / 255, 130 / 255, 246 / 255)
COLOR_MIXED = "#888888"


def _parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _episode_median_fl(ep: dict, flight_tracks: dict) -> float | None:
    """Median flight-level from track pings inside [onset, end]."""
    track = flight_tracks.get(ep["transponder_id"])
    if not track:
        return None
    onset = _parse_iso(ep["onset"])
    end = _parse_iso(ep["end"])
    fls: list[float] = []
    for p in track["pings"]:
        t = _parse_iso(p["wall_time_utc"])
        if t < onset or t > end:
            continue
        alt = p.get("alt_baro_m")
        if alt is None:
            alt = p.get("alt_m")
        if alt is None:
            continue
        fls.append((alt * 3.28084) / 100.0)
    if not fls:
        return None
    fls.sort()
    return fls[len(fls) // 2]


def _classify(ep: dict, label_map: dict[int, str], threshold: float) -> str | None:
    """Human label wins; otherwise auto via peak_score >= threshold."""
    lbl = label_map.get(int(ep["episode_id"]))
    if lbl in VALID_LABELS:
        if lbl == "contrail":
            return "contrail"
        if lbl == "no_contrail":
            return "clear"
        return None  # unsure
    return "contrail" if ep["peak_score"] >= threshold else "clear"


def _load_label_map(labels_path: Path | None) -> dict[int, str]:
    if labels_path is None:
        return {}
    payload = json.loads(labels_path.read_text())
    out: dict[int, str] = {}
    for rec in payload.get("labels", []):
        if rec.get("label") in VALID_LABELS:
            out[int(rec["episode_id"])] = rec["label"]
    return out


def compute_heatmap(manifest: dict, label_map: dict[int, str]):
    """Bin every classifiable episode into (time-bin, FL-bin) cells.

    Returns ``(cells, x_min_ms, n_bins, n_rows)`` where ``cells[bin][row]``
    is a dict ``{"contrail": [...], "clear": [...]}`` of contributing flight
    descriptors — same shape the JS produces, so the visual output matches.
    """
    flight_tracks = manifest.get("flight_tracks", {})
    threshold = manifest.get("detection_threshold", DETECTION_THRESHOLD_FALLBACK)
    bin_ms = BIN_MINUTES * 60_000
    n_rows = (FL_MAX - FL_MIN) // FL_STEP

    obs: list[tuple[int, float, str, dict]] = []  # (t_ms, fl, cls, flight_meta)
    for ep in manifest["episodes"]:
        fl = _episode_median_fl(ep, flight_tracks)
        if fl is None:
            continue
        cls = _classify(ep, label_map, threshold)
        if cls is None:
            continue
        t_mid_ms = int(
            (
                _parse_iso(ep["onset"]).timestamp()
                + _parse_iso(ep["end"]).timestamp()
            )
            * 500  # seconds → ms, /2 baked in
        )
        obs.append(
            (
                t_mid_ms,
                fl,
                cls,
                {
                    "callsign": ep["callsign"],
                    "episode_id": ep["episode_id"],
                    "score": ep["peak_score"],
                    "fl": fl,
                    "label": label_map.get(int(ep["episode_id"])),
                },
            )
        )

    day_start_ms = int(_parse_iso(manifest["date"] + "T00:00:00+00:00").timestamp() * 1000)
    if not obs:
        x_min_ms = day_start_ms
        x_max_ms = day_start_ms + 86_400_000
    else:
        t_min = min(o[0] for o in obs)
        t_max = max(o[0] for o in obs)
        x_min_ms = day_start_ms + ((t_min - day_start_ms) // bin_ms) * bin_ms - bin_ms
        x_max_ms = day_start_ms + (
            -(-(t_max - day_start_ms) // bin_ms)
        ) * bin_ms + bin_ms
    n_bins = max(1, round((x_max_ms - x_min_ms) / bin_ms))

    if not obs:
        n_eps = len(manifest["episodes"])
        n_with_track = sum(1 for ep in manifest["episodes"]
                           if ep["transponder_id"] in flight_tracks)
        print(
            f"[regenerate_heatmap] WARNING: no FL data — "
            f"{n_eps} episodes, {n_with_track} have a flight_track entry, "
            f"none had altitude. The local `concam bundle` output strips "
            f"alt_baro_m; use a public bundle (build_public_bundle.py) instead."
        )

    cells = [
        [{"contrail": [], "clear": []} for _ in range(n_rows)]
        for _ in range(n_bins)
    ]
    for t_ms, fl, cls, meta in obs:
        b = (t_ms - x_min_ms) // bin_ms
        if not (0 <= b < n_bins):
            continue
        if not (FL_MIN <= fl < FL_MAX):
            continue
        r = int((fl - FL_MIN) // FL_STEP)
        cells[b][r][cls].append(meta)
    return cells, x_min_ms, n_bins, n_rows


def render(cells, x_min_ms, n_bins, n_rows, *, date: str, title_suffix: str = ""):
    """Render with matplotlib. Returns the Figure."""
    # Build the colour grid. With imshow(origin="lower"), array row 0 maps to
    # the bottom of the plot, so we index by r directly — FL_MIN sits at the
    # bottom and FL_MAX at the top, matching the labeler.
    rgb = np.zeros((n_rows, n_bins, 4), dtype=np.float32)
    for b in range(n_bins):
        for r in range(n_rows):
            cell = cells[b][r]
            c = len(cell["contrail"])
            k = len(cell["clear"])
            if c == 0 and k == 0:
                rgb[r, b] = (*tuple(int(COLOR_NO_OBS[i:i+2], 16) / 255 for i in (1, 3, 5)), 1.0)
            elif c > 0 and k > 0:
                rgb[r, b] = (*tuple(int(COLOR_MIXED[i:i+2], 16) / 255 for i in (1, 3, 5)), 1.0)
            elif c > 0:
                a = 0.4 + 0.15 * min(c, 4)
                rgb[r, b] = (*COLOR_CONTRAIL, a)
            else:
                a = 0.35 + 0.15 * min(k, 4)
                rgb[r, b] = (*COLOR_CLEAR, a)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    # Black background under semi-transparent cells so alpha mixing matches
    # the labeler (panel bg there is dark inside the plot frame).
    ax.set_facecolor("#1a1a1a")
    ax.imshow(
        rgb,
        aspect="auto",
        interpolation="nearest",
        extent=(0, n_bins, 0, n_rows),
        origin="lower",
    )

    # Y-axis: FL labels every 50 (5 rows).
    yticks = []
    yticklabels = []
    for fl in range(FL_MIN, FL_MAX + 1, 50):
        row_from_bottom = (fl - FL_MIN) // FL_STEP
        yticks.append(row_from_bottom)
        yticklabels.append(f"FL{fl}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # X-axis: hourly labels.
    bin_ms = BIN_MINUTES * 60_000
    day_start_ms = int(_parse_iso(date + "T00:00:00+00:00").timestamp() * 1000)
    hour_ms = 3_600_000
    first_hour = -((-(x_min_ms - day_start_ms)) // hour_ms)  # ceil
    last_hour = (x_min_ms + n_bins * bin_ms - day_start_ms) // hour_ms
    xticks = []
    xticklabels = []
    label_every = 1 if n_bins <= 36 else (2 if n_bins <= 80 else 4)
    for h in range(first_hour, last_hour + 1):
        if h % label_every != 0:
            continue
        t_ms = day_start_ms + h * hour_ms
        x_bin = (t_ms - x_min_ms) / bin_ms
        if 0 <= x_bin <= n_bins:
            xticks.append(x_bin)
            xticklabels.append(f"{h % 24:02d}:00")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, n_rows)
    title = f"Flight levels × time — {date} (UTC)"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=10)

    # Legend: small swatches matching the labeler.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR_CONTRAIL, label="contrail"),
        Patch(facecolor=COLOR_CLEAR, label="clear"),
        Patch(facecolor=COLOR_MIXED, label="mixed"),
        Patch(facecolor=COLOR_NO_OBS, label="no obs"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-svg", type=Path, default=None)
    args = parser.parse_args()

    if args.out_png is None and args.out_svg is None:
        parser.error("at least one of --out-png or --out-svg is required")

    manifest = json.loads(args.manifest.read_text())
    label_map = _load_label_map(args.labels)
    title_suffix = ""
    if args.labels is not None:
        title_suffix = f"labels: {args.labels.name} ({len(label_map)} applied)"

    cells, x_min_ms, n_bins, n_rows = compute_heatmap(manifest, label_map)
    fig = render(
        cells,
        x_min_ms,
        n_bins,
        n_rows,
        date=manifest["date"],
        title_suffix=title_suffix,
    )

    if args.out_png is not None:
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out_png}")
    if args.out_svg is not None:
        args.out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_svg, bbox_inches="tight")
        print(f"wrote {args.out_svg}")
    plt.close(fig)


if __name__ == "__main__":
    main()
