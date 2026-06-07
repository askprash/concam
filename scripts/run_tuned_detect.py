"""Run detect+aggregate+store with a tuned config into a custom variant dir.

The ``concam run`` CLI requires --date to be a strict ``YYYY-MM-DD``, which
prevents writing to e.g. ``output/2026-04-09-tuned/`` for an A/B variant.
This helper calls the stage functions directly so we can emit the tuned
artefacts under any directory name while still pointing at the base date's
cached OCR/ADS-B/projections.

Usage::

    uv run python scripts/run_tuned_detect.py \\
        --base-date 2026-04-09 \\
        --variant-dir output/2026-04-09-tuned \\
        --config configs/mit_green_building.tuned.yaml
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import load_config
from concam.pipeline import resolve_video_path, stage_paths
from concam.pipeline.stages import (
    run_aggregate_stage,
    run_detect_stage,
    run_store_stage,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-date", required=True, type=datetime.date.fromisoformat,
                   help="Base date (YYYY-MM-DD) whose cached OCR/ADS-B/projections we read.")
    p.add_argument("--variant-dir", required=True, type=Path,
                   help="Output directory for the tuned artefacts "
                        "(e.g. output/2026-04-09-tuned). Will be created.")
    p.add_argument("--config", required=True, type=Path,
                   help="Tuned YAML config.")
    p.add_argument("--base-output-root", type=Path, default=REPO_ROOT / "output",
                   help="Pipeline output root containing <base-date>/.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    site_config = load_config(args.config)
    base_paths = stage_paths(args.base_output_root, args.base_date)

    for stage_name in ("ocr", "adsb", "projections"):
        if not base_paths[stage_name].exists():
            print(f"ERROR: missing {stage_name} cache at {base_paths[stage_name]}",
                  file=sys.stderr)
            return 2

    args.variant_dir.mkdir(parents=True, exist_ok=True)
    detections_path = args.variant_dir / "detections.jsonl"
    episodes_path = args.variant_dir / "episodes.jsonl"
    db_path = args.variant_dir / "pipeline.duckdb"

    video_path = resolve_video_path(site_config.video, args.base_date)

    print(f"[tuned-detect] base date: {args.base_date}")
    print(f"[tuned-detect] variant out: {args.variant_dir}")
    print(f"[tuned-detect] config: {args.config}")
    print(f"[tuned-detect] video: {video_path}")
    print()

    print("[tuned-detect] running detect")
    n_det = run_detect_stage(
        video_path=video_path,
        ocr_path=base_paths["ocr"],
        projections_path=base_paths["projections"],
        site_config=site_config,
        out_path=detections_path,
    )
    print(f"  wrote {n_det} detection records → {detections_path}")

    print("[tuned-detect] running aggregate")
    n_ep = run_aggregate_stage(
        detections_path=detections_path,
        site_config=site_config,
        out_path=episodes_path,
    )
    print(f"  wrote {n_ep} episodes → {episodes_path}")

    print("[tuned-detect] running store")
    n_rows = run_store_stage(
        episodes_path=episodes_path,
        date=args.base_date,
        db_path=db_path,
    )
    print(f"  wrote {n_rows} rows → {db_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
