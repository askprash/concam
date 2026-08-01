#!/usr/bin/env python3
"""Regenerate ONLY the OCR stage for a date, reusing every other cached stage.

Used by the archive-wide reprocess after the GitHub #1 fix (derive the
timestamp date from context instead of trusting the OCR read).  The corruption
lived entirely in ``ocr.jsonl``; ``adsb.json`` and ``projections.jsonl`` are
derived from flight data and camera geometry and are completely independent of
the OCR timestamps, so re-running them would burn hours re-deriving byte-identical
output.

``concam run --from-stage ocr`` would redo adsb + project as well, and there is
no ``--only-stage``, hence this helper.  The caller follows it with
``concam run --from-stage detect``, which re-joins the corrected timestamps
against the cached projections.

Usage:
    python scripts/rerun_ocr_stage.py --date 2026-04-11 [--output-dir output]
                                      [--config configs/mit_green_building.yaml]
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mit_green_building.yaml"

# The fix this reprocess exists to apply.  If a checkout without it were used,
# the run would silently rewrite ocr.jsonl with the same corrupt dates, so the
# marker is asserted before any work happens.
FIX_MARKER = "context-derived date"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="date to process (YYYY-MM-DD)")
    ap.add_argument("--output-dir", default="output", help="pipeline output root")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="site YAML")
    ap.add_argument("--seconds-per-frame", type=float, default=1.0,
                    help="1.0 for the daily timelapse, 0.25 for raw 4fps segments")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from concam.config import load_config
    from concam.pipeline import resolve_video_path, run_ocr_stage, stage_paths

    stages_src = (REPO_ROOT / "concam" / "pipeline" / "stages.py").read_text()
    if FIX_MARKER not in stages_src:
        print(
            f"ERROR: {REPO_ROOT}/concam/pipeline/stages.py does not contain the "
            f"OCR date-derivation fix (marker {FIX_MARKER!r}). Refusing to "
            "regenerate OCR with pre-fix code.",
            file=sys.stderr,
        )
        return 3

    date = datetime.date.fromisoformat(args.date)
    site_config = load_config(args.config)
    paths = stage_paths(Path(args.output_dir), date)
    paths["base"].mkdir(parents=True, exist_ok=True)

    video_path = resolve_video_path(site_config.video, date)
    print(f"video: {video_path}")

    n = run_ocr_stage(
        video_path=video_path,
        date=date,
        site_config=site_config,
        out_path=paths["ocr"],
        seconds_per_frame=args.seconds_per_frame,
    )
    print(f"OCR wrote {n} frame records -> {paths['ocr']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
