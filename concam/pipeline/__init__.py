"""Pipeline stage orchestration for the concam batch CLI.

Each stage reads from and writes to a well-defined intermediate path under an
output directory so ``--from-stage`` can resume from a named checkpoint.

Stage outputs, all under ``{output_dir}/{YYYY-MM-DD}/``:

    ocr.jsonl           one line per video frame:
                        {frame_idx, wall_time_utc, confidence, method, status}
    adsb.json           list of flights (callsign, transponder_id, pings)
    projections.jsonl   one line per (wall_time_utc, flight):
                        {wall_time_utc, callsign, transponder_id,
                         pixel_x, pixel_y, path_dx, path_dy,
                         roi: {x, y, w, h}}
    detections.jsonl    one line per (frame, flight) run of the detector:
                        {wall_time_utc, callsign, transponder_id,
                         score, pixel_line, method}
    episodes.jsonl      one line per Episode
    pipeline.duckdb     DuckDB with contrail_episodes populated

Stages are decoupled through these files — unit tests exercise each stage
in isolation and the CLI glues them together.
"""

from concam.pipeline.stages import (
    STAGES,
    resolve_video_path,
    run_adsb_stage,
    run_aggregate_stage,
    run_detect_stage,
    run_ocr_stage,
    run_project_stage,
    run_store_stage,
    stage_paths,
)

__all__ = [
    "STAGES",
    "resolve_video_path",
    "run_ocr_stage",
    "run_adsb_stage",
    "run_project_stage",
    "run_detect_stage",
    "run_aggregate_stage",
    "run_store_stage",
    "stage_paths",
]
