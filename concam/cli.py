"""Command-line interface for the concam pipeline."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import click


DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "mit_green_building.yaml"


@click.group()
@click.option("--config", default=str(DEFAULT_CONFIG), show_default=True,
              type=click.Path(exists=False), help="Path to site YAML config.")
@click.option("-v", "--verbose", is_flag=True, default=False,
              help="Verbose logging.")
@click.pass_context
def main(ctx: click.Context, config: str, verbose: bool) -> None:
    """MIT ConCam pipeline — contrail detection and labeling from sky camera footage."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@main.command()
@click.option("--date", required=True, type=str,
              help="UTC date to process (YYYY-MM-DD).")
@click.option("--output-dir", default="output", show_default=True,
              type=click.Path(), help="Directory for intermediate and final outputs.")
@click.option("--from-stage", default=None,
              type=click.Choice(["ocr", "adsb", "project", "detect", "aggregate", "store"]),
              help="Resume from this stage, loading earlier stage outputs from cache.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Validate inputs and print execution plan without running any stage.")
@click.option("--max-frames", default=None, type=int,
              help="Limit OCR/detect stages to the first N video frames (for testing).")
@click.option("--video", default=None, type=click.Path(exists=False),
              help="Override the video path resolved from config (use to point at a raw segment).")
@click.option("--seconds-per-frame", default=1.0, show_default=True, type=float,
              help="Wall-clock seconds between consecutive video frames. 1.0 for daily timelapse, 0.25 for 4fps raw segments.")
@click.pass_context
def run(ctx: click.Context, date: str, output_dir: str,
        from_stage: str | None, dry_run: bool, max_frames: int | None,
        video: str | None, seconds_per_frame: float) -> None:
    """Run the full detection pipeline for a single UTC date."""
    from concam.config import load_config
    from concam.pipeline import (
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

    try:
        parsed_date = datetime.date.fromisoformat(date)
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got: {date!r}", param_hint="--date")

    site_config = load_config(ctx.obj["config_path"])
    output_root = Path(output_dir)
    paths = stage_paths(output_root, parsed_date)

    start_idx = STAGES.index(from_stage) if from_stage else 0
    plan = STAGES[start_idx:]

    click.echo(f"Pipeline date: {parsed_date}")
    click.echo(f"Config: {ctx.obj['config_path']}")
    click.echo(f"Output dir: {paths['base']}")
    if from_stage:
        click.echo(f"Resuming from stage: {from_stage}")
    if max_frames is not None:
        click.echo(f"Max frames: {max_frames}")

    if dry_run:
        click.echo("[dry-run] Execution plan:")
        for stage in plan:
            click.echo(f"  - {stage}")
        return

    # Verify cached outputs exist for any stages we skip.
    for skipped in STAGES[:start_idx]:
        if skipped in ("ocr", "adsb", "projections", "detections", "episodes", "store"):
            pass  # handled below by per-stage existence check
    # The stages we skip must have their output files present.
    _skip_file_for = {
        "ocr": "ocr",
        "adsb": "adsb",
        "project": "projections",
        "detect": "detections",
        "aggregate": "episodes",
    }
    for skipped in STAGES[:start_idx]:
        key = _skip_file_for.get(skipped)
        if key and not paths[key].exists():
            raise click.ClickException(
                f"--from-stage={from_stage} requires cached {skipped} output at "
                f"{paths[key]}, but it does not exist."
            )

    paths["base"].mkdir(parents=True, exist_ok=True)

    # Resolve video path up-front so OCR and detect stages agree.
    video_path = None
    if "ocr" in plan or "detect" in plan:
        if video is not None:
            video_path = Path(video)
            if not video_path.exists():
                raise click.ClickException(f"--video path does not exist: {video_path}")
        else:
            video_path = resolve_video_path(site_config.video, parsed_date)
        click.echo(f"Video: {video_path}")

    for stage in plan:
        click.echo(f"\n=== Running stage: {stage} ===")
        if stage == "ocr":
            n = run_ocr_stage(
                video_path=video_path,
                date=parsed_date,
                site_config=site_config,
                out_path=paths["ocr"],
                max_frames=max_frames,
                seconds_per_frame=seconds_per_frame,
            )
            click.echo(f"OCR: wrote {n} frame records -> {paths['ocr']}")
        elif stage == "adsb":
            n = run_adsb_stage(
                date=parsed_date,
                site_config=site_config,
                out_path=paths["adsb"],
            )
            click.echo(f"ADS-B: wrote {n} flights -> {paths['adsb']}")
        elif stage == "project":
            n = run_project_stage(
                adsb_path=paths["adsb"],
                site_config=site_config,
                out_path=paths["projections"],
            )
            click.echo(f"PROJECT: wrote {n} projections -> {paths['projections']}")
        elif stage == "detect":
            n = run_detect_stage(
                video_path=video_path,
                ocr_path=paths["ocr"],
                projections_path=paths["projections"],
                site_config=site_config,
                out_path=paths["detections"],
                max_frames=max_frames,
            )
            click.echo(f"DETECT: wrote {n} detections -> {paths['detections']}")
        elif stage == "aggregate":
            n = run_aggregate_stage(
                detections_path=paths["detections"],
                site_config=site_config,
                out_path=paths["episodes"],
            )
            click.echo(f"AGGREGATE: wrote {n} episodes -> {paths['episodes']}")
        elif stage == "store":
            n = run_store_stage(
                episodes_path=paths["episodes"],
                date=parsed_date,
                db_path=paths["store"],
            )
            click.echo(f"STORE: inserted {n} episodes -> {paths['store']}")
        else:
            raise click.ClickException(f"unknown stage: {stage}")

    click.echo(f"\nPipeline complete. DuckDB: {paths['store']}")


@main.command()
@click.option("--date", required=True, type=str, help="UTC date (YYYY-MM-DD).")
@click.option("--labelers", required=True, multiple=True, help="Labeler IDs.")
@click.option("--overlap-fraction", default=0.2, show_default=True, type=float,
              help="Fraction of episodes assigned to both labelers for inter-rater calibration.")
@click.option("--output-dir", default="output", show_default=True,
              type=click.Path(), help="Pipeline output root (must contain <date>/pipeline.duckdb etc).")
@click.option("--bundles-dir", default=None, type=click.Path(),
              help="Where to write per-labeler bundle directories. Defaults to <output-dir>/<date>/bundles.")
@click.option("--video", default=None, type=click.Path(exists=False),
              help="Override the video path resolved from config.")
@click.option("--seconds-per-frame", default=1.0, show_default=True, type=float,
              help="Wall-clock seconds between consecutive video frames.")
@click.pass_context
def bundle(ctx: click.Context, date: str, labelers: tuple[str, ...],
           overlap_fraction: float, output_dir: str,
           bundles_dir: str | None, video: str | None,
           seconds_per_frame: float) -> None:
    """Generate per-labeler annotation bundles for a given date."""
    from concam.bundle import generate_bundles
    from concam.config import load_config
    from concam.pipeline import resolve_video_path, stage_paths

    try:
        parsed_date = datetime.date.fromisoformat(date)
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got: {date!r}", param_hint="--date")

    site_config = load_config(ctx.obj["config_path"])
    paths = stage_paths(Path(output_dir), parsed_date)

    required = [paths["store"], paths["projections"], paths["detections"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise click.ClickException(
            "Bundle requires pipeline outputs; missing:\n  " + "\n  ".join(missing)
        )

    if video is not None:
        video_path = Path(video)
    else:
        try:
            video_path = resolve_video_path(site_config.video, parsed_date)
        except FileNotFoundError as e:
            raise click.ClickException(str(e))

    out_root = Path(bundles_dir) if bundles_dir else paths["base"] / "bundles"
    image_size = tuple(site_config.calibration.calibration_resolution)

    result = generate_bundles(
        date=parsed_date,
        labelers=list(labelers),
        overlap_fraction=overlap_fraction,
        db_path=paths["store"],
        projections_path=paths["projections"],
        detections_path=paths["detections"],
        video_path=video_path,
        image_size=image_size,
        output_dir=out_root,
        seconds_per_frame=seconds_per_frame,
        ocr_path=paths["ocr"] if paths["ocr"].exists() else None,
        detection_threshold=site_config.aggregation.detection_threshold,
    )
    for lbl, bundle_dir in result.items():
        click.echo(f"{lbl}: {bundle_dir}")


@main.command("ingest-labels")
@click.option("--date", required=True, type=str, help="UTC date (YYYY-MM-DD).")
@click.option("--labels", required=True, multiple=True, type=click.Path(exists=True),
              help="Path(s) to completed label JSON files.")
@click.pass_context
def ingest_labels(ctx: click.Context, date: str, labels: tuple[str, ...]) -> None:
    """Ingest completed label JSON files into the DuckDB database."""
    raise click.ClickException("Ingest-labels command not yet implemented.")
