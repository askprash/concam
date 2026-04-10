"""Command-line interface for the concam pipeline."""

from __future__ import annotations

import click
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "mit_green_building.yaml"


@click.group()
@click.option("--config", default=str(DEFAULT_CONFIG), show_default=True,
              type=click.Path(exists=False), help="Path to site YAML config.")
@click.pass_context
def main(ctx: click.Context, config: str) -> None:
    """MIT ConCam pipeline — contrail detection and labeling from sky camera footage."""
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
@click.pass_context
def run(ctx: click.Context, date: str, output_dir: str,
        from_stage: str | None, dry_run: bool) -> None:
    """Run the full detection pipeline for a single UTC date."""
    from datetime import date as date_type
    try:
        parsed_date = date_type.fromisoformat(date)
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got: {date!r}", param_hint="--date")

    click.echo(f"Pipeline date: {parsed_date}")
    click.echo(f"Config: {ctx.obj['config_path']}")
    click.echo(f"Output dir: {output_dir}")
    if from_stage:
        click.echo(f"Resuming from stage: {from_stage}")
    if dry_run:
        click.echo("[dry-run] Execution plan:")
        stages = ["ocr", "adsb", "project", "detect", "aggregate", "store"]
        start_idx = stages.index(from_stage) if from_stage else 0
        for stage in stages[start_idx:]:
            click.echo(f"  - {stage}")
        return

    raise click.ClickException("Pipeline stages not yet implemented. Run after implementing individual modules.")


@main.command()
@click.option("--date", required=True, type=str, help="UTC date (YYYY-MM-DD).")
@click.option("--labelers", required=True, multiple=True, help="Labeler IDs.")
@click.option("--overlap-fraction", default=0.2, show_default=True, type=float,
              help="Fraction of episodes assigned to both labelers for inter-rater calibration.")
@click.pass_context
def bundle(ctx: click.Context, date: str, labelers: tuple[str, ...],
           overlap_fraction: float) -> None:
    """Generate per-labeler annotation bundles for a given date."""
    raise click.ClickException("Bundle command not yet implemented.")


@main.command("ingest-labels")
@click.option("--date", required=True, type=str, help="UTC date (YYYY-MM-DD).")
@click.option("--labels", required=True, multiple=True, type=click.Path(exists=True),
              help="Path(s) to completed label JSON files.")
@click.pass_context
def ingest_labels(ctx: click.Context, date: str, labels: tuple[str, ...]) -> None:
    """Ingest completed label JSON files into the DuckDB database."""
    raise click.ClickException("Ingest-labels command not yet implemented.")
