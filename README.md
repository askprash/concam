# MIT ConCam contrail-detection pipeline

A camera ("ConCam") on the roof of the MIT Green Building records the sky. This
pipeline ingests that video, reads the on-frame timestamp, detects contrails,
cross-references nearby aircraft from ADS-B flight tracks, groups the detections
into per-flight "Episodes", and builds a web "labeler" bundle so human reviewers
can mark ground truth.

For the domain vocabulary and architecture in depth, read
[`CONTEXT.md`](CONTEXT.md) — it is the authoritative glossary, and this README
intentionally stays shallow and points there.

## Status

Work in progress, shared so student contributors can keep improving it. The
detector is not yet well calibrated (it both misses real contrails and fires on
non-contrails — see [`docs/labeling_guide.md`](docs/labeling_guide.md)). Expect
rough edges, open `TODO`s, and parameters that still need tuning.

## Requirements & setup

This project is **UV-managed**. Do not use pip, conda, or hand-activated
venvs — see [`AGENTS.md`](AGENTS.md) for the full conventions.

- Python: `>=3.12` (declared in `pyproject.toml`; `.python-version` currently
  pins 3.14).
- Install dependencies:

  ```bash
  uv sync                  # core pipeline
  uv sync --extra ocr      # adds EasyOCR fallback (heavy, optional)
  uv sync --extra review   # adds JupyterLab + matplotlib for the notebooks
  ```

- Run the test suite:

  ```bash
  uv run pytest
  ```

  Many tests are gated on data that is not in the repo (real camera
  calibration, video, the feder ADS-B store) and will **skip** when run off the
  data machine. A fresh clone runs the rest of the suite cleanly.

## Data prerequisites (not in the repo)

A fresh clone passes the (non-data-gated) tests but **cannot run the full
pipeline** without external inputs that are not committed:

- **Camera calibration `.npz`** — `configs/mit_green_building.yaml` hard-codes
  `npz_path: /home/prash/contrails/LAE_skycam/calibration/pointpicker_calibration_estimate.npz`.
  You must repoint this at a calibration file you have.
- **Raw video** — expected under `/net/d16/data/contrail-camera/...` (daily
  timelapse is 1 fps; raw segments are 4 fps). Override per-run with `--video`.
- **feder ADS-B store** — expected at `/home/mcast/data/feder`.

When a path moves, edit `configs/mit_green_building.yaml` rather than
hard-coding it elsewhere (see the "Calibration / data paths" section of
[`AGENTS.md`](AGENTS.md)).

## Quickstart

The CLI is installed as the `concam` console script (entry point
`concam.cli:main`):

```bash
# See the plan for a date without running anything (works without data):
uv run concam run --date 2026-04-09 --dry-run

# Run the full pipeline for one UTC date (needs video + ADS-B + calibration):
uv run concam run --date 2026-04-09

# Resume from a stage using cached earlier outputs:
uv run concam run --date 2026-04-09 --from-stage detect

# Build per-labeler review bundles after a run:
uv run concam bundle --date 2026-04-09 --labelers alice --labelers bob

# Ingest completed label files and check inter-rater agreement:
uv run concam ingest-labels --date 2026-04-09 --labels labels/2026-04-09_prash.json
uv run concam agreement --date 2026-04-09
```

Outputs land under `output/<date>/` (intermediate `.jsonl`/`.json` per stage
plus `pipeline.duckdb`). A full-day run takes hours — submit it via SLURM
(`sbatch slurm/run_pipeline.sh YYYY-MM-DD`) rather than blocking a session.

## Repo layout

| Path          | What's there |
|---------------|--------------|
| `concam/`     | The package: `cli.py`, `pipeline/`, `ocr/`, `adsb/`, `projection/`, `detection/`, `aggregation/`, `bundle/`, `storage/`, `ingest/`, etc. |
| `configs/`    | Site YAML config (`mit_green_building.yaml`) — the one place for paths and tuned parameters. |
| `scripts/`    | ~30 helpers for sweeps, HPO, validation, OCR-template generation, and publishing. Not part of the core CLI. |
| `slurm/`      | `sbatch` job scripts for full-day runs, backfills, HPO, and clip rendering. |
| `notebooks/`  | Exploration / parameter-tuning notebooks. |
| `labels/`     | Human-labeled ground truth JSONs, `labels/YYYY-MM-DD_<labeler>.json`. |
| `docs/`       | Labeling guide, public-labeler notes, UROP assignment, PTS-drift bug writeup. |
| `tests/`      | `pytest` suite (some tests data-gated). |

## How it works

The `concam run` command executes six decoupled stages, each reading/writing a
named file under `output/<date>/` so `--from-stage` can resume:

1. **ocr** — decode frames and OCR the burned-in timestamp overlay (the
   wall-clock truth, since container PTS can drift). Template-match primary,
   EasyOCR fallback.
2. **adsb** — load nearby flight tracks from the `feder` store within an
   altitude band and radius (config-driven).
3. **project** — project each ADS-B ping into camera pixels using the camera
   calibration, yielding a per-flight ROI and path vector.
4. **detect** — run the contrail detector along each flight's path: a rotated
   ROI around the path, adaptive (percentile) Canny edges, then Hough lines
   constrained to the flight-path angle. Color-channel transforms for
   experimentation (NRBR, CIELAB b\*, etc.) live in
   `concam/detection/transforms.py`.
5. **aggregate** — smooth and gap-split per-flight detections into Episodes.
6. **store** — write Episodes into `pipeline.duckdb`.

See [`CONTEXT.md`](CONTEXT.md) for what each term means and how the clusters
(detection, OCR, ADS-B, projection) are structured.

## Labeling / ground truth

Human labels are the canonical ground truth for tuning the detector. Reviewers
use a browser labeler served from a bundle — start with
[`docs/labeling_guide.md`](docs/labeling_guide.md) (and
[`docs/public_labeler.md`](docs/public_labeler.md),
[`docs/urop_label_assignment.md`](docs/urop_label_assignment.md)). Exported
label JSONs are committed under `labels/` as
`labels/YYYY-MM-DD_<labeler_id>.json` and ingested with
`uv run concam ingest-labels`.

## Notebooks

The `notebooks/` directory holds exploration tools; `filter_playground.ipynb` is
the computer-vision parameter-tuning playground for detection. Install the
notebook deps with `uv sync --extra review`.

## Contributing

Read [`AGENTS.md`](AGENTS.md) for conventions. In short: UV only, work on `main`
unless told otherwise, and run `uv run pytest` (full suite green) before you
commit. A known intermittent issue is documented in
[`docs/pts_drift_bug.md`](docs/pts_drift_bug.md).
