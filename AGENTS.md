# AGENTS.md — repository-local guidance for AI sessions

This file is binding for any Claude / Ralph session run against this repo.
It is the durable version of conventions previously buried in
`.ralph/progress.txt`.

## Environment

UV-managed project. Do **not** use pip, conda, or manually-activated venvs.

- Install / sync: `uv sync` (add `--extra review` for notebooks, `--extra ocr` for EasyOCR fallback)
- Run the CLI: `uv run concam <command>`
- Run tests: `uv run pytest`
- Add a dep: `uv add <package>`

Python >=3.11. OpenCV is `opencv-python-headless` (no display on SLURM nodes).

## Long-running jobs

Full-day pipeline runs (`uv run concam run --date ...` on a 24 h timelapse)
take 2–4 h of wall time. Submit these via SLURM — do **not** block a session
on them:

```
sbatch slurm/run_pipeline.sh YYYY-MM-DD
# or, for resume-only runs:
sbatch slurm/regression_rerun.sh YYYY-MM-DD --from-stage detect
```

Poll `slurm/logs/` and `squeue -u $USER` for status. If you need a result
before the session ends and the job is still running, log the expected
resumption point in `.ralph/progress.txt` and stop.

## Human-review items

Some PRD items have `HUMAN REVIEW REQUIRED` in their steps (typically
anything visual: overlay alignment, clip quality, label judgement). These
cannot be auto-closed.

Do **not** skip or defer these items. The correct handling is:

1. Finish the machine half of the item.
2. Produce the artifacts the human needs to review (PNG panels, rendered
   clips, a served labeler.html URL, etc.).
3. Write explicit step-by-step review instructions into
   `.ralph/progress.txt` under a "HUMAN REVIEW REQUIRED" heading.
4. Leave `passes=false` and flag the item in the next-session priorities.

Items that have historically been deferred this way: 14 (projection
alignment), 15 (browser smoke test), 27 (contrail-length clip box),
29 (April-9 label batch — now closed), 31 (ADS-B misalignment), 32
(labeler UX).

## Tracking files

- **`.ralph/prd.json`** — small-task execution board. Source of truth for
  scope. Tracked in git. Update the completed task's entry with
  `passes=true` and the `commit` hash. Do **not** commit prd.json after
  each session; commit it once at the very end when all tasks are done.
- **`.ralph/progress.txt`** — local-only session notes (gitignored). After
  each session append: task completed, files changed, key decisions,
  blockers, and the next step. Session headings are
  `## [YYYY-MM-DD] Title — STATUS`.

Tag each session by the Claude model running it when non-trivial
architectural decisions are made, so future sessions can weight the
advice appropriately.

## Branching and commits

- Work on `main` is the norm. Do **not** create feature branches unless
  the user explicitly requests one.
- Commit meaningful messages explaining the *why*, not just the *what*.
- Do **not** push. Do **not** change git remotes. Do **not** merge into
  `main` from another branch (you're already on it).
- Ralph `--no-verify` is forbidden unless the user explicitly allows it
  for a specific commit.

## Subagents

Use lightweight subagent models (Haiku via the `general-purpose` or
`Explore` agent types) for targeted exploration — grepping for a symbol,
summarizing a reference document, enumerating file changes. Reserve Opus
for architectural work.

Avoid spawning a subagent to do what a single `Grep` + `Read` could do in
the main session; the context-window savings are not worth the
round-trip.

## Scope discipline

Work on exactly one Ralph-sized task per session, or one tightly related
pair only if they must land together. Prefer risky architectural and
cross-module work ahead of cleanup or polish.

Before finishing, run `uv run pytest` and confirm the full suite passes.
If a new test is added, it must be part of the committed diff.

## Human-labeled ground truth

Human label JSONs (exported from the browser labeler) are the canonical
ground truth for the detector. They are tracked in git under `labels/`
with the naming convention `labels/YYYY-MM-DD_{labeler_id}.json`. The
operational copy at `output/validation/detection/.../label_batch/labels.json`
is the gitignored working copy that the HTML labeler writes to during a
session; always promote a fresh export into `labels/` before any
model-tuning session that depends on it.

## Calibration / data paths

- Raw video: `/net/d16/data/contrail-camera/YYYY_MM_DD_HHMM_HHMM.mp4`
  (daily timelapse 0000_2359 is 1 fps; raw segments are 4 fps)
- ADS-B (feder): `/home/mcast/data/feder`
- Camera calibration: `LAE_skycam/calibration/pointpicker_calibration_estimate.npz`
- Site POIs (projection-alignment ground truth): `LAE_skycam/.../MITSC_POIs.csv`

None of these are copied into the repo. If a path moves, update
`configs/mit_green_building.yaml` rather than hard-coding the new path.

### TEMPORARY: feder 1.0.0 workarounds

`feder` is pinned exactly to `1.0.0`. The release has two known issues we
work around locally; **both must be removed once the maintainer ships a
fix** — do not extend or normalize them as the canonical integration
pattern:

1. The wheel ships an empty `Requires-Dist`, so `lz4` and `pandas` are
   declared directly in `pyproject.toml` to plug feder's transitive
   imports. Drop these declarations once feder fixes its packaging
   metadata.
2. `feder.common.db.DB.__init__` opens each per-day SQLite file with
   `?mode=ro` only, which fails with "attempt to write a readonly
   database" for any reader who is not `mcast` on the Hex data files.
   `concam.adsb._patch_feder_readonly_open` shadows
   `feder.common.db.sqlite3` with a wrapper that appends `&immutable=1`
   to the URI before connect. The helper hard-asserts
   `feder.__version__ == "1.0.0"` so a future `uv lock` that picks up a
   different feder fails loudly instead of silently de-aligning.
   Delete the helper, its call site in `load_flights`, and bump the pin
   once feder ships a release with `&immutable=1` applied upstream.

If you hit feder integration friction, file an issue with the
maintainer; do not paper over it with another reach into private
internals here.
