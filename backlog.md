# Backlog — detector & geometry explorations

Epic-level explorations sitting **above** `.ralph/prd.json`. Each entry is a
candidate epic that is *not yet decomposed* into Ralph-sized tasks — the
subtasks get nailed down in a "grill-me" session and only then land in
`prd.json`. Treat scope/effort here as a signal, not a commitment.

Origin: the 2026-06-25 contrail-box evaluation + background CV-ceiling research
(verdict: Canny+Hough is at its limit; see `memory/project_cv_ceiling_research.md`
and the annotated bibliography in that session). The overlay half of that
research (rectangle → path-following ribbon) is **already shipped** (commit
`008b1cf`); everything below is the *detector* and *evaluation* half.

---

## Baseline to beat (so every epic is measurable)

Current production detector (Canny edges + Hough lines, `cross_grad`
preprocessing, threshold 0.083), scored by `scripts/eval_detector_truth_tables.py`
against `labels/derived/reliable_labels.json` (9 days, 940 P / 1627 N):

- **Pooled ROC-AUC 0.870**, TPR 80.9%, FPR 14.4%, Prec 76.4%.
- Strong days ~0.90–0.96 AUC; **04-03 is 0% recall (AUC 0.500)** and **04-19 is
  near-chance (0.533)** — confident false positives up to score 0.59 on
  hand-labeled negatives.

Any epic below should report the same truth-table metrics before/after, on the
same label set, so they are directly comparable.

---

## Granularity & sequencing (my read on "does each need its own epic?")

- **E0 (evaluation harness) should go first** and is a true prerequisite — you
  cannot fairly compare E1–E5 without a metric better than pooled ROC-AUC and a
  held-out split. Small but enabling.
- **E1 (ridge filter) and E2 (score normalization) are one epic, not two.** The
  ridge response and how you normalize/threshold it are the same change to
  `_preprocess` + scoring; splitting them creates an artificial seam. Keep E2 as
  a named workstream *inside* E1.
- **E3 (LSD/CannyLines) is a spike, not an epic** — a half-day bake-off that
  either feeds E1 or gets discarded. Don't give it epic ceremony.
- **E4 (temporal track-before-detect) is its own epic** — it touches detection
  *and* episode aggregation, and is independent of which feature extractor wins.
- **E5 (learned models) is genuinely large and probably two epics** (patch-CNN
  classifier vs. thin-structure segmenter) — but they share a data-labeling and
  eval substrate, so scope that substrate once.

- **E6 (systematic sweep) largely subsumes E1 as an empirical search.** E1 is
  the *hypothesis* (ridge filter wins); E6 *tests it against everything else*.
  They share the transform-chain kernel wiring (the real prerequisite), so don't
  run them as separate builds — do the wiring once, then E6 is the experiment and
  E1 is "what we expect E6 to confirm." If you only fund one near-term thing,
  fund E6.
- **E6 and E7 are the two "launch-and-leave" autonomous-search epics** (own
  section below) — they're what gets sharded onto SLURM for half a day and comes
  back with options. E0 must land first to give them a frozen split.

Suggested order: **E0 → (wire transform chains) → E6 ‖ E7 (parallel sbatch
searches) → E4 → E5**. E3 (LSD) folds into E6's line-extraction axis. E5's
segmenter is the only path that *unifies* detector + overlay (masks →
polylines), so it's the long-term destination, not the next step.

---

## E0 — Evaluation substrate: consensus set + harness *(prerequisite)*

- **Tier:** enabler · **Scope:** S–M · **Labels:** existing, plus a coverage
  decision
- **Why:** pooled ROC-AUC hides the failure modes we actually care about
  (per-day collapse, confident FPs, thin/faint recall). No held-out split today,
  so any learned model (E5) or sweep (E6) can overfit silently.
- **⚠ Label-coverage reality (measured 2026-06-25):** of 2,567 consensus labels,
  only **88 have ≥2 reviewers agreeing — all on a single day (04-09)**, 34
  contrail / 54 not, with 7 inter-reviewer conflicts. The "extract everything ≥2
  reviewers agreed on" plan yields a tiny, single-day, imbalanced set — too
  narrow to tune against, because the detector's failure is *day-dependent*
  (04-03 is 0% recall, 04-08/09 are ~0.9 AUC). Day-diversity matters more than
  reviewer-redundancy here.
- **Approach:** extend `scripts/eval_detector_truth_tables.py` toward a
  GVCCS-style protocol — per-day + pooled, a frozen train/val/test **day** split,
  precision-recall curves, and a "confident-FP" slice (negatives scoring high).
  Emit the standard consensus subsets as artifacts: the full single-reviewer set
  (day-stratified, for sweeping) and the strict 88-episode ≥2-reviewer set (clean
  held-out sanity check). If/when we get pixel/polyline labels, add Dice/mIoU +
  instance AP.
- **Open decisions (grill-me / now):** (1) Sweep against the full 2,567
  single-reviewer set with the 88 as holdout — **recommended** — or hold out for
  more double-labeling first? (2) Is episode-level the right unit, or do we need
  frame/pixel ground truth we don't have (→ a labeling campaign)? (3) Which days
  are the frozen test set? (4) Single optimization target, or a Pareto front
  (recall on faint days vs. confident-FP rate)?
- **Refs:** `scripts/eval_detector_truth_tables.py`,
  `scripts/build_reliable_label_set.py` (votes/conflicts already computed),
  GVCCS metric protocol (ESSD 18:1037, 2026),
  `memory/project_cv_ceiling_research.md`.

## E1 — Ridge/line-filter detector (+ E2 normalization) *(cheapest real win)*

- **Tier:** 1 · **Scope:** M · **Labels:** none required
- **Why:** a contrail is a thin bright *ridge* (two parallel edges), not a step
  edge. Canny+Hough thresholds gradient magnitude globally → faint contrails
  fall below threshold (04-03 zero recall) while cloud/building step-edges fire
  (the 0.59 false positives). A ridge filter responds to the *line* geometry and
  is the workhorse of vessel/crack/road detection.
- **Approach:** wire the **existing but unconnected** `tf_frangi` (and the
  transform-chain concept) in `concam/detection/transforms.py` into
  `_preprocess` (`concam/detection/_core.py:98–113`, which today only dispatches
  `local_contrast`/`cross_grad`). Integrate the multi-scale ridge response
  *along the projected path* as the score. **E2:** background-normalize the
  response against a contrail-free band in the same ROI (z-score) instead of a
  global threshold — directly targets day-to-day variance.
- **Open questions (grill-me):** Replace `cross_grad` or chain after it? Frangi
  sigma range for our contrail widths? Does the score stay in [0,1] so 0.083-era
  thresholds/HPO still apply, or do we re-tune? Re-run HPO
  (`slurm/hpo_reliable_daytime.sh`) or hand-set? How do we keep the production
  path bit-identical for unaffected modes (the crop-replay anchor bug from
  `b23fe89`)?
- **Refs:** `transforms.py:288 tf_frangi`, `_core.py:_preprocess`,
  curvilinear-structure survey (PMC10042761).

## E3 — Robust line primitives spike *(fold into E1)*

- **Tier:** 1 · **Scope:** S (spike) · **Labels:** none
- **Why:** Canny+Hough fragments lines and produces spurious accumulator peaks;
  LSD / CannyLines are parameter-free and more robust to low contrast/noise.
- **Approach:** bake-off LSD vs. Hough as the line-extraction step on a handful
  of labeled days; keep only if it moves the truth table. Time-boxed.
- **Open questions:** Does it survive as a primary detector or only as a
  secondary feature alongside the ridge response? Straight-line bias acceptable
  given E1 already handles curvature in the score integration?
- **Refs:** LSD (Grompone von Gioi et al.), CannyLines (ICIP 2015).

## E4 — Temporal track-before-detect (grow-then-retract)

- **Tier:** 2 · **Scope:** M–L · **Labels:** none required
- **Why:** single-frame detection misses faint contrails and over-commits to
  one frame's evidence; episode aggregation currently lets `end` outrun the
  visible contrail. TBD accumulates weak evidence across frames before deciding.
- **Approach:** per-flight, maintain a hysteresis/decay state along the path
  (high threshold to start a track, lower to sustain, decay to retract);
  optionally optical-flow-advect the contrail tip between detections. This is the
  *detector-side* analog of the overlay ribbon we already ship.
- **Open questions (grill-me):** Does this live in the detect stage or the
  aggregate stage? Hysteresis thresholds — tuned or learned? Interaction with the
  PTS-drift bug (`memory/project_pts_drift_intermittent.md`) on frame ordering?
  How do we evaluate "retracts correctly" without per-frame labels (→ depends on
  E0)?
- **Refs:** track-before-detect literature (IEEE 5957387, arXiv 2309.13922).

## E5 — Learned detector (likely two epics)

- **Tier:** 3 · **Scope:** L · **Labels:** uses ~2500 consensus + likely new
- **Why:** the research's strongest expected gain. Our ADS-B path prior makes a
  small model far more data-efficient than the blind satellite/Kaggle work.
- **E5a — Patch-CNN-along-path classifier (lower risk):** tile the path ROI,
  classify each tile contrail/no-contrail with an ImageNet-pretrained backbone,
  aggregate to an episode score. Trains on episode labels we already have.
- **E5b — Thin-structure segmenter (the unifier):** fine-tune from a
  Kaggle-contrails / crack / vessel checkpoint, emitting **masks → polylines** —
  this collapses detector and overlay into one representation and gives us the
  GVCCS-style metrics for free.
- **Open questions (grill-me):** Do we have enough labels, or does E5 start with
  a labeling campaign (E5b needs pixel/polyline labels, which we don't have)?
  Compute/runtime budget on the SLURM partition — does an ML model fit the
  per-day pipeline latency? Use GVCCS / SIRTA for pretraining despite fisheye-vs-
  wide-field geometry mismatch? Build vs. keep classical as fallback?
- **Refs:** GVCCS (arXiv 2507.18330), Kaggle 1st/2nd-place writeups,
  small-data transfer-learning (PMC4890616), `memory/project_cv_ceiling_research.md`.

---

# Autonomous search epics *(designed to run unattended on SLURM)*

These two are explicitly built to be **launched in a fresh session, sharded
across sbatch jobs, left to run for ~half a day, and to come back with a ranked
shortlist of options** — not to be hand-iterated. Each must end by writing a
single merged `report.md` of the top N pipelines with per-day truth tables and
the confident-FP slice (from E0), so we can pick by eye. Both depend on E0's
frozen split and consensus subsets.

## E6 — Systematic classical-CV pipeline search ("settle the playground")

- **Tier:** 1–2 · **Scope:** L (but mostly compute, not code) · **Labels:** E0
  consensus set · **Execution:** SLURM job array, ~half a day
- **Why:** we have a 13-transform menu (`concam/detection/transforms.py`:
  `nrbr, hsv_sat_inv, lab_b, grey_excess, local_contrast, dog, tophat,
  tophat_oriented, clahe, cross_grad, temporal_diff, frangi`, …) and an old
  `notebooks/filter_playground.ipynb`, but we **never systematically settled
  which pipeline wins** — the playground was eyeballed on a few frames, and the
  production HPO only ever swept `none/local_contrast/cross_grad`. Now that we
  have a day-diverse labeled set, decide it empirically.
- **Prerequisite (shared with E1):** wire the **transform-chain dispatch** into
  the kernel. Today `_preprocess` (`concam/detection/_core.py:98–113`) hard-codes
  three modes; `transforms.py` already documents a `chain = [...]` concept and
  has all 13 transforms, just unreachable. Make `preprocessing` accept an ordered
  chain so the sweep can exercise e.g. `["temporal_diff","frangi"]` or
  `["clahe","tophat_oriented"]`.
- **Approach (smart decomposition):**
  1. Define the search space: transform **chains** (1–3 stages from the menu) ×
     line-extraction params (Canny percentiles / Hough or LSD from E3) ×
     temporal window for `temporal_diff` (Δt = 1–5 s) × detection threshold.
     Enumerate to a combo list; expect thousands of cells.
  2. **Shard** the combo list into K independent SLURM array tasks (the "smart
     decomposition" — each task owns a disjoint slice, replays `detect()` over
     the E0 episodes for its combos, writes a partial `shard_{i}.json` with
     AUC/Youden/per-day metrics). The expensive part is frame decode per episode;
     decode once per episode and reuse across combos within a shard.
  3. Merge + rank into `report.md`; promote the top few to a confirmation run on
     the held-out days.
- **Reuse, don't rebuild:** `scripts/detection_hpo.py` already does the
  replay→AUC/Youden→`sweep_report.md` ranking for the 3 wired modes;
  `slurm/hpo_reliable_daytime.sh` is the batch template (96G). E6 = generalize
  its grid to chains + temporal + sharding.
- **Open questions (grill-me):** Cap chain length at 2 or 3 (combinatorics)?
  Per-combo decode caching strategy to stay within walltime? Score still
  Mann-Whitney AUC, or switch to the E0 Pareto target? Do we let the sweep pick
  *per-day* thresholds (diagnostic) or force one global threshold (production)?
  Guard against overfitting the chain to 04-09's heavy double-labeling.
- **Deliverable:** ranked shortlist (≈top 5 pipelines) with per-day truth tables
  vs. the 0.870-AUC baseline, plus an explicit "what the winner still misses".
- **Refs:** `transforms.py`, `_core.py:_preprocess`, `detection_hpo.py`,
  `slurm/hpo_reliable_daytime.sh`, `notebooks/filter_playground.ipynb`.

## E7 — Flight-path advection alignment detector

- **Tier:** 2 (novel) · **Scope:** M–L · **Labels:** E0 consensus set ·
  **Execution:** sbatch search over offset grids, ~half a day
- **Why:** a contrail does not sit exactly on the instantaneous projected flight
  path — it **advects with the wind** and is displaced by the time we see it.
  The current detector looks for an edge *on* the path, which both misses
  drifted contrails (faint-recall days) and fires on path-aligned clutter
  (confident FPs). If instead we *search for the displacement that best aligns a
  detected line with the path*, a good alignment is strong evidence the line is
  that flight's contrail — and the best-fit displacement is a free wind estimate.
- **Approach:**
  1. In a **broadened** ROI around the projected path, extract candidate lines
     (Canny/Hough or LSD, or the E6-winning ridge response).
  2. Search a 2-D **advection offset** (dx, dy) — and optionally a small
     along-path shear — over a grid bounded by a max plausible wind displacement
     for the contrail's age/altitude. For each offset, translate the path and
     score alignment: fraction of path length that has a *parallel* detected line
     within a perpendicular tolerance.
  3. Detection score = max alignment over offsets; record `argmax` offset as the
     apparent advection vector. Build an **advected track** from it (no real wind
     data needed) and optionally render it as a second ribbon.
  4. **Cross-check (free validation):** simultaneous flights at similar altitude
     should yield consistent advection vectors — a coherent wind field is
     independent evidence the alignment is real, not coincidental.
- **Optic-flow variant (optional, slower):** estimate the advection field from
  dense optical flow between frames instead of a brute-force offset grid; flag as
  a compute spike — likely too slow for per-episode use, evaluate on a sample.
- **Open questions (grill-me):** Offset-search bound (how much can a contrail
  drift in our frame scale)? Rigid translation vs. affine/shear? Does alignment
  score *replace* or *multiply* the existing intensity score? Per-frame or
  accumulated over the episode? How to evaluate the wind byproduct without
  ground-truth wind (→ the cross-flight-consistency check is the proxy)?
- **Deliverable:** AUC/confident-FP comparison of advection-alignment vs.
  on-path detection on the E0 set, especially the faint-recall and confident-FP
  slices; a sample wind field for one day as a sanity artifact.
- **Refs:** `concam/projection` (path geometry), `tf_temporal_diff`,
  detector ROI in `concam/detection/_core.py`, optical-flow notes in
  `memory/project_cv_ceiling_research.md`.

---

## Related, non-research items (park here so they aren't lost)

- **Low-activity-day coverage** (04-03 0% recall, 04-19 near-chance) — likely
  *resolved as a side effect* of E1/E2, but track it as an explicit acceptance
  criterion, not a separate epic. See
  `memory/project_detection_tuning_blocked_on_labels.md`.
- **Ribbon tuning** (`CONTRAIL_RIBBON_DECAY_MS` 45s, half-width 16px) — pure
  polish on the shipped overlay; not an epic, fold into normal labeler tweaks
  once there's user feedback from the live 04-09 preview.
