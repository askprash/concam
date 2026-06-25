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

Suggested order: **E0 → E1(+E2) → E4 → E5**, with E3 as an early spike folded
into E1. E5's segmenter is the only path that *unifies* detector + overlay
(masks → polylines), so it's the long-term destination, not the next step.

---

## E0 — Evaluation harness upgrade *(prerequisite)*

- **Tier:** enabler · **Scope:** S–M · **Labels:** uses existing ~2500
- **Why:** pooled ROC-AUC hides the failure modes we actually care about
  (per-day collapse, confident FPs, thin/faint recall). No held-out split today,
  so any learned model (E5) can't be trusted.
- **Approach:** extend `scripts/eval_detector_truth_tables.py` toward a
  GVCCS-style protocol — per-day + pooled, a fixed train/val/test day split,
  precision-recall curves, and a "confident-FP" slice (negatives scoring high).
  If/when we have pixel or polyline labels, add Dice/mIoU + instance AP.
- **Open questions (grill-me):** Which days are the test set, and do we freeze
  them now? Is episode-level the right unit, or do we need frame/pixel-level
  ground truth (which we don't have yet — does E0 include a labeling push)? What
  single number, if any, do we optimize?
- **Refs:** `scripts/eval_detector_truth_tables.py`, GVCCS metric protocol
  (ESSD 18:1037, 2026), `memory/project_cv_ceiling_research.md`.

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

## Related, non-research items (park here so they aren't lost)

- **Low-activity-day coverage** (04-03 0% recall, 04-19 near-chance) — likely
  *resolved as a side effect* of E1/E2, but track it as an explicit acceptance
  criterion, not a separate epic. See
  `memory/project_detection_tuning_blocked_on_labels.md`.
- **Ribbon tuning** (`CONTRAIL_RIBBON_DECAY_MS` 45s, half-width 16px) — pure
  polish on the shipped overlay; not an epic, fold into normal labeler tweaks
  once there's user feedback from the live 04-09 preview.
