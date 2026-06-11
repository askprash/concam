# Label reliability analysis — 2026-06-11

Which committed label files are trustworthy, and in which episode-ID space.
Supersedes the "labels unreliable / lrsand outlier" conclusion from the May
HPO round — that conclusion was an **episode-ID-space artifact**, not a
labeler-quality problem.

## The core problem: three episode-ID spaces

Public manifests are regenerated whenever the pipeline re-runs, and
`build_public_bundle.py` renumbers episodes from the projections file. A label
export is only meaningful against the manifest generation it was labeled on.

| Space | What | Who labeled in it |
|---|---|---|
| current | manifests from the 2026-05-01+ outputs (June-9 regen reproduced IDs deterministically — verified by score↔label AUC ≈ 0.9) | lrsand (04-09, 04-19), thendo (03-29, 03-30, 04-08; June exports) |
| April-21 public (04-09 only, 535 eps) | archived: `labels/archive/2026-04-09_manifest_2026-04-21_episodes.json` | reviewer-1 (04-09), thendo (04-09) |
| label-batch (04-09, 35 candidates) | `output/validation/detection/2026-04-09/label_batch/candidates.json` | prash (04-09) |

Only 2 of 434 mappable April-21 episodes kept the same ID after the May-1
rerun — *every* cross-space comparison by raw `episode_id` was garbage.

## Agreement after remapping to current space (2026-04-09)

| Pair | n | raw agree | κ (2×2, unsure dropped) |
|---|---|---|---|
| lrsand vs reviewer-1 | 28 | 1.00 | **1.00** |
| lrsand vs thendo | 150 | 0.55 | **0.84** |
| reviewer-1 vs thendo | 28 | 0.79 | 1.00 |

lrsand — previously flagged as the outlier — agrees *perfectly* with
reviewer-1 and strongly with thendo. The 0.55 raw agreement with thendo is
almost entirely thendo's heavy `unsure` usage (59/150 on that file), which the
labeling guide has since discouraged.

Sanity check (detector peak_score vs label, current space): thendo 04-08
AUC = 0.90, lrsand 04-09 AUC = 0.91 → labels strongly correlate with the
detector in the correct ID space. The April-cohort files in raw current space
read AUC 0.41–0.60 → wrong space, as expected.

## Verdicts

**Usable directly** (current space): `2026-03-29_thendo`, `2026-03-30_thendo`,
`2026-04-08_thendo`, `2026-04-09_lrsand`, `2026-04-19_lrsand`
(lrsand's 04-19 labels are timestamped 2026-05-02, *after* the May-1 rerun).

**Usable after remap** via the archived April-21 skeleton:
`2026-04-09_thendo`, `2026-04-09_reviewer-1`.

**Excluded**: `2026-04-09_prash` (label-batch ID space; remappable via the
batch `candidates.json` if ever needed), `2026-04-15_reviewer-1` (labeled
2026-04-22, before the May-1 rerun; no archived manifest for 04-15 exists, so
it cannot be remapped — 19 labels lost).

## Consolidated set

`scripts/build_reliable_label_set.py` → `labels/derived/reliable_labels.json`:
**1,204 consensus labels across 5 dates** (88 multi-labeler episodes on 04-09;
7 conflicting episodes excluded for adjudication). `unsure` votes are dropped;
conflicting definite votes exclude the episode and are listed under
`conflicts`.

## Process fix going forward

The June-8/9 near-miss (thendo exported hours before a mass manifest
regeneration) shows the failure mode is still live. Two mitigations now exist:

1. The archived manifest skeleton pattern (`labels/archive/`) — archive the
   episode skeleton whenever a labeled date's pipeline is re-run.
2. Episode identity should eventually be content-addressed
   (`transponder_id + onset`) rather than ordinal — tracked as a follow-up.
