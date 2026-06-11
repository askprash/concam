# ADR-0003: Label provenance — episode IDs are manifest-generation-scoped

Date: 2026-06-11 · Status: accepted

## Context

Label exports reference episodes by ordinal `episode_id`, but
`build_public_bundle.py` renumbers episodes every time the pipeline re-runs
(only 2 of 434 mappable IDs survived the 2026-05-01 rerun of 2026-04-09).
Joining label files to manifests across that boundary silently mismatches
flights. This corrupted the May-2026 HPO round and produced the false
"labels unreliable / lrsand outlier" conclusion (see
docs/label_reliability.md — after remapping, lrsand vs reviewer-1 κ = 1.00).

## Decision

1. **Treat (date, manifest generation) as part of a label file's identity.**
   A label export is only joinable to the manifest generation the reviewer
   saw; verify with the peak_score↔label AUC sanity check (~0.9 right space,
   ~0.5 wrong space).
2. **Archive episode skeletons before re-running labeled dates** under
   `labels/archive/<date>_manifest_<gen-date>_episodes.json` (episode_id,
   transponder_id, callsign, onset, end, peak_score) so old labels can be
   remapped by `(transponder_id, onset)`.
3. **Tune only against the consolidated set** produced by
   `scripts/build_reliable_label_set.py` → `labels/derived/reliable_labels.json`,
   which encodes the per-file provenance verdicts and consensus rules
   (`unsure` dropped; conflicting definite votes excluded for adjudication).

## Consequences

- 2026-04-15_reviewer-1 (19 labels) is unusable — no archived skeleton exists
  for its generation. The cost of not archiving is permanent label loss.
- Follow-up (open): make episode identity content-addressed
  (`transponder_id` + onset) in the manifest/export schema so renumbering
  becomes impossible by construction.
