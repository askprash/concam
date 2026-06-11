# ADR-0004: Overlap flagging in pixel space, not 3D distance

Date: 2026-06-11 · Status: accepted

## Context

The public manifest's `overlap_episode_ids` was always empty: overlap was
keyed on 3D aircraft separation, but attribution confusion happens in *pixel*
space — two flights 80+ km apart in 3D can project onto near-identical pixel
tracks. The eval-phase attribution analysis (2026-04-09) found per-frame
attribution itself is sound (only 0.17% of detections have a rival that is
both closer and angle-better), but 9 episode pairs fired simultaneous
detections with line midpoints ≤150 px apart — one physical contrail credited
to two flights — with zero downstream visibility.

## Decision

`build_public_bundle.py` flags episodes via `sustained_overlap_ids`: a pair
is flagged when the **median pixel separation of their flight tracks over the
temporal overlap** is ≤ 100 px (median, so transient perpendicular crossings
don't flag; same-transponder pairs skipped). Flagged ids populate
`overlap_episode_ids` and per-episode `is_overlap`, which the labeler already
renders as the yellow "overlap" badge.

Threshold provenance: 2026-04-09 sensitivity sweep flagged 8/17/30/40% of 665
episodes at 60/100/150/200 px; confirmed double-credits sat at ≤150 px line
separation and per-frame two-flight ambiguity within 100 px was 1.9%. 100 px
(17% flagged) balances signal against badge noise; not yet validated against
adjudicated double-credit ground truth.

## Consequences

- Reviewers see which passes might share one physical contrail; label
  adjudication can treat disagreements on flagged pairs as possible
  same-contrail/different-flight cases.
- Exclusive per-frame detection claiming (Hungarian assignment of lines to
  flights) remains an open follow-up — flagging surfaces the ambiguity but
  does not resolve it.
