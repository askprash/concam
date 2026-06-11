# ADR-0001: Categorical persistence labels, gated on contrail visibility

Date: 2026-06-11 · Status: accepted

## Context

The labeler had a 1–5 "persistence" slider per episode. Reviewers had no
shared rubric for what 3 vs 4 means, the value was set even for passes with no
contrail at all, and nothing downstream consumed it. We want persistence
information that maps to the physical question we actually care about
(does the contrail survive long enough to matter radiatively?) and that
cannot be recorded for a non-existent contrail.

## Decision

Replace the slider with two categorical options, each with hover guidance
that doubles as the operational definition:

- `short` — "You can see the contrail dissipating behind the airplane"
- `potentially_persistent` — "The end of the contrail is outside the frame"

The options are **gated**: interactable only when the auto-detector crossed
threshold for the pass OR the reviewer labeled it `contrail`; otherwise
grayed out. Export drops a persistence choice whose gate no longer holds
(label changed away from contrail), so stale choices can't leak into the
dataset. A free `measurement_km` text box autosaves alongside, autofilled by
the Measure tool's saved measurements.

Legacy `persistence_rating` (1–5) is still accepted by ingest and kept in the
DuckDB schema; new columns `persistence VARCHAR` and `measurement_km DOUBLE`
are added via `ALTER TABLE ... IF NOT EXISTS` migration.

## Consequences

- Old localStorage state and old export files remain ingestible; analyses must
  treat `persistence_rating` and `persistence` as separate generations.
- The two categories are deliberately observational (what the reviewer can
  see in-frame), not durations — frame exit is the camera's hard limit on
  observable persistence.
