# ADR-0002: Static-scene mask — manual outline ∪ edge persistence, applied post-Canny

Date: 2026-06-11 · Status: accepted (amended same day: manual SVG outline added)

## Context

Review showed most detector false positives come from the tall foreground
building: its high-contrast edges survive Canny in every frame, and whenever a
flight track aligns with one of them, the angle-constrained Hough stage emits
"contrail" lines. Two candidate fixes were considered: (a) hand-drawn
building bounds; (b) automatic detection of static regions.

## Decision

The production mask is the **union of two sources**, built by
`scripts/build_static_mask.py` into
`configs/static_mask_mit_green_building.npz` (~28.5% of frame), wired via
`DetectionConfig.static_mask_path`:

1. **Manual outline** (`--svg`, `configs/static_mask_manual_outline.svg`,
   drawn in Inkscape over a 1280×720 frame capture; straight-segment paths
   only, rasterized at frame resolution). This is the authority on building
   *volumes* — glass facades reflect sky and have weak/unstable edges, so
   edge persistence alone left holes a Hough line could thread through
   (auto-only coverage was 10.4%).
2. **Edge persistence** (fixed-threshold Canny over ~40 frames/day across the
   daylight window of two days with different skies; edge in ≥ 50% of samples;
   dilate 12 px). This still catches thin static structure poking *above* the
   drawn outline — crane booms, antennas — and adapts if small fixtures
   appear without redrawing the SVG.

The kernel suppresses masked pixels in the **edge map after Canny**, not in
the input image. Zeroing input pixels (as the timestamp exclusion does)
manufactures fresh straight edges along the mask boundary — exactly the
aligned lines we are removing. Post-Canny suppression has no boundary
artifact and leaves the adaptive-Canny percentile statistics untouched.
Because it only removes edges, scores are monotonically ≤ unmasked scores —
useful for cheap re-evaluation (episodes with old score 0 stay 0).

The labeler hatches the masked regions (manifest `exclusion_regions`, built
by `build_public_bundle.py` via `mask_to_polygons`) so reviewers can see what
the detector ignores.

## Consequences

- The mask is a site artifact like the calibration npz; rebuild it if the
  camera moves or new construction appears (two `--video` days recommended).
- `grow_contrail_length` runs with `apply_exclusion=False` and therefore also
  skips the static mask — consistent with the existing timestamp-exclusion
  behaviour (documented latent inconsistency in CONTEXT.md).
- A residual known bias: the rotated-polygon pre-Canny zeroing can still
  create polygon-boundary edges under fixed-threshold configs (mostly
  suppressed by the adaptive floor in production); flagged during testing,
  not addressed here.
