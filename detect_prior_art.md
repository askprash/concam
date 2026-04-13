# Detection prior-art survey (2026-04-13)

Written while pausing PRD item 6 (detection validation) to redesign the
detector. The current `concam.detection` module scored 3/7 real contrails
at 0/13 false positives on the April 8 labeling pass — useful precision,
43% recall. Investigation of sibling codebases under `/home/prash/contrails/`
showed four core techniques we are missing; a fifth (color-space
preprocessing) is novel and untested anywhere.

## Sibling-code survey

### 1. camera-flight-overlay (production)
Path: `/home/prash/contrails/camera-flight-overlay/contrail_labeler/utils/detection_utils.py`

- **Rotated ROI** via `cv2.minAreaRect()` + `cv2.boxPoints()`. Expandable by
  `border_px`. Not axis-aligned bbox like we currently have.
- **Frame-difference preprocessing** before Canny: subtracts the prior frame
  to kill static clouds, buildings, and the overlay text. Grayscale diff,
  Gaussian blur, no color-space work.
- **Adaptive percentile Canny thresholds:** p96 for low, p99.5 for high, computed
  on the masked ROI pixel distribution. Not fixed 50/150.
- **Angle-constrained Hough scoring:** counted lines must align with the
  flight-path vector within ±8°. Decision is `score ≥ 2 AND long_lines ≥ 2`.
- **Long-line threshold:** lines ≥ 40 px only.
- Quality: production, tuned in notebooks, running in LAE_skycam pipeline.

Quotable entry points:
- `get_directional_rectangle()` — builds the rotated rect from velocity vector.
- `resize_rect_polygon()` — grow/shrink the rect by a pixel delta.
- Canny+Hough path: lines 345-427.

### 2. groundcam_contrail_observatory (production, mirror)
Path: `/home/prash/contrails/groundcam_contrail_observatory/utils/detection_utils.py`

Identical to camera-flight-overlay, plus:
- **Frangi ridge filter** path: `skimage.filters.frangi(gray_diff, sigmas=(1,2,3), black_ridges=False)`.
  Purpose-built for bright-on-dark ridges; alternative to Canny.
- Key entry point: `apply_frangi_to_rectangles()` (Frangi branch 231-274).

### 3. contrail-review-pipeline (tuning harness)
Path: `/home/prash/contrails/contrail-review-pipeline/src/contrail_review_pipeline/detect/directional_lines.py`

Refactored version of the same pipeline with ~20 tunable parameters exposed
via ipywidgets sliders. No new algorithms. Useful as a pattern for how our
sweep script should expose parameters.

### 4. LAE_skycam diagnostic notebooks
Paths:
- `/home/prash/contrails/LAE_skycam/stream_save/interactive_detection_tuning.py`
- `/home/prash/contrails/LAE_skycam/stream_save/SWR22D_Canny_Hough_Tuning.ipynb`

Exploratory 6-panel comparison renderer. No new detection methods.

### What's NOT in any sibling
- No color-space preprocessing (HSV V, LAB L, blue-channel subtraction).
- No ML / CNN detectors.
- No axis-aligned ROI code — we're alone on that.

## User's architectural feedback (2026-04-13)

From the review of the diagnose PNG on April 8 labels:

1. **Rotated ROI, properly.** The oriented rectangle must be rotated along
   the flight-path vector, not wrapped in an axis-aligned bbox. The ROI
   should be long along-track and narrow across-track, sized to the
   expected contrail geometry behind the aircraft.
2. **Color-space experiment.** Contrails are near-white against blue sky;
   grayscale throws away that contrast. V (HSV), L (LAB), or "whiteness"
   (min of R,G,B) might recover signal that Canny-on-grayscale misses.
   This was not done anywhere in the siblings.

User's verdict: "We really need to get this part right because without this
the whole project is broken."

## Gap analysis — what mit-concam-pipeline needs

| Technique                              | Status here          | In siblings |
|----------------------------------------|----------------------|-------------|
| Rotated ROI mask                       | Missing (axis-aligned) | Yes        |
| Frame-difference preprocessing         | Missing              | Yes         |
| Adaptive percentile Canny thresholds   | Missing (fixed)      | Yes         |
| Angle-constrained Hough scoring        | Missing              | Yes         |
| Long-line threshold                    | Missing              | Yes         |
| Frangi ridge filter path               | Missing              | Yes (groundcam) |
| Color-space preprocessing (HSV/LAB/whiteness) | Missing       | No — novel  |
| Temporal median / multi-frame integration | Missing           | No — novel  |

## Recommended redesign path

1. `/grill-me` session to resolve open decisions:
   a. Port camera-flight-overlay wholesale vs clean rewrite with same techniques?
   b. Temporal frame diff as hard requirement vs per-frame fallback (timelapse
      is 1 fps; raw is 4 fps — diff behaviour differs)?
   c. Color-space preprocessing: in-scope for this cycle, or deferred?
   d. Frangi filter: in-scope, or deferred?
   e. Scoring contract: continuous score (what we have) vs sibling's discrete
      `score≥2 AND long_lines≥2` gate?
   f. Item 7 in PRD: reopen, or leave passes=true and add a new supersedes item?

2. Record answers in a new PRD item (or amend existing item 6).

3. Implement the redesign:
   - Port directional-rect utilities from camera-flight-overlay.
   - Add temporal frame-diff preprocessing (with per-frame fallback).
   - Adaptive percentile Canny thresholds.
   - Angle-constrained Hough scoring (±8° from path vector).
   - Optional: Frangi branch, color-space branch.

4. Re-run the sweep on batch-1 labels (and any additional batches collected)
   with the new detector to re-tune parameters and threshold.

5. Update `configs/mit_green_building.yaml` with the new parameters and
   mark PRD item 6 passes=true.

## Reference snippets ready for reuse

- Rotated rect construction:
  `camera-flight-overlay/contrail_labeler/utils/detection_utils.py:get_directional_rectangle`
- Rotated rect masking + Canny+Hough:
  `camera-flight-overlay/contrail_labeler/utils/detection_utils.py:345-427`
- Frangi alternative:
  `groundcam_contrail_observatory/utils/detection_utils.py:apply_frangi_to_rectangles`
- Parameter-exposing tuning harness (UI pattern to mimic in our sweep):
  `LAE_skycam/stream_save/interactive_detection_tuning.py:129-204`

## Current-session artifacts kept

- `output/validation/detection/2026-04-08/manifest.json` — 20 candidates, batch 1
- `output/validation/detection/2026-04-08/labels.json` — 7 positive, 13 negative
- `output/validation/detection/2026-04-08/diagnose_best_combo.png` — per-candidate panels
- `output/validation/detection/2026-04-08-batch2/manifest.json` — 20 more candidates (batch 2, unlabeled)

Batch-2 labeling is **paused** pending the detector redesign — no point
labeling more data for a detector we're about to rewrite.
