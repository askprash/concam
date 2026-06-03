# CONTEXT — domain & architecture glossary

Shared vocabulary for the MIT Green Building sky-camera contrail detection and
labeling pipeline. This file names the concepts the code is organised around so
that future work (human or AI) uses one term per idea. Seeded during the
detection-deepening review; extend it as new concepts crystallise.

## Domain concepts

- **Frame** — one decoded image from a timelapse video. Daily timelapse is 1 fps
  (`..._0000_2359.mp4`); raw segments are 4 fps. A frame's wall-clock time comes
  from the burned-in **timestamp overlay** (read by the OCR cluster), not the
  container PTS, which can drift.
- **ROI / Rect** — an axis-aligned bounding rectangle (`concam.projection.Rect`)
  used as a cheap crop before detection.
- **Polygon** — the *rotated* detection rectangle (4×2 array, full-frame pixel
  coords) oriented along a flight's ground track. The rotated mask, not the ROI,
  does the along-track selection.
- **Path vector (`path_vec`)** — the flight-path unit vector `(vx, vy)` in image
  pixels; detection only keeps Hough lines aligned with it (`±angle_tolerance_deg`).
- **Detection** — running the contrail detector on one `(frame, ROI, polygon,
  path_vec)`; yields a `DetectionResult` (score, pixel_line, line counts,
  `contrail_length_px`).
- **Contrail length** — along-track pixel span of the aligned long Hough lines,
  measured directly (`detect`) or by adaptive ROI growth (`grow_contrail_length`).
- **Candidate** — a stored row in a validation **manifest** (`manifest.json`):
  an ROI + `pixel_x/y` + `path_dx/dy` + saved crop PNG, used to re-run the
  detector against human labels offline.
- **Episode** — a contiguous run of detected frames for one flight, after a
  rolling-median smoothing gate and gap-splitting (`concam.aggregation`).
- **Flight / ADS-B ping** — aircraft trajectory data from `feder`; pings are
  projected to pixels via the camera **calibration** (`concam.projection`).

## Architecture concepts (detection cluster)

The detector keeps the edge+Hough math in one place; callers route through it.

- **Detection Pass** (`concam.detection.run_detection_pass -> DetectionPass`, in
  `concam/detection/_core.py`) — the single canonical application of the
  detection kernel (preprocess via `_prepare_base` → rotated mask → adaptive
  Canny + pixel floor → Canny → angle-constrained Hough). `DetectionPass`
  carries every intermediate array (base, mask, edges, line sets, length).
  `detect` scores from it, `grow_contrail_length` grows from it, and `explain`
  returns it for visualisation — so a panel cannot diverge from what the
  detector saw.
- **explain()** — `detect`'s computation without scoring; returns the
  `DetectionPass`. Visualisers must render this, never re-derive edges by hand.
- **Detection metrics** (`concam/detection/metrics.py`) — `mann_whitney_auc`,
  `youden_threshold` (midpoint operating-point), `youden_at` (point eval),
  `rank_metric` (deterministic ranking when a metric is undefined). The single
  home for detector-evaluation math used by the sweep/HPO scripts.
- **Candidate geometry** (`concam/detection/geometry.py`,
  `candidate_geometry -> CandidateGeometry`) — reconstructs `(Rect, polygon,
  path_vec, center)` from a manifest candidate + crop, as if the crop were the
  full frame.
- **Detection panels** (`concam/detection/viz.py`) — `compose_grid` (pure tile
  compositor) and `render_detection_panels(DetectionPass)` (base / edges /
  overlay, all derived from the pass).
- **Frame decode** (`concam/video.py`) — `decode_frames` (random-access seek)
  and `decode_frames_sequential` (forward scan); I/O only, returns
  `{index: BGR}`, callers crop. The two strategies agree on in-range indices but
  differ past the end of the video (seek clamps to last frame; sequential
  omits) — a documented, not-yet-reconciled inconsistency.

## Known latent inconsistencies (documented, deliberately not auto-"fixed")

- `grow_contrail_length` runs the kernel with `apply_exclusion=False`, so the
  timestamp-exclusion region is not masked during length growth — only `detect`
  excludes it. Threaded explicitly via `apply_exclusion` at the call site.
- `decode_frames` vs `decode_frames_sequential` past-end behaviour (above).
