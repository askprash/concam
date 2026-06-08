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
- **Detection parameter single-source** — `DetectionConfig` and
  `AggregationConfig` (`concam/config.py`) are the one home for every detection
  and aggregation parameter. Dataclass defaults are kept equal to the base site
  YAML (`configs/mit_green_building.yaml`), so a bare `DetectionConfig()` is
  honest about production behaviour, and `tests/test_config.py` fails if code and
  YAML ever drift. Sweep scripts (`detection_validation_sweep.py`) derive their
  ROI from the loaded `det_cfg` rather than module constants; the score-function
  catalogue in `detection_score_sweep.py` keeps fixed normalization constants on
  purpose (each variant must be compared at one common reference geometry).

## Architecture concepts (OCR cluster)

Every Frame's wall-clock truth is the `MM/DD/YYYY HH:MM:SS` overlay the camera
burns into the corner; reading it is a two-engine problem behind one seam.

- **Timestamp engine** (`concam.ocr.engines`) — `TimestampEngine` Protocol
  (`read(frame) -> EngineRead | None`). A `None` means "engine unavailable /
  produced nothing"; an `EngineRead` with `parsed_dt=None` means "ran but text
  didn't parse" — the distinction the composition needs. Two adapters satisfy
  it: `TemplateMatchEngine` (primary, dependency-free, classifies fixed glyph
  slots by cross-correlation against `templates.npz`) and `EasyOcrEngine`
  (fallback, heavy/optional, **lazily** imports `easyocr` only on first use).
  `FixedFormatTimestampReader` *composes* them — primary, then fallback only when
  the primary fails to parse or its confidence is below threshold — instead of
  hardcoding EasyOCR inline. Two adapters = a real seam.
- **OCR preprocessing** (`concam.ocr._preprocess`) — the single home for the
  pixel pipeline (crop → binarize → slot extraction → glyph normalization) and
  slot-layout constants. Both the runtime engine and the one-shot template
  generator (`scripts/generate_ocr_templates.py`) import it — load-bearing,
  because the template bank is only valid if built with *exactly* the
  preprocessing applied at match time.
- **Timestamp cleanup** (`concam.ocr._fallback_clean`) — canonicalises EasyOCR's
  loose tokens. *Resolved correctness note:* `_canon_time` once honoured an
  AM/PM token's hour range but never converted to 24-hour, corrupting every PM
  and midnight reading from the fallback; it now does the 12→24 conversion
  (`tests/test_ocr_fallback_clean.py` pins the truth table).

## Architecture concepts (ADS-B cluster)

The ADS-B loader (`concam/adsb`) keeps the `feder` dependency behind a port/adapter
seam so the convert→filter→upsample pipeline is testable without the live store.

- **`FlightSource`** (port, Protocol) — single method
  `fetch(t_start, t_end, bbox, min_altitude_ft) -> Iterator[RawTrajectory]`,
  exposing exactly the time-window / bounding-box / min-barometric-altitude
  pre-filter the feder query applies.
- **Raw types** (slotted, feder-native units) — `RawPoint(time, lat, lon, alt_ft,
  alt_gnss_ft)` (altitudes in **feet**, mapped 1:1 from feder `Point.alt` /
  `Point.alt_gnss`) and `RawTrajectory(callsign, transponder_id, aircraft_type,
  orig, dest, points)`.
- **`FederFlightSource`** (production adapter) — the only place that imports
  `feder`; owns the version-pinned readonly monkeypatch, the `FEDER_DATA_DIR`
  env, the `FlightQuery(...).with_bounds().spatially_crosses().filter_waypoints()
  .run()` chain, and the feder→Raw mapping.
- **`RecordedFlightSource`** (fake) — replays an in-memory `RawTrajectory` list or
  a committed JSON trace (`tests/fixtures/adsb_raw_trace.json`), so the
  conversion/altitude-policy/radius-filter/1 s-upsample logic runs without feder.
  `load_flights(date, config, timezone=None, source=None)` defaults `source` to
  `FederFlightSource`, so the production path is unchanged.

## Architecture concepts (projection cluster)

- **`Calibration`** (`concam/projection`) holds the camera intrinsics/extrinsics
  and precomputes the ENU rotation + ECEF→ENU pyproj transform from plain arrays
  (no file I/O in the constructor). `load_calibration` is a thin `.npz` reader
  over it.
- **`synthetic_calibration()`** — a module factory that builds a valid
  `Calibration` in memory (simple pinhole at a real GPS; `cam_ecef` derived from
  `camera_gps` via the same pyproj transform, so the two are mutually
  consistent; identity rotation ⇒ camera frame = ENU). It exists so projection
  geometry (`_gps_to_enu`, `project_pings`, `flight_path_vector`,
  `project_pixel_to_meters`) is unit-testable against hand-computable geometry
  without the real `.npz`. The 12 real-camera tests stay `.npz`-gated because
  they assert MIT-Green-Building-specific pixel coordinates.

## Known latent inconsistencies (documented, deliberately not auto-"fixed")

- `grow_contrail_length` runs the kernel with `apply_exclusion=False`, so the
  timestamp-exclusion region is not masked during length growth — only `detect`
  excludes it. Threaded explicitly via `apply_exclusion` at the call site.
- `decode_frames` vs `decode_frames_sequential` past-end behaviour (above).
