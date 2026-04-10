# mit-concam-pipeline

## Problem Statement

The MIT Laboratory for Aviation and the Environment operates a 4K sky camera on top of the Green Building that continuously records overhead airspace. Aircraft contrails—condensation trails left by jet engines—have significant and uncertain climate impacts. A parallel satellite-based ML system detects contrails from orbit, but satellite detections carry a time lag and spatial uncertainty: by the time a satellite passes over, the contrail may have spread, drifted, or dissipated. The ground camera has no such lag; a contrail is observable from onset and can be tracked through its full lifecycle at high spatial resolution.

There is currently no systematic pipeline to extract a labeled contrail dataset from this camera. Existing codebases in the lab are partial, use deprecated dependencies, and were not designed with research-quality ground truth as the goal. As a result, there is no reliable labeled dataset to validate the satellite ML system, and no ground-truth record of which flights produced contrails on which days.

Researchers need a pipeline that is trustworthy enough to stake a publication on: reproducible, checkpointed, diagnostically transparent, and producing labels that can be cross-referenced against satellite detections flight by flight.

---

## Solution

A batch processing pipeline that, given a date, ingests the day's timelapse video and ADS-B flight data, runs automated contrail detection on each flight's projected footprint in the video, and produces a DuckDB database of candidate contrail episodes. A companion browser-based labeling tool allows student labelers to review the full annotated video, confirm or reject auto-detected episodes, and add persistence ratings. The result is a publication-quality labeled dataset of contrail events with inter-rater reliability built in.

The pipeline is implemented as a UV-managed Python package with a CLI, importable modules for use in Jupyter notebooks, and a SLURM submission script for overnight batch runs on the hex cluster.

---

## User Stories

### Researcher (PI / postdoc)

1. As a researcher, I want to run the pipeline for a single date from the command line so that I can produce a complete set of candidate contrail episodes without writing any ad hoc code.

2. As a researcher, I want the pipeline to stop cleanly at any intermediate stage and produce inspectable output so that I can diagnose failures without re-running the entire pipeline from scratch.

3. As a researcher, I want a Jupyter notebook that lets me load a day's video and ADS-B data, visualize the oriented detection bounding boxes frame by frame, and interactively adjust Canny and Hough parameters so that I can tune the detector for the MIT camera without guessing.

4. As a researcher, I want a Jupyter notebook that compares the fixed-format template OCR reader against EasyOCR on a sample of real frames so that I can make a go/no-go decision before committing to the template approach.

5. As a researcher, I want the pipeline's timestamp extraction to be robust to the clock drift present in the timelapse so that frame-to-ADS-B alignment does not silently degrade across a long video.

6. As a researcher, I want each candidate episode in the DuckDB output to carry a detector score, a pixel-space line estimate, the associated flight's callsign, and timing so that I can query episodes by confidence and cross-reference them against satellite detections.

7. As a researcher, I want to be able to query the labeled dataset to answer questions like "what fraction of flights above 30,000 ft produced a contrail on this day?" or "how does contrail persistence correlate with aircraft type?" without writing custom parsing code.

8. As a researcher, I want the camera site and calibration to be fully described in a YAML config file so that the pipeline can be extended to a second camera site with a config change and no code change.

9. As a researcher, I want the labeler bundle to be generated as a self-contained directory I can hand off to student labelers manually so that no server infrastructure is required.

10. As a researcher, I want a small fixed overlap between the two labelers' assignments so that I can compute inter-rater agreement and report it in a publication.

### Student labeler

11. As a student labeler, I want to open a single HTML file in my browser and have the full annotated day's video play back so that I do not need to install any software or run any commands.

12. As a student labeler, I want the video to show ADS-B flight tracks as overlays, with flights where the auto-detector found a contrail highlighted in a distinct color, so that I can immediately see which flights are the high-priority candidates.

13. As a student labeler, I want a timeline panel with jump markers for each flight episode so that I can navigate directly to each candidate without scrubbing through the entire video.

14. As a student labeler, I want to toggle the detected line segment overlay on and off so that I can judge for myself whether a contrail is present before being anchored by the auto-detection.

15. As a student labeler, I want to record my decision as "contrail," "no contrail," or "unsure," add a persistence rating from 1 to 5, and optionally add a free-text note for each flight episode so that my judgment is fully captured.

16. As a student labeler, I want my labels to be saved locally as a JSON file that I can return to the researcher so that no network connection or server is needed during a labeling session.

17. As a student labeler, I want my progress to be saved automatically so that closing the browser does not lose completed work.

---

## Acceptance Criteria

### Pipeline scaffolding

- [ ] The project installs cleanly with `uv sync` on the hex cluster.
- [ ] `uv run concam run --date YYYY-MM-DD` executes the full pipeline for a given date and exits with a non-zero code if any required input is missing.
- [ ] Each pipeline stage writes output to a well-defined intermediate location and can be re-run independently by passing `--from-stage` or equivalent.
- [ ] A SLURM submission script submits the pipeline as a batch job with appropriate resource requests.
- [ ] A YAML config file fully describes the MIT Green Building camera site: intrinsics, extrinsics, ADS-B filter parameters, OCR region, and detection thresholds.

### Timestamp OCR

- [ ] The fixed-format template reader extracts timestamps from a sample of 100 real frames from the MIT camera with accuracy equal to or better than EasyOCR.
- [ ] The template reader runs at least 5x faster than EasyOCR on the same sample.
- [ ] The reader returns a confidence score and a status field for every frame.
- [ ] When confidence is below a configurable threshold, the reader falls back to EasyOCR and tags the result as a fallback.
- [ ] A TrustButVerifyTracker uses the per-frame OCR results to interpolate timestamps across low-confidence frames without timeline collapse.
- [ ] The OCR validation notebook clearly shows the accuracy, speed, and fallback rate comparison, and records the go/no-go decision.

### ADS-B ingestion

- [ ] The pipeline loads ADS-B data for a given UTC date from the local `feder` data store.
- [ ] Flights are filtered to those within a configurable horizontal radius and above a configurable altitude threshold.
- [ ] ADS-B pings are upsampled to one-second resolution by linear interpolation.
- [ ] Each ping is projected from GPS coordinates to pixel coordinates using the camera calibration, accounting for barrel/pincushion distortion.

### Contrail detection

- [ ] For each flight present in the scene during a given frame, the detector computes an oriented bounding rectangle aligned to the flight path vector derived from consecutive ADS-B points.
- [ ] Canny edge detection and Hough line transform are applied only within that bounding rectangle.
- [ ] The detector returns a score between 0 and 1 and a pixel-space line segment estimate for each (frame, flight) pair.
- [ ] The detection validation notebook shows the detector output on at least 20 real frame-flight pairs from April 9, 2026, with a parameter sweep across Canny and Hough thresholds.
- [ ] The detection notebook records the chosen parameters and the go/no-go decision.

### Episode aggregation and storage

- [ ] Consecutive frames where a flight scores above a configurable threshold are grouped into a single episode.
- [ ] Each episode is written to a DuckDB table with all fields in the agreed schema: `flight_id`, `date`, `contrail_onset_time`, `contrail_end_time`, `detector_score`, `pixel_line`, and the label fields initialized to null.
- [ ] The DuckDB file for a single day can be queried from a Jupyter notebook using standard SQL.
- [ ] Re-running the pipeline for a date that already has data either overwrites or appends cleanly, without producing duplicate rows.

### Labeler bundle

- [ ] The bundle generator produces a directory containing the annotated video (or a reference to it), a JSON manifest of all flight episodes with ADS-B tracks and detection overlays, and a self-contained HTML/JS labeler file.
- [ ] The labeler plays the full day's video in a browser with no server or installation required.
- [ ] ADS-B flight tracks are drawn as canvas overlays on the video.
- [ ] Flights where the detector score exceeds the configurable threshold are highlighted in a distinct color; undetected flights are shown in a neutral color.
- [ ] The detected line segment overlay can be toggled on and off independently of the ADS-B track overlay.
- [ ] The timeline panel shows a marker for each flight episode, labeled with callsign and score.
- [ ] Clicking a marker seeks the video to the episode onset.
- [ ] The labeler records: label (`contrail` / `no_contrail` / `unsure`), persistence rating (1–5), labeler ID, and an optional free-text note per episode.
- [ ] Labels are autosaved to localStorage and can be exported as a JSON file.
- [ ] Loading the HTML file again in the same browser restores previously saved labels.

### Dataset and inter-rater reliability

- [ ] Completed label JSON files from student labelers can be ingested into DuckDB, populating the label fields for each episode.
- [ ] Approximately 20% of episodes are assigned to both labelers; the remainder are divided.
- [ ] A query or notebook cell computes pairwise inter-rater agreement (Cohen's kappa or percent agreement) on the overlapping set.

---

## Implementation Decisions

### Package and dependency management
UV is used for all dependency management. The package is installable as a library (`import concam`) for use in Jupyter notebooks and as a CLI (`uv run concam`). No conda environment is required for new development, though the existing `adsb_overlay` conda environment may be used for reference comparisons.

### Timestamp OCR architecture
The fixed-format template reader is the primary path. It crops a fixed region of interest from the frame, applies threshold-based binarization, and matches character glyphs against pre-built templates using normalized cross-correlation. It returns a `TimestampRead` dataclass with `parsed_dt`, `text`, `confidence`, `per_char_confidence`, `method`, and `status`. EasyOCR is retained as an explicit fallback, called only when confidence is below a threshold. The `TrustButVerifyTracker` from the prior codebase is preserved as the second line of defense for interpolation across low-confidence frames.

### ADS-B data source
`feder` is the primary ADS-B data source. The ADS-B loader must abstract the data source behind an interface so that an alternative source (such as adsb.lol) can be substituted via config without changing the detection or projection code.

### Detection ROI
The detection region of interest for each (frame, flight) pair is an oriented bounding rectangle. Its orientation is computed from the vector between the two most recent ADS-B pings projected into pixel space, and its length and width are configurable in the YAML config. Canny edge detection and Hough line transform run only within this rectangle. This avoids processing full 4K frames and focuses detection where a contrail would actually appear.

### Detection score
The score is derived from the Hough accumulator: the ratio of the maximum Hough bin value in the oriented ROI to a configurable reference value, clipped to [0, 1]. The pixel line segment is the line corresponding to the highest Hough bin, expressed in full-frame pixel coordinates.

### Database schema
A single DuckDB file per camera site holds the `contrail_episodes` table. Label fields are nullable and populated in a second pass after labelers return their JSON files. The schema is versioned; migration logic is required if the schema changes.

### Labeler overlay rendering
The labeler HTML/JS file renders ADS-B track overlays and detection line segments on an HTML5 canvas element synchronized to the video's `timeupdate` event. The overlay data is embedded in the JSON manifest. The video file is referenced by relative path; the bundle directory must be kept together. The toggle controls hide/show canvas layers independently.

### Video for the labeler bundle
For Phase 1, the bundle references the existing daily timelapse AV1 file directly. The bundle generator does not re-encode or copy the video; it writes a relative path in the JSON manifest. The researcher copies or symlinks the video into the bundle directory before hand-off.

### Inter-rater assignment
The bundle generator accepts a `--labelers` argument with two or more labeler IDs and an `--overlap-fraction` argument (default 0.2). It produces one bundle per labeler with the appropriate episode assignments pre-filtered in the JSON manifest.

### Checkpointing
Each pipeline stage writes output to a well-defined intermediate path. The CLI accepts a `--from-stage` flag to resume from a checkpoint. Stages are: `ocr`, `adsb`, `project`, `detect`, `aggregate`, `bundle`.

---

## Testing Decisions

The two highest-risk components—OCR and detection—are validated in Jupyter notebooks against real data before being wired into the pipeline. These notebooks serve as the primary go/no-go gate and as living documentation of the parameter choices.

Unit tests cover:
- The timestamp parser: round-trip property (format(parse(text)) == text), impossible date rejection, separator enforcement.
- The TrustButVerifyTracker: perfect monotone sequences never trigger re-anchor; one bad read amid good reads produces a projected timestamp; a sustained new offset triggers re-anchor.
- The episode aggregator: consecutive above-threshold frames are merged; a gap resets the episode; score is the peak over the episode.
- The DuckDB schema: inserting a valid episode row and querying it back returns identical values; inserting a row with an invalid label value fails.

Integration tests cover:
- The ADS-B loader round-trip: load a known date, filter by altitude and radius, upsample, and verify the number of pings and their timestamps.
- The projection round-trip: project a known GPS coordinate to pixels and verify it falls within the expected image region given the calibration.

The labeler HTML/JS is tested manually in a browser against a synthetic bundle with a short video and a small number of fabricated episodes.

No mocking of the `feder` data store; tests that require ADS-B data use a checked-in small fixture file.

---

## Out of Scope

- Real-time or near-real-time detection. The pipeline is batch only.
- Any changes to the recording pipeline in `LAE_skycam/`.
- Support for cameras other than the MIT Green Building camera in Phase 1.
- Automated upload of labeled data to any external service.
- A web server for the diagnostic or labeling interface in Phase 1.
- ML-based contrail detection. Hough+Canny is the detection method.
- Automated comparison against satellite detections. The database schema supports it, but the comparison logic is out of scope.
- Automated nightly scheduling in Phase 1. The SLURM script is run manually.
- adsb.lol as an ADS-B source in Phase 1.
- Contrail segmentation masks or polygon annotations. Line segments are sufficient.
- Any modification to the existing `TrustButVerifyTracker` logic.

---

## Further Notes

- The April 9, 2026 raw hourly segments and the April 8, 2026 daily timelapse are the primary test inputs. The OCR and detection notebooks should use these files.
- The camera native frame rate is 4 fps. The daily timelapse is sampled at 1 frame per real second and played back at 30 fps. Frame N of the timelapse corresponds to real time = video_start + N seconds, subject to OCR-corrected drift.
- The existing `TrustButVerifyTracker` in `camera-flight-overlay` is known to work. Copy it with attribution and add property-based tests rather than rewriting it.
- The existing `detection_utils.py` in `groundcam_contrail_observatory` is a reference implementation. Review it carefully before adopting any parameter values; the original was tuned for a different camera and site.
- All timestamps in the database are UTC. The MIT camera timestamps are US/Eastern; the OCR reader must convert to UTC using the site timezone from the YAML config.
- The `feder` package documentation is at the URL recorded in the design decisions document. Read the API before writing the loader; the data layout is not yet known and should not be assumed.
- Student labelers are undergraduates. The HTML labeler must be self-explanatory with no training required beyond a short written guide. Error states (missing video, corrupt JSON) must produce a clear human-readable message, not a JavaScript exception.
