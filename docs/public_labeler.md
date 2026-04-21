# Public labeler site on hex.mit.edu

A BasicAuth-protected web UI at `https://hex.mit.edu/~prash/concam/` that
lets external collaborators review daily ConCam output (video + ADS-B
overlay + automatic detections) without needing server access.

## What reviewers see

- **Landing page** (`/concam/`) listing every published date with
  per-date flight-pass counts and detector-positive counts.
- **Per-date labeler** (`/concam/<date>/labeler.html`) — the existing
  browser labeler (`concam/bundle/templates/labeler.html`), enhanced
  this session with:
  - Video fits the viewport height (CSS aspect-ratio + max-width/height
    100%, canvas overlay stays pixel-aligned).
  - Sidebar entry per **in-scene ADS-B pass**, not per detector-positive
    episode — so false negatives are visible.
  - `?user=<kerb>` in the URL sets the reviewer identity, falling back to
    localStorage and then a first-load prompt.
  - Header pills: reviewer ID, live "N contrails found" tally,
    date-switch dropdown.
  - Sort by time asc or peak_score desc; "detected only" filter.
  - Flight dots show `CALLSIGN FL{nnn}` using the nearest ping's
    barometric altitude.
  - Default overlays: ADS-B dot only (no track, no detection lines) —
    reviewers turn them on deliberately.
  - Export filename tagged with reviewer: `<date>_<user>_labels.json`.

## Publishing a new date

Full loop from a fresh raw video to a live URL:

```
# 1. Run the full pipeline (~2-4 h). See AGENTS.md for the sbatch rule.
sbatch slurm/run_pipeline.sh 2026-04-22

# 2. Once it finishes, publish:
./scripts/publish_public_date.sh 2026-04-22
```

`publish_public_date.sh` is idempotent — re-running regenerates the
bundle, manifest, symlink, and landing page without breaking earlier
dates. Under the hood:

1. `concam bundle --date <date> --labelers prash` → produces
   `output/<date>/bundles/prash/`.
2. `scripts/build_public_bundle.py` → synthesises the
   all-flight-passes manifest by walking `projections.jsonl` (one
   sidebar entry per in-scene pass, split on
   `AggregationConfig.max_gap_seconds`) and attaching per-frame
   detection scores/lines from `detections.jsonl`. Rebuilds
   `flight_tracks` from the **current** projections so sidebar entries
   and overlay dots stay in sync (stale source bundles caused the
   AAL126-no-dot bug during this session).
3. Symlinks `/net/d16/data/contrail-camera/YYYY_MM_DD_0000_2359.mp4`
   into `~/public_html/concam/<date>/video.mp4`. Apache's userdir
   `Options SymLinksIfOwnerMatch` follows owner-matching symlinks, so
   no disk-quota hit.
4. Regenerates `~/public_html/concam/dates.json` and `index.html`
   from whatever subdirectories have a readable `manifest.json`.

### Non-standard video filenames

The publisher's raw-video path is hardcoded to the
`YYYY_MM_DD_0000_2359.mp4` pattern. Dates with shorter or offset
captures (e.g. `2025_10_19_1200_2359.mp4`) need either:

- A 3-line patch to `scripts/publish_public_date.sh` to resolve the
  actual file via a glob, OR
- A manual symlink + direct call to `build_public_bundle.py`.

The pipeline run itself also hits the same assumption in
`concam.pipeline.stages.resolve_video_path`; pass `--video <path>`
explicitly via `sbatch --wrap` for non-standard files.

## Managing reviewers

`scripts/manage_public_reviewers.py` owns `~/public_html/concam/.htpasswd`
and `~/public_html/concam/credentials.txt` (mode 0600, human-readable
record). Apache only reads `.htpasswd`; editing `credentials.txt`
directly does nothing to auth.

```
# Add one reviewer (random password)
python3 scripts/manage_public_reviewers.py add alice

# Add one with a chosen password
python3 scripts/manage_public_reviewers.py add alice --password spring2026

# Remove
python3 scripts/manage_public_reviewers.py remove reviewer2

# See everyone
python3 scripts/manage_public_reviewers.py list

# Wipe and start over
python3 scripts/manage_public_reviewers.py reset alice bob carol
```

## Collecting labels

Reviewers hit **Export labels** and it downloads to their machine as
`<date>_<username>_labels.json`. They email it back; you store it
under `labels/public_reviewers/<date>/<user>.json` (per the
feedback memory: human labels live under `labels/`, not `output/`)
and ingest:

```
uv run concam ingest-labels --date 2026-04-09 \
    --labels labels/public_reviewers/2026-04-09/*.json
```

Once multiple reviewers' labels are in, `uv run concam agreement
--date <date>` computes Cohen's κ per pair and flags disagreements.

## Known limitations

1. **No server-side upload.** hex's Apache config disables PHP and CGI
   in user directories via `php_admin_flag engine Off` + Options
   without `ExecCGI`. Neither is overridable from `.htaccess`.
   Enabling either requires a sysadmin edit to
   `/etc/apache2/mods-available/*.conf`. Tracked in PRD item 33.
2. **No auto-username from BasicAuth.** Browsers don't expose the
   `Authorization` header to JS, and hex's config won't let us inject
   `REMOTE_USER` into a response (would need `mod_headers` + userdir
   allow). Workaround: give each reviewer a personalised URL with
   `?user=<kerb>`, or let them answer the first-load prompt.
3. **Public_html is open by default**, so BasicAuth is the only thing
   between the raw daily video and the internet. Don't put anything
   sensitive in `~/public_html/concam/` besides what the pipeline
   already publishes.

## File map

```
~/public_html/concam/
├── index.html              # regenerated by publish_public_date.sh
├── dates.json              # regenerated; read by the date dropdown
├── .htaccess               # BasicAuth config
├── .htpasswd               # hashed passwords (managed by script)
├── credentials.txt         # plaintext record, mode 0600
└── <date>/
    ├── labeler.html        # template copy
    ├── manifest.json       # synthesised all-flight-passes manifest
    └── video.mp4           # symlink to /net/d16/data/contrail-camera/...

~/contrails/mit-concam-pipeline/
├── concam/bundle/templates/labeler.html    # canonical template — edit here
├── scripts/build_public_bundle.py          # manifest synthesiser
├── scripts/publish_public_date.sh          # end-to-end publisher
└── scripts/manage_public_reviewers.py      # BasicAuth user manager
```

## PRD items added this session

- **33** — public reviewer website (HUMAN REVIEW: pending off-MIT
  confirmation by an external reviewer).
- **34** — wire the filter_playground's chosen transform chain into
  `DetectionConfig` (explicit follow-up to item 29).
- **35** — ingest + Cohen's κ reporting for public reviewer
  submissions.

Item 32 (callsign legibility + FL) is partly landed by this session —
the FL display works; the font bump + dark pill backdrop are still open.
