#!/bin/bash
#SBATCH --job-name=concam-hpo-reliable-daytime
#SBATCH --time=8:00:00  # 120-combo grid x ~5.5k daylight crops (04-08 + 04-09)
                        # at ~20 ms/detect ~= 4h replay + 2 whole-day 4K decodes;
                        # 8h leaves headroom over the 6h the May round needed.
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G  # crop-extraction caches thousands of padded 4K crops; the
                   # May round OOM'd at 16G (21G RSS) — see hpo_tune_validate.sh.
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/hpo-reliable-daytime-%j.log
#
# Detector hyperparameter retune on the RELIABLE consensus label set
# (labels/derived/reliable_labels.json, ADR-0003 — the May-2026 round was
# invalidated by mixed episode-ID spaces), restricted to daytime episodes
# (onset in DAYLIGHT_UTC), with the static building mask ON (base config),
# sweeping preprocessing variants (cross_grad gains on the finest grid,
# local_contrast sigmas, none) x canny_percentile_high x roi_along_px.
#
#   tune on   2026-04-08 (357 daylight labels: 125 contrail / 232 no_contrail)
#   hold out  2026-04-09 (329 daylight labels: 143 contrail / 186 no_contrail)
#
# 2026-04-19 is QUARANTINED (corrupted detection-to-frame join, see
# docs/label_reliability.md). 2026-03-30's labeled episodes all fall OUTSIDE
# the daylight window (0/72) — unusable here. 2026-03-29 (46 daylight labels)
# can be added as an extra holdout report:
#   EXTRA_DATES="2026-03-29" sbatch slurm/hpo_reliable_daytime.sh
# (each extra date costs a whole-day decode + a full-grid replay).
#
# Does NOT write the production/tuned config and does NOT publish — it stops
# at a report for human review (same philosophy as hpo_tune_validate.sh).
#
# Usage: sbatch slurm/hpo_reliable_daytime.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
mkdir -p slurm/logs output/hpo_reliable

TRAIN_DATE=2026-04-08
VAL_DATE=2026-04-09
EXTRA_DATES="${EXTRA_DATES:-}"
BASE_CONFIG=configs/mit_green_building.yaml
RELIABLE_LABELS=labels/derived/reliable_labels.json
# Daylight window (UTC, episode onset). Civil-daylight extract convention for
# April at Boston is ~10:30-23:30 UTC (scripts/detection_validation_extract.py);
# the detector was strongest ~12:00-20:00 UTC in the 2026-06 labeled-episode
# review. 11:00-22:30 trims grazing-light dawn/dusk while keeping most labels.
DAYLIGHT_UTC="${DAYLIGHT_UTC:-11:00,22:30}"
OUT_ROOT=output/hpo_reliable

echo "=== [1/3] tuning sweep on ${TRAIN_DATE} (reliable labels, ${DAYLIGHT_UTC} UTC) ==="
uv run python scripts/detection_hpo.py \
    --date "${TRAIN_DATE}" \
    --reliable-labels "${RELIABLE_LABELS}" \
    --daylight-utc "${DAYLIGHT_UTC}" \
    --manifest "${HOME}/public_html/concam/${TRAIN_DATE}/manifest.json" \
    --config "${BASE_CONFIG}" \
    --out-dir "${OUT_ROOT}/${TRAIN_DATE}"

echo "=== [2/3] held-out sweep on ${VAL_DATE} (same grid, reliable labels) ==="
uv run python scripts/detection_hpo.py \
    --date "${VAL_DATE}" \
    --reliable-labels "${RELIABLE_LABELS}" \
    --daylight-utc "${DAYLIGHT_UTC}" \
    --manifest "${HOME}/public_html/concam/${VAL_DATE}/manifest.json" \
    --config "${BASE_CONFIG}" \
    --out-dir "${OUT_ROOT}/${VAL_DATE}"

echo "=== [3/3] select winner on ${TRAIN_DATE}, report held-out on ${VAL_DATE} ==="
uv run python scripts/hpo_select_and_validate.py \
    --train-results "${OUT_ROOT}/${TRAIN_DATE}/sweep_results.json" \
    --val-results   "${OUT_ROOT}/${VAL_DATE}/sweep_results.json" \
    --base-config   "${BASE_CONFIG}" \
    --out           "${OUT_ROOT}/holdout_validation.md"

for d in ${EXTRA_DATES}; do
    echo "=== [extra] holdout sweep + report on ${d} ==="
    uv run python scripts/detection_hpo.py \
        --date "${d}" \
        --reliable-labels "${RELIABLE_LABELS}" \
        --daylight-utc "${DAYLIGHT_UTC}" \
        --manifest "${HOME}/public_html/concam/${d}/manifest.json" \
        --config "${BASE_CONFIG}" \
        --out-dir "${OUT_ROOT}/${d}"
    uv run python scripts/hpo_select_and_validate.py \
        --train-results "${OUT_ROOT}/${TRAIN_DATE}/sweep_results.json" \
        --val-results   "${OUT_ROOT}/${d}/sweep_results.json" \
        --base-config   "${BASE_CONFIG}" \
        --out           "${OUT_ROOT}/holdout_validation_${d}.md"
done

echo "=== done. Review: ${OUT_ROOT}/${TRAIN_DATE}/sweep_report.md,"
echo "             ${OUT_ROOT}/${VAL_DATE}/sweep_report.md,"
echo "             ${OUT_ROOT}/holdout_validation.md ==="
echo "No config was written and nothing was published (review-only run)."
