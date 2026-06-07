#!/bin/bash
#SBATCH --job-name=concam-hpo-publish
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --output=slurm/logs/hpo-publish-%j.log

# End-to-end: run the 4-knob HPO sweep against the unified prash+thendo+reviewer-1
# label set on 2026-04-09, write a tuned config from the winning combo, then
# publish the tuned variant to ~/public_html/concam/2026-04-09-tuned/ so it
# appears in the dropdown alongside the production 2026-04-09 entry.
#
# Usage: sbatch slurm/hpo_and_publish_tuned.sh
#
# Single-job design (vs job dependencies) so the whole chain survives any
# upstream session disconnect.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

DATE="2026-04-09"
HPO_OUT="output/validation/detection/${DATE}/hpo"
TUNED_CONFIG="configs/mit_green_building.tuned.yaml"

mkdir -p "${REPO_DIR}/slurm/logs"

echo "=== HPO sweep on ${DATE} ==="
echo "Node:   $(hostname)"
echo "Start:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

uv run python scripts/detection_hpo.py \
    --date "${DATE}" \
    --labels \
        labels/${DATE}_prash.json \
        labels/${DATE}_thendo.json \
        labels/${DATE}_reviewer-1.json \
    --manifest "${HOME}/public_html/concam/${DATE}/manifest.json" \
    --out-dir "${HPO_OUT}" \
    --frames-per-episode 8

echo
echo "=== writing tuned config from sweep winner ==="
uv run python - <<PY
import json
import shutil
import sys
from pathlib import Path

import yaml

results_path = Path("${HPO_OUT}/sweep_results.json")
data = json.loads(results_path.read_text())
top = data["results"][0]
print(f"Top combo: gain={top['cross_grad_gain']}, "
      f"pct_high={top['canny_percentile_high']}, "
      f"roi_along={top['roi_along_px']}, "
      f"thr={top['best_threshold']}")
print(f"  AUC={top['auc']:.3f}, YJ={top['best_youden_j']:.3f}, "
      f"TP={top['best_tp']}/{top['n_pos']}, FP={top['best_fp']}/{top['n_neg']}")

prod = Path("configs/mit_green_building.yaml")
tuned = Path("${TUNED_CONFIG}")
shutil.copy(prod, tuned)

cfg = yaml.safe_load(tuned.read_text())
cfg["detection"]["cross_grad_gain"] = float(top["cross_grad_gain"])
cfg["detection"]["canny_percentile_high"] = float(top["canny_percentile_high"])
cfg["detection"]["roi_along_px"] = int(top["roi_along_px"])
cfg["aggregation"]["detection_threshold"] = float(top["best_threshold"])
tuned.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"Wrote {tuned}")
PY

echo
echo "=== publishing tuned variant to ~/public_html/concam/${DATE}-tuned/ ==="
scripts/publish_tuned_variant.sh "${DATE}" tuned "${TUNED_CONFIG}"

echo
echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Public URL: https://hex.mit.edu/~prash/concam/${DATE}-tuned/labeler.html"
