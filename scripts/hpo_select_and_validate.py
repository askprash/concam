"""Select the HPO winner on the tuning date and report its HELD-OUT metrics.

The detector "model" is a DetectionConfig.  We tune it on one date (the sweep
ranks configs by best Youden-J, AUC tiebreak) and must report how that *single
chosen* config + threshold performs on a DIFFERENT date the tuner never saw —
otherwise the headline number is one the sweep optimised into existence.

Because ``detection_hpo.py`` evaluates the SAME grid on every date, the winning
combo from the tuning date already has a row in the validation date's results;
we just read it out at the threshold we committed to on the tuning date.

Usage::

    uv run python scripts/hpo_select_and_validate.py \\
        --train-results output/hpo/2026-04-09/sweep_results.json \\
        --val-results   output/hpo/2026-04-19/sweep_results.json \\
        --base-config   configs/mit_green_building.yaml \\
        --out           output/hpo/holdout_validation.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concam.config import load_config

_TOL = 1e-6


def _same_combo(a: dict, b: dict) -> bool:
    return (
        abs(a["cross_grad_gain"] - b["cross_grad_gain"]) < _TOL
        and abs(a["canny_percentile_high"] - b["canny_percentile_high"]) < _TOL
        and int(a["roi_along_px"]) == int(b["roi_along_px"])
    )


def _row_for_combo(results: list[dict], combo: dict) -> dict:
    for r in results:
        if _same_combo(r, combo):
            return r
    raise SystemExit(
        f"combo {combo['cross_grad_gain']}/{combo['canny_percentile_high']}/"
        f"{combo['roi_along_px']} not found in the other sweep — were the grids "
        f"identical across dates?"
    )


def _pt_at(row: dict, threshold: float) -> dict:
    for pt in row["per_threshold"]:
        if abs(pt["threshold"] - threshold) < _TOL:
            return pt
    raise SystemExit(
        f"threshold {threshold} not in sweep grid {[p['threshold'] for p in row['per_threshold']]}"
    )


def _recall(pt: dict) -> float:
    p = pt["tp"] + pt["fn"]
    return pt["tp"] / p if p else float("nan")


def _fpr(pt: dict) -> float:
    n = pt["fp"] + pt["tn"]
    return pt["fp"] / n if n else float("nan")


def _precision(pt: dict) -> float:
    pp = pt["tp"] + pt["fp"]
    return pt["tp"] / pp if pp else float("nan")


def _scorecard(label: str, row: dict, threshold: float) -> dict:
    pt = _pt_at(row, threshold)
    return {
        "label": label,
        "gain": row["cross_grad_gain"],
        "pct": row["canny_percentile_high"],
        "roi": int(row["roi_along_px"]),
        "thr": threshold,
        "auc": row["auc"],
        "tp": pt["tp"], "fp": pt["fp"], "fn": pt["fn"], "tn": pt["tn"],
        "recall": _recall(pt),
        "fpr": _fpr(pt),
        "precision": _precision(pt),
        "youden_j": pt["youden_j"],
    }


def _md_row(s: dict) -> str:
    return (
        f"| {s['label']} | {s['gain']}/{s['pct']}/{s['roi']} | {s['thr']} | "
        f"{s['auc']:.3f} | {s['recall']:.2f} ({s['tp']}/{s['tp'] + s['fn']}) | "
        f"{s['fpr']:.2f} ({s['fp']}/{s['fp'] + s['tn']}) | {s['precision']:.2f} | "
        f"{s['youden_j']:.3f} |"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-results", required=True, type=Path)
    ap.add_argument("--val-results", required=True, type=Path)
    ap.add_argument("--base-config", type=Path,
                    default=REPO_ROOT / "configs" / "mit_green_building.yaml")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    train = json.loads(args.train_results.read_text())
    val = json.loads(args.val_results.read_text())

    prod_thr = load_config(str(args.base_config)).aggregation.detection_threshold

    winner = train["results"][0]
    win_thr = winner["best_threshold"]
    val_winner = _row_for_combo(val["results"], winner)
    val_oracle = val["results"][0]

    cards = [
        # In-sample: what the tuner achieved on the tuning date.
        _scorecard(f"prod @ {train['date']} (train)", train["baseline"], prod_thr),
        _scorecard(f"tuned @ {train['date']} (train, in-sample)", winner, win_thr),
        # Held-out: applied UNCHANGED to the validation date.
        _scorecard(f"prod @ {val['date']} (HELD-OUT)", val["baseline"], prod_thr),
        _scorecard(f"tuned @ {val['date']} (HELD-OUT)", val_winner, win_thr),
        # Context: best the grid could do on the validation date (oracle).
        _scorecard(f"oracle @ {val['date']} (upper bound)", val_oracle, val_oracle["best_threshold"]),
    ]
    held_prod = cards[2]
    held_tuned = cards[3]

    d_recall = held_tuned["recall"] - held_prod["recall"]
    d_auc = held_tuned["auc"] - held_prod["auc"]
    d_fpr = held_tuned["fpr"] - held_prod["fpr"]
    generalizes = (d_recall >= 0 and d_auc >= -0.02) or (d_auc > 0.02)

    lines = [
        "# Held-out tuning validation",
        "",
        f"- **Tuned on:** {train['date']}  ({train['n_pos']} contrail / "
        f"{train['n_neg']} no_contrail; labels: "
        f"{', '.join(Path(p).name for p in train['labels_merged_from'])})",
        f"- **Validated on (held out):** {val['date']}  ({val['n_pos']} contrail / "
        f"{val['n_neg']} no_contrail; labels: "
        f"{', '.join(Path(p).name for p in val['labels_merged_from'])})",
        f"- **Chosen config:** cross_grad_gain={winner['cross_grad_gain']}, "
        f"canny_percentile_high={winner['canny_percentile_high']}, "
        f"roi_along_px={int(winner['roi_along_px'])}, detection_threshold={win_thr}",
        f"- **Production threshold (for baseline rows):** {prod_thr}",
        "",
        "| config | gain/pct/roi | thr | AUC | recall (TP/P) | FPR (FP/N) | prec | YoudenJ |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    lines += [_md_row(s) for s in cards]
    lines += [
        "",
        "## Generalisation verdict (held-out, tuned vs production)",
        "",
        f"- Δ recall: **{d_recall:+.2f}**  ({held_prod['recall']:.2f} → {held_tuned['recall']:.2f})",
        f"- Δ AUC: **{d_auc:+.3f}**  ({held_prod['auc']:.3f} → {held_tuned['auc']:.3f})",
        f"- Δ FPR: **{d_fpr:+.2f}**  ({held_prod['fpr']:.2f} → {held_tuned['fpr']:.2f})  (lower is better)",
        "",
        f"**{'GENERALISES — tuned config improves the unseen day.' if generalizes else 'DOES NOT clearly generalise — tuned config overfits the tuning date; do not promote.'}**",
        "",
        "_Oracle row = the best the grid could do on the validation date if we had_",
        "_tuned on it directly; the gap (tuned-held-out vs oracle) shows how much_",
        "_a single-day tune leaves on the table._",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[validate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
