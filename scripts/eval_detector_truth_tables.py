#!/usr/bin/env python3
"""Detector truth tables against the reliable consensus label set.

Joins ``labels/derived/reliable_labels.json`` (per-episode consensus
contrail / no_contrail) to the per-episode detector ``peak_score`` carried in
each day's public ``manifest.json`` and reports confusion matrices.

This is a *scoring* pass over existing detector output — it does NOT re-run
detection. Use it after the label set changes to re-judge performance.

Convention (matches concam.detection.metrics): predict-positive iff
``peak_score >= threshold``; contrail -> positive, no_contrail -> negative.
``unsure``/conflict episodes are already excluded by the reliable-set builder.

Usage:
    uv run python scripts/eval_detector_truth_tables.py
    uv run python scripts/eval_detector_truth_tables.py --threshold 0.083
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from concam.detection.metrics import mann_whitney_auc, youden_threshold

REPO_ROOT = Path(__file__).resolve().parent.parent
RELIABLE = REPO_ROOT / "labels" / "derived" / "reliable_labels.json"
PUBLIC_ROOT = Path.home() / "public_html" / "concam"


def confusion(pos: list[float], neg: list[float], t: float) -> dict:
    tp = sum(1 for s in pos if s >= t)
    fn = len(pos) - tp
    fp = sum(1 for s in neg if s >= t)
    tn = len(neg) - fp
    tpr = tp / len(pos) if pos else float("nan")
    tnr = tn / len(neg) if neg else float("nan")
    fpr = fp / len(neg) if neg else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, tpr=tpr, tnr=tnr, fpr=fpr, prec=prec)


def fmt_pct(x: float) -> str:
    return "  n/a" if x != x else f"{x*100:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.083,
                    help="operating threshold (default 0.083, the documented production point)")
    args = ap.parse_args()
    T = args.threshold

    rl = json.loads(RELIABLE.read_text())["labels"]

    per_day: dict[str, tuple[list[float], list[float]]] = {}
    for date in sorted(rl):
        man = PUBLIC_ROOT / date / "manifest.json"
        score = {str(e["episode_id"]): e["peak_score"]
                 for e in json.loads(man.read_text())["episodes"]}
        pos, neg = [], []
        for eid, rec in rl[date].items():
            s = score.get(eid)
            if s is None:
                continue
            (pos if rec["label"] == "contrail" else neg).append(s)
        per_day[date] = (pos, neg)

    hdr = (f"{'date':<12} {'P':>4} {'N':>4}  {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>5}  "
           f"{'TPR':>6} {'TNR':>6} {'FPR':>6} {'Prec':>6}  {'AUC':>5}")
    print(f"\nDetector truth tables vs reliable consensus labels  (predict-positive iff peak_score >= {T})")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    all_pos, all_neg = [], []
    for date, (pos, neg) in per_day.items():
        all_pos += pos
        all_neg += neg
        c = confusion(pos, neg, T)
        auc = mann_whitney_auc(pos, neg)
        print(f"{date:<12} {len(pos):>4} {len(neg):>4}  {c['tp']:>4} {c['fn']:>4} {c['fp']:>4} {c['tn']:>5}  "
              f"{fmt_pct(c['tpr'])} {fmt_pct(c['tnr'])} {fmt_pct(c['fpr'])} {fmt_pct(c['prec'])}  "
              f"{auc:5.3f}" if auc == auc else
              f"{date:<12} {len(pos):>4} {len(neg):>4}  {c['tp']:>4} {c['fn']:>4} {c['fp']:>4} {c['tn']:>5}  "
              f"{fmt_pct(c['tpr'])} {fmt_pct(c['tnr'])} {fmt_pct(c['fpr'])} {fmt_pct(c['prec'])}    n/a")

    print("-" * len(hdr))
    c = confusion(all_pos, all_neg, T)
    auc = mann_whitney_auc(all_pos, all_neg)
    print(f"{'OVERALL':<12} {len(all_pos):>4} {len(all_neg):>4}  {c['tp']:>4} {c['fn']:>4} {c['fp']:>4} {c['tn']:>5}  "
          f"{fmt_pct(c['tpr'])} {fmt_pct(c['tnr'])} {fmt_pct(c['fpr'])} {fmt_pct(c['prec'])}  {auc:5.3f}")

    t_opt, j = youden_threshold(all_pos, all_neg)
    c_opt = confusion(all_pos, all_neg, t_opt)
    print(f"\nThreshold-free pooled ROC-AUC: {auc:.3f}   (P={len(all_pos)}, N={len(all_neg)})")
    print(f"Youden-optimal threshold on this label set: {t_opt:.3f}  (J={j:.3f}) -> "
          f"TPR {fmt_pct(c_opt['tpr'])}, FPR {fmt_pct(c_opt['fpr'])}, Prec {fmt_pct(c_opt['prec'])}")


if __name__ == "__main__":
    main()
