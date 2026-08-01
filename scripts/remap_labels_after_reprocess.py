#!/usr/bin/env python3
"""Re-key human labels onto the episode IDs produced by the OCR-fix reprocess.

Episode IDs are positional: the store stage assigns 1..N in ``episodes.jsonl``
order.  The OCR date fix restores the detection hours that used to be lost
after the first date misread, so a reprocessed day gains episodes *in the
middle* of the day and every later episode shifts up.  ``labels/*.json`` key on
a bare ``episode_id``, so without remapping a label silently starts describing
a different aircraft pass.

The stable identity of an episode is ``(transponder_id, onset)`` -- the same
natural key scripts/build_reliable_label_set.py already uses to reconcile the
April-21 public ID space.  This script reads the ``episodes.pre-ocrfix.jsonl``
baseline that slurm/reprocess_ocr_fix_array.sh saved, matches it against the
regenerated ``episodes.jsonl``, and rewrites the label files.

Labels that cannot be matched are never guessed at: they are reported and (with
--apply) dropped into a ``unmapped`` block in the rewritten file so the
information is preserved rather than silently discarded.

**ID-space guard.** Not every label file is in the ID space this baseline
describes.  labels/ mixes several historical spaces (see the provenance notes
in scripts/build_reliable_label_set.py) -- e.g. 2026-04-15_reviewer-1.json was
exported 2026-04-22 against a manifest that two later regenerations have since
replaced.  Remapping such a file would translate IDs *from the wrong space* and
quietly corrupt real human work.  A file is therefore only rewritten when
almost all of its labels resolve against the baseline; anything above
``MAX_UNMAPPED_FRACTION`` is reported as a different ID space and left
untouched unless --force says otherwise.

Dry-run by default.  Usage:
    python scripts/remap_labels_after_reprocess.py               # report only
    python scripts/remap_labels_after_reprocess.py --apply       # rewrite files
    python scripts/remap_labels_after_reprocess.py --date 2026-04-03 --apply
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO_ROOT / "labels"
ARCHIVE_DIR = LABELS_DIR / "archive"

# Onset timestamps come from the same detection rows on both sides for any
# episode that predates the corruption cliff, so an exact match is the norm.
# A small tolerance covers an episode whose boundary frame changed when the
# restored hours altered aggregation at the edges.
ONSET_TOLERANCE_S = 2.0

# A label file genuinely in the baseline's ID space resolves essentially all of
# its episode IDs -- the reprocess only *adds* episodes, it does not remove the
# ones the labeller saw.  A large unmapped fraction therefore means the file was
# exported against a different manifest generation, not that the labels are bad.
MAX_UNMAPPED_FRACTION = 0.10


def _load_episodes(path: Path) -> list[dict]:
    episodes = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def _key(ep: dict) -> tuple[str, str]:
    return (ep.get("transponder_id") or "", ep.get("onset") or "")


def _parse(ts: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def build_remap(old: list[dict], new: list[dict]) -> tuple[dict[int, int], list[int]]:
    """Map old positional episode_id -> new positional episode_id.

    Returns (mapping, unmatched_old_ids).
    """
    # Exact (transponder_id, onset) first.
    new_by_key: dict[tuple[str, str], int] = {}
    for i, ep in enumerate(new):
        new_by_key.setdefault(_key(ep), i + 1)

    # Fallback index for near-miss onsets, bucketed by transponder.
    by_transponder: dict[str, list[tuple[dt.datetime, int]]] = {}
    for i, ep in enumerate(new):
        onset = _parse(ep.get("onset"))
        if onset is not None:
            by_transponder.setdefault(ep.get("transponder_id") or "", []).append(
                (onset, i + 1)
            )

    mapping: dict[int, int] = {}
    unmatched: list[int] = []
    for i, ep in enumerate(old):
        old_id = i + 1
        hit = new_by_key.get(_key(ep))
        if hit is None:
            onset = _parse(ep.get("onset"))
            candidates = by_transponder.get(ep.get("transponder_id") or "", [])
            if onset is not None and candidates:
                best, best_delta = None, None
                for cand_onset, cand_id in candidates:
                    delta = abs((cand_onset - onset).total_seconds())
                    if best_delta is None or delta < best_delta:
                        best, best_delta = cand_id, delta
                if best_delta is not None and best_delta <= ONSET_TOLERANCE_S:
                    hit = best
        if hit is None:
            unmatched.append(old_id)
        else:
            mapping[old_id] = hit
    return mapping, unmatched


def process_date(date: str, output_dir: Path, apply: bool, force: bool = False) -> dict:
    base = output_dir / date
    old_path = base / "episodes.pre-ocrfix.jsonl"
    new_path = base / "episodes.jsonl"
    result = {"date": date, "status": None, "label_files": []}

    if not old_path.exists():
        result["status"] = "no pre-ocrfix baseline (day not reprocessed)"
        return result
    if not new_path.exists():
        result["status"] = "no regenerated episodes.jsonl"
        return result

    old, new = _load_episodes(old_path), _load_episodes(new_path)
    mapping, unmatched = build_remap(old, new)
    identity = all(k == v for k, v in mapping.items()) and not unmatched and len(old) == len(new)

    result.update(
        episodes_before=len(old),
        episodes_after=len(new),
        matched=len(mapping),
        unmatched_old_ids=unmatched,
        identity=identity,
    )

    label_files = sorted(LABELS_DIR.glob(f"{date}_*.json"))
    if not label_files:
        result["status"] = "no label files for this date"
        return result
    if identity:
        result["status"] = "episode IDs unchanged; labels left alone"
        result["label_files"] = [p.name for p in label_files]
        return result

    for path in label_files:
        doc = json.loads(path.read_text())
        labels = doc.get("labels", [])
        remapped, dropped = [], []
        for entry in labels:
            old_id = entry.get("episode_id")
            new_id = mapping.get(old_id)
            if new_id is None:
                dropped.append(entry)
            else:
                remapped.append({**entry, "episode_id": new_id})
        changed = sum(
            1 for a, b in zip(labels, remapped) if a.get("episode_id") != b.get("episode_id")
        )
        unmapped_fraction = len(dropped) / len(labels) if labels else 0.0
        wrong_space = unmapped_fraction > MAX_UNMAPPED_FRACTION
        entry_report = {
            "file": path.name,
            "exported_at": doc.get("exported_at"),
            "labels": len(labels),
            "remapped": len(remapped),
            "id_changed": changed,
            "unmapped": len(dropped),
            "unmapped_fraction": round(unmapped_fraction, 3),
        }
        if wrong_space and not force:
            entry_report["action"] = (
                f"SKIPPED - {unmapped_fraction:.0%} of labels do not resolve against "
                "the pre-fix baseline, so this file is in a different episode-ID "
                "space (exported against an earlier manifest generation). "
                "Remapping it would translate from the wrong space. Needs manual "
                "provenance work; see scripts/build_reliable_label_set.py."
            )
            result["label_files"].append(entry_report)
            continue
        entry_report["action"] = "remapped" if apply else "would remap"
        result["label_files"].append(entry_report)
        if apply:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            archived = ARCHIVE_DIR / f"{path.stem}.pre-ocrfix{path.suffix}"
            if not archived.exists():
                shutil.copy2(path, archived)
            doc["labels"] = remapped
            if dropped:
                doc["unmapped_pre_ocrfix_labels"] = dropped
            doc["remapped_from"] = {
                "reason": "episode renumbering after the GitHub #1 OCR date-fix reprocess",
                "key": "(transponder_id, onset)",
                "baseline": str(old_path.relative_to(REPO_ROOT)),
                "remapped_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(doc, indent=2) + "\n")

    result["status"] = "remapped" if apply else "would remap (dry run)"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", action="append", help="limit to this date (repeatable)")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the label files (default is a dry run)")
    ap.add_argument("--force", action="store_true",
                    help="remap even files that look like a different episode-ID "
                         "space (see MAX_UNMAPPED_FRACTION); use only with "
                         "verified provenance")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if args.date:
        dates = args.date
    else:
        pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
        dates = sorted({
            m.group(1)
            for p in LABELS_DIR.glob("*.json")
            if (m := pattern.match(p.name))
        })

    if not dates:
        print("no labelled dates found", file=sys.stderr)
        return 1

    results = [process_date(d, output_dir, args.apply, args.force) for d in dates]
    print(json.dumps(results, indent=2))

    needs = [r for r in results if r.get("status", "").startswith(("remapped", "would remap"))]
    skipped = [
        (r["date"], lf["file"])
        for r in results
        for lf in r.get("label_files", [])
        if isinstance(lf, dict) and str(lf.get("action", "")).startswith("SKIPPED")
    ]
    print(f"\n{len(needs)} of {len(dates)} labelled dates need remapping", file=sys.stderr)
    if skipped:
        print(f"{len(skipped)} label file(s) SKIPPED as a different ID space "
              f"(needs manual provenance work):", file=sys.stderr)
        for date, name in skipped:
            print(f"  {date}: {name}", file=sys.stderr)
    if not args.apply and needs:
        print("re-run with --apply to rewrite them", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
