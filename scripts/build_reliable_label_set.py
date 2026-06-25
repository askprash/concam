#!/usr/bin/env python3
"""Build the consolidated *reliable* human-label set for detector tuning.

Provenance analysis (2026-06-11; see docs/label_reliability.md) showed the
labels/ directory mixes three episode-ID spaces:

  * **current** — manifests built from the 2026-05-01+ pipeline outputs (the
    June-2026 mass regeneration reproduced these IDs deterministically).
    Files exported after the relevant rerun are directly usable.
  * **April-21 public space** (2026-04-09 only) — 535-episode manifest;
    archived skeleton at labels/archive/2026-04-09_manifest_2026-04-21_episodes.json.
    Labels are remapped onto the current space by (transponder_id, onset).
  * **label-batch space** (2026-04-09_prash.json) — synthetic 35-candidate
    batch numbering; excluded here (remappable via
    output/validation/detection/2026-04-09/label_batch/candidates.json).

The earlier "lrsand is an outlier / labels unreliable" conclusion was an
artifact of comparing across these spaces: after remapping, lrsand vs
reviewer-1 agrees perfectly (kappa = 1.0, n = 28) and lrsand vs thendo reaches
kappa = 0.84 on definite labels.

Output: labels/derived/reliable_labels.json
  {date: {episode_id: {"label", "labelers", "votes"}}}
with `unsure` votes dropped and episodes with conflicting definite votes
excluded (recorded under "conflicts").
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO_ROOT / "labels"
ARCHIVE_MANIFEST = (
    LABELS_DIR / "archive" / "2026-04-09_manifest_2026-04-21_episodes.json"
)
PUBLIC_ROOT = Path.home() / "public_html" / "concam"
OUT_PATH = LABELS_DIR / "derived" / "reliable_labels.json"

# Files whose episode IDs match the current public manifests (exported after
# the last pipeline rerun for their date).
DIRECT_FILES = [
    "2026-03-29_thendo_labels.json",
    "2026-03-30_thendo_labels.json",
    "2026-03-31_thendo_labels.json",
    "2026-04-03_lrsand_labels.json",
    "2026-04-08_thendo_labels.json",
    "2026-04-09_lrsand_labels.json",
    "2026-04-19_lrsand_labels.json",
    "2026-06-04_thendo_labels.json",
    "2026-06-09_thendo_labels.json",
]
# 2026-04-09 files in the April-21 ID space, remapped via the archived skeleton.
REMAP_FILES = ["2026-04-09_thendo.json", "2026-04-09_reviewer-1.json"]
# Excluded: 2026-04-09_prash.json (label-batch ID space),
# 2026-04-15_reviewer-1.json (pre-rerun space, no archived manifest to remap).


def load_labels(path: Path) -> tuple[str, str, dict[int, str]]:
    d = json.loads(path.read_text())
    return d["date"], d["labeler_id"], {
        rec["episode_id"]: rec["label"] for rec in d["labels"]
    }


def build_april9_remap() -> dict[int, int]:
    """old (April-21) episode_id -> current episode_id, by (tid, onset) then
    unique same-tid window overlap."""
    old = json.loads(ARCHIVE_MANIFEST.read_text())
    cur = json.loads((PUBLIC_ROOT / "2026-04-09" / "manifest.json").read_text())

    def key(e):
        return (e["transponder_id"], e["onset"][:19])

    cur_by_key = {key(e): e["episode_id"] for e in cur["episodes"]}
    cur_by_tid: dict[str, list[dict]] = {}
    for e in cur["episodes"]:
        cur_by_tid.setdefault(e["transponder_id"], []).append(e)

    mapping: dict[int, int] = {}
    for e in old["episodes"]:
        k = key(e)
        if k in cur_by_key:
            mapping[e["episode_id"]] = cur_by_key[k]
            continue
        overlaps = [
            c for c in cur_by_tid.get(e["transponder_id"], [])
            if min(c["end"], e["end"]) > max(c["onset"], e["onset"])
        ]
        if len(overlaps) == 1:
            mapping[e["episode_id"]] = overlaps[0]["episode_id"]
    return mapping


def consensus(votes_by_episode: dict[int, dict[str, str]]) -> tuple[dict, dict]:
    """Per-episode consensus over definite labels.

    `unsure` votes are dropped (the labeling guide tells reviewers to minimise
    them; they carry no contrail/no-contrail information). Episodes whose
    definite votes conflict are excluded from the consensus and reported
    separately — adjudicate by re-watching, don't average.
    """
    out: dict = {}
    conflicts: dict = {}
    for eid, votes in sorted(votes_by_episode.items()):
        definite = {lid: lab for lid, lab in votes.items() if lab != "unsure"}
        if not definite:
            continue
        counts = Counter(definite.values())
        if len(counts) > 1:
            conflicts[str(eid)] = definite
            continue
        label = next(iter(counts))
        out[str(eid)] = {
            "label": label,
            "labelers": sorted(definite),
            "votes": len(definite),
        }
    return out, conflicts


def main() -> None:
    votes: dict[str, dict[int, dict[str, str]]] = {}  # date -> eid -> labeler -> label

    for name in DIRECT_FILES:
        date, lid, labels = load_labels(LABELS_DIR / name)
        for eid, lab in labels.items():
            votes.setdefault(date, {}).setdefault(eid, {})[lid] = lab

    remap = build_april9_remap()
    for name in REMAP_FILES:
        date, lid, labels = load_labels(LABELS_DIR / name)
        assert date == "2026-04-09"
        dropped = 0
        for eid, lab in labels.items():
            cid = remap.get(eid)
            if cid is None:
                dropped += 1
                continue
            votes.setdefault(date, {}).setdefault(cid, {})[lid] = lab
        if dropped:
            print(f"[reliable] {name}: {dropped} labels had no current-space episode")

    dates_out = {}
    all_conflicts = {}
    for date, by_ep in sorted(votes.items()):
        cons, conf = consensus(by_ep)
        dates_out[date] = cons
        if conf:
            all_conflicts[date] = conf
        n_multi = sum(1 for v in cons.values() if v["votes"] > 1)
        print(f"[reliable] {date}: {len(cons)} consensus labels "
              f"({n_multi} multi-labeler), {len(conf)} conflicts excluded")

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "scripts/build_reliable_label_set.py",
        "labels": dates_out,
        "conflicts": all_conflicts,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    total = sum(len(v) for v in dates_out.values())
    print(f"[reliable] wrote {OUT_PATH} ({total} labels across {len(dates_out)} dates)")


if __name__ == "__main__":
    main()
