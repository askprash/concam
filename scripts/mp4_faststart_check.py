#!/usr/bin/env python3
"""Report whether an MP4 is 'faststart' (moov box before mdat).

A browser cannot begin playback until it has the moov box.  When moov is muxed
*after* mdat the player must range-fetch the tail of the file first, which on a
multi-GB 4K daily costs seconds of stall before the first frame appears -- the
"video failed to load" / slow-seek symptom reviewers hit on the labeler site.

Reads only the top-level box headers (a few hundred bytes seeked through the
file), so it is cheap enough to run across the whole archive.

Usage:
    python scripts/mp4_faststart_check.py FILE [FILE ...]

Prints one JSON object per line: {"path", "faststart", "boxes", "size"}.
Exit status is 0 only if every file given is faststart, so it doubles as a
post-remux verification gate in shell scripts.
"""
from __future__ import annotations

import json
import os
import struct
import sys

# A malformed/truncated file could otherwise walk forever; no sane MP4 has
# hundreds of top-level boxes before the payload.
MAX_BOXES = 64


def top_level_boxes(path: str, max_boxes: int = MAX_BOXES) -> list[str]:
    """Return the ordered top-level box types of an ISO-BMFF (MP4) file."""
    size = os.path.getsize(path)
    boxes: list[str] = []
    with open(path, "rb") as f:
        offset = 0
        while offset < size and len(boxes) < max_boxes:
            f.seek(offset)
            header = f.read(16)
            if len(header) < 8:
                break
            box_size = struct.unpack(">I", header[0:4])[0]
            box_type = header[4:8].decode("latin-1", errors="replace")
            if box_size == 1:
                # 64-bit extended size lives in the 8 bytes after the type.
                if len(header) < 16:
                    break
                box_size = struct.unpack(">Q", header[8:16])[0]
            elif box_size == 0:
                # Box runs to end of file.
                box_size = size - offset
            boxes.append(box_type)
            if box_size < 8:
                break  # corrupt length; stop rather than loop
            offset += box_size
    return boxes


def is_faststart(boxes: list[str]) -> bool:
    if "moov" not in boxes or "mdat" not in boxes:
        return False
    return boxes.index("moov") < boxes.index("mdat")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    all_ok = True
    for path in argv[1:]:
        try:
            boxes = top_level_boxes(path)
            ok = is_faststart(boxes)
            record = {
                "path": path,
                "faststart": ok,
                "boxes": boxes[:8],
                "size": os.path.getsize(path),
            }
        except OSError as exc:
            all_ok = False
            record = {"path": path, "faststart": False, "error": str(exc)}
        else:
            all_ok = all_ok and ok
        print(json.dumps(record))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
