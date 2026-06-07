"""Token-stitching helper for the EasyOCR fallback.

Adapted from the reference camera-flight-overlay codebase at
``camera-flight-overlay/utilities/ocr_utils.py`` (function
``clean_easyocr_output``).  The intent and regexes are unchanged; the only
edits are to remove unused imports and to always return a 24-hour string.

The output is always 24-hour ``HH:MM:SS``.  The MIT Green Building overlay
itself is 24-hour, but EasyOCR can still emit a spurious ``AM``/``PM`` token
(it is in the allowlist), so :func:`_canon_time` honours such a suffix and
converts it to 24-hour rather than dropping it -- emitting the 12-hour value
verbatim would corrupt every PM reading and midnight.
"""

from __future__ import annotations

import re


_DATE_SPLIT = re.compile(r"\b(?P<m>\d{1,2})\s*[-/]\s*(?P<d>\d{1,2})\s*/\s*(?P<y>\d{2,4})\b")
_DATE_MERGE = re.compile(r"\b(?P<m>\d{2})(?P<d>\d{2})(?P<y>\d{4})\b")
_DATE_STD = re.compile(r"\b(?P<m>\d{1,2})[/-](?P<d>\d{1,2})[/-](?P<y>\d{2,4})\b")
_TIME_ANY = re.compile(r"\b(?P<h>\d{1,2})[:.](?P<mi>\d{2})(?:[:.](?P<s>\d{2}))?\s*(?P<ampm>(?i:AM|PM))?\b")
_WEEKDAY = re.compile(r"\b(?i:MON|TUE|WED|THU|FRI|SAT|SUN)\b")


def _norm(s: str) -> str:
    s = s.strip()
    s = s.replace("O", "0").replace("o", "0").replace("l", "1")
    return re.sub(r"\s+", " ", s)


def _canon_date(m, d, y) -> str:
    m, d = int(m), int(d)
    y = int(("20" + y) if len(y) == 2 else y)
    if not (1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
        return ""
    return f"{m:02d}/{d:02d}/{y:04d}"


def _canon_time(h, mi, s, ampm) -> str:
    h, mi = int(h), int(mi)
    s = int(s) if s is not None else 0
    if not (0 <= mi <= 59 and 0 <= s <= 59):
        return ""
    if ampm:
        # 12-hour clock: validate the 1-12 range, then convert to 24-hour.
        # 12 AM -> 00, 1-11 AM -> unchanged, 12 PM -> 12, 1-11 PM -> +12.
        if not (1 <= h <= 12):
            return ""
        is_pm = ampm.strip().upper().startswith("P")
        if is_pm:
            h = h if h == 12 else h + 12
        else:
            h = 0 if h == 12 else h
    else:
        if not (0 <= h <= 23):
            return ""
    return f"{h:02d}:{mi:02d}:{s:02d}"


def clean_easyocr_output(results) -> str:
    """Stitch EasyOCR tokens into a canonical ``MM/DD/YYYY HH:MM:SS`` string."""
    tokens: list[str] = []
    for it in results:
        try:
            txt = str(it[1])
        except Exception:
            continue
        txt = _norm(txt)
        if txt:
            tokens.append(_WEEKDAY.sub("", txt).strip())

    if not tokens:
        return "UNREADABLE"

    space_joined = " ".join(tokens)
    tight_joined = "".join(tokens)
    windows = tokens[:] + ["".join(p) for p in zip(tokens, tokens[1:])]
    candidates = [space_joined, tight_joined] + windows

    date_str, time_str = None, None
    for s in candidates:
        if not date_str:
            m = _DATE_SPLIT.search(s) or _DATE_STD.search(s) or _DATE_MERGE.search(s)
            if m:
                dcanon = _canon_date(m.group("m"), m.group("d"), m.group("y"))
                if dcanon:
                    date_str = dcanon
        if not time_str:
            m = _TIME_ANY.search(s)
            if m:
                tcanon = _canon_time(
                    m.group("h"), m.group("mi"), m.group("s"), m.group("ampm")
                )
                if tcanon:
                    time_str = tcanon
        if date_str and time_str:
            break

    if date_str and time_str:
        return f"{date_str} {time_str}"
    if date_str:
        return date_str
    if time_str:
        return time_str
    return "UNREADABLE"
