"""Timestamp string parsing helpers for the fixed-format reader.

The camera overlay renders dates in the canonical form ``MM/DD/YYYY HH:MM:SS``
(24-hour clock).  The parser here accepts only that exact shape; anything else
is rejected.  This keeps the grammar tight so that a confident-but-wrong
reading from the template matcher surfaces as a ``ValueError`` rather than a
silently incorrect datetime.
"""

from __future__ import annotations

import re
from datetime import datetime


# The canonical form the fixed-format reader emits when all slots classify
# successfully: ``MM/DD/YYYY HH:MM:SS``.  Two digits, slash, two digits, slash,
# four digits, space, HH:MM:SS.
CANONICAL_RE = re.compile(
    r"^(?P<m>\d{2})/(?P<d>\d{2})/(?P<y>\d{4}) "
    r"(?P<hh>\d{2}):(?P<mm>\d{2}):(?P<ss>\d{2})$"
)


def parse_canonical_timestamp(text: str) -> datetime:
    """Parse a canonical ``MM/DD/YYYY HH:MM:SS`` string.

    Raises ``ValueError`` on any structural mismatch, out-of-range component,
    or impossible calendar date (e.g. February 30th).  The check for
    impossible dates is delegated to :func:`datetime`, which is strict.
    """
    if not isinstance(text, str):
        raise ValueError(f"expected str, got {type(text).__name__}")

    match = CANONICAL_RE.match(text)
    if not match:
        raise ValueError(f"not a canonical timestamp: {text!r}")

    # datetime() validates month/day/hour/minute/second ranges for us, and
    # raises ValueError for impossible calendar dates like 02/30.
    return datetime(
        year=int(match.group("y")),
        month=int(match.group("m")),
        day=int(match.group("d")),
        hour=int(match.group("hh")),
        minute=int(match.group("mm")),
        second=int(match.group("ss")),
    )
