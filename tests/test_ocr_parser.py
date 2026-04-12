"""Property tests for the canonical timestamp parser."""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given, strategies as st

from concam.ocr.parser import parse_canonical_timestamp


# ---------------------------------------------------------------------------
# Round-trip property: every valid datetime in the overlay range must format
# to a canonical string that parses back to the same datetime.

# The overlay shows four-digit years, so 1900-2100 is a safe window.
_dt_strategy = st.datetimes(
    min_value=datetime(1900, 1, 1),
    max_value=datetime(2100, 12, 31, 23, 59, 59),
).map(lambda d: d.replace(microsecond=0))


@given(dt=_dt_strategy)
def test_round_trip(dt: datetime) -> None:
    text = dt.strftime("%m/%d/%Y %H:%M:%S")
    assert parse_canonical_timestamp(text) == dt


# ---------------------------------------------------------------------------
# Calendar validity: impossible dates must raise.

@pytest.mark.parametrize(
    "bad",
    [
        "02/30/2026 00:00:00",   # Feb 30
        "04/31/2026 00:00:00",   # Apr 31
        "13/01/2026 00:00:00",   # month 13
        "00/15/2026 00:00:00",   # month 0
        "01/00/2026 00:00:00",   # day 0
        "02/29/2027 00:00:00",   # non-leap year
    ],
)
def test_impossible_dates_fail(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_canonical_timestamp(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "04-12-2026 00:00:00",   # dashes not slashes
        "04/12/2026 00-00-00",   # dashes in time
        "04/12/2026T00:00:00",   # T separator (ISO, not ours)
        "04/12/26 00:00:00",     # 2-digit year
        "04/12/2026 00:00",      # missing seconds
        "4/12/2026 00:00:00",    # 1-digit month (canonical is always 2)
        "04/12/2026  00:00:00",  # double space
        "",
        "not a timestamp",
        "04/12/2026 25:00:00",   # hour 25
        "04/12/2026 00:60:00",   # minute 60
        "04/12/2026 00:00:60",   # second 60
    ],
)
def test_invalid_structure_fails(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_canonical_timestamp(bad)


def test_non_string_fails() -> None:
    with pytest.raises(ValueError):
        parse_canonical_timestamp(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_canonical_timestamp(123456)  # type: ignore[arg-type]
