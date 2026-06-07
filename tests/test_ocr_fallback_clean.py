"""Truth-table tests for the EasyOCR token-stitching fallback.

The overlay is nominally 24-hour, but EasyOCR can still emit a stray
``AM``/``PM`` token (both are in the reader's allowlist).  When that happens the
suffix MUST drive a 12->24 hour conversion; emitting the 12-hour hour verbatim
silently corrupts every afternoon reading and midnight.  These cases pin that
conversion.

We drive the public entry point (:func:`clean_easyocr_output`) for the
end-to-end wiring and exercise :func:`_canon_time` directly for the corner-case
truth table -- the public path can only reach the same code, and the direct
calls keep each corner unambiguous.
"""

from __future__ import annotations

import pytest

from concam.ocr._fallback_clean import _canon_time, clean_easyocr_output


def _tokens(*texts: str):
    """Shape plain strings like EasyOCR ``(bbox, text, conf)`` results."""
    return [([[0, 0], [1, 1]], t, 0.99) for t in texts]


# ---------------------------------------------------------------------------
# _canon_time: the 12->24 hour truth table that the bug got wrong.

@pytest.mark.parametrize(
    "h, mi, s, ampm, expected",
    [
        ("12", "00", "00", "AM", "00:00:00"),   # midnight: 12 AM -> 00
        ("12", "30", "00", "PM", "12:30:00"),   # noon: 12 PM stays 12
        ("01", "00", "00", "PM", "13:00:00"),   # 1 PM -> 13
        ("11", "59", "00", "AM", "11:59:00"),   # last AM hour unchanged
        ("11", "59", "00", "PM", "23:59:00"),   # last PM hour -> 23
        # 24-hour path (no suffix) is untouched.
        ("23", "15", "07", None, "23:15:07"),
        ("00", "00", "00", None, "00:00:00"),
        # Invalid inputs return "" (contract preserved).
        ("13", "00", "00", "PM", ""),           # 13 invalid on a 12h clock
        ("00", "00", "00", "AM", ""),           # 0 invalid on a 12h clock
        ("24", "00", "00", None, ""),           # 24 invalid on a 24h clock
        ("10", "60", "00", "AM", ""),           # minute out of range
    ],
)
def test_canon_time_truth_table(h, mi, s, ampm, expected) -> None:
    assert _canon_time(h, mi, s, ampm) == expected


# ---------------------------------------------------------------------------
# Public entry point: a stray PM token must reach the 24-hour conversion, and
# case must not matter.

def test_public_path_pm_converts_to_24h() -> None:
    out = clean_easyocr_output(_tokens("04/12/2026", "01:30:00", "PM"))
    assert out == "04/12/2026 13:30:00"


def test_public_path_12am_is_midnight() -> None:
    out = clean_easyocr_output(_tokens("04/12/2026", "12:00:00", "am"))
    assert out == "04/12/2026 00:00:00"


def test_public_path_plain_24h_unchanged() -> None:
    out = clean_easyocr_output(_tokens("04/12/2026", "23:15:07"))
    assert out == "04/12/2026 23:15:07"
