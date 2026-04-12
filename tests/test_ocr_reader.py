"""Integration tests for the FixedFormatTimestampReader.

These run against a real raw-segment video when it is available on the host;
otherwise they skip cleanly so the suite still passes on machines without the
camera data share mounted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from concam.config import OcrConfig
from concam.ocr import FixedFormatTimestampReader, TimestampRead


VIDEO_PATH = Path(
    "/net/d16/data/contrail-camera/raw_segments_clean/2026-04-12_00-00-00.mp4"
)


@pytest.fixture(scope="module")
def reader() -> FixedFormatTimestampReader:
    return FixedFormatTimestampReader(OcrConfig())


@pytest.fixture(scope="module")
def real_frame():
    if not VIDEO_PATH.exists():
        pytest.skip(f"{VIDEO_PATH} not available on this host")
    import cv2

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            pytest.skip("could not read frame 0")
        return frame
    finally:
        cap.release()


def test_read_returns_timestamp_read(reader, real_frame) -> None:
    result = reader.read(real_frame)
    assert isinstance(result, TimestampRead)
    assert result.parsed_dt is not None
    # The zero-th frame of the 2026-04-12 midnight segment shows 00:00:01.
    assert result.parsed_dt.year == 2026
    assert result.parsed_dt.month == 4
    assert result.parsed_dt.day == 12
    assert result.parsed_dt.hour == 0
    assert result.parsed_dt.minute == 0
    assert result.status == "ok"
    assert result.method == "template"
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.per_char_confidence) == 18


def test_read_broad_sweep_is_accurate(reader) -> None:
    """Sample 30 frames spread across the segment; all must parse cleanly."""
    if not VIDEO_PATH.exists():
        pytest.skip(f"{VIDEO_PATH} not available on this host")
    import cv2

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample = np.linspace(0, n - 1, 30).astype(int)
        parsed_count = 0
        for fn in sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fn))
            ok, frame = cap.read()
            if not ok:
                continue
            r = reader.read(frame)
            if r.parsed_dt is not None and r.status == "ok":
                parsed_count += 1
        # Allow up to one missed frame across the sweep (codec decode
        # transients are occasionally noisier than the template bank).
        assert parsed_count >= 29, (
            f"only {parsed_count}/30 frames parsed cleanly"
        )
    finally:
        cap.release()


def test_read_on_blank_frame_flags_failure(reader) -> None:
    """An all-black frame has no overlay; parsing must report failure."""
    # ROI is read from the top-right of a 3840x2160 frame by default.
    blank = np.zeros((2160, 3840, 3), dtype=np.uint8)
    result = reader.read(blank)
    # No bright pixels → every glyph normalizes to zeros → parse fails.
    assert result.parsed_dt is None
    assert result.status in {"parse_failed", "low_confidence"}


def test_read_accepts_small_exact_roi(reader) -> None:
    """Feeding a pre-cropped ROI (height=80, width=875) still works.

    This is important for the visual-spot-check script (PRD item 3) which
    stores crops rather than whole frames.
    """
    if not VIDEO_PATH.exists():
        pytest.skip(f"{VIDEO_PATH} not available on this host")
    import cv2

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    try:
        ok, frame = cap.read()
        if not ok:
            pytest.skip("could not read frame 0")
        h, w = frame.shape[:2]
        roi = frame[0:80, w - 875 : w]
        # Crop is the whole frame from the reader's perspective; feed it as-is
        # and the reader will "crop" a top-right region of the same size.
        result = reader.read(roi)
        assert result.parsed_dt is not None
    finally:
        cap.release()
