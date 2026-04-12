"""Tests for the Hough+Canny contrail detection module."""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from concam.config import DetectionConfig
from concam.detection import DetectionResult, detect
from concam.projection import Rect


@pytest.fixture
def config() -> DetectionConfig:
    """Default detection config with relaxed thresholds for synthetic tests."""
    return DetectionConfig(
        score_threshold=0.3,
        canny_low=50,
        canny_high=150,
        hough_threshold=20,
        hough_min_line_length=20,
        hough_max_line_gap=5,
        roi_padding=20,
    )


def _make_frame_with_line(
    width: int = 200,
    height: int = 200,
    pt1: tuple[int, int] = (10, 100),
    pt2: tuple[int, int] = (190, 100),
    thickness: int = 2,
    bg: int = 30,
    fg: int = 220,
) -> np.ndarray:
    """Create a grayscale frame with a single bright line on a dark background."""
    frame = np.full((height, width), bg, dtype=np.uint8)
    cv2.line(frame, pt1, pt2, fg, thickness)
    return frame


class TestDetectClearLine:
    """A synthetic frame with a clear line should score high."""

    def test_horizontal_line_scores_high(self, config: DetectionConfig) -> None:
        frame = _make_frame_with_line(width=200, height=200)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, config)
        assert result.score > 0.5
        assert result.pixel_line is not None
        assert result.method == "hough_canny"

    def test_diagonal_line_scores_high(self, config: DetectionConfig) -> None:
        frame = _make_frame_with_line(
            width=300, height=300, pt1=(10, 10), pt2=(290, 290), thickness=2
        )
        roi = Rect(x=0, y=0, w=300, h=300)
        result = detect(frame, roi, config)
        assert result.score > 0.5
        assert result.pixel_line is not None

    def test_line_in_subroi(self, config: DetectionConfig) -> None:
        """Line drawn inside a sub-ROI; pixel_line should be in full-frame coords."""
        frame = np.full((400, 400), 30, dtype=np.uint8)
        # Draw line from (110, 200) to (290, 200) in frame coords
        cv2.line(frame, (110, 200), (290, 200), 220, 2)
        roi = Rect(x=100, y=150, w=200, h=100)
        result = detect(frame, roi, config)
        assert result.score > 0.4
        assert result.pixel_line is not None
        # Line endpoints should be in full-frame coordinates (x >= 100, y >= 150)
        x1, y1, x2, y2 = result.pixel_line
        assert x1 >= 100 and x2 >= 100
        assert y1 >= 150 and y2 >= 150


class TestDetectBlankFrame:
    """A blank or uniform frame should score near zero."""

    def test_black_frame(self, config: DetectionConfig) -> None:
        frame = np.zeros((200, 200), dtype=np.uint8)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, config)
        assert result.score < 0.1
        assert result.pixel_line is None

    def test_white_frame(self, config: DetectionConfig) -> None:
        frame = np.full((200, 200), 255, dtype=np.uint8)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, config)
        assert result.score < 0.1

    def test_uniform_gray_frame(self, config: DetectionConfig) -> None:
        frame = np.full((200, 200), 128, dtype=np.uint8)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, config)
        assert result.score < 0.1


class TestDetectCorruptedFrame:
    """Corrupted or degenerate inputs should not raise exceptions."""

    def test_random_noise_frame(self, config: DetectionConfig) -> None:
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(frame, roi, config)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0

    def test_zero_area_roi(self, config: DetectionConfig) -> None:
        frame = _make_frame_with_line()
        roi = Rect(x=100, y=100, w=0, h=0)
        result = detect(frame, roi, config)
        assert result.score == 0.0
        assert result.pixel_line is None

    def test_roi_outside_frame(self, config: DetectionConfig) -> None:
        frame = _make_frame_with_line(width=200, height=200)
        roi = Rect(x=300, y=300, w=100, h=100)
        result = detect(frame, roi, config)
        assert result.score == 0.0
        assert result.pixel_line is None

    def test_roi_partially_outside_frame(self, config: DetectionConfig) -> None:
        """ROI extends beyond frame edges — should clip and not crash."""
        frame = _make_frame_with_line(width=200, height=200)
        roi = Rect(x=150, y=150, w=200, h=200)
        result = detect(frame, roi, config)
        assert isinstance(result, DetectionResult)


class TestDetectBGRInput:
    """Detector should handle both grayscale and BGR input."""

    def test_bgr_frame(self, config: DetectionConfig) -> None:
        gray = _make_frame_with_line()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        roi = Rect(x=0, y=0, w=200, h=200)
        result = detect(bgr, roi, config)
        assert result.score > 0.5
        assert result.pixel_line is not None


class TestScoreNormalization:
    """Score should always be in [0, 1]."""

    def test_score_capped_at_one(self, config: DetectionConfig) -> None:
        """Even with a line spanning the full diagonal, score <= 1.0."""
        frame = _make_frame_with_line(
            width=100, height=100, pt1=(0, 0), pt2=(99, 99), thickness=3
        )
        roi = Rect(x=0, y=0, w=100, h=100)
        result = detect(frame, roi, config)
        assert result.score <= 1.0
