"""Hough+Canny contrail detector module.

Detects linear features (contrails) within an oriented ROI of a sky camera frame
using Canny edge detection followed by a probabilistic Hough line transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from concam.config import DetectionConfig
from concam.projection import Rect


@dataclass
class DetectionResult:
    """Result of running the contrail detector on one (frame, ROI) pair."""

    score: float  # 0-1 normalized detection confidence
    pixel_line: tuple[float, float, float, float] | None  # (x1, y1, x2, y2) full-frame coords, or None
    method: str  # e.g. "hough_canny"


def detect(
    frame: np.ndarray,
    roi: Rect,
    config: DetectionConfig,
) -> DetectionResult:
    """Run Hough+Canny contrail detection within an ROI.

    Args:
        frame: Full camera frame as a numpy array (H, W) grayscale or (H, W, 3) BGR.
        roi: Axis-aligned bounding rectangle to search within.
        config: Detection parameters (Canny thresholds, Hough params).

    Returns:
        DetectionResult with score in [0, 1], pixel_line in full-frame coordinates
        (or None if no line detected), and method="hough_canny".
    """
    # Convert to grayscale if needed
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Extract ROI, clipping to frame bounds
    h, w = gray.shape[:2]
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(w, roi.x + roi.w)
    y2 = min(h, roi.y + roi.h)

    if x2 <= x1 or y2 <= y1:
        return DetectionResult(score=0.0, pixel_line=None, method="hough_canny")

    crop = gray[y1:y2, x1:x2]

    # Canny edge detection
    edges = cv2.Canny(crop, config.canny_low, config.canny_high)

    # Probabilistic Hough line transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_line_length,
        maxLineGap=config.hough_max_line_gap,
    )

    if lines is None or len(lines) == 0:
        return DetectionResult(score=0.0, pixel_line=None, method="hough_canny")

    # Score: longest detected line normalized by the ROI diagonal.
    # This gives a 0-1 measure of how prominent the linear feature is.
    roi_diag = np.hypot(x2 - x1, y2 - y1)
    if roi_diag < 1:
        return DetectionResult(score=0.0, pixel_line=None, method="hough_canny")

    best_length = 0.0
    best_line = None
    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        length = np.hypot(lx2 - lx1, ly2 - ly1)
        if length > best_length:
            best_length = length
            best_line = (lx1, ly1, lx2, ly2)

    score = min(1.0, best_length / roi_diag)

    # Convert from ROI-local to full-frame coordinates
    if best_line is not None:
        pixel_line = (
            float(best_line[0] + x1),
            float(best_line[1] + y1),
            float(best_line[2] + x1),
            float(best_line[3] + y1),
        )
    else:
        pixel_line = None

    return DetectionResult(score=score, pixel_line=pixel_line, method="hough_canny")
