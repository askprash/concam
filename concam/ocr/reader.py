"""Fixed-format timestamp reader for the MIT Green Building camera overlay.

The overlay renders ``MM/DD/YYYY HH:MM:SS <DAY>`` in a fixed font at a fixed
position in every frame.  Rather than run a general-purpose OCR engine per
frame, we exploit the fact that the grammar and glyph set are completely
closed: 10 digits, ``/``, and ``:``.  Each character is classified by template
matching against a tiny dictionary captured once during calibration.

Flow:
    1. Crop the overlay ROI from the top-right of the frame.
    2. Binarize by thresholding on brightness.
    3. For each of 18 fixed slots, crop, tight-normalize the glyph, and score
       it against every template of the expected *kind* (digit vs. slash vs.
       colon), using normalized cross-correlation.
    4. Assemble the characters into a canonical string and parse.
    5. Report per-slot and overall confidence.  When confidence is below a
       threshold the caller can route to the optional EasyOCR fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from typing import Optional

import cv2
import numpy as np

from concam.config import OcrConfig
from concam.ocr.parser import parse_canonical_timestamp


# Slot layout is inherent to the camera overlay font and duplicates the
# constants in ``scripts/generate_ocr_templates.py``.  These are the physical
# truths of the overlay, so there is no benefit to plumbing them through
# config — a change in the overlay font requires regenerating templates, at
# which point this file and the script change together.
SLOT_X_STARTS = (
    128, 160, 192, 224, 256, 288, 320, 352, 384, 416,   # MM/DD/YYYY
    480, 512, 544, 576, 608, 640, 672, 704,             # HH:MM:SS
)
SLOT_KINDS = (
    "digit", "digit", "slash", "digit", "digit", "slash",
    "digit", "digit", "digit", "digit",
    "digit", "digit", "colon", "digit", "digit", "colon", "digit", "digit",
)
SLOT_WIDTH = 32
SLOT_Y_TOP = 12
SLOT_HEIGHT = 48

TEMPLATE_H = 44
TEMPLATE_W = 28

# When the canonical string is built we insert a space between date and time;
# the slot sequence itself contains no space.
_SPACE_AFTER_SLOT = 9


@dataclass
class TimestampRead:
    """Outcome of reading the timestamp on a single frame."""

    parsed_dt: Optional[datetime]
    text: str
    confidence: float
    per_char_confidence: tuple[float, ...]
    method: str
    status: str

    @property
    def ok(self) -> bool:
        return self.parsed_dt is not None and self.status == "ok"


@dataclass
class _Templates:
    digit: dict[str, np.ndarray] = field(default_factory=dict)
    slash: np.ndarray | None = None
    colon: np.ndarray | None = None


def _load_templates_npz(path) -> _Templates:
    """Load templates from a numpy .npz file, returning a typed container."""
    data = np.load(path)
    tpl = _Templates()
    for key in data.files:
        arr = data[key].astype(np.float32)
        if key == "slash":
            tpl.slash = arr
        elif key == "colon":
            tpl.colon = arr
        elif len(key) == 1 and key.isdigit():
            tpl.digit[key] = arr
    if set(tpl.digit.keys()) != set("0123456789"):
        raise RuntimeError(
            f"template bank missing digits: "
            f"{sorted(set('0123456789') - set(tpl.digit.keys()))}"
        )
    if tpl.slash is None or tpl.colon is None:
        raise RuntimeError("template bank missing slash or colon")
    return tpl


def _default_templates_path():
    return resources.files("concam.ocr").joinpath("templates.npz")


def _crop_roi(frame: np.ndarray, region: tuple[int, int], position: str) -> np.ndarray:
    h, w = frame.shape[:2]
    rh, rw = region
    if position == "top_right":
        return frame[0:rh, w - rw : w]
    if position == "top_left":
        return frame[0:rh, 0:rw]
    if position == "bottom_right":
        return frame[h - rh : h, w - rw : w]
    if position == "bottom_left":
        return frame[h - rh : h, 0:rw]
    raise ValueError(f"unsupported timestamp_position: {position!r}")


def _binarize(roi: np.ndarray, threshold: int) -> np.ndarray:
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def _extract_slot(binary: np.ndarray, slot_idx: int) -> np.ndarray:
    x = SLOT_X_STARTS[slot_idx]
    bh, bw = binary.shape
    # Clamp to the ROI bounds in case a caller passes an undersized ROI.
    y0 = min(SLOT_Y_TOP, bh)
    y1 = min(SLOT_Y_TOP + SLOT_HEIGHT, bh)
    x0 = min(x, bw)
    x1 = min(x + SLOT_WIDTH, bw)
    return binary[y0:y1, x0:x1]


def _normalize_glyph(slot: np.ndarray) -> np.ndarray:
    """Tight-crop and center-pad a glyph to the canonical template size."""
    rows = np.where(slot.any(axis=1))[0]
    cols = np.where(slot.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=np.float32)
    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1
    cropped = slot[y0:y1, x0:x1]
    out = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=np.float32)
    ch, cw = cropped.shape
    if ch > TEMPLATE_H or cw > TEMPLATE_W:
        cropped = cv2.resize(cropped, (min(cw, TEMPLATE_W), min(ch, TEMPLATE_H)))
        ch, cw = cropped.shape
    dy = (TEMPLATE_H - ch) // 2
    dx = (TEMPLATE_W - cw) // 2
    out[dy : dy + ch, dx : dx + cw] = cropped.astype(np.float32)
    return out


def _score(glyph: np.ndarray, template: np.ndarray) -> float:
    """Normalized cross-correlation between a glyph and a template.

    Both inputs are float32 arrays of the same shape.  Result is in ``[0, 1]``
    for matched polarity and identical shape; we only ever compare same-shape
    arrays here.
    """
    g = glyph.flatten()
    t = template.flatten()
    g_norm = np.linalg.norm(g)
    t_norm = np.linalg.norm(t)
    if g_norm == 0 or t_norm == 0:
        return 0.0
    return float(np.dot(g, t) / (g_norm * t_norm))


def _classify_slot(
    glyph: np.ndarray, kind: str, templates: _Templates
) -> tuple[str, float]:
    """Return ``(character, confidence)`` for a single slot.

    For digits, confidence is the margin between the best and second-best
    template score (scaled to ``[0, 1]``), which captures how unambiguous the
    decision is.  For slash/colon there is only one template of that kind, so
    confidence is simply the NCC score with its template.
    """
    if kind == "slash":
        return "/", _score(glyph, templates.slash)
    if kind == "colon":
        return ":", _score(glyph, templates.colon)
    # Digit: pick the arg-max over the 10 templates.
    scores = [(ch, _score(glyph, tpl)) for ch, tpl in templates.digit.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    best_ch, best_score = scores[0]
    runner_up = scores[1][1] if len(scores) > 1 else 0.0
    # Margin-based confidence: widely separated top-2 → high confidence.
    # Clamp to [0, 1].  An exact template match for the winner with no
    # separation from the runner-up still reports moderate confidence.
    if best_score <= 0:
        return best_ch, 0.0
    margin = (best_score - runner_up) / best_score
    # Blend margin with absolute match quality so a poor best-match with a
    # wide margin doesn't masquerade as high confidence.
    confidence = float(max(0.0, min(1.0, best_score * (0.5 + 0.5 * margin))))
    return best_ch, confidence


class FixedFormatTimestampReader:
    """Read ``MM/DD/YYYY HH:MM:SS`` from a fixed overlay region.

    Parameters
    ----------
    config
        OCR configuration; controls ROI position/size and the confidence
        threshold at which the EasyOCR fallback is triggered.
    templates_path
        Optional override for the templates ``.npz`` file.  Defaults to the
        one shipped with the package.
    easyocr_reader
        Optional prebuilt EasyOCR ``Reader`` instance.  If omitted and
        fallback is ever needed, one is lazily constructed.
    """

    def __init__(
        self,
        config: OcrConfig,
        templates_path=None,
        easyocr_reader=None,
    ) -> None:
        self.config = config
        self._templates = _load_templates_npz(
            templates_path if templates_path is not None else _default_templates_path()
        )
        self._easyocr_reader = easyocr_reader  # lazy

    # -- public API -------------------------------------------------------

    def read(self, frame: np.ndarray) -> TimestampRead:
        """Read the timestamp from a single full frame.

        Returns a :class:`TimestampRead`; ``parsed_dt`` is ``None`` when both
        the primary and (if enabled) fallback paths fail.
        """
        roi = _crop_roi(frame, self.config.timestamp_region, self.config.timestamp_position)
        binary = _binarize(roi, threshold=200)

        chars: list[str] = []
        confs: list[float] = []
        for slot_idx, kind in enumerate(SLOT_KINDS):
            slot = _extract_slot(binary, slot_idx)
            glyph = _normalize_glyph(slot)
            ch, conf = _classify_slot(glyph, kind, self._templates)
            chars.append(ch)
            confs.append(conf)

        text = "".join(chars[:_SPACE_AFTER_SLOT + 1]) + " " + "".join(chars[_SPACE_AFTER_SLOT + 1:])
        overall = float(np.mean(confs)) if confs else 0.0

        parsed: Optional[datetime] = None
        status = "ok"
        try:
            parsed = parse_canonical_timestamp(text)
        except ValueError:
            status = "parse_failed"

        # Primary template path result.
        if parsed is not None and overall >= self.config.fallback_confidence_threshold:
            return TimestampRead(
                parsed_dt=parsed,
                text=text,
                confidence=overall,
                per_char_confidence=tuple(confs),
                method="template",
                status="ok",
            )

        # Fallback path: either parsing failed or confidence too low.
        fb = self._fallback(frame)
        if fb is not None:
            fb_dt, fb_text, fb_conf = fb
            return TimestampRead(
                parsed_dt=fb_dt,
                text=fb_text,
                confidence=fb_conf,
                per_char_confidence=tuple(confs),
                method="easyocr_fallback",
                status="ok",
            )

        # Both paths failed; return whatever the template path produced so the
        # tracker can decide how to proceed.
        return TimestampRead(
            parsed_dt=parsed,
            text=text,
            confidence=overall,
            per_char_confidence=tuple(confs),
            method="template",
            status=status if parsed is None else "low_confidence",
        )

    # -- fallback ---------------------------------------------------------

    def _fallback(self, frame: np.ndarray):
        """Optional EasyOCR fallback.  Returns ``(dt, text, confidence)`` or None.

        Silently disabled if EasyOCR is not installed.
        """
        try:
            import easyocr  # noqa: F401  (probe only)
        except Exception:
            return None

        if self._easyocr_reader is None:
            import easyocr  # type: ignore
            self._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        roi = _crop_roi(frame, self.config.timestamp_region, self.config.timestamp_position)
        # Use the same bright-text mask the reference code uses.
        if roi.ndim == 2:
            rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        else:
            rgb = roi
        mask = cv2.inRange(rgb, (230, 230, 230), (255, 255, 255))
        masked = cv2.bitwise_and(rgb, cv2.merge([mask, mask, mask]))
        result = self._easyocr_reader.readtext(
            masked,
            paragraph=False,
            allowlist="0123456789:/-AMPONTUEWDHFRISamp ",
            rotation_info=[0],
        )
        if not result:
            return None

        # Import inside to keep the reference copy close to its usage.  This
        # stitches tokens together into a canonical string.
        from concam.ocr._fallback_clean import clean_easyocr_output

        stitched = clean_easyocr_output(result)
        try:
            parsed = parse_canonical_timestamp(
                _coerce_to_canonical(stitched)
            )
        except ValueError:
            return None

        # Average EasyOCR's per-token confidences for reporting.
        confs = [float(item[2]) for item in result if len(item) >= 3]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return parsed, _coerce_to_canonical(stitched), avg_conf


def _coerce_to_canonical(text: str) -> str:
    """Normalize an EasyOCR-cleaned string to ``MM/DD/YYYY HH:MM:SS`` form.

    The reference :func:`clean_easyocr_output` returns a handful of shapes
    depending on what it could recover; here we only accept the
    date-plus-24h-time shape and drop any trailing AM/PM (the camera overlay
    is 24-hour, so an AM/PM suffix is noise from the token stitcher).
    """
    # Strip optional trailing AM/PM token.
    cleaned = text.strip()
    if cleaned.endswith(" AM") or cleaned.endswith(" PM"):
        cleaned = cleaned[:-3].rstrip()
    return cleaned
