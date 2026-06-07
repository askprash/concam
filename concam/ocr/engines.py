"""Timestamp OCR engines behind a common adapter seam.

There are two genuinely different ways to read the wall-clock overlay off a
frame:

* :class:`TemplateMatchEngine` -- the fast, dependency-free primary path that
  classifies each fixed glyph slot by normalized cross-correlation against a
  tiny template bank.
* :class:`EasyOcrEngine` -- a heavy, optional fallback that runs a general OCR
  model and stitches its tokens into a canonical string.

Both implement the :class:`TimestampEngine` protocol, so
:class:`~concam.ocr.reader.FixedFormatTimestampReader` can *compose* them
(primary, then fallback) instead of hard-coding EasyOCR inline.

EasyOCR is an optional extra (``uv sync --extra ocr``).  Its adapter must NOT
import ``easyocr`` at module load -- the import and the model construction stay
lazy, happening only on the first ``read`` that actually needs the fallback, so
template-only users never pull in the heavy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from concam.ocr._preprocess import (
    SLOT_KINDS,
    binarize,
    crop_roi,
    extract_slot,
    normalize_glyph,
)
from concam.ocr.parser import parse_canonical_timestamp


# When the canonical string is built we insert a space between date and time;
# the slot sequence itself contains no space.
_SPACE_AFTER_SLOT = 9


@dataclass
class EngineRead:
    """A single engine's attempt at reading the timestamp on one frame.

    ``parsed_dt`` is ``None`` when the engine produced text that did not parse.
    ``per_char_confidence`` is empty for engines that have no per-glyph notion
    (e.g. EasyOCR).  ``status`` is ``"ok"`` when the text parsed and
    ``"parse_failed"`` otherwise; the composing reader uses it, together with
    ``confidence``, to decide whether to fall through to the next engine.
    """

    parsed_dt: Optional[datetime]
    text: str
    confidence: float
    per_char_confidence: tuple[float, ...]
    status: str


@runtime_checkable
class TimestampEngine(Protocol):
    """Reads a timestamp from a full frame, or returns ``None`` if it cannot run.

    Returning ``None`` means "this engine is unavailable / produced nothing"
    (e.g. EasyOCR is not installed, or it returned no tokens) -- distinct from
    returning an :class:`EngineRead` whose ``parsed_dt`` is ``None`` (the engine
    ran but its text did not parse).
    """

    def read(self, frame: np.ndarray) -> Optional[EngineRead]:
        ...


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


def _classify_slot(glyph: np.ndarray, kind: str, templates) -> tuple[str, float]:
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
    # Margin-based confidence: widely separated top-2 -> high confidence.
    # Clamp to [0, 1].  An exact template match for the winner with no
    # separation from the runner-up still reports moderate confidence.
    if best_score <= 0:
        return best_ch, 0.0
    margin = (best_score - runner_up) / best_score
    # Blend margin with absolute match quality so a poor best-match with a
    # wide margin doesn't masquerade as high confidence.
    confidence = float(max(0.0, min(1.0, best_score * (0.5 + 0.5 * margin))))
    return best_ch, confidence


class TemplateMatchEngine:
    """Primary engine: per-slot template matching against the glyph bank."""

    def __init__(self, templates, region: tuple[int, int], position: str) -> None:
        self._templates = templates
        self._region = region
        self._position = position

    def read(self, frame: np.ndarray) -> Optional[EngineRead]:
        roi = crop_roi(frame, self._region, self._position)
        binary = binarize(roi, threshold=200)

        chars: list[str] = []
        confs: list[float] = []
        for slot_idx, kind in enumerate(SLOT_KINDS):
            slot = extract_slot(binary, slot_idx)
            glyph = normalize_glyph(slot)
            ch, conf = _classify_slot(glyph, kind, self._templates)
            chars.append(ch)
            confs.append(conf)

        text = (
            "".join(chars[: _SPACE_AFTER_SLOT + 1])
            + " "
            + "".join(chars[_SPACE_AFTER_SLOT + 1 :])
        )
        overall = float(np.mean(confs)) if confs else 0.0

        parsed: Optional[datetime] = None
        status = "ok"
        try:
            parsed = parse_canonical_timestamp(text)
        except ValueError:
            status = "parse_failed"

        return EngineRead(
            parsed_dt=parsed,
            text=text,
            confidence=overall,
            per_char_confidence=tuple(confs),
            status=status,
        )


def _coerce_to_canonical(text: str) -> str:
    """Normalize an EasyOCR-cleaned string to ``MM/DD/YYYY HH:MM:SS`` form.

    :func:`~concam.ocr._fallback_clean.clean_easyocr_output` returns a handful
    of shapes depending on what it could recover; here we drop any trailing
    AM/PM token left on the canonical string (the overlay is 24-hour, and the
    stitcher has already folded any AM/PM into the hour).
    """
    cleaned = text.strip()
    if cleaned.endswith(" AM") or cleaned.endswith(" PM"):
        cleaned = cleaned[:-3].rstrip()
    return cleaned


class EasyOcrEngine:
    """Fallback engine wrapping EasyOCR.  Lazily imports / builds the model.

    Construction is cheap and never touches ``easyocr``.  The first ``read``
    that needs it imports the package and builds the ``Reader``; if the package
    is not installed, ``read`` returns ``None`` (fallback silently disabled).
    """

    def __init__(
        self, region: tuple[int, int], position: str, easyocr_reader=None
    ) -> None:
        self._region = region
        self._position = position
        self._reader = easyocr_reader  # lazy unless injected

    def _ensure_reader(self):
        """Return the EasyOCR reader, building it lazily, or ``None`` if absent."""
        if self._reader is not None:
            return self._reader
        try:
            import easyocr  # type: ignore
        except Exception:
            return None
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return self._reader

    def read(self, frame: np.ndarray) -> Optional[EngineRead]:
        reader = self._ensure_reader()
        if reader is None:
            return None

        roi = crop_roi(frame, self._region, self._position)
        # Use the same bright-text mask the reference code uses.
        if roi.ndim == 2:
            rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        else:
            rgb = roi
        mask = cv2.inRange(rgb, (230, 230, 230), (255, 255, 255))
        masked = cv2.bitwise_and(rgb, cv2.merge([mask, mask, mask]))
        result = reader.readtext(
            masked,
            paragraph=False,
            allowlist="0123456789:/-AMPONTUEWDHFRISamp ",
            rotation_info=[0],
        )
        if not result:
            return None

        from concam.ocr._fallback_clean import clean_easyocr_output

        stitched = clean_easyocr_output(result)
        canonical = _coerce_to_canonical(stitched)
        try:
            parsed = parse_canonical_timestamp(canonical)
        except ValueError:
            return None

        # Average EasyOCR's per-token confidences for reporting.
        confs = [float(item[2]) for item in result if len(item) >= 3]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return EngineRead(
            parsed_dt=parsed,
            text=canonical,
            confidence=avg_conf,
            per_char_confidence=(),
            status="ok",
        )
