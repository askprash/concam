"""Fixed-format timestamp reader for the MIT Green Building camera overlay.

The overlay renders ``MM/DD/YYYY HH:MM:SS <DAY>`` in a fixed font at a fixed
position in every frame.  Rather than run a general-purpose OCR engine per
frame, we exploit the fact that the grammar and glyph set are completely
closed: 10 digits, ``/``, and ``:``.  Each character is classified by template
matching against a tiny dictionary captured once during calibration.

This module owns the *composition*: it loads the template bank, builds the two
:class:`~concam.ocr.engines.TimestampEngine` adapters (template matcher as the
primary, EasyOCR as the optional fallback), and runs them primary-then-fallback
to produce a :class:`TimestampRead`.  The per-engine pixel work lives in
:mod:`concam.ocr.engines`; the shared preprocessing in
:mod:`concam.ocr._preprocess`.

Flow:
    1. The template engine crops the ROI, binarizes, classifies all 18 fixed
       slots, assembles a canonical string, and parses it.
    2. If it parsed AND its overall confidence clears the threshold, that
       result is used.
    3. Otherwise the EasyOCR fallback engine runs (lazily; silently skipped if
       EasyOCR is not installed).  If it yields a parseable timestamp, that is
       used instead.
    4. If both fail, the template engine's (un-parseable / low-confidence)
       result is returned so the tracker can decide how to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from typing import Optional

import numpy as np

from concam.config import OcrConfig
from concam.ocr.engines import EasyOcrEngine, EngineRead, TemplateMatchEngine


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


class FixedFormatTimestampReader:
    """Read ``MM/DD/YYYY HH:MM:SS`` from a fixed overlay region.

    Composes two :class:`~concam.ocr.engines.TimestampEngine` adapters: a
    template matcher (primary) and EasyOCR (optional fallback).

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
        templates = _load_templates_npz(
            templates_path if templates_path is not None else _default_templates_path()
        )
        self._primary = TemplateMatchEngine(
            templates, config.timestamp_region, config.timestamp_position
        )
        self._fallback_engine = EasyOcrEngine(
            config.timestamp_region,
            config.timestamp_position,
            easyocr_reader=easyocr_reader,
        )

    # -- public API -------------------------------------------------------

    def read(self, frame: np.ndarray) -> TimestampRead:
        """Read the timestamp from a single full frame.

        Returns a :class:`TimestampRead`; ``parsed_dt`` is ``None`` when both
        the primary and (if enabled) fallback paths fail.
        """
        primary: EngineRead = self._primary.read(frame)

        # Primary template path result.
        if (
            primary.parsed_dt is not None
            and primary.confidence >= self.config.fallback_confidence_threshold
        ):
            return TimestampRead(
                parsed_dt=primary.parsed_dt,
                text=primary.text,
                confidence=primary.confidence,
                per_char_confidence=primary.per_char_confidence,
                method="template",
                status="ok",
            )

        # Fallback path: either parsing failed or confidence too low.
        fb = self._fallback_engine.read(frame)
        if fb is not None:
            return TimestampRead(
                parsed_dt=fb.parsed_dt,
                text=fb.text,
                confidence=fb.confidence,
                # Carry the template engine's per-slot confidences for parity
                # with the pre-seam behaviour (EasyOCR has no per-glyph notion).
                per_char_confidence=primary.per_char_confidence,
                method="easyocr_fallback",
                status="ok",
            )

        # Both paths failed; return whatever the template path produced so the
        # tracker can decide how to proceed.
        return TimestampRead(
            parsed_dt=primary.parsed_dt,
            text=primary.text,
            confidence=primary.confidence,
            per_char_confidence=primary.per_char_confidence,
            method="template",
            status=primary.status if primary.parsed_dt is None else "low_confidence",
        )
