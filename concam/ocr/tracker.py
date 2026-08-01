"""Trust-but-verify timestamp tracker.

Adapted verbatim (with light cleanups) from the reference camera-flight-overlay
codebase at ``camera-flight-overlay/utilities/timestamp_utils.py`` authored by
the LAE SkyCam team.  The algorithm is unchanged: maintain a trusted timeline
of recent good reads, project through transient OCR failures, and re-anchor to
a new timeline only after several consistent anomalous reads confirm that the
video clock actually jumped.

The only notable deviation from the reference is that the verbose ``print``
statements have been removed; callers that want to observe re-anchor events
should read the ``status`` string returned from :meth:`TrustButVerifyTracker.validate`.
"""

from __future__ import annotations

import datetime as _dt
from collections import deque
from typing import Optional


class TrustButVerifyTracker:
    """Maintain a trusted timeline of OCR reads and project through failures.

    Parameters
    ----------
    start_local
        A seed datetime used only to produce a projected timestamp in the rare
        case where OCR fails before any trusted reading has been established.
    frame_interval_seconds
        Nominal wall-clock interval between the frames we actually *process*
        (i.e. after any upstream subsampling).  Used as the projection step
        when we do not yet have any history.
    seconds_per_frame
        Wall-clock seconds per raw video frame.  Combined with the gap in
        frame numbers, this lets us project a precise timestamp for a given
        frame from the last trusted anchor.
    """

    def __init__(
        self,
        start_local: _dt.datetime,
        frame_interval_seconds: float,
        seconds_per_frame: float,
    ) -> None:
        self.start_local = start_local
        self.frame_interval_seconds = frame_interval_seconds
        self.seconds_per_frame = seconds_per_frame
        self.history_size = 5
        self.reanchor_threshold = 3
        self.anomaly_multiplier = 10
        # Largest calendar-date jump (in days) a contender may have relative to
        # the trusted timeline and still be eligible for re-anchoring.  A
        # legitimate midnight rollover is +1 day (time wraps 23:59:59 -> 00:00:00),
        # so 1 day of slack is allowed; anything larger is OCR date corruption
        # (e.g. a confident year/day misread such as 2026-04-11 -> 2026-04-14 or
        # 8026-04-11) and must never be promoted regardless of how
        # seconds-delta-consistent the corrupt frames are with each other.
        self.max_reanchor_date_delta_days = 1
        # Count of contender reads rejected purely because their calendar date
        # was implausibly far from the trusted timeline.  Surfaced by callers so
        # date corruption never vanishes silently (no silent truncation).
        self.date_rejected_count = 0

        self.current_timestamp: Optional[_dt.datetime] = None
        self.prev_timestamp: Optional[_dt.datetime] = None

        self.trusted_timeline: deque = deque(maxlen=self.history_size)
        self.contender_timeline: list = []

    def _is_consistent_with_last(
        self,
        new_frame: int,
        new_ts: _dt.datetime,
        last_frame: int,
        last_ts: _dt.datetime,
    ) -> bool:
        if not last_ts:
            return True
        expected_dt = (new_frame - last_frame) * self.seconds_per_frame
        actual_dt = (new_ts - last_ts).total_seconds()
        is_backward = actual_dt < 0
        is_huge_jump = actual_dt > (expected_dt * self.anomaly_multiplier)
        is_stuck = actual_dt < (expected_dt / 2)
        return not (is_backward or is_huge_jump or is_stuck)

    def _date_delta_ok(
        self, new_ts: _dt.datetime, ref_ts: _dt.datetime
    ) -> bool:
        """True iff ``new_ts``'s calendar date is within slack of ``ref_ts``'s.

        The seconds-delta consistency checks in :meth:`_is_consistent_with_last`
        only look at the *elapsed* time between reads, so a run of frames that
        all share the *same wrong date* (a confident OCR year/day misread) is
        internally consistent and would otherwise re-anchor onto garbage.  This
        guard rejects any contender whose date is more than
        ``max_reanchor_date_delta_days`` away from the trusted date, while still
        admitting a legitimate midnight rollover (date + 1).
        """
        delta_days = abs((new_ts.date() - ref_ts.date()).days)
        return delta_days <= self.max_reanchor_date_delta_days

    def _is_non_decreasing(self, new_ts: Optional[_dt.datetime]) -> bool:
        if self.prev_timestamp is None or new_ts is None:
            return True
        return new_ts >= self.prev_timestamp

    def validate(
        self,
        frame_num: int,
        ocr_ts: Optional[_dt.datetime],
        is_valid: bool,
    ) -> tuple[_dt.datetime, str]:
        """Vet an OCR reading for ``frame_num`` and return ``(timestamp, status)``.

        ``is_valid`` should be False when OCR failed entirely or produced an
        unparseable string; ``ocr_ts`` may still be populated from a partial
        recovery but it will be ignored.
        """
        self.prev_timestamp = self.current_timestamp

        # Phase 1: building the initial trusted timeline.
        if len(self.trusted_timeline) < self.history_size:
            if is_valid and ocr_ts is not None:
                if not self.trusted_timeline:
                    self.current_timestamp = ocr_ts
                    self.trusted_timeline.append((frame_num, ocr_ts))
                    status = (
                        f"building_history "
                        f"({len(self.trusted_timeline)}/{self.history_size})"
                    )
                else:
                    last_frame, last_ts = self.trusted_timeline[-1]
                    if self._is_consistent_with_last(
                        frame_num, ocr_ts, last_frame, last_ts
                    ):
                        self.current_timestamp = ocr_ts
                        self.trusted_timeline.append((frame_num, ocr_ts))
                        status = (
                            f"building_history "
                            f"({len(self.trusted_timeline)}/{self.history_size})"
                        )
                    else:
                        gap = (frame_num - last_frame) * self.seconds_per_frame
                        self.current_timestamp = last_ts + _dt.timedelta(seconds=gap)
                        status = "building_history_invalid_ocr"
            else:
                if self.trusted_timeline:
                    last_frame, last_ts = self.trusted_timeline[-1]
                    gap = (frame_num - last_frame) * self.seconds_per_frame
                    self.current_timestamp = last_ts + _dt.timedelta(seconds=gap)
                else:
                    base = self.current_timestamp or self.start_local
                    self.current_timestamp = base + _dt.timedelta(
                        seconds=self.frame_interval_seconds
                    )
                status = "building_history_invalid_ocr"
            return self.current_timestamp, status

        # Phase 2: trusted timeline exists; verify.
        last_frame, last_ts = self.trusted_timeline[-1]
        if (
            is_valid
            and ocr_ts is not None
            and self._is_consistent_with_last(frame_num, ocr_ts, last_frame, last_ts)
            and self._is_non_decreasing(ocr_ts)
        ):
            # Happy path: OCR is consistent with the trusted timeline.
            self.current_timestamp = ocr_ts
            self.trusted_timeline.append((frame_num, ocr_ts))
            if self.contender_timeline:
                self.contender_timeline.clear()
            status = "consistent"
        else:
            # Anomaly: project from the trusted timeline for this frame.
            gap = (frame_num - last_frame) * self.seconds_per_frame
            self.current_timestamp = last_ts + _dt.timedelta(seconds=gap)
            status = "anomaly_projected"
            # Defense-in-depth: a contender whose calendar date is implausibly
            # far from the trusted date is OCR date corruption, not a real clock
            # jump.  Refuse to build a contender from it (so it stays projected)
            # regardless of how seconds-delta-consistent it is.  This also guards
            # the raw-segment (4 fps) path, which never goes through
            # ``run_ocr_stage``'s out-of-day gate.
            if (
                is_valid
                and ocr_ts is not None
                and not self._date_delta_ok(ocr_ts, last_ts)
            ):
                self.date_rejected_count += 1
                if self.contender_timeline:
                    self.contender_timeline.clear()
                return self.current_timestamp, status
            if is_valid and ocr_ts is not None and self._is_non_decreasing(ocr_ts):
                # Track whether this anomaly is the start of a new timeline.
                if not self.contender_timeline:
                    self.contender_timeline.append((frame_num, ocr_ts))
                else:
                    c_frame, c_ts = self.contender_timeline[-1]
                    if self._is_consistent_with_last(
                        frame_num, ocr_ts, c_frame, c_ts
                    ):
                        self.contender_timeline.append((frame_num, ocr_ts))
                    else:
                        self.contender_timeline = [(frame_num, ocr_ts)]
                if len(self.contender_timeline) >= self.reanchor_threshold:
                    self.trusted_timeline = deque(
                        self.contender_timeline, maxlen=self.history_size
                    )
                    self.contender_timeline.clear()
                    self.current_timestamp = ocr_ts
                    status = "RE-ANCHORED"
            elif self.contender_timeline:
                self.contender_timeline.clear()

        return self.current_timestamp, status
