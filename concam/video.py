"""Video frame extraction utilities.

Provides a single canonical interface for decoding frames from an MP4 into
BGR arrays via PyAV, eliminating the divergent copies that previously lived
in individual scripts.

Interface
---------
Both functions share the same contract:

``path`` : str | Path
    Path to the video file.
``indices`` : Iterable[int]
    Frame indices to decode.  Duplicates are silently deduplicated.
    Negative indices are silently dropped by both functions.
Returns : dict[int, np.ndarray]
    Maps requested frame index → HxWx3 uint8 BGR array.

    **Out-of-range behaviour differs between the two functions** (preserved
    from the originals):

    * ``decode_frames`` (seek strategy): the PyAV seek is clamped by the
      codec to the last keyframe in the file.  Consequently, indices past
      the end of the video are **included** in the returned dict and map to
      the last decodable frame — callers that supply only in-range indices
      are unaffected, but callers should not rely on out-of-range indices
      being omitted.

    * ``decode_frames_sequential`` (scan strategy): the scan loop exits
      when the first wanted index that has not yet been seen is larger than
      any remaining decoded frame.  Out-of-range indices are therefore
      silently **omitted** from the result.

    In practice the two strategies agree for every index that is actually
    present in the video.

``upscale_to`` : (width, height) | None
    When set and the decoded frame is smaller than the requested size,
    each frame is bilinearly upscaled (``cv2.INTER_LINEAR``) to match.
    Used when an archive video is lower-resolution than the projection
    calibration.

Function choice
---------------
* ``decode_frames`` — random-access seek-per-frame strategy.  Use for
  sparse / small sets of frame indices (< ~dozens) spread across the video.
  Ports the seek logic from ``scripts/detection_validation_extract.py``
  exactly (same PTS arithmetic, same seek flags, same BGR output).

* ``decode_frames_sequential`` — single forward scan strategy.  Use when
  indices are many or spread across the full day; decodes the video once
  linearly and collects frames as they arrive.  Ports the scan loop from
  ``scripts/tune_from_episode_labels.extract_crops`` (decode half only;
  cropping remains in the caller).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import av
import cv2
import numpy as np


def decode_frames(
    path: str | Path,
    indices: Iterable[int],
    *,
    upscale_to: tuple[int, int] | None = None,
) -> dict[int, np.ndarray]:
    """Decode specific frame indices via random-access seek-per-frame.

    For each requested index the container is seeked backward to the nearest
    keyframe at or before the target PTS, then frames are decoded until one
    whose PTS meets or exceeds the target is found.  The PTS→frame-index
    mapping is::

        target_time_s = (target_idx / total_frames) * duration_s   # if total_frames > 0
        target_pts    = int(target_time_s / float(time_base))

    This arithmetic is preserved verbatim from the reference implementation in
    ``scripts/detection_validation_extract._decode_frames``.

    NOTE ON BEHAVIOUR FIDELITY
    --------------------------
    The reference implementation uses ``if total_frames else 0.0`` as its
    zero-guard, which means if total_frames is 0 the seek always targets PTS=0
    for every index.  This edge case is preserved here; ``decode_frames_sequential``
    uses a different guard (``max(1, …)``).  On real MIT timelapse videos the two
    strategies agree for every index actually present in the video.
    """
    path = Path(path)
    wanted = sorted(set(i for i in indices if i >= 0))
    if not wanted:
        return {}

    out: dict[int, np.ndarray] = {}
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        duration_s = (
            float(stream.duration * stream.time_base) if stream.duration else 0.0
        )
        total_frames = (
            int(stream.frames)
            if stream.frames
            else int(round(duration_s * float(stream.average_rate or 30)))
        )
        for target_idx in wanted:
            target_time_s = (
                (target_idx / total_frames) * duration_s if total_frames else 0.0
            )
            target_pts = int(target_time_s / float(time_base))
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            decoded = None
            for frame in container.decode(stream):
                decoded = frame
                if frame.pts is not None and frame.pts >= target_pts:
                    break
            if decoded is not None:
                arr = decoded.to_ndarray(format="bgr24")
                if upscale_to is not None and (arr.shape[1], arr.shape[0]) != upscale_to:
                    arr = cv2.resize(arr, upscale_to, interpolation=cv2.INTER_LINEAR)
                out[target_idx] = arr
    finally:
        container.close()
    return out


def decode_frames_sequential(
    path: str | Path,
    indices: Iterable[int],
    *,
    upscale_to: tuple[int, int] | None = None,
) -> dict[int, np.ndarray]:
    """Decode specific frame indices via a single sequential forward scan.

    Seeks to just before the first wanted frame (to avoid replaying the
    whole preceding video), then iterates frames in decode order, collecting
    each wanted frame as it arrives.  The PTS→frame-index mapping is::

        fidx = int(round(float(frame.pts * stream.time_base * stream.average_rate)))

    This arithmetic is preserved verbatim from ``scripts/tune_from_episode_labels.
    extract_crops`` (decode strategy; cropping is removed — callers receive raw
    BGR arrays and apply their own crops).

    Use this strategy when the index set is large or spread across the full
    day; it is significantly faster than seek-per-frame in that regime.
    """
    path = Path(path)
    wanted_set = set(i for i in indices if i >= 0)
    wanted = sorted(wanted_set)
    if not wanted:
        return {}

    out: dict[int, np.ndarray] = {}
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Seek to just before the first wanted frame to avoid decoding the
        # whole preceding video (same strategy as tune_from_episode_labels).
        first = wanted[0]
        if stream.time_base and stream.average_rate:
            pts = int(first / float(stream.average_rate) / float(stream.time_base))
            container.seek(max(pts - 10, 0), stream=stream, any_frame=False)
        wi = 0
        for pkt_frame in container.decode(stream):
            fidx = pkt_frame.pts * stream.time_base * stream.average_rate
            fidx = int(round(float(fidx)))
            while wi < len(wanted) and wanted[wi] < fidx:
                wi += 1
            if wi >= len(wanted):
                break
            if wanted[wi] != fidx:
                continue
            arr = pkt_frame.to_ndarray(format="bgr24")
            if upscale_to is not None and (arr.shape[1], arr.shape[0]) != upscale_to:
                arr = cv2.resize(arr, upscale_to, interpolation=cv2.INTER_LINEAR)
            out[fidx] = arr
            wi += 1
    finally:
        container.close()
    return out
