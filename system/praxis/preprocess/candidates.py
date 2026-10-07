# -*- coding: utf-8 -*-
"""Where each candidate track is, so a reviewer can see which person it is.

The confirmation screen asks one question - which of these people is the teacher - and it cannot
be answered by a whole-frame still. A reviewer shown four unmarked frames and asked to pick
track 2 has no way to tell which person that is, so a screen built that way would collect
endorsements of something nobody could identify. That is the one thing confirmation exists to
prevent.

So a candidate is shown as a crop, and the crop comes from the pose artefact: the keypoints of
that track in that frame, hulled and padded. The geometry is derived here and nothing is taken
from the client, which keeps the screen honest in both directions - the reviewer sees the person
the track number refers to, and no caller can ask for an arbitrary region of the footage.

**Pixels, and whose.** `keypoints` are stored un-letterboxed at the source resolution (see
`praxis.preprocess.pose`), so a box computed here indexes the video's own coordinate system and
needs no rescaling before it reaches an ffmpeg crop filter. `frame_width` and `frame_height`
come from the artefact's provenance rather than from the file, because the file on disk is the
blurred copy and a mismatch between the two is a fact worth failing on, not papering over.

Nothing here decodes a frame or touches ffmpeg. It converts an artefact into timestamps and
rectangles, which is what makes it testable without a video.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from praxis.preprocess.artifacts import LoadedPose

# A hull drawn through the keypoints alone ends at the wrists and the eyes, which crops the head
# and hands off the person it is meant to identify. A quarter of the hull on each side restores
# enough of the body, and enough of the room behind it to place them.
PAD_FRACTION = 0.25

# Below this a crop is a smear rather than a person, and a reviewer cannot identify it. A track
# whose hull is smaller is widened to it, which is honest: the person really is that far away.
MIN_BOX_PIXELS = 48


@dataclass(frozen=True)
class Thumbnail:
    """One still of one candidate track, as a timestamp and a rectangle to cut from it."""

    track_id: int
    index: int
    frame: int
    at_seconds: float
    box: tuple[int, int, int, int]

    def as_json(self) -> dict[str, object]:
        x, y, width, height = self.box
        return {"index": self.index, "at_seconds": round(self.at_seconds, 3),
                "frame": self.frame, "box": {"x": x, "y": y, "width": width, "height": height}}


def _even(value: int) -> int:
    """Rounded up to even. Some encoders refuse odd dimensions, and a crop is cheaper to widen
    by a pixel here than to debug as an ffmpeg exit code."""
    return value + (value % 2)


def hull(keypoints: np.ndarray, *, frame_width: int, frame_height: int,
         pad: float = PAD_FRACTION) -> tuple[int, int, int, int] | None:
    """The padded bounding box of one person's visible keypoints, clamped to the frame.

    `keypoints` is `(17, 3)` as x, y, confidence. A keypoint with zero confidence was discarded
    by the estimator and contributes nothing - included, it would drag the box to the origin and
    crop a corner of the room. Returns None when nothing is visible, which is how an absent
    track is told apart from one at the top-left.
    """
    visible = keypoints[keypoints[:, 2] > 0]
    if not len(visible):
        return None

    left, top = float(visible[:, 0].min()), float(visible[:, 1].min())
    right, bottom = float(visible[:, 0].max()), float(visible[:, 1].max())

    margin_x = max((right - left) * pad, MIN_BOX_PIXELS / 2)
    margin_y = max((bottom - top) * pad, MIN_BOX_PIXELS / 2)

    x = max(0, int(left - margin_x))
    y = max(0, int(top - margin_y))
    width = _even(min(frame_width - x, int(right + margin_x) - x))
    height = _even(min(frame_height - y, int(bottom + margin_y) - y))

    # Clamping the far edge can push the box back past the frame once the width was rounded up.
    x = max(0, min(x, frame_width - width))
    y = max(0, min(y, frame_height - height))
    if width < 2 or height < 2:
        return None
    return x, y, width, height


def _geometry(pose: LoadedPose) -> tuple[int, int, float]:
    """Frame size and sample rate out of the provenance, refusing a blank.

    A zero width or sample rate would produce boxes of nothing and timestamps that all land at
    the start, and both would look like a video problem rather than a missing provenance field.
    """
    width = int(pose.provenance.get("frame_width", 0))
    height = int(pose.provenance.get("frame_height", 0))
    sampled_fps = float(pose.provenance.get("sampled_fps", 0.0))
    if width <= 0 or height <= 0 or sampled_fps <= 0:
        raise ValueError(
            f"the artefact's provenance records {width}x{height} at {sampled_fps} fps, "
            f"which is not a geometry a crop or a timestamp can be derived from")
    return width, height, sampled_fps


def box_at(pose: LoadedPose, track_id: int, frame: int,
           *, pad: float = PAD_FRACTION) -> tuple[int, int, int, int] | None:
    """Where one track was in one sampled frame, or None if it was not locatable there.

    This is what a thumbnail request resolves, and it is addressed by frame rather than by a
    position in a strip. A position is not a stable address: the strip is `count` frames spread
    across the track's presence, so position 1 of three frames and position 1 of two frames are
    different frames, and a URL carrying the position would return a different picture depending
    on how long a strip the caller had last asked for. Two of them did.

    The frame is the caller's and the rectangle is not, which is the property that matters: a
    client can ask for any frame of this track and gets that track in it, and cannot ask for an
    arbitrary region of the room under a track number.
    """
    frames, _ = pose.track_ids.shape
    if not 0 <= frame < frames:
        return None
    slots = np.nonzero(pose.track_ids[frame] == track_id)[0]
    if not len(slots):
        return None

    frame_width, frame_height, _ = _geometry(pose)
    return hull(pose.keypoints[frame, int(slots[0])], frame_width=frame_width,
                frame_height=frame_height, pad=pad)


def present_frames(pose: LoadedPose, track_id: int) -> np.ndarray:
    """Sampled-frame indices where this track was tracked, ascending and without repeats.

    A track can occupy more than one person slot in a frame only if tracking assigned it twice,
    which it does not; `unique` is here so a malformed artefact yields one entry rather than a
    duplicated thumbnail.
    """
    rows, _ = np.nonzero(pose.track_ids == track_id)
    return np.unique(rows)


def thumbnails(pose: LoadedPose, track_id: int, *, count: int,
               pad: float = PAD_FRACTION) -> list[Thumbnail]:
    """Up to `count` stills of one track, spread across the frames it appears in.

    Spread rather than consecutive: a strip cut from one second of a thirty-minute lesson shows
    a reviewer one pose of one person, and the thing that identifies a teacher is where they are
    over the session. A frame whose keypoints are all discarded contributes nothing and is
    skipped rather than yielding an empty crop, so a track can return fewer than `count` - or,
    if it was never visible, none, which the caller reports as its own state.
    """
    if count < 1:
        return []

    _, _, sampled_fps = _geometry(pose)
    frames = present_frames(pose, track_id)
    if not len(frames):
        return []

    # The session's last sampled frame sits within one frame interval of the end of the video -
    # 151.75s of a 151.76s recording in this corpus - so a seek to it decodes nothing and the
    # still is a dead entry in the strip. Dropped when the track has anywhere else to be. A
    # track present only in that frame keeps it and 404s honestly: there is no other moment to
    # show, and offering none would say the track was never visible.
    if len(frames) > 1 and frames[-1] == pose.track_ids.shape[0] - 1:
        frames = frames[:-1]

    # The midpoint of each of `count` equal segments of the *presence* list, so a track that
    # appears for a tenth of the session is sampled across that tenth rather than across the
    # session.
    #
    # Midpoints rather than endpoints, which is load-bearing twice over. The last sampled frame
    # of a session sits within a frame interval of the end of the video - 151.75s of a 151.76s
    # recording in this corpus - and a seek there decodes nothing, so a strip that ended on it
    # had one dead still on every session. And an endpoint spacing rounds two positions onto one
    # frame for a short presence, which served the same picture twice.
    wanted = (np.arange(min(count, len(frames))) + 0.5) * len(frames) / min(count, len(frames))
    wanted = np.unique(wanted.astype(int))

    out: list[Thumbnail] = []
    for frame in frames[wanted]:
        row = int(frame)
        box = box_at(pose, track_id, row, pad=pad)
        if box is None:
            continue
        out.append(Thumbnail(track_id=track_id, index=len(out), frame=row,
                             at_seconds=row / sampled_fps, box=box))
    return out
