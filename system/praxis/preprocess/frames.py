# -*- coding: utf-8 -*-
"""Single frames out of a video, for a person to look at or a measurement to read.

Two callers need the same thing and would otherwise each shell out to ffmpeg their own way:
the teacher-confirmation screen, which shows a reviewer what each candidate track looks like
before they identify one, and `scripts/camera_motion.py`, which needs a sequence of small
grayscale frames to estimate how far the frame drifted. L20 is the reason they share this: two
spellings of one ffmpeg invocation is how one of them ends up without `-protocol_whitelist`.

**Seeking, not filtering.** `-ss` before `-i` jumps in the container and decodes forward from
the preceding keyframe, which costs the same whether the file is two minutes or thirty-two. A
filter that selects sparse frames has to walk every frame to emit them, and on the 1080p
recording in this corpus that exceeded the timeout without reaching a verdict. D78 settled
this for the brightness probe and the same arithmetic applies here.

**Only the blurred copy is ever read.** D18 deletes the original after preprocessing, so there
is normally nothing else on disk; this module takes whatever path it is handed and the caller is
responsible for having resolved it from `media_objects.blurred_relative_path`. Nothing here
reaches for a file by session id, which keeps the one place that maps a session to a path in
`praxis/api/locate.py`.
"""
from __future__ import annotations

import subprocess

import numpy as np

from praxis.ingest.errors import IngestRefused, refuse
from praxis.tools import Tool

# A single seek-and-decode. Generous enough for a keyframe interval on phone footage and short
# enough that a damaged region fails rather than hanging a request.
SEEK_TIMEOUT_S = 30

# JPEG quality for a review thumbnail. 3 is visually clean on a person-sized crop and about a
# fifth the bytes of 1; the reviewer is identifying which person is the teacher, not reading
# text off a board.
JPEG_QUALITY = 3


def frame_unreadable(detail: str) -> IngestRefused:
    """404 rather than 500: the frame is a resource, and a timestamp past the end of a file is a
    request for something that is not there rather than a server fault."""
    return refuse("frame-unreadable", 404, detail)


def scaled_height(width: int, height: int, to_width: int) -> int:
    """The height that keeps the aspect ratio, rounded to even for the encoders that need it.

    Computed here and passed to ffmpeg explicitly rather than using `scale=W:-2`, because a
    caller decoding raw bytes has to know the shape to reshape them and `-2` does not say.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"cannot scale a {width}x{height} frame")
    return max(2, round(height * to_width / width / 2) * 2)


def _decode(ffmpeg: Tool, path, *, at_seconds: float, arguments: tuple[str, ...]) -> bytes:
    offset = max(0.0, at_seconds)
    try:
        done = subprocess.run(
            ffmpeg.command("-v", "error", "-ss", f"{offset:.3f}", "-i", str(path),
                           "-frames:v", "1", *arguments, "-"),
            capture_output=True, stdin=subprocess.DEVNULL, timeout=SEEK_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise frame_unreadable(
            f"the frame at {offset:.3f}s could not be decoded within {SEEK_TIMEOUT_S}s, which "
            f"for a single seek means the file is damaged rather than slow") from exc

    if done.returncode != 0 or not done.stdout:
        raise frame_unreadable(
            f"no frame could be decoded at {offset:.3f}s; the seek landed past the last "
            f"decodable frame or in a damaged region")
    return done.stdout


def jpeg_at(ffmpeg: Tool, path, *, at_seconds: float, width: int,
            box: tuple[int, int, int, int] | None = None) -> bytes:
    """One frame as JPEG bytes, scaled to `width` with the aspect ratio kept.

    With `box` as `(x, y, w, h)` in source pixels, the frame is cropped to it first and `width`
    then applies to the crop. Crop precedes scale in the filter chain and the order is not
    interchangeable: scaling first would move the rectangle, so a box derived from the pose
    artefact's coordinates would land somewhere else in the room.
    """
    chain = f"scale={width}:-2"
    if box is not None:
        x, y, crop_width, crop_height = box
        if crop_width < 2 or crop_height < 2:
            raise frame_unreadable(
                f"a {crop_width}x{crop_height} crop is not an image; the region asked for is "
                f"smaller than a pixel pair")
        chain = f"crop={crop_width}:{crop_height}:{x}:{y},{chain}"

    return _decode(ffmpeg, path, at_seconds=at_seconds,
                   arguments=("-vf", chain, "-q:v", str(JPEG_QUALITY), "-f", "mjpeg"))


def gray_at(ffmpeg: Tool, path, *, at_seconds: float, width: int,
            height: int) -> np.ndarray:
    """One frame as a `(height, width)` array of 8-bit luminance.

    The shape is supplied rather than inferred, which is what makes the raw buffer safe to
    reshape. A buffer whose length disagrees with the shape asked for is a decode that produced
    something else, and is refused rather than reshaped into noise.
    """
    raw = _decode(ffmpeg, path, at_seconds=at_seconds,
                  arguments=("-vf", f"scale={width}:{height}", "-pix_fmt", "gray",
                             "-f", "rawvideo"))
    expected = width * height
    buffer = np.frombuffer(raw, dtype=np.uint8)
    if buffer.size < expected:
        raise frame_unreadable(
            f"the decoded frame is {buffer.size} bytes and a {width}x{height} grayscale frame "
            f"is {expected}; the scale filter did not produce the size it was asked for")
    return buffer[:expected].reshape(height, width)


def gray_sequence(ffmpeg: Tool, path, *, count: int, duration_s: float, width: int,
                  height: int) -> list[np.ndarray]:
    """`count` frames spread evenly across the file, as grayscale arrays.

    A seek that lands past the end or inside a damaged region contributes nothing instead of
    failing the sequence, because a measurement over the frames that did decode is still a
    measurement; the caller is told how many it got and decides whether that is enough. Only a
    file that yields no frame at all is unreadable.
    """
    if count < 1 or duration_s <= 0:
        return []

    frames = []
    for index in range(count):
        try:
            frames.append(gray_at(ffmpeg, path, at_seconds=index * duration_s / count,
                                  width=width, height=height))
        except IngestRefused:
            continue
    if not frames:
        raise frame_unreadable(
            f"none of the {count} sampled frames could be decoded, so the file is unreadable "
            f"rather than partly damaged")
    return frames
