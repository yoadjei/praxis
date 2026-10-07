# -*- coding: utf-8 -*-
"""Technical metadata, read from the file rather than taken from the uploader.

Everything the quality gate decides on comes from here, and everything here comes from ffprobe
or ffmpeg. The uploader supplies covariates and consent; the file supplies its own duration,
resolution, frame rate and whether it has sound.

Both binaries are invoked through `praxis.tools`, which pins `-protocol_whitelist
file,crypto,data`. The path being probed arrived from outside the system, and without that flag
a filename that is secretly a URL would be fetched from inside the ingest path. D43.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import pstdev

from praxis.ingest.errors import unreadable_media
from praxis.tools import Tool

PROBE_TIMEOUT_S = 120

# Per seek, not per file. A container-level seek plus one frame is quick at any file size, so a
# call that takes this long has hit something malformed rather than something large.
LUMA_SEEK_TIMEOUT_S = 30

# Frames are decoded to this many pixels a side before averaging. Mean luminance does not need
# resolution, and a 90-minute file at full size would be gigabytes through a pipe.
LUMA_EDGE = 64


@dataclass(frozen=True)
class MediaMetadata:
    """What `media_objects` stores, as read from the container.

    `width` and `height` are the geometry **after** the container's rotation is applied, which
    is the frame the decoder hands the pipeline, and therefore the only geometry any consumer
    reason about. A phone recording held upright stores a 1024x576 stream with a -90 display
    matrix and decodes to 576x1024; recording the stream figures would describe a landscape
    recording that does not exist. D80.
    """

    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    container: str
    bytes: int
    rotation_degrees: int = 0

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    @property
    def orientation(self) -> str:
        if self.height > self.width:
            return "portrait"
        return "landscape" if self.width > self.height else "square"

    @property
    def pixels(self) -> int:
        """Frame area. The resolution floor is expressed against this and the shorter edge,
        because a width threshold asks a different question of portrait and landscape video."""
        return self.width * self.height

    @property
    def shorter_edge(self) -> int:
        return min(self.width, self.height)


def _run(argv: list[str], timeout: int, *, text: bool = True) -> subprocess.CompletedProcess:
    """Every ffmpeg and ffprobe call goes through here, so a timeout is always a refusal.

    `text=False` for the one caller that reads decoded pixels back: raw grayscale put
    through a UTF-8 decoder is not the frame that went in.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=text,
                              stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise unreadable_media(
            f"the file could not be read within {timeout}s, which for a container this size "
            f"means it is malformed rather than slow") from exc


def _fraction(value: str | None) -> float:
    """`r_frame_rate` arrives as "25/1", and as "0/0" when the container does not say."""
    if not value:
        return 0.0
    try:
        return float(Fraction(value))
    except (ZeroDivisionError, ValueError):
        return 0.0


def _frame_rate(video: dict) -> float:
    """Frames actually delivered per second of content, which is what the gate asks about.

    `avg_frame_rate` is frame count over duration. `r_frame_rate` is the lowest rate whose
    timebase can express every timestamp in the stream exactly - the least common multiple
    of the rates present - so on variable-frame-rate footage it is an upper bound and not a
    measurement. The corpus is phone recordings, and ffprobe reports `r_frame_rate` 120 for a
    file that delivers 30 frames a second, and 31.58 for several that deliver 29.98. A
    `min_fps` check against that number cannot fail: it is asking whether the container
    *could* have been smooth. D79.

    `r_frame_rate` is the fallback, not the preference. A stream whose duration the container
    does not state reports `avg_frame_rate` as 0/0, and for that file the nominal rate is the
    only figure there is.
    """
    average = _fraction(video.get("avg_frame_rate"))
    return average if average > 0 else _fraction(video.get("r_frame_rate"))


def _rotation(video: dict) -> int:
    """Display rotation in degrees, normalised to 0, 90, 180 or 270.

    Two places carry it and they do not agree across ffmpeg versions: the old `TAG:rotate`
    string, and the display matrix in `side_data_list`, which is where ffmpeg 7 and later put
    it. Both are read and the side data wins, because a file can carry a stale tag.

    The sign is discarded deliberately. A -90 and a 270 describe the same quarter turn, and what
    the geometry needs to know is only whether the axes swap.
    """
    degrees = 0
    for side_data in video.get("side_data_list") or ():
        if "rotation" in side_data:
            degrees = int(float(side_data["rotation"]))
            break
    else:
        tag = (video.get("tags") or {}).get("rotate")
        if tag is not None:
            try:
                degrees = int(float(tag))
            except ValueError:
                degrees = 0
    return degrees % 360


def probe(ffprobe: Tool, path: Path) -> MediaMetadata:
    """Read the container, or refuse with a reason naming what ffprobe objected to."""
    result = _run(ffprobe.command(
        "-v", "error",
        "-show_entries",
        "format=duration,format_name,size:"
        "stream=codec_type,width,height,r_frame_rate,avg_frame_rate:"
        "stream_side_data=rotation:stream_tags=rotate",
        "-of", "json", str(path)), PROBE_TIMEOUT_S)

    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise unreadable_media(
            f"ffprobe could not read the file: {detail[-1] if detail else 'no output'}")

    try:
        parsed = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise unreadable_media(f"ffprobe produced output that is not JSON: {exc}") from exc

    container = parsed.get("format", {})
    streams = parsed.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise unreadable_media("the file contains no video stream")

    duration = container.get("duration")
    if duration is None:
        raise unreadable_media(
            "the container declares no duration, so the file is truncated or malformed")

    width, height = video.get("width"), video.get("height")
    if not width or not height:
        raise unreadable_media("the video stream declares no frame size")

    # A quarter turn swaps the axes, so the recorded geometry is the decoded geometry and not
    # the stored one. Everything downstream - the resolution check, the zones, the artefact -
    # then describes the frame the decoder actually produces. D80.
    rotation = _rotation(video)
    if rotation in (90, 270):
        width, height = height, width

    return MediaMetadata(
        duration_s=float(duration),
        width=int(width),
        height=int(height),
        fps=_frame_rate(video),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        container=container.get("format_name", "unknown"),
        bytes=int(container.get("size") or path.stat().st_size),
        rotation_degrees=rotation,
    )


def frame_rate_jitter(ffprobe: Tool, path: Path, max_packets: int = 6000) -> float:
    """Standard deviation of instantaneous frame rate, in frames per second.

    `ingest.quality_gate.max_fps_jitter` is expressed that way. Presentation timestamps arrive
    out of order whenever the codec uses B-frames, so they are sorted before differencing;
    without that, every ordinary h264 file looks violently unstable.

    Returns 0.0 when there are too few packets to have a spread, which is a statement about the
    sample rather than a claim of perfect stability, and the duration check is what catches a
    file that short.
    """
    from itertools import pairwise

    result = _run(ffprobe.command(
        "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)), PROBE_TIMEOUT_S)
    if result.returncode != 0:
        raise unreadable_media("the video stream's packet timing could not be read")

    times = sorted(float(line) for line in result.stdout.split() if line.strip())[:max_packets]
    gaps = [later - earlier for earlier, later in pairwise(times) if later > earlier]
    if len(gaps) < 2:
        return 0.0
    return pstdev([1.0 / gap for gap in gaps])


def mean_luminance(ffmpeg: Tool, path: Path, sample_frames: int, duration_s: float) -> float:
    """Average Y over frames sampled evenly across the file, on the usual 0 to 255 scale.

    Decoded to raw grayscale and averaged here rather than read out of ffmpeg's `signalstats`
    filter. The filter route needs commas and quotes escaped through a filtergraph, a shell and
    an argument list at once, and it fails by silently probing a file whose name is the whole
    filter expression.

    **One seek per sample, not one decode of the file.** The sample times are the same ones an
    `fps=n/duration` filter would select, and the average over them is therefore the same
    measurement; what changes is how the frames are reached. A filter has to walk every frame in
    the container to emit the sparse ones, so a 32-minute recording at 1080p spent longer
    decoding 3.5 GB than the timeout allowed and the gate never reached a verdict on the one
    file in the corpus that would have passed it. Seeking before `-i` moves in the container
    instead, which costs the same whether the file is two minutes or two hours. D78.
    """
    import numpy as np

    if sample_frames < 1 or duration_s <= 0:
        return 0.0

    total, decoded = 0.0, 0
    for index in range(sample_frames):
        offset = index * duration_s / sample_frames
        # `-ss` ahead of `-i` is an input seek: ffmpeg jumps to the preceding keyframe and
        # decodes forward from there, rather than from the start of the file.
        result = _run(ffmpeg.command(
            "-v", "error", "-ss", f"{offset:.3f}", "-i", str(path),
            "-vf", f"scale={LUMA_EDGE}:{LUMA_EDGE}", "-frames:v", "1",
            "-pix_fmt", "gray", "-f", "rawvideo", "-"), LUMA_SEEK_TIMEOUT_S, text=False)
        if result.returncode != 0 or not result.stdout:
            # A seek landing in a damaged region or past the last decodable frame contributes
            # nothing. Only a file that yields no frame at all is unreadable.
            continue
        total += float(np.frombuffer(result.stdout, dtype=np.uint8).mean())
        decoded += 1

    if decoded == 0:
        raise unreadable_media(
            "the video could not be decoded far enough to sample its brightness")

    return total / decoded
