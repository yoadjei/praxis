# -*- coding: utf-8 -*-
"""Rejoining a recording that a file splitter cut into parts.

One lesson in the corpus arrived as seven files of exactly 208.0 seconds and 10.5 MB each,
produced by a transfer tool that caps file size. They are not seven sessions. Ingesting them as
seven would put one teacher in several partitions, which is R2 broken by a filename convention.

**Joining is a restoration, not a transformation.** The parts are stream-copied back together
with no re-encode, so the result is the bytes the camera produced, in order. If the streams
cannot be copied into one container - different codec, geometry or frame rate - that is not a
split recording and this refuses rather than re-encoding them into something that looks like
one. A re-encode here would silently become the thing the thesis measures.

**The check is strict on purpose.** A loader that joined any group it was handed would, given
a mistaken manifest, fabricate a session out of two unrelated lessons and nothing downstream
could detect it. Every property that must match is compared before anything is written. D82.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from praxis.ingest.errors import IngestRefused, unreadable_media
from praxis.ingest.probe import MediaMetadata, probe
from praxis.tools import Tool

# Two parts of one recording never differ by more than this in frames per second. The parts in
# the corpus vary in the third decimal because each carries its own duration rounding; a genuine
# difference in capture rate is far larger and means these are different recordings.
FPS_TOLERANCE = 0.5

# The joined container's duration against the sum of the parts'. Concatenation adds a little
# container overhead and each part's duration is itself rounded, so exact equality is not
# available; a part silently dropped would show up as a whole part's worth of difference.
DURATION_TOLERANCE_S = 1.0


class PartsNotJoinable(IngestRefused):
    """The files handed over are not parts of one recording."""


@dataclass(frozen=True)
class PartGroup:
    """Several files that together are one session, in playback order."""

    group: str
    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if len(self.paths) < 2:
            raise ValueError(
                f"part group {self.group!r} has {len(self.paths)} file; a group is two or "
                f"more. A single file is an ordinary session and needs no joining.")

    @property
    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.paths)


def _refuse(detail: str) -> PartsNotJoinable:
    """Build the refusal. It is returned rather than raised, so every call site reads
    `raise _refuse(...)` and can attach a cause with `from`."""
    return PartsNotJoinable(unreadable_media(detail).refusal)


def verify_parts(group: PartGroup, ffprobe: Tool) -> tuple[MediaMetadata, ...]:
    """Read every part and refuse unless they are the same recording cut up.

    Returns the parts' metadata in the order given, so the caller does not probe them twice.
    """
    missing = [p for p in group.paths if not p.is_file()]
    if missing:
        raise _refuse(
            f"part group {group.group!r} names {len(missing)} file(s) that are not on disk: "
            f"{', '.join(p.name for p in missing)}. A group missing a part is a shorter "
            f"recording than the one the manifest describes, which is why this refuses rather "
            f"than joining what is present.")

    try:
        parts = tuple(probe(ffprobe, path) for path in group.paths)
    except IngestRefused as unreadable:
        raise _refuse(
            f"part group {group.group!r} contains a file that cannot be read: "
            f"{unreadable}") from unreadable

    first, *rest = parts
    for path, part in zip(group.paths[1:], rest, strict=True):
        for label, mine, theirs in (
            ("frame size", (first.width, first.height), (part.width, part.height)),
            ("rotation", first.rotation_degrees, part.rotation_degrees),
            ("container", first.container, part.container),
            ("audio", first.has_audio, part.has_audio),
        ):
            if mine != theirs:
                raise _refuse(
                    f"part group {group.group!r} is not one recording: {group.paths[0].name} "
                    f"and {path.name} differ in {label} ({mine} against {theirs}). Joining "
                    f"them would need a re-encode, and a re-encoded session is not the "
                    f"footage the camera produced.")

        if abs(first.fps - part.fps) > FPS_TOLERANCE:
            raise _refuse(
                f"part group {group.group!r} is not one recording: {group.paths[0].name} runs "
                f"at {first.fps:.3f} fps and {path.name} at {part.fps:.3f}.")

    return parts


def join_parts(group: PartGroup, destination: Path, *, ffmpeg: Tool,
               ffprobe: Tool) -> MediaMetadata:
    """Stream-copy the parts into `destination` and return the joined file's metadata.

    The caller owns `destination`; this writes it and verifies it, and does not delete the
    parts. Deleting the originals is `preprocess.blur`'s job and happens after blurring, which
    is a privacy commitment rather than a cleanup step.
    """
    import subprocess

    parts = verify_parts(group, ffprobe)
    expected = sum(part.duration_s for part in parts)

    destination.parent.mkdir(parents=True, exist_ok=True)
    listing = destination.with_suffix(".parts.txt")
    # ffmpeg's concat demuxer reads this listing; single quotes are its escape and a path
    # containing one would break the file, so such a path is refused rather than mangled.
    for path in group.paths:
        if "'" in str(path):
            raise _refuse(
                f"{path} contains a single quote, which ffmpeg's concat listing cannot escape. "
                f"Rename the file before joining it.")
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in group.paths), encoding="utf-8")

    try:
        result = subprocess.run(
            ffmpeg.command("-v", "error", "-y", "-f", "concat", "-safe", "0",
                           "-i", str(listing), "-c", "copy", str(destination),
                           file_only=False),
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=1800)
    finally:
        listing.unlink(missing_ok=True)

    # One cleanup site, not one per failure. A join that does not verify leaves nothing behind:
    # the file on disk is not a session, and a later run that found it would have no way to tell
    # it from one. Probing is inside the guard because it fails on exactly the files worth
    # removing - a truncated or unreadable join - and it was the path originally left out.
    try:
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()
            raise _refuse(
                f"joining part group {group.group!r} failed: "
                f"{detail[-1] if detail else 'ffmpeg produced no error text'}")

        joined = probe(ffprobe, destination)
        if abs(joined.duration_s - expected) > DURATION_TOLERANCE_S:
            raise _refuse(
                f"part group {group.group!r} joined to {joined.duration_s:.2f}s but its parts "
                f"sum to {expected:.2f}s. A part was dropped or truncated, and a session "
                f"shorter than its own footage would be scored as if the missing minutes were "
                f"never recorded.")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    return joined
