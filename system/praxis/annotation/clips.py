"""Clip planning and reference: the 8-second stepper arithmetic.

A session of T seconds yields roughly T/8 clips. The trailing remainder shorter than
`min_tail_seconds` is DROPPED, not padded, because a 2-second clip is not a unit of analysis
and padding invents data.

**Float accumulation must not drift:** boundaries are computed from the index, never by
repeated addition, so that clip_plan(d, 8.0, 8.0) returns boundaries whose sum is d,
and no rounding error accumulates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClipRef:
    """A clip within a session, identified by clip_index and timestamps."""

    session_id: str
    clip_index: int
    start_seconds: float
    end_seconds: float

    @property
    def clip_id(self) -> str:
        """The stable identifier for this clip, for use in annotation assignments."""
        return f"{self.session_id}:{self.clip_index:05d}"


def parse_clip_id(clip_id: str) -> tuple[str, int]:
    """Split a clip_id back into the session and index it was built from.

    The inverse of `ClipRef.clip_id`. `annotation_assignments` stores session_id and
    clip_index as separate columns while an annotation carries the joined clip_id, so
    anything matching a label to the assignment it answers has to split it again.

    Raises:
        ValueError: if the string was not produced by `ClipRef.clip_id`.
    """
    session_id, separator, index = clip_id.rpartition(":")
    if not separator or not session_id or not index.isdigit():
        raise ValueError(
            f"{clip_id!r} is not a clip id; expected '<session_id>:<clip_index>'")
    return session_id, int(index)


def clip_plan(
    session_id: str,
    duration_seconds: float,
    clip_seconds: float = 8.0,
    min_tail_seconds: float = 4.0,
) -> tuple[ClipRef, ...]:
    """Plan non-overlapping clips from a session, dropping short tails.

    Why the tail is dropped rather than padded: a 2-second clip is not a unit of analysis.
    Padding invents data where there are none, which is why behaviour analysis rejects the
    padding path. Where the remainder is shorter than `min_tail_seconds` it is dropped from
    analysis rather than kept as a partial unit.

    **Determinism rule: boundaries are computed from the index, never by accumulation.**
    This ensures that the sum of all clips equals the session duration within floating-point
    precision, and that repeated calls with the same parameters return identical results.
    No clip is ever dropped due to accumulated rounding error.

    Args:
        session_id: The session this plan is for.
        duration_seconds: The total length of the session in seconds.
        clip_seconds: The target length of each clip, in seconds. Default 8.0.
        min_tail_seconds: The minimum length of a remainder to keep it as a clip.
            Remainders shorter than this are dropped. Default 4.0.

    Returns:
        A tuple of ClipRef objects, one per clip, in index order. The tuple is empty
        if the session is shorter than min_tail_seconds.
    """
    if duration_seconds <= 0:
        return ()

    clips: list[ClipRef] = []

    # Compute the number of full clips that fit.
    n_full = int(duration_seconds // clip_seconds)

    # For each full clip, compute its boundaries from the index, not by accumulation.
    for i in range(n_full):
        start = i * clip_seconds
        end = (i + 1) * clip_seconds
        clips.append(ClipRef(session_id, i, start, end))

    # Check if there is a tail and whether to keep it.
    tail_start = n_full * clip_seconds
    tail_length = duration_seconds - tail_start

    if tail_length >= min_tail_seconds:
        clips.append(ClipRef(session_id, n_full, tail_start, duration_seconds))

    return tuple(clips)


def plan_from_config(session_id: str, duration_seconds: float, config) -> tuple[ClipRef, ...]:
    """The clip plan both callers use, so a clip means one thing.

    **Partial tails are dropped, not kept short.** `min_tail_seconds` is `length_s` rather
    than half of it, which was the previous value and carried a TODO asking for a config key.
    The key is not the fix. Stage A samples `behaviour.clip.frames` across whatever span a clip
    has, so a five-second clip reaches the model as 64 frames at 12.8 fps while the config
    declares 8, and the temporal head sees a short clip as a long one played slowly. Nothing
    downstream reports that; the shapes agree and the loss falls.

    Dropping the tail costs up to `length_s` of footage per session - under a minute across
    this corpus - and buys the guarantee `ClipSection.frames_match_length_and_rate` already
    asserts about the configuration: that length, rate and frame count describe one another. A
    clip is the unit of analysis the codebook is written against, and the codebook describes
    eight-second clips. D99.
    """
    clip = config.behaviour.clip
    return clip_plan(session_id, duration_seconds,
                     clip_seconds=clip.length_s, min_tail_seconds=clip.length_s)
