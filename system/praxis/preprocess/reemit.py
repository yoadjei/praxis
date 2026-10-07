# -*- coding: utf-8 -*-
"""Rebuild a candidate ranking from a stored pose artefact, for rows that predate the column.

Migration 0009 added `teacher_tracks.candidates` and gave every existing row a default saying
the ranking was never recorded, with a reason telling whoever read it to re-emit from the pose
artefact. This is that re-emission. A migration may not do it - no DML backfills, and inventing
scores inside one would be exactly the silent rewrite that rule exists to prevent - so it is a
script, it is audited, and it says in the row itself where its numbers came from.

**The numbers are not the original run's, and the row must not pretend they are.** The first run
measured each track from its detection boxes: the centroid and area of the box the estimator
emitted. Those boxes are not in the artefact - only keypoints are - so a ranking rebuilt from
what survives is measured from the keypoint hull instead. The two are close and they are not
the same number, which is why `source` is a required field of both the stored candidates and
the audit event, and why this writes:

  * `candidates` - new information, previously the string `unrecorded`;
  * `reason` - replacing the placeholder 0009 wrote, and naming the hull as its source;
  * `track_id` and `heuristic_score` - **only when the row did not exist at all**.

A row that already carries a proposal keeps its `track_id` and `heuristic_score` untouched.
Overwriting a stored score with one measured differently would redefine a recorded metric while
every column still looked consistent, and the proposal is the fact the original run asserted.

For a session that has no row - the pre-0009 declined proposals - `heuristic_score` is written
null rather than filled with the hull-derived best. The original run did compute a score and
decided it was below the floor; that number was never stored and cannot be recovered, and
putting a different measurement in its place would read as the score the decision was made on.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np
from sqlalchemy import Connection, insert, select, update

from praxis.audit.write import append
from praxis.db.schema import teacher_tracks
from praxis.preprocess.artifacts import LoadedPose
from praxis.preprocess.candidates import hull
from praxis.preprocess.teacher import TeacherProposal, TrackSignals, propose
from praxis.preprocess.zones import CameraSetup, normalise

# What `candidates.source` says for a ranking measured this way. Read by the API to tell a
# recovered ranking from the original run's, and by anyone comparing the two.
SOURCE = "keypoint-hull"


class ReemitError(RuntimeError):
    """The ranking could not be rebuilt, or rebuilding it would have meant something false."""


@dataclass(frozen=True)
class Recovery:
    """What was written for one session, so a caller can report it without re-reading."""

    session_id: str
    proposal: TeacherProposal
    row_created: bool
    candidate_count: int


def signals_from_pose(pose: LoadedPose, *, setup: CameraSetup | None) -> list[TrackSignals]:
    """The three teacher signals per track, measured from keypoints instead of boxes.

    Presence is the share of sampled frames the track was tracked in, which is the same
    quantity the original run measured and is unaffected by the change of source. Area and
    front-zone time are the hull's, and differ from the box's by however much the box exceeded
    the person - typically a little, and not nothing.

    A frame in which every keypoint was discarded contributes to neither area nor position, but
    still counts as presence: the tracker saw the person there, and dropping the frame would
    understate how long they were in the room.
    """
    frames, _ = pose.track_ids.shape
    if frames <= 0:
        raise ReemitError("the artefact covers no frames, so presence cannot be measured")

    frame_width = int(pose.provenance.get("frame_width", 0))
    frame_height = int(pose.provenance.get("frame_height", 0))
    if frame_width <= 0 or frame_height <= 0:
        raise ReemitError(
            f"the artefact records a {frame_width}x{frame_height} frame, so no area or zone "
            f"fraction measured against it would mean anything")
    frame_area = float(frame_width * frame_height)

    out: list[TrackSignals] = []
    for track_id in sorted(int(t) for t in np.unique(pose.track_ids) if t >= 0):
        rows, columns = np.nonzero(pose.track_ids == track_id)
        areas: list[float] = []
        in_front = 0
        located = 0

        for row, column in zip(rows, columns, strict=True):
            box = hull(pose.keypoints[row, column], frame_width=frame_width,
                       frame_height=frame_height, pad=0.0)
            if box is None:
                continue
            x, y, width, height = box
            areas.append(width * height / frame_area)
            located += 1
            if setup is not None:
                centre = (x + width / 2, y + height / 2)
                if setup.zone_front.contains(normalise(centre, frame_width, frame_height)):
                    in_front += 1

        out.append(TrackSignals(
            track_id=track_id,
            presence_fraction=min(len(rows) / frames, 1.0),
            median_area_fraction=min(median(areas), 1.0) if areas else 0.0,
            # Over the frames the person could be located in, not over every frame they were
            # tracked in. Dividing by the larger number would report a teacher who stood at the
            # front throughout as having been there for only part of it, because the frames
            # where the estimator saw nobody clearly would count against them.
            front_zone_fraction=(in_front / located) if located and setup is not None else 0.0,
        ))
    return out


def rebuild(pose: LoadedPose, *, teacher_config, setup: CameraSetup | None) -> TeacherProposal:
    """The ranking the stored artefact supports, through the same `propose` the pipeline uses.

    Shared rather than reimplemented: the floor, the scaling when zones are missing and the
    wording of the reason are one rule, and a second copy here would be the one that kept the
    unscaled floor after the first was fixed (L20).
    """
    signals = signals_from_pose(pose, setup=setup)
    return propose(signals, teacher_config, zones_available=setup is not None)


def recover(connection: Connection, *, session_id: str, pose: LoadedPose, pose_sha256: str,
            teacher_config, setup: CameraSetup | None, max_candidates: int,
            actor_user_id: str | None = None,
            actor_role: str | None = None) -> Recovery:
    """Write the rebuilt ranking for one session. Commits nothing; the caller owns the
    transaction, so the row and its audit event land together (D47).

    Refuses a session whose ranking is already recorded. A re-emission over a real ranking would
    replace measurements from the detections with measurements from the hull, which is the one
    thing this module exists not to do.
    """
    existing = connection.execute(
        select(teacher_tracks.c.track_id, teacher_tracks.c.candidates,
               teacher_tracks.c.confirmed_by)
        .where(teacher_tracks.c.session_id == session_id)).first()

    if existing is not None:
        stored = existing.candidates or {}
        if stored.get("source") not in (None, "unrecorded"):
            raise ReemitError(
                f"session {session_id} already has a ranking from {stored['source']!r}. "
                f"Re-emitting over it would replace measurements the original run made with "
                f"measurements of a different quantity.")

    proposal = rebuild(pose, teacher_config=teacher_config, setup=setup)
    payload = proposal.candidates(source=SOURCE, limit=max_candidates)
    reason = (
        f"recovered from the stored pose artefact, so the signals below are measured from each "
        f"track's keypoint hull rather than from the detection boxes the original run used. "
        f"{proposal.reason}")

    if existing is None:
        # No row at all: a declined proposal from before 0009 had somewhere to put one. The
        # track and the score stay null - see the module docstring on why a hull-derived best
        # score must not stand in for the one the original decision was made on.
        connection.execute(insert(teacher_tracks).values(
            session_id=session_id, track_id=None, proposed_by="heuristic",
            heuristic_score=None, confirmed_by=None, confirmed_at=None,
            reason=reason, candidates=payload))
    else:
        connection.execute(
            update(teacher_tracks).where(teacher_tracks.c.session_id == session_id)
            .values(reason=reason, candidates=payload))

    append(connection, "teacher_track.ranking_recovered", session_id,
           {"session_id": session_id, "source": SOURCE, "pose_sha256": pose_sha256,
            "candidate_count": len(payload["ranked"]),
            "truncated": payload["truncated"],
            "zones_available": setup is not None,
            # The proposal this ranking would support, recorded without being written to the
            # row. It is what the heuristic would say today from what survives, which is worth
            # being able to compare against the original and is not the original.
            "hull_proposed_track_id": proposal.track_id,
            "hull_best_score": proposal.score,
            "row_created": existing is None},
           actor_user_id=actor_user_id, actor_role=actor_role)

    return Recovery(session_id=session_id, proposal=proposal, row_created=existing is None,
                    candidate_count=len(payload["ranked"]))
