# -*- coding: utf-8 -*-
"""Preprocessing results into the database, and the human confirmation that follows.

Takes a `Connection` and commits nothing, like `praxis.audit.write`: D47 requires the audit row
and the fact it describes to commit together, and a function that opened its own transaction
could not offer that.

What is written is deliberately small. Pose is bulk numeric data and stays on disk; what lands
in Postgres is a pointer and a hash. Learner evidence lands as counts with no identifier,
because there is no column anywhere that could hold one - R1 is enforced by the absence of a
place to violate it, and this module is the boundary where that absence has to hold.

`teacher_tracks` is written with `confirmed_by` null. That is not an oversight and not a state
to be cleaned up later: it is how the system distinguishes a heuristic's guess from a person's
judgement. Every downstream phase filters on the confirmed column, so a session nobody has
reviewed cannot be trained on.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Connection, func, insert, select, update

from praxis.audit.write import append
from praxis.db.schema import (
    camera_setups,
    learner_aggregates,
    media_objects,
    pose_artifacts,
    sessions,
    teacher_tracks,
)
from praxis.preprocess.pipeline import PreprocessResult
from praxis.preprocess.zones import CameraSetup


class PersistError(RuntimeError):
    """The results could not be stored, or storing them would have meant something false."""


def save_camera_setup(connection: Connection, setup: CameraSetup, *, college_id: str,
                      label: str, defined_by: str, actor_role: str | None = None) -> str:
    """Store the zones an operator marked, refusing a setup whose zones overlap.

    Refused rather than resolved. `zone_of` takes the first matching zone, so an overlap is
    silently decided by check order and a band of the room is attributed to the wrong zone for
    every session recorded from this camera position - including the front-zone fraction, which
    is 30 per cent of the teacher heuristic's score.
    """
    overlapping = setup.overlaps()
    if overlapping:
        pairs = ", ".join(f"{a} and {b}" for a, b in overlapping)
        raise PersistError(
            f"camera setup {setup.setup_id} has overlapping seating zones ({pairs}). A "
            f"point in the shared region belongs to whichever is checked first, which is not "
            f"a decision this system is entitled to make on an operator's behalf.")

    connection.execute(insert(camera_setups).values(
        setup_id=setup.setup_id, college_id=college_id, label=label,
        defined_by=defined_by, created_at=func.now(), **setup.as_row()))
    append(connection, "camera_setup.defined", setup.setup_id,
           {"setup_id": setup.setup_id, "college_id": college_id, "defined_by": defined_by,
            "label": label},
           actor_user_id=defined_by, actor_role=actor_role)
    return setup.setup_id


def persist(connection: Connection, *, session_id: str, result: PreprocessResult,
            relative_path: str, max_candidates: int,
            blurred_relative_path: str | None = None,
            setup_id: str | None = None,
            actor_user_id: str | None = None, actor_role: str | None = None) -> None:
    """Write the pose pointer, the teacher proposal and the learner bins for one session.

    `relative_path` is where the `.npz` sits under the artefacts root, supplied by the caller
    because this module has no opinion about the filesystem layout. `blurred_relative_path` is
    where the surviving video sits under the media root, and omitting it leaves `media_objects`
    pointing only at the original this run destroyed. `max_candidates` caps the ranking stored
    for the reviewer, and is the caller's because it is a configured value and this module reads
    no configuration.

    Refuses a session that has already been preprocessed. Re-running is not a repair: the
    original was deleted the first time, so a second run would be reading the blurred video and
    would quietly produce pose from destroyed faces and overwrite the hash that proves what the
    first run saw.
    """
    if result.artifact.session_id != session_id:
        raise PersistError(
            f"the artefact was produced for session {result.artifact.session_id} and is being "
            f"filed under {session_id}. Writing it would attribute one teacher's pose to "
            f"another's session, and the hash would still verify.")

    if not result.original_deleted:
        raise PersistError(
            f"session {session_id} is not eligible to be recorded as preprocessed: the "
            f"unblurred original is still on disk. Nothing is written while that is true.")

    existing = connection.execute(
        select(pose_artifacts.c.session_id).where(pose_artifacts.c.session_id == session_id)
    ).first()
    if existing is not None:
        raise PersistError(
            f"session {session_id} already has a pose artefact. Preprocessing is not "
            f"repeatable: the original was deleted by the first run.")

    artifact = result.artifact
    connection.execute(insert(pose_artifacts).values(
        **{**artifact.as_row(), "relative_path": relative_path, "created_at": func.now()}))

    # The blurred copy is now the only copy, so the row that points at this session's
    # footage has to say where it is. `relative_path` and `media_sha256` are left describing
    # the file as ingested, because the hash is the provenance claim and repointing the path
    # at bytes that hash differently would break it while every column still looked consistent.
    # 0008.
    if blurred_relative_path is not None:
        connection.execute(
            update(media_objects)
            .where(media_objects.c.media_sha256 == select(sessions.c.media_sha256)
                   .where(sessions.c.session_id == session_id).scalar_subquery())
            .values(is_blurred=True, blurred_relative_path=blurred_relative_path))

    # One row per preprocessed session, proposal or not. A declined proposal used to write
    # nothing, on the reasoning that `track_id` was NOT NULL and inventing one would hand a
    # reviewer a track the heuristic never chose. The invention would have been wrong; the
    # silence was too. The row now carries a null track, the score that failed, the heuristic's
    # stated reason and the ranking it ranked, which is what a reviewer needs in order to
    # disagree - and without it `confirm_teacher` has nothing to confirm against. 0009, D90.
    #
    # `source` is "detection" and not a parameter: this function refuses a session that already
    # has a pose artefact, so by construction it only ever runs while the detections are in
    # hand. A run re-emitted from a stored artifact cannot reach here and says so for itself.
    connection.execute(insert(teacher_tracks).values(
        session_id=session_id,
        **result.proposal.as_row(source="detection", limit=max_candidates)))

    if result.learner_bins:
        connection.execute(insert(learner_aggregates),
                           [bin_.as_row(session_id) for bin_ in result.learner_bins])

    if setup_id is not None:
        connection.execute(update(sessions).where(sessions.c.session_id == session_id)
                           .values(setup_id=setup_id))

    append(connection, "session.preprocessed", session_id,
           {"session_id": session_id, "pose_sha256": artifact.sha256,
            "frames_processed": result.frames_processed, "faces_blurred": result.faces_blurred,
            "original_deleted": result.original_deleted,
            "config_sha256": artifact.config_sha256,
            "proposed_track_id": result.proposal.track_id,
            "zones_available": result.zones_available},
           actor_user_id=actor_user_id, actor_role=actor_role)


def confirm_teacher(connection: Connection, *, session_id: str, track_id: int,
                    confirmed_by: str, actor_role: str | None = None) -> None:
    """A person states which track is the teacher. The only way `confirmed_by` is ever set.

    The reviewer may name a track the heuristic did not propose, which is the entire point of
    the step: `require_human_confirmation` is documented NEVER set false because the heuristic
    is a guess. When they disagree both numbers reach the audit trail, and that disagreement is
    what makes the heuristic's field accuracy measurable rather than assumed.

    A later confirmation overwrites an earlier one. The row is current state; the chain is the
    record of who said what and when, and it cannot be rewritten.

    **The no-proposal case is the point, not the exception.** This function used to refuse a
    session whose heuristic cleared nothing, because before 0009 such a session had no row at
    all - so the one case the paragraph above calls the entire point of the step was the case
    the code would not accept, and three of the corpus's seven sessions could not be given a
    teacher by anybody. The refusal that remains is narrower and still worth keeping: a session
    with no row has not been preprocessed, and there is nothing a reviewer could have looked at.

    `proposed_by` becomes 'human' when the heuristic proposed nothing, because the track number
    in the row then originated with the reviewer. Agreeing with a proposal leaves it
    'heuristic'; who agreed is `confirmed_by`'s business and the chain's, not this column's.
    """
    row = connection.execute(
        select(teacher_tracks.c.track_id, teacher_tracks.c.confirmed_by)
        .where(teacher_tracks.c.session_id == session_id)).first()
    if row is None:
        raise PersistError(
            f"session {session_id} has not been preprocessed, so there is no proposal, no "
            f"candidate ranking and nothing a reviewer could have been shown. A confirmation "
            f"invented here would be a human endorsement of something no human saw.")

    proposed, previously = row
    connection.execute(
        update(teacher_tracks).where(teacher_tracks.c.session_id == session_id)
        .values(track_id=track_id, confirmed_by=confirmed_by,
                confirmed_at=datetime.now(UTC),
                proposed_by="heuristic" if proposed is not None else "human"))

    # `agreed_with_heuristic` is null, not false, when there was no proposal: there was nothing
    # to agree or disagree with, and counting an abstention as a disagreement would make a
    # cautious heuristic look wrong in exactly the way `accuracy_on_labelled` refuses to.
    append(connection, "teacher_track.confirmed", session_id,
           {"session_id": session_id, "track_id": track_id, "confirmed_by": confirmed_by,
            "proposed_track_id": proposed,
            "agreed_with_heuristic": None if proposed is None else proposed == track_id,
            "reconfirmation_of": previously},
           actor_user_id=confirmed_by, actor_role=actor_role)


def confirmed_teacher_track(connection: Connection, session_id: str) -> int | None:
    """The confirmed track, or None. What every downstream phase must call.

    Returns None for an unconfirmed proposal rather than the proposal itself, so a caller cannot
    reach a heuristic's guess through a function whose name says confirmed. R1 depends on the
    teacher being the only person classified, and on somebody having said who that is.
    """
    row = connection.execute(
        select(teacher_tracks.c.track_id)
        .where(teacher_tracks.c.session_id == session_id,
               teacher_tracks.c.confirmed_by.isnot(None))).first()
    return None if row is None else int(row[0])
