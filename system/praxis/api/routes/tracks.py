# -*- coding: utf-8 -*-
"""GET /api/v1/sessions/{id}/tracks/candidates, POST .../tracks/confirm - who the teacher is.

API.md section 4 fixes the shape: the candidate list carries each track's heuristic score and a
thumbnail strip, confirmation answers 200, and both are limited to supervisor, researcher and
admin. **Confirmation is mandatory before a session enters annotation or inference**, which
`scripts/plan_annotation.py` enforces by refusing to plan an unconfirmed session - so until this
router existed the gate had no gate-keeper, and a corpus could be preprocessed and then go no
further.

The heuristic's proposal and the reviewer's judgement are different facts and are reported as
different fields. Nothing here merges them into one "teacher" key: `praxis.preprocess.teacher`
exists precisely so that no object in this system can carry an unconfirmed identification, and
an endpoint that flattened the two would undo that at the boundary where it matters most.

**Three ways a session has no ranking, and they are not the same.** It was never preprocessed;
it was preprocessed before 0009 gave a declined proposal a row to live in; or it has a row whose
`candidates` column predates the column and reads `unrecorded`. Each gets its own slug and its
own remedy, because the first needs a phase run, the second and third need
`scripts/reemit_candidates.py`, and a single "not available" would send an operator looking for
the wrong thing. D48: the abstention is its own verdict, and a verdict says what it is.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from starlette.responses import Response

from praxis.api.identity import actor
from praxis.api.locate import blurred_video, not_found, pose_artifact
from praxis.db.schema import pose_artifacts, sessions, teacher_tracks
from praxis.ids import is_ulid
from praxis.ingest.errors import IngestRefused
from praxis.preprocess import artifacts as artifacts_module
from praxis.preprocess import candidates as candidates_module
from praxis.preprocess.frames import jpeg_at
from praxis.preprocess.store import PersistError, confirm_teacher

router = APIRouter(prefix="/api/v1", tags=["tracks"])

ACTION = "confirming the teacher track"

# The value 0009 wrote into every row that predated the column. A ranking cannot be invented for
# those rows, so the endpoint names the state rather than serving an empty list that would read
# as "the heuristic found nobody".
UNRECORDED = "unrecorded"


class ConfirmRequest(BaseModel):
    """The one thing a reviewer supplies: which track is the teacher.

    `extra="forbid"` because a client that sent `confirmed_by` would expect it to be honoured,
    and it is taken from the authorization header instead. Accepting and discarding it would
    file a confirmation under the wrong person while answering 200.
    """

    model_config = ConfigDict(extra="forbid")

    track_id: int = Field(ge=0, description="the track the reviewer identifies as the teacher")


def _row(request: Request, session_id: str) -> tuple[Any, Any]:
    """The session's proposal row and its pose artefact row, or a refusal explaining which of
    the three unreviewable states it is in."""
    if not is_ulid(session_id):
        raise not_found(session_id)

    with request.app.state.engine.begin() as connection:
        exists = connection.execute(
            select(sessions.c.session_id)
            .where(sessions.c.session_id == session_id)).first()
        if exists is None:
            raise not_found(session_id)

        proposal = connection.execute(
            select(teacher_tracks.c.track_id, teacher_tracks.c.proposed_by,
                   teacher_tracks.c.heuristic_score, teacher_tracks.c.confirmed_by,
                   teacher_tracks.c.confirmed_at, teacher_tracks.c.reason,
                   teacher_tracks.c.candidates)
            .where(teacher_tracks.c.session_id == session_id)).first()
        artifact = connection.execute(
            select(pose_artifacts.c.relative_path, pose_artifacts.c.sha256)
            .where(pose_artifacts.c.session_id == session_id)).first()

    if proposal is None:
        if artifact is None:
            raise HTTPException(status_code=409, detail={
                "type": "/errors/not-preprocessed",
                "detail": f"session {session_id} has no pose artefact, so phase 3 has not run "
                          f"on it and there are no tracks to choose between. Run preprocessing "
                          f"before asking who the teacher is."})
        raise HTTPException(status_code=409, detail={
            "type": "/errors/proposal-not-recorded",
            "detail": f"session {session_id} was preprocessed before a declined proposal had a "
                      f"row to live in, so the heuristic's ranking exists only in the pose "
                      f"artefact. Run scripts/reemit_candidates.py to recover it; nothing here "
                      f"will invent one."})
    return proposal, artifact


def _load_pose(request: Request, artifact) -> tuple[Any, str | None]:
    """The artefact, or None and the reason it could not be read.

    A missing or altered artefact does not fail the request. The stored ranking is still worth
    serving - a reviewer who knows the footage can act on it - and the screen is told the crops
    are unavailable so nobody mistakes an empty strip for a track that was never visible.
    """
    if artifact is None:
        return None, ("this session has a proposal and no pose artefact row, so there is "
                      "nothing to locate the tracks in")
    try:
        return artifacts_module.load(pose_artifact(request, artifact.relative_path),
                                     artifact.sha256), None
    except (artifacts_module.ArtifactError, ValueError, OSError) as failure:
        return None, str(failure)


@router.get("/sessions/{session_id}/tracks/candidates")
def get_candidates(
    request: Request,
    session_id: str,
    thumbnails: Annotated[int, Query(ge=1, le=12,
                                     description="stills per candidate")] = 5,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Every track the heuristic ranked, with its score, its signals and where to see it.

    The proposal, the confirmation and the ranking are three separate keys. `is_proposed` marks
    the heuristic's pick inside the list so the screen does not have to compare ids, and
    `is_confirmed` marks a reviewer's existing choice - which may be a track the heuristic
    never proposed, that being the entire point of the step.

    Arguments:
        session_id: the session (ULID).
        thumbnails: how many stills to locate per candidate, 1 to 12.

    Returns:
        the session's proposal state, its confirmation state, and the ranked candidates.

    Raises:
        404 if the session does not exist.
        409 if it was never preprocessed, or its ranking was never recorded. Distinct slugs.
    """
    actor(authorization, action=ACTION)
    proposal, artifact = _row(request, session_id)

    stored = proposal.candidates or {}
    ranked = stored.get("ranked") or []
    source = stored.get("source", UNRECORDED)
    pose, thumbnails_unavailable = _load_pose(request, artifact)

    rows = []
    for candidate in ranked:
        track_id = int(candidate["track_id"])
        strip = (candidates_module.thumbnails(pose, track_id, count=thumbnails)
                 if pose is not None else [])
        rows.append({
            "track_id": track_id,
            "score": candidate.get("score"),
            "presence_fraction": candidate.get("presence_fraction"),
            "median_area_fraction": candidate.get("median_area_fraction"),
            "front_zone_fraction": candidate.get("front_zone_fraction"),
            "is_proposed": proposal.track_id is not None and track_id == proposal.track_id,
            "is_confirmed": (proposal.confirmed_by is not None
                             and track_id == proposal.track_id),
            # The URL carries the frame, which is what determines the picture, so it fetches
            # the still this entry describes however long a strip was asked for.
            "thumbnails": [
                {**thumbnail.as_json(),
                 "url": f"/api/v1/sessions/{session_id}/tracks/{track_id}"
                        f"/thumbnail?frame={thumbnail.frame}"}
                for thumbnail in strip],
        })

    return {
        "session_id": session_id,
        "proposal": {
            "track_id": proposal.track_id,
            "score": proposal.heuristic_score,
            "proposed_by": proposal.proposed_by,
            "reason": proposal.reason,
        },
        "confirmation": {
            "confirmed_by": proposal.confirmed_by,
            "confirmed_at": (proposal.confirmed_at.isoformat()
                             if proposal.confirmed_at is not None else None),
        },
        "ranking": {
            "source": source,
            "zones_available": stored.get("zones_available"),
            "truncated": stored.get("truncated", 0),
            # Not derived from `ranked` being empty: a session in which no track was formed has
            # an empty ranking that *is* recorded, and telling an operator to re-emit it would
            # send them after a file that holds the same nothing.
            "available": source != UNRECORDED,
            "detail": (None if source != UNRECORDED else
                       "this row predates the candidates column, so the scores the heuristic "
                       "ranked were never stored; scripts/reemit_candidates.py recovers them "
                       "from the pose artefact"),
        },
        "thumbnails_unavailable": thumbnails_unavailable,
        "candidates": rows,
    }


@router.get("/sessions/{session_id}/tracks/{track_id}/thumbnail", response_model=None)
def get_thumbnail(
    request: Request,
    session_id: str,
    track_id: int,
    frame: Annotated[int, Query(ge=0, description="sampled-frame index to cut from")] = 0,
    width: Annotated[int, Query(ge=64, le=1024, description="output width")] = 320,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """One still of one candidate, cropped to where that track was in one frame.

    **Addressed by frame, not by a position in the strip.** A position is not a stable address:
    the strip is `thumbnails` frames spread across a track's presence, so position 1 of three
    and position 1 of two are different frames. Addressing by position made this endpoint serve
    a different picture from the one the candidate list described, and byte-identical pictures
    for two different positions. The frame is what determines the image, so the frame is what
    the URL carries, and `/tracks/candidates` bakes it in.

    The crop geometry is still recomputed here from the pose artefact rather than accepted from
    the caller. That is what makes the picture evidence about this track: a client-supplied
    rectangle would let the screen show a reviewer any region of the footage under a track
    number, and the confirmation it collected would mean nothing.

    Raises:
        404 if the session, its footage or the artefact is unavailable, or the track was not
            locatable in that frame.
        409 if the session has no recorded proposal.
    """
    actor(authorization, action=ACTION)
    _, artifact = _row(request, session_id)

    pose, unavailable = _load_pose(request, artifact)
    if pose is None:
        raise HTTPException(status_code=404, detail={
            "type": "/errors/pose-artifact-unreadable",
            "detail": f"no crop can be located for session {session_id}: {unavailable}"})

    box = candidates_module.box_at(pose, track_id, frame)
    if box is None:
        raise HTTPException(status_code=404, detail={
            "type": "/errors/thumbnail-not-found",
            "detail": f"track {track_id} is not locatable in frame {frame} of session "
                      f"{session_id}: either it was not tracked there or every keypoint in it "
                      f"was discarded. A track is only visible in the frames it appears in."})

    sampled_fps = float(pose.provenance.get("sampled_fps") or 0.0)
    if sampled_fps <= 0:
        raise HTTPException(status_code=404, detail={
            "type": "/errors/pose-artifact-unreadable",
            "detail": f"session {session_id} records no sample rate, so a frame index cannot "
                      f"be turned into a timestamp"})

    try:
        image = jpeg_at(request.app.state.ffmpeg, blurred_video(request, session_id),
                        at_seconds=frame / sampled_fps, width=width, box=box)
    except IngestRefused as refused:
        raise HTTPException(status_code=refused.refusal.status,
                            detail=refused.refusal.as_problem()) from refused

    return Response(content=image, media_type="image/jpeg")


@router.post("/sessions/{session_id}/tracks/confirm")
def post_confirm(
    request: Request,
    session_id: str,
    body: ConfirmRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """A person states which track is the teacher. 200, not 201: the row already existed.

    The reviewer may name a track the heuristic did not rank. That is permitted
    deliberately - `require_human_confirmation` is documented NEVER set false because the
    heuristic is a guess - and the disagreement reaches the audit trail, which is what
    makes the heuristic's field accuracy measurable rather than assumed.

    Raises:
        401 if no actor is claimed, 403 if the claimed role may not confirm.
        404 if the session does not exist.
        409 if it was never preprocessed or its proposal was never recorded.
    """
    actor_user_id, actor_role = actor(authorization, action=ACTION)
    if actor_user_id is None:
        raise HTTPException(status_code=401, detail={
            "type": "/errors/unauthenticated",
            "detail": "a confirmation is attributed to a person, and the token names a role "
                      "with no user id; `role:user_id` is the form"})

    _row(request, session_id)

    with request.app.state.engine.begin() as connection:
        try:
            confirm_teacher(connection, session_id=session_id, track_id=body.track_id,
                            confirmed_by=actor_user_id, actor_role=actor_role)
        except PersistError as refused:
            raise HTTPException(status_code=409, detail={
                "type": "/errors/confirmation-refused",
                "detail": str(refused)}) from refused

    return {"session_id": session_id, "track_id": body.track_id,
            "confirmed_by": actor_user_id}
