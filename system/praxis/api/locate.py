# -*- coding: utf-8 -*-
"""Session id to file on disk, with the containment check that has to accompany it.

Two routers need a session's footage - the media router streams it, the tracks router cuts a
still out of it - and the lookup is not just a join: a path out of the database has to be proven
to sit inside its configured root before anything opens it. L20 is the argument for this being
one module rather than a function in each: the copy that drifts is the one that keeps the join
and loses the check, and nothing about the response would look different.

The refusals are `HTTPException` because every caller is a route and the distinctions matter to
the client. A row that promises a blurred file which is not on disk is not "session not found":
by then D18 has deleted the original, so the footage is gone rather than merely unindexed, and
an operator who reads the right message goes looking for a backup instead of an ingest bug.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request
from sqlalchemy import select

from praxis.db.schema import media_objects, sessions
from praxis.ids import is_ulid


def not_found(session_id: str) -> HTTPException:
    """The one refusal several routes share. A malformed id and an unknown one are the
    same answer deliberately: whether a string is a ULID tells the caller nothing about
    the corpus."""
    return HTTPException(status_code=404, detail={
        "type": "/errors/not-found", "detail": f"session {session_id} not found"})


def _contained(root: Path, relative: str) -> Path:
    """`root / relative`, refusing a value that escapes. `root / path` yields the path itself
    when it is absolute, so an absolute value in the column escapes exactly as `..` does, and
    `is_relative_to` compares path components so a sibling directory whose name merely starts
    with the root's cannot pass."""
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{relative} resolves outside {root}")
    return resolved


def blurred_video(request: Request, session_id: str) -> Path:
    """Where this session's surviving footage is.

    Raises:
        404 if the session is unknown, is_blurred is false, no path is recorded, the path
            escapes the media root, or the file is not there. Each with its own message.
    """
    if not is_ulid(session_id):
        raise not_found(session_id)

    with request.app.state.engine.begin() as connection:
        row = connection.execute(
            select(sessions.c.session_id, media_objects.c.is_blurred,
                   media_objects.c.blurred_relative_path, media_objects.c.bytes)
            .join(media_objects, media_objects.c.media_sha256 == sessions.c.media_sha256)
            .where(sessions.c.session_id == session_id)).first()

    if row is None:
        raise not_found(session_id)

    if not row.is_blurred:
        raise HTTPException(status_code=404, detail={
            "type": "/errors/not-blurred",
            "detail": f"session {session_id} is not blurred"})

    if row.blurred_relative_path is None:
        raise HTTPException(status_code=404, detail={
            "type": "/errors/missing-path",
            "detail": f"session {session_id} has no blurred path"})

    try:
        path = _contained(request.app.state.media_root, row.blurred_relative_path)
    except (OSError, ValueError) as exc:
        raise not_found(session_id) from exc

    if not path.is_file():
        raise HTTPException(status_code=404, detail={
            "type": "/errors/blurred-file-missing",
            "detail": f"session {session_id} records a blurred video that is not on disk"})
    return path


def pose_artifact(request: Request, relative_path: str) -> Path:
    """Where a pose artefact sits under the configured root.

    Existence is not checked here. `praxis.preprocess.artifacts.load` verifies the file
    against the hash the database recorded, and a missing file and an altered one are the
    same question - is this the artefact the row describes - answered in the one place
    that can answer it.
    """
    return _contained(Path(request.app.state.config.paths.pose_artifacts), relative_path)
