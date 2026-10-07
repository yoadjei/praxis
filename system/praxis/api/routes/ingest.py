# -*- coding: utf-8 -*-
"""POST /api/v1/sessions - the upload endpoint.

API.md section 3 fixes the shape: multipart, a `file` part and a JSON `metadata` part matching
the `Session` contract, `201` carrying the quality verdict, `422` for consent, `409` for a
duplicate. The status codes live in `praxis.ingest.errors` rather than here, because whether a
withdrawn consent is a 422 is a fact about the rule and a second route that ingested media would
otherwise have to decide it again.

The route is `def`, not `async def`, deliberately. Reading a 20 GiB upload and running ffprobe
are both blocking, and FastAPI runs a synchronous route in a worker thread; written `async` the
same code would stall the event loop for the length of the upload.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from praxis.api.identity import actor
from praxis.contracts.session import Domain, missing_school
from praxis.ingest.errors import IngestRefused
from praxis.ingest.service import IngestRequest, ingest

router = APIRouter(prefix="/api/v1", tags=["ingest"])


class SessionMetadata(BaseModel):
    """The JSON part. A subset of the `Session` contract: the fields an operator supplies.

    The rest of a session is measured, not declared. `media_sha256` is computed from the bytes,
    `quality_verdict` comes from the gate, and `session_id` is minted by the server, so none of
    them is accepted here - a client that could name its own hash could name the wrong one.

    `extra="forbid"` is what makes that true. Ignoring an unknown key would accept a request
    that names a hash and silently discard it, so the client is told its upload succeeded on
    terms the server never agreed to.
    """

    model_config = ConfigDict(extra="forbid")

    teacher_id: str = Field(min_length=1)
    college_id: str
    consent_id: str
    domain: Domain
    # Required for a classroom session. Checked here as well as on `Session`, because `Session`
    # is built after the upload has been stored and probed, and this route's whole shape is that
    # a refusal happens before a byte is written. One predicate, `contracts.session.
    # missing_school`, so the two cannot drift. D98.
    school_id: str | None = None
    recorded_on: date
    subject: str | None = None
    grade_level: str | None = None

    camera_distance_m: float | None = Field(default=None, gt=0)
    room_area_m2: float | None = Field(default=None, gt=0)
    pupil_count: int | None = Field(default=None, ge=0)
    ambient_noise_dba: float | None = Field(default=None, ge=0)
    teacher_movement_range_m: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def a_classroom_session_names_its_school(self) -> SessionMetadata:
        refusal = missing_school(self.domain, self.school_id)
        if refusal:
            raise ValueError(refusal)
        return self


def _chunks(upload: UploadFile, chunk_bytes: int) -> Iterator[bytes]:
    """The raw stream, read in configured slices so a large upload is never held in memory."""
    while chunk := upload.file.read(chunk_bytes):
        yield chunk


@router.post("/sessions", status_code=201)
def create_session(
    request: Request,
    file: Annotated[UploadFile, File(description="mp4 or mov")],
    metadata: Annotated[str, Form(description="JSON matching the Session contract")],
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Ingest one recording. Consent is checked before a single byte is written."""
    actor_user_id, actor_role = actor(authorization, action="ingest")

    try:
        declared = SessionMetadata.model_validate(json.loads(metadata))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail={
            "type": "/errors/malformed-metadata",
            "detail": f"the metadata part is not JSON: {exc}"}) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={
            "type": "/errors/invalid-metadata",
            "detail": exc.errors(include_url=False)}) from exc

    state = request.app.state
    try:
        result = ingest(
            IngestRequest(
                filename=file.filename or "upload.mp4",
                chunks=_chunks(file, state.config.ingest.hash_chunk_bytes),
                actor_user_id=actor_user_id, actor_role=actor_role,
                **declared.model_dump()),
            engine=state.engine, config=state.config, ffprobe=state.ffprobe,
            ffmpeg=state.ffmpeg, media_root=state.media_root, enqueue=state.enqueue)
    except IngestRefused as refused:
        raise HTTPException(status_code=refused.refusal.status,
                            detail=refused.refusal.as_problem()) from refused

    return {
        "session_id": result.session.session_id,
        "media_sha256": result.media.media_sha256,
        "quality": {
            "verdict": result.quality.verdict,
            "checks": [{"name": check.name, **check.as_dict()}
                       for check in result.quality.checks],
        },
    }
