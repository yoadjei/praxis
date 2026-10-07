# -*- coding: utf-8 -*-
"""Ingest, in the order the specification fixes and for the reasons it gives.

    1. resolve consent            - before a single byte is written
    2. stream to a temp file      - hashing as it goes; the hash is not known until the end
    3. probe and gate             - a file that cannot be read is refused here
    4. os.replace into place      - atomic, same filesystem
    5. one transaction            - media_objects, sessions, and the audit row together
    6. hand off to the queue      - after the commit, never inside it

Steps 4 and 5 are in that order because a row naming a file that is not there is unrecoverable
by inspection, while a file with no row is inert and findable by scanning. D46. Step 5 is one
transaction because R5 is worth nothing if the trail and the facts can disagree, and D47 is why
the audit append serialises.

Step 6 is outside the transaction because a queue is not transactional: enqueuing inside would
publish work for a session that might still roll back, and the worker is fast enough to arrive
before the commit does.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, insert, select
from sqlalchemy.exc import IntegrityError

from praxis.audit.write import append
from praxis.config_schema import PraxisConfig
from praxis.contracts.session import Session
from praxis.db.engine import transaction
from praxis.db.schema import media_objects, sessions
from praxis.ids import new_ulid
from praxis.ingest import consent as consent_module
from praxis.ingest.errors import IngestRefused, duplicate_session
from praxis.ingest.quality import QualityReport, measure
from praxis.ingest.storage import StoredMedia, store
from praxis.tools import Tool


@dataclass(frozen=True)
class IngestRequest:
    """What the operator supplies. The file supplies everything else."""

    filename: str
    chunks: Iterable[bytes]
    teacher_id: str
    college_id: str
    consent_id: str
    domain: str
    recorded_on: object
    # Required for a classroom session and meaningless for microteaching. Refused by `Session`
    # and by `a_classroom_session_names_its_school`, because the school is what places a
    # practicum session on the shift ladder. D98.
    school_id: str | None = None
    subject: str | None = None
    grade_level: str | None = None
    camera_distance_m: float | None = None
    room_area_m2: float | None = None
    pupil_count: int | None = None
    ambient_noise_dba: float | None = None
    teacher_movement_range_m: float | None = None
    actor_user_id: str | None = None
    actor_role: str | None = None


@dataclass(frozen=True)
class IngestResult:
    session: Session
    media: StoredMedia
    quality: QualityReport
    audit_row_hash: str


def ingest(request: IngestRequest, *, engine: Engine, config: PraxisConfig,
           ffprobe: Tool, ffmpeg: Tool, media_root: Path,
           enqueue: Callable[[str], None] | None = None) -> IngestResult:
    """Run the six steps. Any refusal leaves every table exactly as it was."""
    session_id = new_ulid()

    # 1. Consent, on its own connection, before the stream is touched. A refusal here has
    #    written nothing to any domain table, which is the property API.md states in so many
    #    words. It does write to the trail: SCHEMA.md section 9 lists session.rejected among the
    #    events that must be logged, and an upload turned away without a record is exactly the
    #    thing an audit is asked about later.
    try:
        with transaction(engine) as connection:
            granted = consent_module.resolve(connection, request.consent_id, request.domain)
    except IngestRefused as refused:
        _record_refusal(engine, session_id, request, refused)
        raise

    try:
        # 2 and 3. The temp file is removed on every path out of `store`, and a file the prober
        #    cannot read raises before anything has been placed.
        stored = store(
            request.chunks, media_root, request.filename,
            accepted_containers=config.ingest.accepted_containers,
            max_bytes=config.ingest.max_upload_bytes,
            chunk_bytes=config.ingest.hash_chunk_bytes)
        metadata, report = measure(
            stored.absolute(media_root), config.ingest.quality_gate, ffprobe, ffmpeg)
    except IngestRefused as refused:
        _record_refusal(engine, session_id, request, refused)
        raise

    session = Session(
        session_id=session_id, teacher_id=request.teacher_id, college_id=request.college_id,
        consent_id=granted.consent_id, media_sha256=stored.media_sha256,
        domain=request.domain, school_id=request.school_id,
        subject=request.subject, grade_level=request.grade_level,
        recorded_on=request.recorded_on,
        camera_distance_m=request.camera_distance_m, room_area_m2=request.room_area_m2,
        pupil_count=request.pupil_count, ambient_noise_dba=request.ambient_noise_dba,
        teacher_movement_range_m=request.teacher_movement_range_m,
        quality_verdict=report.verdict, quality_detail=report.as_detail(),
        created_at=datetime.now(UTC))

    # 5. One transaction. If any part of it fails the session does not exist, and neither does
    #    the claim in the trail that it does.
    try:
        with transaction(engine) as connection:
            _upsert_media(connection, stored, metadata)
            connection.execute(insert(sessions).values(**_session_row(session)))
            record = append(
                connection, "session.ingested", session.session_id,
                {"session_id": session.session_id, "teacher_id": session.teacher_id,
                 "domain": session.domain, "media_sha256": session.media_sha256,
                 "quality_verdict": report.verdict},
                actor_user_id=request.actor_user_id, actor_role=request.actor_role)
    except IntegrityError as exc:
        if _is_duplicate_session(exc):
            refused = duplicate_session(stored.media_sha256)
            _record_refusal(engine, session_id, request, refused)
            raise refused from exc
        raise

    # 6. After the commit. A worker that arrived first would look for a session that is not
    #    there yet and fail for a reason that has nothing to do with the session.
    if enqueue is not None:
        enqueue(session.session_id)

    return IngestResult(session=session, media=stored, quality=report,
                        audit_row_hash=record.row_hash)


def _session_row(session: Session) -> dict:
    row = session.model_dump()
    row["quality_verdict"] = session.quality_verdict
    return row


def _upsert_media(connection, stored: StoredMedia, metadata) -> None:
    """One row per hash. The same footage registered twice is one media object and two
    sessions, which BUILD-SPEC requires and D51 bounds."""
    existing = connection.execute(
        select(media_objects.c.media_sha256)
        .where(media_objects.c.media_sha256 == stored.media_sha256)).scalar()
    if existing is not None:
        return

    connection.execute(insert(media_objects).values(
        media_sha256=stored.media_sha256, relative_path=stored.relative_path,
        bytes=stored.bytes_written, duration_s=metadata.duration_s,
        width=metadata.width, height=metadata.height, fps=metadata.fps,
        has_audio=metadata.has_audio, is_blurred=False, reachable=True,
        created_at=datetime.now(UTC)))


def _is_duplicate_session(exc: IntegrityError) -> bool:
    return "uq_sessions_media_teacher_date" in str(getattr(exc, "orig", exc))


def _record_refusal(engine: Engine, session_id: str, request: IngestRequest,
                    refused: IngestRefused) -> None:
    """Write `session.rejected` for a session that will never exist.

    The identifier was minted before the first step precisely so that a refusal has something to
    name. Nothing is written to `sessions`, so this is the only record that the attempt happened
    at all, which is what makes "how many uploads were turned away, and why" a question the
    trail can answer.
    """
    with transaction(engine) as connection:
        append(connection, "session.rejected", session_id,
               {"session_id": session_id, "reason": refused.refusal.slug,
                "detail": refused.refusal.detail, "teacher_id": request.teacher_id},
               actor_user_id=request.actor_user_id, actor_role=request.actor_role)
