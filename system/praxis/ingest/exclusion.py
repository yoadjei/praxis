# -*- coding: utf-8 -*-
"""A researcher decides a session will not be annotated, and says why.

Takes a `Connection` and commits nothing, like `praxis.audit.write` and
`praxis.preprocess.store`: D47 requires the audit row and the fact it describes to commit
together, and a function that opened its own transaction could not offer that.

**An exclusion is a judgement, not a measurement.** `sessions.quality_verdict` records what the
quality gate measured about the file; this records that a person will not use it. The two are
kept apart deliberately. A session can pass the gate and be useless - the corpus contains one
whose camera moves 23.82 per cent of the frame diagonal between sampled frames, fragmenting the
teacher across 1248 track ids - and recording that as a quality failure would let a research
decision masquerade as something the gate detected.

Excluding is reversible and leaves everything in place: the footage, the pose artefact, the
teacher proposal and the clip plan all stay. What changes is that `plan_annotation` stops
offering the session and says, with the ground, why. The chain keeps both the exclusion and any
later reinstatement, because the row is current state and the trail is the record of who decided
what (R5). D96.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Connection, select, update

from praxis.audit.write import append
from praxis.db.schema import sessions


class ExclusionError(RuntimeError):
    """The session could not be excluded, or excluding it would have meant something false."""


@dataclass(frozen=True)
class Exclusion:
    """What was recorded, so a caller can report it without re-reading the row."""

    session_id: str
    reason: str
    excluded_by: str
    excluded_at: datetime


def exclude(connection: Connection, *, session_id: str, reason: str, excluded_by: str,
            actor_role: str | None = None) -> Exclusion:
    """Record that this session will not be annotated, on the stated ground.

    `reason` is required and is not defaulted. An exclusion with no ground cannot be questioned
    by the next reader and is indistinguishable from unfinished work, which is the state this
    whole mechanism exists to end - so a blank one is refused here and by
    `an_exclusion_ground_is_not_blank` at the database.

    Re-excluding an already-excluded session is refused rather than silently overwriting. The
    second ground might differ from the first, and quietly replacing it would lose the reasoning
    that was acted on. Reinstate first if the decision has genuinely changed.
    """
    if not reason or not reason.strip():
        raise ExclusionError(
            f"session {session_id} cannot be excluded without a stated ground. An exclusion "
            f"recorded as a bare flag reads as unfinished work to everyone who comes after, "
            f"which is the thing it is supposed to distinguish itself from.")

    row = connection.execute(
        select(sessions.c.session_id, sessions.c.excluded_at, sessions.c.exclusion_reason)
        .where(sessions.c.session_id == session_id)).first()
    if row is None:
        raise ExclusionError(f"session {session_id} does not exist, so there is nothing to "
                             f"exclude from annotation.")
    if row.excluded_at is not None:
        raise ExclusionError(
            f"session {session_id} was already excluded on {row.excluded_at:%Y-%m-%d}: "
            f"{row.exclusion_reason!r}. Overwriting that would discard the ground somebody "
            f"acted on. Reinstate it first if the decision has changed.")

    at = datetime.now(UTC)
    connection.execute(
        update(sessions).where(sessions.c.session_id == session_id)
        .values(excluded_at=at, excluded_by=excluded_by,
                exclusion_reason=reason.strip()))

    append(connection, "session.excluded", session_id,
           {"session_id": session_id, "reason": reason.strip(), "excluded_by": excluded_by},
           actor_user_id=excluded_by, actor_role=actor_role)

    return Exclusion(session_id=session_id, reason=reason.strip(), excluded_by=excluded_by,
                     excluded_at=at)


def reinstate(connection: Connection, *, session_id: str, reason: str, reinstated_by: str,
              actor_role: str | None = None) -> None:
    """Undo an exclusion, on a stated ground of its own.

    The row returns to the state of a session nobody has excluded, which is what the three NULL
    columns mean. The chain keeps both decisions: `session.excluded` is still there, and this
    appends beside it rather than in place of it, so the record is that somebody changed their
    mind rather than that nobody ever excluded it.
    """
    row = connection.execute(
        select(sessions.c.excluded_at, sessions.c.exclusion_reason)
        .where(sessions.c.session_id == session_id)).first()
    if row is None:
        raise ExclusionError(f"session {session_id} does not exist.")
    if row.excluded_at is None:
        raise ExclusionError(
            f"session {session_id} is not excluded, so there is nothing to reinstate. A "
            f"reinstatement recorded against a session nobody excluded would put a decision "
            f"in the trail that was never made.")

    connection.execute(
        update(sessions).where(sessions.c.session_id == session_id)
        .values(excluded_at=None, excluded_by=None, exclusion_reason=None))

    append(connection, "session.excluded", session_id,
           {"session_id": session_id, "excluded_by": reinstated_by,
            "reason": f"REINSTATED: {reason.strip()}",
            "reinstates": row.exclusion_reason},
           actor_user_id=reinstated_by, actor_role=actor_role)


def excluded(connection: Connection) -> dict[str, str]:
    """Every excluded session and its ground, for a report that must not silently omit them."""
    return {
        row.session_id: row.exclusion_reason
        for row in connection.execute(
            select(sessions.c.session_id, sessions.c.exclusion_reason)
            .where(sessions.c.excluded_at.isnot(None))
            .order_by(sessions.c.session_id))
    }
