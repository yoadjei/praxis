"""Persistence for the annotation workflow.

This module reads and writes the tables that track clip assignments, rater labels, and
codebook revisions. Everything is append-only by design, so no UPDATE or DELETE ever runs
from here. Corrections are new rows, not modifications, so that the audit trail is
immutable and the history of calibration changes is visible.

The module works at the level of raw data contracts (dataclasses from
`praxis.annotation.server` and plain dicts), not ORM objects. SQLAlchemy Core is used
throughout. D52 explains why ORM is deliberately absent here.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from praxis.annotation.clips import ClipRef, parse_clip_id
from praxis.annotation.server import (
    AnnotationRefused,
    Assignment,
)
from praxis.db.schema import (
    annotation_assignments,
    annotations,
    codebook_revisions,
    sessions,
)


def record_assignments(conn: Connection, assignments: tuple[Assignment, ...]) -> int:
    """Insert clip assignments for a calibration or production round.

    Arguments:
        conn: an active database connection
        assignments: the assignments to issue, typically from
            `praxis.annotation.server.build_assignments`

    Returns:
        the number of rows inserted

    Raises:
        IntegrityError if a rater is assigned the same clip+behaviour twice in the same
        round, which violates the UNIQUE constraint. This is a data error, not a
        concurrency race, because the assignments are typically built in one go before
        they are issued.
    """
    rows = [
        {
            "assignment_id": a.assignment_id,
            "rater_id": a.rater_id,
            "session_id": a.clip.session_id,
            "clip_index": a.clip.clip_index,
            "clip_start_s": a.clip.start_seconds,
            "clip_end_s": a.clip.end_seconds,
            "behaviour": a.behaviour,
            "round_name": a.round_name,
            "created_at": datetime.now(UTC),
        }
        for a in assignments
    ]
    if not rows:
        return 0
    conn.execute(annotation_assignments.insert(), rows)
    # len(rows), not result.rowcount. psycopg reports -1 for an executemany INSERT unless the
    # statement carries RETURNING, so rowcount would silently report failure on a successful
    # write. The count is already known: the insert is one statement in one transaction, so it
    # either wrote every row or raised.
    return len(rows)


def find_assignment(
    conn: Connection,
    *,
    rater_id: str,
    clip_id: str,
    behaviour: str,
) -> str | None:
    """Find the assignment a label answers, if one was issued for it.

    A label recorded without its assignment_id is invisible to every round-filtered query,
    because `load_annotations(round_name=...)` reaches the round through the assignment
    table. That is the whole calibration workflow, so anything recording a label on a rater's
    behalf resolves the assignment first and passes it to `record_annotation`.

    The lookup is exact, not a guess: `UNIQUE (rater_id, session_id, clip_index, behaviour)`
    means at most one assignment can match, across every round.

    Arguments:
        conn: an active database connection
        rater_id: the rater who made the label
        clip_id: the clip the label is about, in `ClipRef.clip_id` form
        behaviour: the behaviour the label scores

    Returns:
        the assignment_id, or None when the label was made outside any assignment. The
        column is nullable precisely so that external and corrective labels can be filed.

    Raises:
        ValueError: if clip_id is not a clip id. A malformed one is a caller error, and
        swallowing it would file an orphan annotation that no round can ever see.
    """
    session_id, clip_index = parse_clip_id(clip_id)
    row = conn.execute(
        select(annotation_assignments.c.assignment_id).where(
            annotation_assignments.c.rater_id == rater_id,
            annotation_assignments.c.session_id == session_id,
            annotation_assignments.c.clip_index == clip_index,
            annotation_assignments.c.behaviour == behaviour,
        ),
    ).first()
    return row.assignment_id if row is not None else None


def record_annotation(
    conn: Connection,
    annotation: Any,
    assignment_id: str | None = None,
) -> str:
    """Record a rater's label.

    The annotation object is expected to carry these fields:
    - annotation_id (str, ULID)
    - clip_id (str)
    - rater_id (str, ULID)
    - behaviour (str, BehaviourId)
    - codebook_version (str)
    - labels (dict)
    - is_nonscorable (bool, default False)
    - note (str, optional)
    - rater_confidence (str, default "certain")
    - session_college_id (str, ULID, optional)

    Arguments:
        conn: an active database connection
        annotation: a Pydantic model or dict carrying the label. Must have annotation_id set.
        assignment_id: the assignment_id this label belongs to, if any. May be None for
            external or corrective labels that come without an assignment context.

    Returns:
        the annotation_id

    Raises:
        AnnotationRefused (HTTP 409) if this rater has already labelled this clip+behaviour.
            The UNIQUE constraint is checked at the database level, and a duplicate is caught
            here and rephrased as a 409 rather than letting an IntegrityError bubble.
    """
    annotation_id = annotation.annotation_id
    try:
        conn.execute(
            annotations.insert(),
            {
                "annotation_id": annotation_id,
                "assignment_id": assignment_id,
                "clip_id": annotation.clip_id,
                "rater_id": annotation.rater_id,
                "behaviour": annotation.behaviour,
                "codebook_version": annotation.codebook_version,
                "labels": annotation.labels,
                "is_nonscorable": getattr(annotation, "is_nonscorable", False),
                "note": getattr(annotation, "note", None),
                "rater_confidence": getattr(annotation, "rater_confidence", "certain"),
                "session_college_id": getattr(annotation, "session_college_id", None),
                "created_at": datetime.now(UTC),
            },
        )
    except Exception as e:
        # Check if this is a UNIQUE constraint violation. The constraint is on
        # (rater_id, clip_id, behaviour), which means this rater has already labelled
        # this clip+behaviour. This is a 409 Conflict, not a 422 Unprocessable Entity.
        error_str = str(e).lower()
        if "unique" in error_str or "duplicate" in error_str:
            raise AnnotationRefused(
                reason=f"rater {annotation.rater_id} already labelled "
                f"clip {annotation.clip_id} behaviour {annotation.behaviour}",
                http_status=409,
            ) from e
        raise

    return annotation_id


def load_annotations(
    conn: Connection,
    *,
    round_name: str | None = None,
    clip_ids: tuple[str, ...] | None = None,
) -> tuple[Any, ...]:
    """Load annotations, optionally filtered by round and clips.

    Arguments:
        conn: an active database connection
        round_name: if given, load only annotations from this calibration or production
            round, by joining against annotation_assignments. If None, load all annotations
            regardless of round.
        clip_ids: if given, load only annotations for these clips

    Returns:
        a tuple of row dicts, each carrying the annotation's fields. Use a dict rather
        than a dataclass here so the caller can construct whatever contract object they
        need; the annotation itself is immutable so the dict can be taken as the source of
        truth.
    """
    query = select(annotations)

    if clip_ids:
        query = query.where(annotations.c.clip_id.in_(clip_ids))

    if round_name is not None:
        # Join against assignments to filter by round. This requires assignment_id to be
        # non-null; annotations without an assignment are kept separate.
        query = query.where(
            annotations.c.assignment_id.in_(
                select(annotation_assignments.c.assignment_id).where(
                    annotation_assignments.c.round_name == round_name
                ),
            ),
        )

    result = conn.execute(query)
    return tuple(dict(row._mapping) for row in result)


def queue_for(
    conn: Connection,
    rater_id: str,
    *,
    limit: int = 25,
    round_name: str | None = None,
) -> tuple[Assignment, ...]:
    """Fetch the clips a rater is assigned to label.

    The rater receives assignments they have not yet submitted an annotation for. An
    assignment is "done" when an annotation row exists with the same assignment_id
    and behaviour, so we exclude those.

    **An excluded session's assignments are withheld.** A round planned before the exclusion
    still holds its rows - this table is append-only, so retiring them is not available and
    would not be wanted either, since the plan is the record of what was asked for. But D96
    decided that session is not annotated, and `plan_annotation.py` already refuses to put it
    in a new round. A queue that served it anyway would make the exclusion mean one thing when
    planning and another when labelling, and the rater is the one place where that costs hours
    of somebody's attention.

    Arguments:
        conn: an active database connection
        rater_id: the rater to queue for
        limit: maximum number of assignments to return, typically 25 or so for a single
            fetch
        round_name: if given, load only from this round. If None, load from any round.

    Returns:
        a tuple of Assignment objects, ordered by clip_index so the UI can walk them
        in sequence
    """
    query = select(annotation_assignments).where(
        annotation_assignments.c.rater_id == rater_id,
        select(sessions.c.session_id)
        .where(sessions.c.session_id == annotation_assignments.c.session_id,
               sessions.c.excluded_at.is_(None))
        .exists(),
    )

    if round_name is not None:
        query = query.where(annotation_assignments.c.round_name == round_name)

    # Exclude assignments that already have an annotation for this behaviour.
    # An assignment carries one behaviour, so checking behaviour + assignment_id is enough.
    #
    # NOT EXISTS rather than NOT IN, and `==` rather than `is_`. Both matter:
    #
    #   `is_` renders SQL `IS`, which PostgreSQL accepts only against NULL, TRUE or FALSE.
    #   Against a column it is a syntax error, so this query could never have run.
    #
    #   `annotations.assignment_id` is nullable, because an annotation can be made outside any
    #   assignment. Under NOT IN, a single NULL in the subquery makes the predicate UNKNOWN for
    #   every row and the queue comes back empty for every rater - silently, with no error to
    #   notice. NOT EXISTS compares row by row and is unaffected.
    query = query.where(
        ~select(annotations.c.annotation_id)
        .where(annotations.c.assignment_id == annotation_assignments.c.assignment_id)
        .exists(),
    )

    query = query.order_by(annotation_assignments.c.clip_index).limit(limit)

    result = conn.execute(query)
    assignments = []
    for row in result:
        row_dict = dict(row._mapping)
        clip = ClipRef(
            session_id=row_dict["session_id"],
            clip_index=row_dict["clip_index"],
            start_seconds=row_dict["clip_start_s"],
            end_seconds=row_dict["clip_end_s"],
        )
        assignments.append(
            Assignment(
                assignment_id=row_dict["assignment_id"],
                rater_id=row_dict["rater_id"],
                clip=clip,
                behaviour=row_dict["behaviour"],
                round_name=row_dict["round_name"],
            ),
        )

    return tuple(assignments)


def adopt_codebook_revision(
    conn: Connection,
    codebook: Any,
    *,
    supersedes: str | None = None,
    rationale: str,
) -> None:
    """Record a new codebook version.

    The codebook is the authority on what labels mean. When it changes, the change is
    recorded here with the codebook's SHA-256 hash so a later reader can verify what
    the raters saw. Every annotation records the codebook version it was made under, so
    a revision cannot silently reinterpret an earlier label. If this exact codebook
    version has been adopted before, this call is idempotent.

    Arguments:
        conn: an active database connection
        codebook: the codebook object, which must have a `version` field and a
            `document_sha256` field (computed from the prose the raters read)
        supersedes: the version this one replaces, if any. None for the first version.
        rationale: a free-text explanation of what changed and why, for the audit trail

    Raises:
        IntegrityError if a different version has already been adopted with the same
        SHA-256 hash, which means two different codebook objects somehow generated the
        same hash. This is a data integrity error, not a conflict to handle gracefully.
    """
    # Check if this version exists already. If it does, this is idempotent.
    query = select(codebook_revisions).where(
        codebook_revisions.c.version == codebook.version,
    )
    existing = conn.execute(query).first()

    if existing:
        # Idempotent: the version exists already with the same hash.
        if existing.document_sha256 == codebook.document_sha256:
            return
        # Otherwise it is a corrupt record: same version, different content.
        raise ValueError(
            f"codebook version {codebook.version} exists with different content: "
            f"stored {existing.document_sha256}, supplied {codebook.document_sha256}",
        )

    # New version: insert it.
    conn.execute(
        codebook_revisions.insert(),
        {
            "version": codebook.version,
            "document_sha256": codebook.document_sha256,
            "adopted_at": datetime.now(UTC),
            "supersedes": supersedes,
            "rationale": rationale,
        },
    )
