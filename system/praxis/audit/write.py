# -*- coding: utf-8 -*-
"""Appending to the audit trail inside somebody else's transaction.

`praxis.audit.chain` knows how a row is hashed and how a chain is verified, and holds no opinion
about storage. This module is the storage half, and it takes a `Connection` rather than opening
one, because D47 requires the audit row and the fact it describes to commit together. A function
that managed its own transaction could not offer that.

The chain is an ordered structure and the order cannot be recovered afterwards, so appenders
are serialised before the head is read. Two concurrent uploads that both read `prev_hash` before
either writes produce a fork, `verify()` reports it, and nobody can repair it because the table
refuses `UPDATE`.

**The serialisation is an advisory lock, not `SELECT ... FOR UPDATE` on the newest row.** That
was the first design and it does not work. A row lock cannot prevent an insert: the second
writer blocks on row N, and when the first writer commits row N+1 the lock is granted on row N,
which is still what its query returns. Both then chain from N. On an empty table there is no row
to lock and both writers believe they are first. `pg_advisory_xact_lock` locks the *chain*
rather than a row, covers the empty case, and is released by commit or rollback. D47.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Connection, func, insert, select, text

from praxis.audit.chain import GENESIS_PREV_HASH, AuditRecord
from praxis.audit.events import EVENTS, EventSpec
from praxis.db.schema import audit_log
from praxis.ids import is_ulid


class AuditWriteError(RuntimeError):
    """The row could not be written, or would not have meant what it claimed."""


# One lock for the whole chain, held for the transaction. The value is arbitrary but must be
# stable and must not collide with another advisory lock in this database, so it is derived from
# the table name rather than picked.
CHAIN_LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"praxis.audit_log.chain").digest()[:8], "big", signed=True)


def _spec(event_type: str) -> EventSpec:
    spec = EVENTS.get(event_type)
    if spec is None:
        raise AuditWriteError(
            f"{event_type!r} is not in the event taxonomy. Add it to praxis/audit/events.py "
            f"with the fields a reader would need, or the trail records something nobody can "
            f"interpret.")
    return spec


def _validated(spec: EventSpec, payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in spec.required_fields if field not in payload]
    if missing:
        raise AuditWriteError(
            f"{spec.event_type} requires {', '.join(spec.required_fields)}; missing "
            f"{', '.join(missing)}")
    return dict(payload)


def _storable_actor(actor_user_id: str | None) -> str | None:
    """Refuse an actor id the column cannot hold without altering it.

    `audit_log.actor_user_id` is `CHAR(26)`, which is blank-padded: a shorter value comes back
    from `read_chain` with trailing spaces, `AuditRecord.digest()` recomputes the hash over the
    padded string, and `verify` reports a content break on a row nobody touched. The chain is
    then unverifiable from that row onward and the only symptom is R5 failing for a reason that
    looks like tampering.

    So this is refused at the boundary rather than stripped on the way back. Stripping would
    mean the trail hashed one value and stored another, and a chain whose verification depends
    on normalising what it reads is not proving much. The column says the actor is a ULID; a
    caller that has no ULID passes None, which is a recorded absence and stores faithfully.
    """
    if actor_user_id is None:
        return None
    if not is_ulid(actor_user_id):
        raise AuditWriteError(
            f"actor_user_id {actor_user_id!r} is not a ULID, and audit_log.actor_user_id is "
            f"CHAR(26). A shorter value is stored blank-padded, so the row would come back "
            f"differing from what was hashed and the chain would report a break at every "
            f"verification from here on (R5). Pass a ULID, or None to record that the actor "
            f"is unknown.")
    return actor_user_id


def lock_chain(connection: Connection) -> None:
    """Take the chain lock for the rest of this transaction.

    Every appender takes the same lock, so appends queue instead of racing. Released by commit
    or rollback, including a rollback nobody wrote, which is why it is the transaction-scoped
    variant and not `pg_advisory_lock`.
    """
    connection.execute(select(func.pg_advisory_xact_lock(CHAIN_LOCK_KEY)))


def current_head(connection: Connection) -> str:
    """The last row's hash. Call `lock_chain` first if you intend to append."""
    return connection.execute(
        select(audit_log.c.row_hash).order_by(audit_log.c.audit_id.desc()).limit(1)
    ).scalar() or GENESIS_PREV_HASH


def append(connection: Connection, event_type: str, entity_id: str,
           payload: Mapping[str, Any], *, actor_user_id: str | None = None,
           actor_role: str | None = None, occurred_at: datetime | None = None) -> AuditRecord:
    """Seal one event onto the chain and insert it. Commits nothing; the caller owns that."""
    # Every check happens before the lock is taken. Holding the chain lock while discovering the
    # payload is malformed would make every other appender wait on a transaction already doomed.
    spec = _spec(event_type)
    checked = _validated(spec, payload)
    actor_user_id = _storable_actor(actor_user_id)

    lock_chain(connection)
    record = AuditRecord(
        occurred_at=occurred_at or datetime.now(UTC),
        event_type=event_type,
        entity_type=spec.entity_type,
        entity_id=entity_id,
        payload=checked,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
    ).sealed(current_head(connection))

    connection.execute(insert(audit_log).values(**record.as_row()))
    return record


def read_chain(connection: Connection, limit: int | None = None) -> list[AuditRecord]:
    """The stored rows as records, oldest first, for `praxis.audit.chain.verify`.

    The stored `row_hash` *is* carried across, and has to be: `AuditRecord.digest()` recomputes
    the digest from content and never reads it from storage, so the stored value is the other
    half of the comparison that detects tampering. Dropping it would leave `verify` comparing
    each recomputed digest against an empty string and calling every row broken.

    An earlier version of this docstring claimed the opposite - that the hash was deliberately
    not carried, lest a tampered row vouch for itself - while the code below carried it. The
    code was right. The sentence is recorded here because acting on it would have disabled R5's
    detection in the course of making the code agree with its own comment.
    """
    statement = select(audit_log).order_by(audit_log.c.audit_id)
    if limit is not None:
        statement = statement.limit(limit)

    records = []
    for row in connection.execute(statement).mappings():
        records.append(AuditRecord(
            occurred_at=row["occurred_at"],
            event_type=row["event_type"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            payload=row["payload"],
            actor_user_id=row["actor_user_id"],
            actor_role=row["actor_role"],
            prev_hash=row["prev_hash"] or GENESIS_PREV_HASH,
            row_hash=row["row_hash"],
        ))
    return records


def assert_append_only(connection: Connection) -> None:
    """Confirm the database itself still refuses to change this table.

    Cheap, and worth doing at startup rather than discovering during an incident that the
    trigger was dropped by a migration nobody reviewed. R5.
    """
    present = connection.execute(text(
        "SELECT count(*) FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
        "WHERE NOT t.tgisinternal AND c.relname = 'audit_log'")).scalar()
    if not present:
        raise AuditWriteError(
            "R5: audit_log has no mutation trigger. The append-only guarantee is not in force; "
            "re-apply migration 0001 before writing anything else.")
