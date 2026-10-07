# -*- coding: utf-8 -*-
"""The hash-chained audit log: append-only in Python, tamper-evident against anyone.

Two guarantees, and they stop different attackers. The database's `REVOKE` plus trigger stops
the application role and an ordinary mistake. Neither stops a superuser, and a research record
that a superuser can silently edit is not a research record. The chain is what makes that
edit *detectable*: every row's hash covers the previous row's hash, so altering, deleting or
reordering anything invalidates every hash after it and `verify` names the first break.

**SCHEMA.md gives the formula but not the encoding, and an unencoded formula is not a
specification.** `row_hash = sha256(prev_hash || occurred_at || actor || event_type ||
entity_id || payload)` leaves four things open, and two independent implementations that
disagree on any of them produce different hashes for the same row, which is a false alarm that
teaches people to ignore the alarm. So this module pins all four:

* **Framing.** Fields are length-prefixed, not concatenated. Plain concatenation cannot tell
  `("ab", "c")` from `("a", "bc")`, so an attacker could move a character across a field
  boundary and keep the hash. Each field is written as an 8-byte big-endian length followed by
  its bytes, which makes the encoding injective.
* **`actor`.** The table has both `actor_user_id` and `actor_role`, and the formula names
  neither. Both are framed, separately, so a role change cannot pass unnoticed.
* **The genesis row.** `prev_hash` is NULL for the first row. It is framed as the empty string,
  which is a value, rather than the text "NULL", which is a different row's plausible content.
* **Serialisation.** Timestamps are ISO 8601 in UTC with microseconds and a `Z`; a naive
  datetime is refused rather than assumed local. Payloads are compact JSON with sorted keys.

**`entity_type` is included although the formula omits it.** Without it, editing a row's
`entity_type` in place leaves `row_hash` valid, and "which kind of thing this event was about"
is exactly the sort of field an edit would target. Recorded as a deviation.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from praxis.audit.events import AuditError, spec_for

GENESIS_PREV_HASH = ""


def canonical_timestamp(moment: datetime) -> str:
    """ISO 8601, UTC, microsecond precision, `Z`. Naive datetimes are refused.

    A naive timestamp would be hashed as whatever the writing machine's clock meant at the
    time, and the same event replayed elsewhere would hash differently.
    """
    if moment.tzinfo is None:
        raise AuditError(
            f"{moment!r} has no timezone. An audit timestamp without one cannot be compared "
            f"across machines, and the chain would break the first time it was verified "
            f"somewhere else.")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def canonical_payload(payload: Mapping[str, Any]) -> str:
    """Compact JSON with sorted keys, so two equal payloads have one representation.

    A value JSON cannot represent natively is **refused, not coerced**. Coercing with
    `default=str` looks harmless and breaks injectivity: `Decimal("1.5")` and the string
    `"1.5"` would canonicalise identically, so two genuinely different rows would share a
    hash. The column is `JSONB`, so a payload that is not JSON was going to lose information
    at the database anyway; failing here says so while the caller is still in scope.
    """
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AuditError(
            f"an audit payload must be JSON: {exc}. Convert the value at the call site — a "
            f"datetime to an ISO string, a Decimal to a float or a string — rather than "
            f"letting the encoder guess, because two different values that stringify alike "
            f"would then share a row_hash.") from exc


def _framed(*parts: str) -> bytes:
    """Length-prefix each field so the concatenation is injective."""
    out = bytearray()
    for part in parts:
        encoded = part.encode("utf-8")
        out += len(encoded).to_bytes(8, "big") + encoded
    return bytes(out)


@dataclass(frozen=True)
class AuditRecord:
    """One row. Frozen, because R5 is not a convention here either."""

    occurred_at: datetime
    event_type: str
    entity_type: str
    entity_id: str
    payload: dict[str, Any]
    actor_user_id: str | None = None
    actor_role: str | None = None
    prev_hash: str = GENESIS_PREV_HASH
    row_hash: str = ""

    def __post_init__(self) -> None:
        # `frozen=True` stops the field being reassigned; it does nothing about the dict the
        # field points at. A caller who keeps a reference could mutate the payload after the
        # row was sealed, leaving row_hash describing content the record no longer holds. The
        # copy is taken here rather than in `append` so that building an AuditRecord directly
        # is safe too. Round-tripping through canonical JSON also copies nested structures and
        # refuses anything the column could not store.
        object.__setattr__(self, "payload", json.loads(canonical_payload(self.payload)))

    def digest(self) -> str:
        """This row's hash, recomputed from its content. Never read from storage."""
        return hashlib.sha256(_framed(
            self.prev_hash or GENESIS_PREV_HASH,
            canonical_timestamp(self.occurred_at),
            self.actor_user_id or "",
            self.actor_role or "",
            self.event_type,
            self.entity_type,
            self.entity_id,
            canonical_payload(self.payload),
        )).hexdigest()

    def sealed(self, prev_hash: str) -> AuditRecord:
        """This row placed after `prev_hash`, with its own hash computed.

        `digest()` does not read `row_hash`, so linking first and hashing second is safe and
        there is no order in which a row could hash its own hash.
        """
        linked = replace(self, prev_hash=prev_hash, row_hash="")
        return replace(linked, row_hash=linked.digest())

    def as_row(self) -> dict[str, Any]:
        """The shape the `audit_log` table stores."""
        return {
            "occurred_at": canonical_timestamp(self.occurred_at),
            "actor_user_id": self.actor_user_id,
            "actor_role": self.actor_role,
            "event_type": self.event_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "payload": json.loads(canonical_payload(self.payload)),
            "prev_hash": self.prev_hash or None,
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class ChainBreak:
    """Where the chain first stopped being consistent, and in which of the two ways."""

    index: int
    kind: str                    # "content" | "link"
    expected: str
    found: str
    record: AuditRecord

    def describe(self) -> str:
        if self.kind == "content":
            return (f"row {self.index} ({self.record.event_type} on "
                    f"{self.record.entity_id}): its content no longer hashes to its stored "
                    f"row_hash. Expected {self.expected[:16]}..., stored {self.found[:16]}... "
                    f"The row was edited in place.")
        return (f"row {self.index} ({self.record.event_type} on {self.record.entity_id}): its "
                f"prev_hash does not match row {self.index - 1}'s row_hash. Expected "
                f"{self.expected[:16]}..., found {self.found[:16]}... A row was deleted, "
                f"inserted or reordered.")


class AuditLog:
    """An append-only, hash-chained sequence of events.

    There is no update and no delete, here or at the database. A correction is a new row, which
    is why `append` is the only mutating method this class has.
    """

    def __init__(self, records: Iterable[AuditRecord] = ()) -> None:
        self._records: list[AuditRecord] = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, index: int) -> AuditRecord:
        return self._records[index]

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    @property
    def head(self) -> str:
        return self._records[-1].row_hash if self._records else GENESIS_PREV_HASH

    def append(self, event_type: str, entity_id: str, payload: Mapping[str, Any],
               occurred_at: datetime, actor_user_id: str | None = None,
               actor_role: str | None = None) -> AuditRecord:
        """Record an event. Refuses anything a replay could not read back.

        The payload check is here rather than at the call site because the call sites are the
        places least likely to be revisited: an event recorded without the fields a replay
        needs is a hole that only shows up when the trail is finally asked a question.
        """
        spec = spec_for(event_type)
        spec.check(spec.entity_type, payload)

        record = AuditRecord(
            occurred_at=occurred_at, event_type=event_type, entity_type=spec.entity_type,
            entity_id=entity_id, payload=dict(payload), actor_user_id=actor_user_id,
            actor_role=actor_role).sealed(self.head)
        self._records.append(record)
        return record

    def verify(self) -> ChainBreak | None:
        """Walk the chain and return the first break, or None if it is intact.

        The first break, not all of them: everything after a break is expected to fail, so a
        list of them would be one real finding followed by noise.
        """
        return verify(self._records)


def verify(records: Sequence[AuditRecord]) -> ChainBreak | None:
    """Check a sequence of rows without needing an `AuditLog` to hold them."""
    previous = GENESIS_PREV_HASH
    for index, record in enumerate(records):
        if (record.prev_hash or GENESIS_PREV_HASH) != previous:
            return ChainBreak(index, "link", previous,
                              record.prev_hash or GENESIS_PREV_HASH, record)
        recomputed = record.digest()
        if recomputed != record.row_hash:
            return ChainBreak(index, "content", recomputed, record.row_hash, record)
        previous = record.row_hash
    return None
