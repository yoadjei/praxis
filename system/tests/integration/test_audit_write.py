# -*- coding: utf-8 -*-
"""Writing the audit chain to PostgreSQL, including the part that only fails under load.

The single-threaded behaviour of an append-only chain is easy and every implementation gets it
right. The property that matters is D47: concurrent appenders must not both chain from the same
predecessor, because the fork they produce is permanent - `verify()` reports it and nobody can
repair it, since the table refuses `UPDATE`.

The first implementation took `SELECT ... FOR UPDATE` on the newest row, which reads correctly
and does nothing: a row lock cannot prevent an insert, so the blocked writer wakes holding a
lock on row N while row N+1 already exists. `test_concurrent_appends_do_not_fork_the_chain`
fails against that version and passes against the advisory lock.

Every test here runs against the `scratch_database` fixture rather than the developer's, and
that is not tidiness. Audit rows cannot be removed - the trigger refuses DELETE and D53 closed
TRUNCATE - so a test that writes a badly formed row breaks the chain permanently in whatever
database it touched. Mutation-testing this very module did exactly that, twice. D54.

Without a configured database these degrade to the parts that need no server. Configured but
unreachable is a failure: see tests/conftest.py.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import func, select

from praxis.audit.chain import verify
from praxis.audit.write import AuditWriteError, append, assert_append_only, read_chain
from praxis.db import build_engine, keywords_to_url, transaction
from praxis.ids import new_ulid
from tests.integration.conftest import requires_database

WRITERS = 12


def engine_or_none(scratch_database: str | None):
    return build_engine(keywords_to_url(scratch_database)) if scratch_database else None


def ingested(session_id: str) -> dict:
    """A session.ingested payload with every field the taxonomy demands."""
    return {"session_id": session_id, "teacher_id": "T-7", "domain": "classroom"}


# ---------------------------------------------------------------------------
# Taxonomy enforcement, which needs no database
# ---------------------------------------------------------------------------

def test_an_event_outside_the_taxonomy_is_refused() -> None:
    with pytest.raises(AuditWriteError, match="not in the event taxonomy"):
        append(None, "session.vibed", new_ulid(), {})


def test_a_payload_missing_a_required_field_is_refused() -> None:
    """A field declared required and then omitted makes the trail unable to answer the question
    it was written to answer, and nothing downstream would notice."""
    with pytest.raises(AuditWriteError, match="missing teacher_id, domain"):
        append(None, "session.ingested", new_ulid(), {"session_id": new_ulid()})


@pytest.mark.parametrize("actor", ["alice", "01M2H", "0" * 27,
                                   "01M2HDIRECT000000000000000"])
def test_an_actor_the_column_would_pad_is_refused(actor) -> None:
    """The row would come back differing from what was hashed, so every later verification
    would report a break. The last case is the one that found this: twenty-six characters and
    still not a ULID, because Crockford base32 excludes I, L, O and U."""
    with pytest.raises(AuditWriteError, match="is not a ULID"):
        append(None, "session.ingested", new_ulid(), ingested(new_ulid()),
               actor_user_id=actor)


def test_an_unknown_actor_is_none_rather_than_a_placeholder() -> None:
    """None stores faithfully as null and is a recorded absence. A placeholder string would be
    padded, which is the defect, and would also claim a person who does not exist."""
    with pytest.raises(AuditWriteError, match="not in the event taxonomy"):
        # Reaches the taxonomy check, which means the None actor got past the actor check.
        append(None, "session.nonexistent", new_ulid(), {}, actor_user_id=None)


def test_a_short_actor_would_otherwise_break_the_chain(scratch_database: str | None) -> None:
    """The defect itself, demonstrated against the column rather than argued about.

    Written through raw SQL because `append` now refuses it, which is the fix. What this pins
    is *why* it refuses: a 24-character actor stored in `CHAR(26)` comes back padded, and a
    digest recomputed over the padded value does not match the one that was stored.
    """
    engine = engine_or_none(scratch_database)
    if not requires_database(engine):
        return

    from sqlalchemy import text as sql

    with transaction(engine) as connection:
        stored = connection.execute(sql(
            "select cast(:value as char(26)) as padded"), {"value": "01M2H0000000000000000000"}
        ).scalar()
    assert stored == "01M2H0000000000000000000  "
    assert stored != "01M2H0000000000000000000", (
        "if this ever becomes equal the column stopped blank-padding and the guard in "
        "praxis.audit.write could be relaxed; until then it is load-bearing")


# ---------------------------------------------------------------------------
# Against PostgreSQL
# ---------------------------------------------------------------------------

def test_appending_links_each_row_to_the_one_before(scratch_database: str | None) -> None:
    engine = engine_or_none(scratch_database)
    if not requires_database(engine):
        return

    with transaction(engine) as connection:
        first = append(connection, "session.ingested", new_ulid(), ingested(new_ulid()))
        second = append(connection, "session.ingested", new_ulid(), ingested(new_ulid()))

        assert second.prev_hash == first.row_hash
        assert verify(read_chain(connection)) is None


def test_the_stored_chain_verifies_from_genesis(scratch_database: str | None) -> None:
    """Reads the whole table from genesis. `read_chain` deliberately does not carry the stored
    `row_hash` across, so a tampered row cannot vouch for itself."""
    engine = engine_or_none(scratch_database)
    if not requires_database(engine):
        return

    with transaction(engine) as connection:
        for _ in range(5):
            append(connection, "session.ingested", new_ulid(), ingested(new_ulid()))
        stored = read_chain(connection)
        assert len(stored) >= 5
        assert verify(stored) is None, "the chain as stored does not verify"


def test_the_database_still_refuses_to_change_the_trail(scratch_database: str | None) -> None:
    engine = engine_or_none(scratch_database)
    if not requires_database(engine):
        return
    with transaction(engine) as connection:
        assert_append_only(connection)


def test_concurrent_appends_do_not_fork_the_chain(scratch_database: str | None) -> None:
    """D47, and the only test here that a plausible wrong implementation fails.

    Each writer opens its own connection and its own transaction, so the serialisation has to
    come from the database rather than from the GIL.
    """
    engine = engine_or_none(scratch_database)
    if not requires_database(engine):
        return

    def write(index: int) -> str:
        with transaction(engine) as connection:
            record = append(connection, "session.ingested", new_ulid(), ingested(new_ulid()))
            return record.row_hash

    with ThreadPoolExecutor(max_workers=WRITERS) as pool:
        hashes = list(pool.map(write, range(WRITERS)))

    assert len(set(hashes)) == WRITERS, "two writers produced the same row"

    with transaction(engine) as connection:
        chain = read_chain(connection)
        break_found = verify(chain)
        assert break_found is None, (
            f"D47: concurrent appends forked the chain: "
            f"{break_found.describe() if break_found else ''}")

        predecessors = [record.prev_hash for record in chain]
        assert len(predecessors) == len(set(predecessors)), (
            "D47: two rows claim the same predecessor, which is a fork the chain cannot "
            "represent and the table cannot repair")


def test_nobody_can_tidy_up_after_these_tests(scratch_database: str | None) -> None:
    """These tests leave rows behind permanently, and that is the correct behaviour rather than
    a shortcoming. The first version of the concurrency test deleted its own rows in a finally
    block; the database refused, as superuser, which is exactly what R5 promises. An audit trail
    a test suite can tidy is not an audit trail."""
    engine = engine_or_none(scratch_database)
    if not requires_database(engine):
        return

    with transaction(engine) as connection:
        granted = connection.execute(select(func.has_table_privilege(
            "praxis_app", "audit_log", "DELETE"))).scalar()
    assert granted is False, "R5: praxis_app has been granted DELETE on audit_log"


# ---------------------------------------------------------------------------
# The two sources scripts/verify_audit_chain.py reads, which must agree
# ---------------------------------------------------------------------------

def test_the_database_and_the_json_export_give_the_same_chain(
    scratch_database: str | None, tmp_path
) -> None:
    """`--from-db` and `--from-json` are two readings of one chain, and must not diverge.

    The verifier was written against an export because there was no database to read when it was
    written, and it said so: "when the database lands, `--from-db` selects the same rows ordered
    by `audit_id` and hands them to the same `verify`; the check must not be reimplemented for
    the second source". The database landed and the flag was added, which creates the thing this
    test exists to prevent - two paths to the same answer, either of which can be changed on its
    own. The export shape is already covered in tests/unit/test_audit.py; what is checked here
    is that the shape and the table produce identical records.
    """
    engine = engine_or_none(scratch_database)
    if not requires_database(engine):
        return

    with transaction(engine) as connection:
        for _ in range(4):
            append(connection, "session.ingested", new_ulid(), ingested(new_ulid()))

    from scripts.verify_audit_chain import load, load_from_db

    from_db = load_from_db(keywords_to_url(scratch_database))
    assert from_db, "nothing was read back from audit_log"
    assert verify(from_db) is None, "the chain this test wrote does not verify from the table"

    # Export exactly what an operator would hand a reader, then read it back the other way.
    export = tmp_path / "audit.json"
    export.write_text(json.dumps([
        {
            "occurred_at": record.occurred_at.isoformat(),
            "event_type": record.event_type,
            "entity_type": record.entity_type,
            "entity_id": record.entity_id,
            "payload": record.payload,
            "actor_user_id": record.actor_user_id,
            "actor_role": record.actor_role,
            "prev_hash": record.prev_hash,
            "row_hash": record.row_hash,
        }
        for record in from_db
    ]), encoding="utf-8")

    from_json = load(export)
    assert verify(from_json) is None, "the exported chain does not verify"
    assert [record.row_hash for record in from_json] == [
        record.row_hash for record in from_db], "the two sources disagree about the chain"
    assert from_json == from_db, "the two sources build different records from the same rows"
