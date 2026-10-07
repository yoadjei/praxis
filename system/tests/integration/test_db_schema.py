# -*- coding: utf-8 -*-
"""The Core table definitions against the database they claim to describe.

`praxis/db/schema.py` restates what the migrations create. That is a third copy of the same
facts, after the DDL and the pydantic contracts, and this project has twice shipped a
disagreement between two such copies that every local reading made look correct.

So this compares code against the running database rather than against another document. A
migration that adds a column, or a hand-edit that drops one, fails here rather than at the first
insert that mentions it.
"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect

from praxis.db import build_engine, keywords_to_url, metadata
from praxis.db.schema import APPEND_ONLY

# Tables an earlier migration owns and Phase 1 does not describe in Core yet. Named rather than
# skipped over silently, so adding one is a deliberate act.
NOT_YET_MAPPED = {"adjudications", "alembic_version"}


def inspector_or_none(scratch_database: str | None):
    if scratch_database is None:
        return None
    return inspect(build_engine(keywords_to_url(scratch_database)))


def test_every_core_table_exists_in_the_database(scratch_database: str | None) -> None:
    inspector = inspector_or_none(scratch_database)
    if inspector is None:
        return

    actual = set(inspector.get_table_names()) | set(inspector.get_view_names())
    declared = set(metadata.tables)
    assert declared <= actual, (
        f"praxis/db/schema.py declares tables the migrations do not create: "
        f"{sorted(declared - actual)}")


def test_no_table_is_missing_from_the_core_definitions(scratch_database: str | None) -> None:
    """The direction that actually catches drift. A table the migrations create and the code
    does not know about is a table nothing in this suite is checking."""
    inspector = inspector_or_none(scratch_database)
    if inspector is None:
        return

    actual = set(inspector.get_table_names()) - NOT_YET_MAPPED
    unmapped = actual - set(metadata.tables)
    assert not unmapped, (
        f"{sorted(unmapped)} exist in the database and are absent from praxis/db/schema.py. "
        f"Add them, or add them to NOT_YET_MAPPED with a reason.")


@pytest.mark.parametrize("table_name", sorted(
    set(metadata.tables) - {"active_consents"}))
def test_columns_and_nullability_agree(scratch_database: str | None, table_name: str) -> None:
    """Names, and whether each may be null. Types are deliberately not compared: SQLAlchemy
    reflects a CHAR(26) and a TEXT into different objects than the ones declared here, and a
    comparison that needed a translation table would be asserting about the translation."""
    inspector = inspector_or_none(scratch_database)
    if inspector is None:
        return

    live = {column["name"]: column["nullable"]
            for column in inspector.get_columns(table_name)}
    declared = {column.name: column.nullable
                for column in metadata.tables[table_name].columns}

    assert set(declared) == set(live), (
        f"{table_name}: only in code {sorted(set(declared) - set(live))}, "
        f"only in the database {sorted(set(live) - set(declared))}")

    disagreements = [f"{name}: code says {'null' if declared[name] else 'not null'}, "
                     f"database says {'null' if live[name] else 'not null'}"
                     for name in declared if declared[name] != live[name]]
    assert not disagreements, f"{table_name}:\n  " + "\n  ".join(disagreements)


def test_the_append_only_tables_are_still_protected(scratch_database: str | None) -> None:
    """`APPEND_ONLY` is a claim this module makes about two tables. R5 depends on it being
    true of the database and not merely written in a frozenset."""
    if scratch_database is None:
        return

    import psycopg
    with psycopg.connect(scratch_database) as conn:
        protected = {row[0] for row in conn.execute(
            "SELECT DISTINCT c.relname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
            "WHERE NOT t.tgisinternal")}

    unprotected = APPEND_ONLY - protected
    assert not unprotected, (
        f"R5: {sorted(unprotected)} are declared append-only in praxis/db/schema.py and carry "
        f"no trigger in the database")
