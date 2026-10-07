# -*- coding: utf-8 -*-
"""The migrations, against a real database wherever one is configured.

SCHEMA.md section 12 rule 2: every migration has a tested downgrade, and an untested downgrade
is not a downgrade. 0001 is the single exception and earns it from R5 - an audit trail that a
migration can un-make was never append-only. 0002 inherits nothing from that argument, so its
downgrade runs here.

Without `PRAXIS_TEST_DSN` these degrade to reading the migration files, which still catches a
migration that stops creating a table or starts granting a privilege R5 forbids. Configured but
unreachable is a failure, never a downgrade to the weaker check: see tests/conftest.py.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from praxis.ids import new_ulid
from tests.conftest import database_dsn

PHASE_1_TABLES = ("colleges", "teachers", "consent_records", "media_objects", "sessions")
PHASE_1_OBJECTS = (*PHASE_1_TABLES, "active_consents")


def alembic(repo_root: Path, *args: str, dsn: str | None = None) -> subprocess.CompletedProcess:
    """Alembic reads DATABASE_URL, never alembic.ini, so a password cannot be committed."""
    dsn = dsn or database_dsn()
    assert dsn is not None
    parts = dict(piece.split("=", 1) for piece in dsn.split())
    url = (f"postgresql+psycopg://{parts['user']}:{parts['password']}"
           f"@{parts['host']}:{parts['port']}/{parts['dbname']}")
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args], cwd=repo_root, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=180, env={**os.environ, "DATABASE_URL": url})


def objects_present(dsn: str, names: tuple[str, ...]) -> set[str]:
    import psycopg
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(%s)", (list(names),)).fetchall()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# Static: these run whether or not a database is configured
# ---------------------------------------------------------------------------

def test_every_phase_1_table_is_created_by_a_migration(repo_root: Path) -> None:
    sources = "\n".join(p.read_text(encoding="utf-8")
                        for p in (repo_root / "migrations" / "versions").glob("*.py"))
    for table in PHASE_1_TABLES:
        assert re.search(rf"CREATE TABLE {table}\b", sources), (
            f"no migration creates {table}, so Phase 1 cannot run against a fresh database")


def test_0001_still_refuses_to_be_reversed(repo_root: Path) -> None:
    """The exception to section 12, and it has to stay exceptional."""
    source = (repo_root / "migrations" / "versions" / "0001_append_only_audit.py").read_text(
        encoding="utf-8")
    body = source.split("def downgrade()", 1)[1]
    assert "raise" in body, (
        "R5: 0001 now has a downgrade that runs. Reversing it drops the trigger and restores "
        "the privileges, which makes the audit trail mutable.")


def test_0002_downgrade_is_not_a_stub(repo_root: Path) -> None:
    source = (repo_root / "migrations" / "versions" / "0002_base_schema.py").read_text(
        encoding="utf-8")
    body = source.split("def downgrade()", 1)[1]
    assert "raise" not in body, (
        "SCHEMA.md section 12 rule 2: 0002 creates no append-only table, so it does not "
        "inherit 0001's exemption and must be reversible")
    for table in reversed(PHASE_1_TABLES):
        assert f"DROP TABLE IF EXISTS {table}" in body, f"downgrade leaves {table} behind"


# ---------------------------------------------------------------------------
# Live: the same claims, against PostgreSQL
# ---------------------------------------------------------------------------

def head_revision(repo_root: Path) -> str:
    """The newest revision on disk. Read rather than named, so adding a migration does not
    quietly leave this test asserting something about an older one."""
    revisions = {}
    for path in (repo_root / "migrations" / "versions").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        current = re.search(r'^revision = "([^"]+)"', source, re.MULTILINE)
        previous = re.search(r'^down_revision = (?:"([^"]+)"|None)', source, re.MULTILINE)
        if current:
            revisions[current.group(1)] = previous.group(1) if previous else None
    parents = {parent for parent in revisions.values() if parent}
    heads = set(revisions) - parents
    assert len(heads) == 1, f"migrations have {len(heads)} heads: {sorted(heads)}"
    return heads.pop()


def test_the_migration_actually_applied(repo_root: Path) -> None:
    dsn = database_dsn()
    if dsn is None:
        return

    assert objects_present(dsn, PHASE_1_OBJECTS) == set(PHASE_1_OBJECTS), (
        "run `alembic upgrade head` with DATABASE_URL set; the configured database is behind "
        "the migrations")
    assert head_revision(repo_root) in alembic(repo_root, "current").stdout


def test_0002_downgrade_is_real(repo_root: Path) -> None:
    """The section 12 requirement, executed rather than asserted about.

    Run against a scratch database rather than the developer's. Two reasons, and the second is
    the interesting one: a destructive test should not be able to damage the database somebody
    is working in, and 0003 refuses to be reversed on purpose, so a linear downgrade from head
    could never reach 0001 anyway. Building up to 0002 in a fresh database tests exactly the
    path a new deployment takes.
    """
    dsn = database_dsn()
    if dsn is None:
        return

    scratch = f"praxis_downgrade_{new_ulid()[-12:].lower()}"
    admin = re.sub(r"dbname=\S+", "dbname=postgres", dsn)
    scratch_dsn = re.sub(r"dbname=\S+", f"dbname={scratch}", dsn)

    import psycopg
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{scratch}"')
    try:
        assert alembic(repo_root, "upgrade", "0002_base_schema",
                       dsn=scratch_dsn).returncode == 0
        assert objects_present(scratch_dsn, PHASE_1_OBJECTS) == set(PHASE_1_OBJECTS)

        down = alembic(repo_root, "downgrade", "0001_append_only_audit", dsn=scratch_dsn)
        assert down.returncode == 0, f"downgrade failed:\n{down.stderr}"
        assert objects_present(scratch_dsn, PHASE_1_OBJECTS) == set(), (
            "downgrade left objects behind: "
            f"{sorted(objects_present(scratch_dsn, PHASE_1_OBJECTS))}")

        # R5 is untouched by an ordinary downgrade. The trail outlives the schema around it.
        with psycopg.connect(scratch_dsn) as conn:
            assert conn.execute("SELECT to_regclass('audit_log')").fetchone()[0] is not None, (
                "R5: downgrading the base schema destroyed the audit trail")
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                         "WHERE datname = %s AND pid <> pg_backend_pid()", (scratch,))
            conn.execute(f'DROP DATABASE IF EXISTS "{scratch}"')


def test_0003_refuses_to_be_reversed(repo_root: Path) -> None:
    """0003 removes the last way to empty the trail, so reversing it would put it back."""
    source = (repo_root / "migrations" / "versions" / "0003_forbid_truncate.py").read_text(
        encoding="utf-8")
    assert "raise" in source.split("def downgrade()", 1)[1]


def test_active_consents_hides_withdrawn_and_expired(repo_root: Path) -> None:
    """The view is the only definition of "live consent". Ingest asks it, nothing reimplements
    the predicate in Python, and this is what stops the two drifting."""
    dsn = database_dsn()
    if dsn is None:
        return

    import psycopg

    from praxis.ids import new_ulid

    college, live, withdrawn, expired = new_ulid(), new_ulid(), new_ulid(), new_ulid()
    with psycopg.connect(dsn) as conn:
        conn.execute("INSERT INTO colleges (college_id, code) VALUES (%s, %s)",
                     (college, f"COL-{college[-6:]}"))
        for consent_id, withdrawn_on, expires_on in (
                (live, None, None),
                (withdrawn, "2026-01-01", None),
                (expired, None, "2026-01-01")):
            conn.execute(
                "INSERT INTO consent_records (consent_id, subject_type, college_id, purpose, "
                "recipients, scope, granted_on, expires_on, withdrawn_on, document_ref) "
                "VALUES (%s,'teacher',%s,'research','supervisors','both',"
                "'2025-01-01',%s,%s,'f/1')",
                (consent_id, college, expires_on, withdrawn_on))

        visible = {row[0] for row in conn.execute(
            "SELECT consent_id FROM active_consents WHERE college_id = %s", (college,))}
        conn.rollback()

    assert live in visible
    assert withdrawn not in visible, "a withdrawn consent is still live in active_consents"
    assert expired not in visible, "an expired consent is still live in active_consents"
