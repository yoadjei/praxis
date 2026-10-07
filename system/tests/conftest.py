# -*- coding: utf-8 -*-
"""Shared fixtures. Synthetic data only; never real participant data."""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.annotation.codebook import ACTIVE_CODEBOOK, ScaleType
from praxis.config import load_config
from praxis.contracts import ConfidenceState, Detection
from praxis.ids import new_ulid
from praxis.vocabulary import BehaviourId

PRAXIS_PACKAGE = REPO_ROOT / "praxis"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config():
    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="session")
def python_sources() -> list[Path]:
    """Every module under praxis/, for the tests that read the source rather than run it."""
    return sorted(PRAXIS_PACKAGE.rglob("*.py"))


@pytest.fixture
def confidence() -> ConfidenceState:
    return ConfidenceState(
        raw_prob=0.80, calibrated_prob=0.91, method="ensemble", epistemic=0.04,
        ood_score=0.12, ood_flag=False, in_validated_domain="microteaching")


def example_prediction(behaviour: BehaviourId, nonscorable: bool = False) -> dict:
    """A complete, in-domain prediction for one behaviour, built from the codebook itself.

    Derived rather than hand-written, so that adding a field to the codebook cannot leave the
    fixtures silently producing a prediction that no longer validates. The first level of an
    ordered field and the midpoint of a numeric one are arbitrary but legal, which is all a
    fixture needs.
    """
    spec = ACTIVE_CODEBOOK.behaviour(behaviour)
    labels: dict = {}
    for field in spec.fields:
        if field.name.endswith("_nonscorable"):
            labels[field.name] = nonscorable
        elif field.levels is not None:
            labels[field.name] = field.levels[-1] if len(field.levels) == 2 else field.levels[0]
        else:
            value = (field.minimum + field.maximum) / 2.0
            labels[field.name] = value if field.scale is ScaleType.INTERVAL else round(value)
    return labels


@pytest.fixture
def prediction() -> dict:
    return example_prediction("B1")


@pytest.fixture
def detection(confidence: ConfidenceState) -> Detection:
    return Detection(
        detection_id=new_ulid(), session_id=new_ulid(), behaviour="B1",
        t_start_s=128.0, t_end_s=136.0, predicted=example_prediction("B1"),
        confidence=confidence,
        evidence_ref="/api/v1/evidence/synthetic", model_version="b-resnet50-tcn-v3-ens5",
        gate_outcome="present")


@pytest.fixture
def utc_now() -> datetime:
    return datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)


# Tests that need a real database read this. Absent, the tests that can degrade to a static
# check say so and do it; **set but unreachable is a failure, never a silent downgrade**,
# because "the database test did not run" and "the database test passed" must never look alike.
TEST_DSN_VAR = "PRAXIS_TEST_DSN"


def database_dsn() -> str | None:
    """The DSN to test against, or None if none is configured.

    Deliberately not a fixture: `tests/test_invariants.py` calls it inside a test that must run
    either way, and a fixture that skipped would make an unenforced invariant invisible — the
    exact failure mode R1 to R7 exist to prevent.
    """
    dsn = os.environ.get(TEST_DSN_VAR, "").strip()
    if not dsn:
        return None

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            f"{TEST_DSN_VAR} is set but psycopg is not installed, so the database-level "
            f"invariants cannot be checked. Install it or unset the variable.") from exc

    try:
        with psycopg.connect(dsn, connect_timeout=5):
            pass
    except psycopg.Error as exc:
        raise RuntimeError(
            f"{TEST_DSN_VAR} is set to a database that cannot be reached: {exc}. This is a "
            f"failure rather than a skip: a configured database that is down must not look "
            f"like a database that was never configured.") from exc
    return dsn


# A disposable database, migrated from nothing, for tests that write to the append-only tables.
#
# Those writes cannot be undone. The trigger refuses DELETE, D53 closed TRUNCATE, and a row that
# was written wrongly - by a hand-built INSERT, or by a mutation test exercising a broken
# implementation - breaks the chain permanently in whatever database it landed in. That is not
# hypothetical: it happened twice, and the second time was a mutation test proving the audit
# code was correct. See D54.
#
# Creating a fresh database per session costs about a second and makes the whole class of
# accident impossible.
def _admin_dsn(dsn: str) -> str:
    import re
    return re.sub(r"dbname=\S+", "dbname=postgres", dsn)


@contextmanager
def _migrated_database(dsn: str) -> Iterator[str]:
    """Create a database beside `dsn`, bring it to head, yield its DSN, then drop it.

    Written once and used at two scopes. `scratch_database` holds one of these open for the
    whole run; `private_database` gives a single test one of its own. Two copies of the
    create-migrate-drop sequence is the shape L20 warns about, and the one that would drift is
    the teardown, which is the half that matters.
    """
    import re
    import subprocess

    import psycopg

    from praxis.ids import new_ulid

    name = f"praxis_scratch_{new_ulid()[-12:].lower()}"
    scratch = re.sub(r"dbname=\S+", f"dbname={name}", dsn)
    parts = dict(piece.split("=", 1) for piece in scratch.split())
    url = (f"postgresql+psycopg://{parts['user']}:{parts['password']}"
           f"@{parts['host']}:{parts['port']}/{name}")

    with psycopg.connect(_admin_dsn(dsn), autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    try:
        applied = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=REPO_ROOT,
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=180,
            env={**os.environ, "DATABASE_URL": url})
        if applied.returncode != 0:
            raise RuntimeError(f"could not migrate the scratch database:\n{applied.stderr}")
        yield scratch
    finally:
        with psycopg.connect(_admin_dsn(dsn), autocommit=True) as conn:
            conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                         "WHERE datname = %s AND pid <> pg_backend_pid()", (name,))
            conn.execute(f'DROP DATABASE IF EXISTS "{name}"')


@pytest.fixture(scope="session")
def scratch_database() -> str | None:
    """A migrated, empty database that is dropped afterwards, or None if none is configured."""
    dsn = database_dsn()
    if dsn is None:
        # `yield`, not `return`: this is a generator fixture, and returning before the first
        # yield makes pytest error out instead of handing the test None to degrade on.
        yield None
        return
    with _migrated_database(dsn) as scratch:
        yield scratch


@pytest.fixture
def private_database() -> str | None:
    """A migrated database that one test alone writes to, dropped when that test ends.

    `scratch_database` is shared by the whole run on purpose, and for almost every test that is
    right: it costs a second once instead of once per test. But rows accumulate in it for the
    length of the run and cannot be cleared - the append-only triggers refuse DELETE and D53
    closed TRUNCATE - so a test whose subject is a corpus-wide aggregate cannot use it. The
    dashboard endpoints group over every session and page the list 25 at a time; the annotation
    planner reads the whole corpus to decide what is annotatable. A test asserting what those
    return is asserting something about the contents of the entire database, and in the shared
    one that is whatever ran earlier.

    That is not a hypothetical either. Six tests here passed run by file and failed run as a
    suite: "an empty corpus" saw twenty-six sessions, and two tests looked up their own session
    ids in a page of twenty-five that no longer held them. Paying a second for a database of
    one's own is what makes the assertion mean what it says.
    """
    dsn = database_dsn()
    if dsn is None:
        yield None
        return
    with _migrated_database(dsn) as scratch:
        yield scratch
