# -*- coding: utf-8 -*-
"""Split manifest persistence, against a live PostgreSQL.

The point of these tests is the layer the unit tests cannot reach. `SplitManifest` refuses a
contaminated split in Python and `plan_splits` refuses to produce one, but both are code an
agent can route around by writing the row itself. The `teachers_disjoint` CHECK constraint is
not, and that is what is exercised here: R2 asserted against the database, not against another
Python object.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, text

from praxis.contracts.manifest import SplitManifest
from praxis.db import transaction
from praxis.db.schema import colleges, split_manifests
from praxis.ids import new_ulid
from praxis.splits.store import (
    ManifestNotFound,
    load_manifest,
    manifests_for_config,
    save_manifest,
)
from tests.integration.conftest import requires_database

CONFIG_HASH = "c" * 64


def a_manifest(**overrides) -> SplitManifest:
    payload = dict(
        manifest_id=new_ulid(),
        created_at=datetime.now(UTC),
        config_sha256=CONFIG_HASH,
        train_teachers=["T1", "T2", "T3"],
        val_teachers=["T4"],
        test_teachers=["T5"],
        shift_eval_sessions=["S9"],
    )
    payload.update(overrides)
    return SplitManifest(**payload)


def test_a_manifest_round_trips_through_the_contract(engine) -> None:
    """Saved and read back identical, including the shift_sessions column rename."""
    if not requires_database(engine):
        return

    manifest = a_manifest()
    with transaction(engine) as conn:
        save_manifest(conn, manifest)
        loaded = load_manifest(conn, manifest.manifest_id)

    assert loaded.train_teachers == manifest.train_teachers
    assert loaded.val_teachers == manifest.val_teachers
    assert loaded.test_teachers == manifest.test_teachers
    assert loaded.config_sha256 == CONFIG_HASH
    assert loaded.shift_eval_sessions == ["S9"], (
        "the contract field is shift_eval_sessions and the column is shift_sessions; the "
        "mapping lives in the store and nowhere else")


def test_the_heldout_college_foreign_key_is_real(engine) -> None:
    """A manifest naming a college that does not exist is refused by the database.

    S1 is defined by that college. A manifest pointing at nothing would make institutional
    shift unmeasurable in a way that only surfaces in Phase 6.
    """
    if not requires_database(engine):
        return

    college_id = new_ulid()
    with transaction(engine) as conn:
        conn.execute(colleges.insert().values(
            college_id=college_id, code=f"SPLIT-{college_id[-8:]}", created_at=func.now()))
        save_manifest(conn, a_manifest(heldout_college=college_id))

    with pytest.raises(Exception, match=r"foreign key|violates"), \
            transaction(engine) as conn:
        save_manifest(conn, a_manifest(heldout_college=new_ulid()))


def test_the_database_refuses_a_contaminated_split(engine) -> None:
    """R2 against PostgreSQL, bypassing every Python check.

    The INSERT is built by hand precisely because `SplitManifest` would never produce it. This
    is the layer that holds when an agent writes its own SQL, which is the failure mode the
    CHECK constraint exists for.
    """
    if not requires_database(engine):
        return

    for label, train, val, test in (
        ("train and val", ["T1", "T2"], ["T2"], ["T3"]),
        ("train and test", ["T1"], ["T2"], ["T1"]),
        ("val and test", ["T1"], ["T2"], ["T2"]),
    ):
        with pytest.raises(Exception, match="teachers_disjoint") as refused, \
                transaction(engine) as conn:
            conn.execute(split_manifests.insert().values(
                manifest_id=new_ulid(), config_sha256=CONFIG_HASH,
                train_teachers=train, val_teachers=val, test_teachers=test,
                shift_sessions=[], created_at=func.now()))
        assert "teachers_disjoint" in str(refused.value), label


def test_the_database_refuses_an_empty_partition(engine) -> None:
    """A train or test partition nobody is in fails here rather than in a training run."""
    if not requires_database(engine):
        return

    with pytest.raises(Exception, match="partitions_are_populated"), \
            transaction(engine) as conn:
        conn.execute(split_manifests.insert().values(
            manifest_id=new_ulid(), config_sha256=CONFIG_HASH,
            train_teachers=[], val_teachers=["T1"], test_teachers=["T2"],
            shift_sessions=[], created_at=func.now()))


def test_a_frozen_manifest_cannot_be_edited(engine) -> None:
    """D73. Append-only, so a result stays traceable to the partition it was measured on."""
    if not requires_database(engine):
        return

    manifest = a_manifest()
    with transaction(engine) as conn:
        save_manifest(conn, manifest)

    for statement in (
        f"UPDATE split_manifests SET config_sha256 = '{'d' * 64}' "
        f"WHERE manifest_id = '{manifest.manifest_id}'",
        f"DELETE FROM split_manifests WHERE manifest_id = '{manifest.manifest_id}'",
    ):
        with pytest.raises(
            Exception, match=r"append-only|forbid|not permitted|denied"
        ), transaction(engine) as conn:
            conn.execute(text(statement))


def test_a_missing_manifest_is_a_named_failure_not_a_none(engine) -> None:
    """A model version pointing at a manifest that is not there is unusable, and saying so is
    the difference between an error and a silently unevaluated result."""
    if not requires_database(engine):
        return

    with transaction(engine) as conn, \
            pytest.raises(ManifestNotFound, match="no split manifest"):
        load_manifest(conn, new_ulid())


def test_manifests_for_one_config_come_back_newest_first(engine) -> None:
    """More than one manifest per config is normal: the seed lives in the config, so two runs
    of the generator at different seeds share a hash. A result must therefore name its
    manifest_id, which this makes visible."""
    if not requires_database(engine):
        return

    config_hash = "e" * 64
    older = a_manifest(config_sha256=config_hash,
                       created_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = a_manifest(config_sha256=config_hash,
                       created_at=datetime(2026, 6, 1, tzinfo=UTC),
                       train_teachers=["T7", "T8"], val_teachers=["T9"],
                       test_teachers=["T10"])

    with transaction(engine) as conn:
        save_manifest(conn, older)
        save_manifest(conn, newer)
        found = manifests_for_config(conn, config_hash)

    assert [m.manifest_id for m in found] == [newer.manifest_id, older.manifest_id]
