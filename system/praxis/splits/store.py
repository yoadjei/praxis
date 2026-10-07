# -*- coding: utf-8 -*-
"""Persistence for split manifests. Append-only, because a split is frozen before modelling.

One module, one mapping. `SplitManifest.shift_eval_sessions` is stored in a column SCHEMA.md
names `shift_sessions`, and that rename exists in exactly these two functions so that nothing
else has to know about it.

No update and no delete. D73: a manifest that can be edited after results were measured against
it makes every one of those results unverifiable, and migration 0006 enforces that with the same
triggers D64 used for annotations. A corrected split is a new manifest with a new id, and the
model version that was trained on the old one still points at the old one.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.engine import Connection

from praxis.contracts.manifest import SplitManifest
from praxis.db.schema import split_manifests


class ManifestNotFound(LookupError):
    """No manifest with that id. A model version pointing at one is unusable, not degraded."""


def save_manifest(conn: Connection, manifest: SplitManifest) -> str:
    """Write a manifest and return its id.

    The R2 check runs three times over by the time a row lands: `SplitManifest` validates on
    construction, `plan_splits` puts its output through the Phase 6 leakage gate, and the
    `teachers_disjoint` CHECK constraint refuses the INSERT. That is deliberate. The first two
    can be bypassed by a caller that builds the row itself; the third cannot.

    Arguments:
        conn: an active database connection
        manifest: the manifest to freeze

    Returns:
        the manifest_id

    Raises:
        IntegrityError if the same manifest_id is written twice, or if the partitions overlap.
        Neither is recoverable here: a duplicate id means the caller reused one, and an overlap
        means R2 was violated upstream of this call.
    """
    conn.execute(
        split_manifests.insert(),
        {
            "manifest_id": manifest.manifest_id,
            "config_sha256": manifest.config_sha256,
            "train_teachers": list(manifest.train_teachers),
            "val_teachers": list(manifest.val_teachers),
            "test_teachers": list(manifest.test_teachers),
            "shift_axis": manifest.shift_axis,
            "heldout_college": manifest.heldout_college,
            "heldout_school": manifest.heldout_school,
            "shift_sessions": list(manifest.shift_eval_sessions),
            "classroom_teachers_trained": manifest.classroom_teachers_trained,
            "created_at": manifest.created_at or datetime.now(UTC),
        },
    )
    return manifest.manifest_id


def load_manifest(conn: Connection, manifest_id: str) -> SplitManifest:
    """Read a manifest back, reconstructed through the contract.

    Through `SplitManifest` rather than as a row dict, so that a manifest which somehow became
    contaminated in storage fails on the way out as well as on the way in. The validator is
    cheap and the alternative is trusting a table because it was trusted once.

    Raises:
        ManifestNotFound: if no manifest carries that id.
        ValueError: from SplitManifest, if the stored partitions overlap.
    """
    row = conn.execute(
        select(split_manifests).where(split_manifests.c.manifest_id == manifest_id),
    ).first()
    if row is None:
        raise ManifestNotFound(
            f"no split manifest {manifest_id!r}. A model version referencing it cannot be "
            f"evaluated, because the partition its numbers were measured on is unknown.")

    return SplitManifest(
        manifest_id=row.manifest_id,
        created_at=row.created_at,
        config_sha256=row.config_sha256,
        train_teachers=list(row.train_teachers),
        val_teachers=list(row.val_teachers),
        test_teachers=list(row.test_teachers),
        shift_axis=row.shift_axis,
        heldout_college=row.heldout_college,
        heldout_school=row.heldout_school,
        shift_eval_sessions=list(row.shift_sessions),
        classroom_teachers_trained=row.classroom_teachers_trained,
    )


def manifests_for_config(conn: Connection, config_sha256: str) -> tuple[SplitManifest, ...]:
    """Every manifest generated from one config, newest first.

    More than one is normal and is not a fault: the seed can change without the config hash
    changing, because the seed is in the config. What it does mean is that a result must name
    its manifest_id and not merely its config hash, which is why `model_versions` references
    the manifest rather than recomputing the split.
    """
    rows = conn.execute(
        select(split_manifests)
        .where(split_manifests.c.config_sha256 == config_sha256)
        .order_by(split_manifests.c.created_at.desc()),
    ).all()
    return tuple(load_manifest(conn, row.manifest_id) for row in rows)
