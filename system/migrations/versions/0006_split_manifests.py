"""Teacher-disjoint split manifests. R2, enforced by the database.

Revision ID: 0006_split_manifests
Revises: 0005_annotation
Create Date: 2026-09-26

SCHEMA.md section 6. The `&&` array-overlap operator makes R2 a constraint PostgreSQL refuses
to violate, so a contaminated split cannot be written even by a caller that skipped every
application-level check. `SplitManifest` validates the same rule in Python and
`praxis/splits/manifest.py` runs the Phase 6 leakage gate over its own output; this is the third
layer, and it is the only one an agent cannot bypass by importing something else.

**Append-only, which SCHEMA.md does not state and D73 adds.** `SplitManifest` describes itself
as "frozen before any modelling", and that was procedure rather than structure: nothing stopped
a manifest being edited after results had been measured against it. A split that can change
after the fact makes every number traced to it unverifiable, which is the same argument D64 made
for annotations. The triggers 0001 installed are reused.

`shift_sessions` is the column name SCHEMA.md defines; the contract field is
`shift_eval_sessions`, and `praxis/splits/store.py` is the single place the two are mapped.
"""
from alembic import op

revision = "0006_split_manifests"
down_revision = "0005_annotation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE split_manifests (
            manifest_id     CHAR(26) PRIMARY KEY,
            config_sha256   CHAR(64) NOT NULL,
            train_teachers  TEXT[] NOT NULL,
            val_teachers    TEXT[] NOT NULL,
            test_teachers   TEXT[] NOT NULL,
            heldout_college CHAR(26) REFERENCES colleges(college_id),
            shift_sessions  TEXT[] NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

            -- R2. Not a convention, not a code path: a condition the database checks on
            -- every write. `&&` is array overlap.
            CONSTRAINT teachers_disjoint CHECK (
                NOT (train_teachers && val_teachers) AND
                NOT (train_teachers && test_teachers) AND
                NOT (val_teachers && test_teachers)
            ),

            -- A partition nobody is in is not a partition. An empty train or test array
            -- would otherwise be written happily and fail much later, in a training run.
            CONSTRAINT partitions_are_populated CHECK (
                cardinality(train_teachers) > 0 AND
                cardinality(test_teachers) > 0
            )
        );
    """)

    op.execute("CREATE INDEX split_manifests_config ON split_manifests (config_sha256);")

    # D73. Frozen before any modelling, made structural rather than procedural.
    op.execute("""
        CREATE TRIGGER split_manifests_no_update
            BEFORE UPDATE OR DELETE ON split_manifests
            FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
    """)
    op.execute("""
        CREATE TRIGGER split_manifests_no_truncate
            BEFORE TRUNCATE ON split_manifests
            FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation();
    """)

    op.execute("REVOKE UPDATE, DELETE, TRUNCATE ON split_manifests FROM praxis_app;")


def downgrade() -> None:
    raise RuntimeError(
        "0006_split_manifests cannot be reversed. A split manifest is the partition a result "
        "was measured on, and one that can be unmade after the fact makes every number traced "
        "to it unverifiable. If the schema must change, write a new forward migration and "
        "record why in docs/DECISIONS.md.")
