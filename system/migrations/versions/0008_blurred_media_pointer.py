"""Record where the blurred copy is, once the original has been destroyed.

Revision ID: 0008_blurred_media_pointer
Revises: 0007_split_rule_provenance
Create Date: 2026-10-01

Phase 3 blurs every face and then deletes the unblurred original, which D18 makes a privacy
requirement rather than an option - `preprocess.blur.delete_original_after_blur` is in the
config's `locked` list and the comment beside it says NEVER set false.
`praxis/preprocess/store.persist` refuses to record a session as preprocessed while the
original is still on disk.

What nothing did was say where the surviving file went. `media_objects.relative_path` is
derived from the original's SHA-256, so after preprocessing it named a file that no longer
existed, and `is_blurred` was written false at ingest and never updated by anything but a test
fixture. The result was a database whose only pointer to a session's footage was dangling, on
a corpus where the blurred copy is the only copy that may legally be kept.

`relative_path` is deliberately left alone. The contract that `media_sha256` is the hash of
the bytes at `relative_path` is what lets a reader verify provenance, and quietly repointing
it at a file with a different hash would break that while every column still looked
consistent. The blurred path is a second, separate fact and gets its own column.

`is_blurred` keeps its meaning and finally acquires a writer: true means the original is
gone and `blurred_relative_path` is where the footage is. `retention.blurred_media_years`
depends on being able to tell the two apart.
"""
import sqlalchemy as sa
from alembic import op

revision = "0008_blurred_media_pointer"
down_revision = "0007_split_rule_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "media_objects",
        sa.Column("blurred_relative_path", sa.Text(), nullable=True),
    )
    # Every row that exists now was written by ingest, which sets is_blurred false, so
    # there is nothing to backfill. The constraint states the pairing rather than trusting
    # callers to keep it: a blurred row without a path is the dangling pointer this
    # migration exists to remove.
    op.create_check_constraint(
        "media_objects_blurred_has_a_path",
        "media_objects",
        "(is_blurred = false AND blurred_relative_path IS NULL) "
        "OR (is_blurred = true AND blurred_relative_path IS NOT NULL)",
    )


def downgrade() -> None:
    raise NotImplementedError(
        "dropping blurred_relative_path would leave rows whose only surviving file is unnamed, "
        "on a corpus where the blurred copy is the only copy that may be kept. Restore from a "
        "backup instead.")
