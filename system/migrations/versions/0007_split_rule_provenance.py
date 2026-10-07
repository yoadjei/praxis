"""Record which split rule built a manifest.

Revision ID: 0007_split_rule_provenance
Revises: 0006_split_manifests
Create Date: 2026-10-01

D83. `splits.classroom_teachers_may_train` lets teachers who contribute classroom sessions into
the training and validation partitions. It exists because a corpus that is almost entirely
classroom footage has no microteaching-only teachers to train on, and the strict rule of D73
then refuses to produce any split at all.

The weaker rule narrows what the S0-to-S2 drop measures: degradation across teacher-disjoint
partitions where some training teachers were also recorded in a classroom, rather than
degradation in a model that never saw one. Those are different claims and a stored manifest had
no way to say which it supported. This column is that record.

It is on the manifest and not in a config file because the config is a setting and can be
changed after the fact, while the manifest is the artefact a result is traced to. R2 is
unaffected: teachers remain disjoint across partitions and the CHECK constraints added by 0006
still enforce it, here as everywhere.

`server_default false` so that every manifest written before this column existed reads back as
having been built under the strict rule, which is what it was.
"""
import sqlalchemy as sa
from alembic import op

revision = "0007_split_rule_provenance"
down_revision = "0006_split_manifests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "split_manifests",
        sa.Column("classroom_teachers_trained", sa.Boolean(),
                  nullable=False, server_default=sa.text("false")),
    )
    op.execute("""
        COMMENT ON COLUMN split_manifests.classroom_teachers_trained IS
        'Whether teachers contributing classroom (S2) sessions were eligible for the train and '
        'validation partitions. False is the strict rule of D73. True narrows what the '
        'S0-to-S2 drop measured on this manifest means. See D83.'
    """)


def downgrade() -> None:
    # 0006 made this table append-only and said why: a split that can change after results were
    # measured against it makes those results unverifiable. Dropping this column would not
    # change a row, but it would erase the only record of which rule a stored manifest was built
    # under, which has the same effect on anything already traced to it.
    raise NotImplementedError(
        "split_manifests records the rule each manifest was built under and cannot give it up. "
        "Restore from a backup taken before 0007 if the schema must go back.")
