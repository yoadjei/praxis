"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

SCHEMA.md §12: one logical change per migration, and an untested downgrade is not a
downgrade. A migration that drops or renames a column touched by an existing model version
needs an entry in docs/DECISIONS.md saying what happens to prior detections.

Never grant UPDATE, DELETE or TRUNCATE on audit_log or adjudications. CI greps for it.
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
