# -*- coding: utf-8 -*-
"""close the TRUNCATE hole in the append-only tables

Revision ID: 0003_forbid_truncate
Revises: 0002_base_schema
Create Date: 2026-09-15

0001 installs `BEFORE UPDATE OR DELETE ... FOR EACH ROW` and its docstring claims the trigger
"stops everything short of a superuser dropping the trigger itself, including a later migration
that grants the privileges back by mistake". That is not true of `TRUNCATE`, which fires no row
triggers at all. Confirmed against the running cluster before this was written: `TRUNCATE
audit_log` succeeded with both of 0001's triggers in place, and emptied the table.

`REVOKE ... TRUNCATE` in 0001 does stop `praxis_app`, and `test_audit_trail_immutable` greps
migrations for a `GRANT` that would give it back. Neither helps against any other role that
holds the privilege, and the trigger was supposed to be the layer that did not depend on who
was asking.

A truncate trigger is `FOR EACH STATEMENT` by necessity: there are no rows to visit. It reuses
`forbid_mutation`, whose message already names `TG_OP` and the table.
"""
from alembic import op

revision = "0003_forbid_truncate"
down_revision = "0002_base_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TRIGGER audit_log_no_truncate
            BEFORE TRUNCATE ON audit_log
            FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation();
    """)
    op.execute("""
        CREATE TRIGGER adjudications_no_truncate
            BEFORE TRUNCATE ON adjudications
            FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation();
    """)


def downgrade() -> None:
    raise RuntimeError(
        "0003_forbid_truncate cannot be reversed, for the same reason 0001 cannot. Dropping "
        "these triggers restores the ability to empty the audit trail in one statement, and a "
        "trail a migration can empty was never append-only.")
