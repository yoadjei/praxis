# -*- coding: utf-8 -*-
"""append-only audit trail and adjudications

Revision ID: 0001_append_only_audit
Revises:
Create Date: 2026-09-13

Rule R5 lives here, not in application code. Two layers, and they stop different things:

* `REVOKE UPDATE, DELETE, TRUNCATE` stops the ordinary application role, which is what an
  accidental `UPDATE` in a service would come through.
* the `forbid_mutation` trigger stops everything short of a superuser dropping the trigger
  itself, including a later migration that grants the privileges back by mistake.

`tests/test_invariants.py::test_audit_trail_immutable` greps every migration for a `GRANT` of a
mutating privilege on either table, so a migration that undoes this fails CI rather than being
noticed later. That grep is the reason this file names the privileges it revokes explicitly.

**Both append-only tables are created here, and the foreign keys are not.** A trigger can only
be created on a table that exists, so protecting `adjudications` means creating it, and its
`REFERENCES detections(detection_id)` and `REFERENCES users(user_id)` would then require the
whole schema to exist first. The columns are declared with the right types and the constraints
are added by the base-schema migration in Phase 9, which is the migration that creates the
tables they point at. Deferring a foreign key is ordinary; creating a trigger on a table that
does not exist is a migration that cannot run, which is what this file did before the review
caught it.

**Downgrade deliberately refuses.** An audit trail that can be un-made by running a migration
backwards is not append-only, and `alembic downgrade` is exactly the command someone reaches for
when something has gone wrong, which is the moment the trail matters most.
"""
from alembic import op

revision = "0001_append_only_audit"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE audit_log (
            audit_id        BIGSERIAL PRIMARY KEY,
            occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            actor_user_id   CHAR(26),
            actor_role      TEXT,
            event_type      TEXT NOT NULL,
            entity_type     TEXT NOT NULL,
            entity_id       TEXT NOT NULL,
            payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
            prev_hash       CHAR(64),
            row_hash        CHAR(64) NOT NULL
        );
    """)
    op.execute("CREATE INDEX idx_audit_entity ON audit_log(entity_type, entity_id);")
    op.execute("CREATE INDEX idx_audit_time   ON audit_log(occurred_at);")

    op.execute("""
        CREATE TABLE adjudications (
            adjudication_id CHAR(26) PRIMARY KEY,
            detection_id    CHAR(26) NOT NULL,
            reviewer_id     CHAR(26) NOT NULL,
            action          TEXT NOT NULL CHECK (action IN ('confirm','edit','reject')),
            edited_value    JSONB,
            rationale       TEXT,

            -- Reliance-study instrumentation. Not product features; do not remove as unused.
            seconds_on_item DOUBLE PRECISION NOT NULL,
            evidence_replays INTEGER NOT NULL DEFAULT 0,
            revealed_indication BOOLEAN NOT NULL,
            arm             TEXT CHECK (arm IN ('dossier','no_dossier')),

            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

            CONSTRAINT rationale_required_for_change
                CHECK (action = 'confirm' OR (rationale IS NOT NULL
                                              AND length(rationale) >= 10))
        );
    """)
    op.execute("CREATE INDEX idx_adjud_detection ON adjudications(detection_id);")
    op.execute("CREATE INDEX idx_adjud_reviewer  ON adjudications(reviewer_id);")

    op.execute("""
        CREATE OR REPLACE FUNCTION forbid_mutation() RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'append-only table: % not permitted on %', TG_OP, TG_TABLE_NAME;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER audit_log_no_update
            BEFORE UPDATE OR DELETE ON audit_log
            FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
    """)
    op.execute("""
        CREATE TRIGGER adjudications_no_update
            BEFORE UPDATE OR DELETE ON adjudications
            FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
    """)

    op.execute("REVOKE UPDATE, DELETE, TRUNCATE ON audit_log, adjudications FROM praxis_app;")


def downgrade() -> None:
    raise RuntimeError(
        "0001_append_only_audit cannot be reversed. Dropping the trigger and restoring the "
        "privileges would make the audit trail mutable, and a trail that a migration can "
        "un-make was never append-only. If the schema genuinely has to change, write a new "
        "forward migration and record why in docs/DECISIONS.md.")
