"""Let a session be excluded from annotation on a stated ground, rather than by never finishing.

Revision ID: 0010_session_exclusion
Revises: 0009_teacher_proposal_evidence
Create Date: 2026-10-07

`scripts/plan_annotation.py` refuses to plan a session whose teacher track nobody has
confirmed, which is right: a clip cut on the heuristic's guess can show somebody who is not the
teacher, and the label is then attached to their conduct (R1). The consequence is that a session
a researcher has deliberately decided not to use is indistinguishable from one nobody has got to
yet. Both read as "awaiting teacher confirmation", for ever.

**That gap has a session in it.** `01M3VY6DJA12BF7WNY2E1CPSA4` is 1947.1 seconds - 68 per cent
of the 2864.7 seconds of real footage in the corpus - and its camera moves 23.82 per cent of the
frame diagonal between sampled frames. The tracker fragments the teacher across 1248 track ids
and the best single track covers 14.3 per cent of frames, so no track in it is the teacher in
any useful sense. The decision not to annotate it is a research finding with a number behind it.
Recorded as silence it would read as unfinished work, and the next person to open the dashboard
would try to confirm it.

So an exclusion is a thing the schema can hold: who decided, when, and on what ground. The three
columns move together - `an_exclusion_names_a_person_a_time_and_a_ground` refuses a reason with
no author and an author with no reason - because an exclusion whose ground was not written down
is the same problem one layer along.

**This is not a deletion and not a quality verdict.** The footage stays, the pose artefact
stays, the teacher proposal stays, and `quality_verdict` still says what the gate measured about
the file. Excluding a session says a researcher will not label it, which is a different claim
from the file being bad, and conflating the two would let a judgement about research use
masquerade as a measurement. D96.

No backfill. The columns are nullable and default to NULL, which is the truthful state for every
existing row: nobody has excluded any of them. `scripts/exclude_session.py` writes the first one
and emits `session.excluded` onto the chain, so the decision is in the audit trail and not only
in the row that records its outcome (R5).

`tests/integration/test_db_schema.py::test_columns_and_nullability_agree` holds the three
columns against `information_schema`, and `tests/integration/test_plan_annotation.py` holds the
CHECK and the planner's refusal against real inserts.
"""
import sqlalchemy as sa
from alembic import op

revision = "0010_session_exclusion"
down_revision = "0009_teacher_proposal_evidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable and no default. NULL is what every existing row truthfully says - no exclusion
    # has been recorded - so there is nothing to backfill and nothing to invent.
    op.add_column("sessions", sa.Column("excluded_at", sa.DateTime(timezone=True)))
    op.add_column("sessions", sa.Column("excluded_by", sa.CHAR(26)))
    op.add_column("sessions", sa.Column("exclusion_reason", sa.Text()))

    # All three or none. An exclusion with no author cannot be questioned by anyone later, and
    # one with no ground is the silence this migration exists to end. Enforced here rather than
    # in the writer, because a second writer would be a second place for the rule to be
    # forgotten (L20).
    op.create_check_constraint(
        "an_exclusion_names_a_person_a_time_and_a_ground",
        "sessions",
        "(excluded_at IS NULL AND excluded_by IS NULL AND exclusion_reason IS NULL) "
        "OR (excluded_at IS NOT NULL AND excluded_by IS NOT NULL "
        "AND exclusion_reason IS NOT NULL)",
    )

    # A blank string would satisfy the constraint above and say nothing, which is the failure
    # mode this pair is built to prevent rather than a hypothetical.
    op.create_check_constraint(
        "an_exclusion_ground_is_not_blank",
        "sessions",
        "exclusion_reason IS NULL OR length(btrim(exclusion_reason)) > 0",
    )

    op.execute("""
        COMMENT ON COLUMN sessions.exclusion_reason IS
        'Why a researcher decided not to annotate this session, in their own words. Not a '
        'quality verdict: quality_verdict records what the gate measured about the file, and '
        'this records a decision about research use. See D96.'
    """)


def downgrade() -> None:
    raise NotImplementedError(
        "dropping these columns erases the stated ground on which a session was excluded, "
        "leaving it indistinguishable from one nobody has reviewed - which is exactly the "
        "state this migration exists to end. The audit chain keeps the session.excluded event "
        "either way, so the row and the trail would then disagree. Write a new forward "
        "migration and record why in docs/DECISIONS.md.")
