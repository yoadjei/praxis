"""Give the corpus a basic school, so the shift ladder can hold one out.

Revision ID: 0011_schools
Revises: 0010_session_exclusion
Create Date: 2026-10-07

The shift ladder put the shift in the *domain*: S1 was a held-out college's microteaching and S2
was basic-school practicum, with the rule that classroom footage is never trained on. The
revised scope makes the corpus practicum throughout, so classroom footage is now all of the
data. The rule would then refuse every manifest, and the fix nobody should reach for is to relax
it.

So the shift moves from the domain to the **site**. S1 holds out a basic school whose college
still appears in training, which isolates room, camera placement, pupil cohort and acoustics. S2
holds out a whole college, which shifts the teacher population and the sites together. That
ladder needs a school to hold out, and the schema has no school in it. SCHEMA.md section 3 and
section 6. D98.

**A school is not a child of a college.** One basic school hosts teaching-practice students from
several Colleges of Education, and the reverse holds too, so `schools` carries no `college_id`
and the two keys are crossed on the session. Nesting the school under the college would make a
held-out college silently hold out its schools as well, which is S2 collapsing into S1 - the one
confusion this ladder exists to prevent.

`district` and `region` are on the table because they are what a thesis uses to say how far
apart two sites are, and they are the covariates an attribution model would reach for if a drop
between S0 and S1 needs explaining. Nullable: a school can be recorded before its district is
confirmed, and a column that forces a guess gets a guess.

No backfill, and nothing to backfill. `sessions.school_id` is NULL for every existing row, which
is the truthful state: all nine are microteaching, recorded on a college campus, and a
microteaching session has no basic school. That is also why the column is nullable rather than
NOT NULL - the domain decides whether a school exists, and
`a_classroom_session_names_its_school` says so directly instead of making every caller remember
it. All nine existing rows satisfy it today
(`SELECT domain, count(*) FROM sessions GROUP BY domain` returns `microteaching: 9`), so the
constraint is added against real data rather than hoped over.

`split_manifests.heldout_school` is nullable too, and NULL means S1 is not being measured - the
same thing `heldout_college` NULL has always meant for its level. `heldout_college` keeps its
name and its meaning, "the college withheld entirely"; what changed is which rung it labels.
Renaming it
would break three stored manifests for a cosmetic gain, and `split_manifests` is append-only
(D73) so they cannot be rewritten.

**`shift_axis` is the column that makes the two ladders readable apart.** S2 means "all
classroom footage" on the old domain axis and "the held-out college" on the new site axis.
Without a stored axis the same `shift_sessions` array would carry two different level
definitions and nothing could say which, so a result traced back to a manifest could not be
read. It defaults to `'domain'`, which is what the three existing manifests are. That is a
`server_default` and not a backfill: every existing row reads back as the design it was written
under, and no UPDATE is issued.

`tests/integration/test_db_schema.py::test_columns_and_nullability_agree` holds the three
additions against `information_schema`, and
`tests/unit/test_shift.py::TestTheLadderForAPracticumCorpus` holds the leakage rules the ladder
rests on.
"""
import sqlalchemy as sa
from alembic import op

revision = "0011_schools"
down_revision = "0010_session_exclusion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "schools",
        sa.Column("school_id", sa.CHAR(26), primary_key=True),
        sa.Column("code", sa.Text(), nullable=False, unique=True),
        sa.Column("district", sa.Text()),
        sa.Column("region", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.execute("""
        COMMENT ON TABLE schools IS
        'A basic school hosting teaching practicum. Deliberately not a child of colleges: one '
        'school hosts students from several Colleges of Education, and nesting it would make a '
        'held-out college hold out its schools too, collapsing S2 into S1. See D98.'
    """)

    # Nullable because a microteaching session has no basic school, and every row that exists
    # today is microteaching. The constraint below carries the rule that does apply.
    op.add_column("sessions",
                  sa.Column("school_id", sa.CHAR(26), sa.ForeignKey("schools.school_id")))

    op.create_check_constraint(
        "a_classroom_session_names_its_school",
        "sessions",
        "domain <> 'classroom' OR school_id IS NOT NULL",
    )

    # S1 is "held out this school". NULL means S1 is not being measured, which is what NULL has
    # always meant for heldout_college and its level.
    op.add_column("split_manifests",
                  sa.Column("heldout_school", sa.CHAR(26), sa.ForeignKey("schools.school_id")))

    op.execute("""
        COMMENT ON COLUMN split_manifests.heldout_school IS
        'The basic school withheld entirely, which is S1. NULL means S1 is not measured. '
        'Independent of heldout_college, which is S2: a college and a school are crossed, not '
        'nested. See D98.'
    """)

    # Which ladder built this manifest. S2 means "all classroom footage" on the domain axis and
    # "the held-out college" on the site axis, so a manifest that does not say which axis it
    # used cannot be read: one column would carry two level definitions. The default is the
    # domain axis, so the three manifests written before this column read back as exactly what
    # they were, and append-only (D73) means they could not be rewritten anyway.
    #
    # The default is unquoted. SQLAlchemy renders a plain string server_default as a SQL
    # literal itself, so "'domain'" here reaches the database as the eight-character value
    # `'domain'`, quotes included, and `a_shift_axis_is_domain_or_site` then refuses every
    # insert that relied on the default. 0009 writes its text defaults the same way.
    op.add_column("split_manifests",
                  sa.Column("shift_axis", sa.Text(), nullable=False, server_default="domain"))

    op.create_check_constraint(
        "a_shift_axis_is_domain_or_site", "split_manifests",
        "shift_axis IN ('domain', 'site')",
    )

    # On the site axis the levels are named sites, and a manifest that names neither declares a
    # ladder with nothing on it. On the domain axis a held-out school has no rung to belong to,
    # and storing one would suggest it was measured.
    op.create_check_constraint(
        "a_shift_axis_names_the_sites_it_holds_out", "split_manifests",
        "(shift_axis = 'domain' AND heldout_school IS NULL) "
        "OR (shift_axis = 'site' "
        "AND (heldout_school IS NOT NULL OR heldout_college IS NOT NULL))",
    )


def downgrade() -> None:
    raise NotImplementedError(
        "dropping heldout_school erases which school a stored manifest held out, and "
        "split_manifests is append-only (D73) so the manifest cannot be rewritten to say it "
        "again. An S0-to-S1 result would then be unattributable to a site. Dropping "
        "sessions.school_id loses where the footage was recorded, which no other column holds. "
        "Write a new forward migration and record why in docs/DECISIONS.md.")
