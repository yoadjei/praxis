"""Let a declined teacher proposal be a row, and give a reviewer something to disagree with.

Revision ID: 0009_teacher_proposal_evidence
Revises: 0008_blurred_media_pointer
Create Date: 2026-10-02

SCHEMA.md section 4 declares `teacher_tracks.track_id` NOT NULL, and
`praxis/preprocess/store.persist` therefore skips the INSERT entirely when the heuristic clears
no track above its floor. The comment there says so plainly: no proposal means no row, because
inventing a track id so the table has something in it would be worse. That reasoning was right
about the invention and wrong about the row.

**The effect was that an abstention left no trace and could not be acted on.** D48 says an
unavailable check abstains and that the abstention is its own verdict, but a verdict whose
evidence is discarded is not one: of the seven sessions in the corpus, three cleared no track,
and their computed score, their ranked candidates and the heuristic's stated reason were all
dropped at this boundary. `praxis/preprocess/store.confirm_teacher` then refused those three
sessions outright, because it looks for a row before it will write a confirmation - so the one
case its own docstring calls "the entire point of the step", a reviewer naming a track the
heuristic did not propose, was the case the code would not accept. R1 says only the teacher is
classified and that somebody has to have said who that is. Three sessions had no way for
anybody to say.

So `track_id` becomes nullable and means what its absence always meant: nobody has identified
a teacher in this session yet. A NULL track plus a confirmation would be a person endorsing
nothing, which `nothing_is_confirmed_without_a_track` refuses at the database rather than in
a comment.

`reason` and `candidates` are the evidence. Both already exist on `TeacherProposal` - its
`reason` string and its ranked `signals` tuple - and both were computed and then thrown away
because `as_row()` did not emit them. A reviewer confirming a teacher is answering a question
the heuristic already ranked; this is that ranking, with the three measured fractions per
candidate and a note of whether the front-zone term was measurable at all. They are
instrumentation for the confirmation step, not product features, and are not to be removed as
unused. D90.

`proposed_by` acquires the CHECK it never had. It has only ever been written as the literal
`'heuristic'` from one line of Python, and `confirm_teacher` leaves it alone, so 'human' was
representable and never written. It is written now: a reviewer who identifies a track in a
session the heuristic declined is that track's origin.

**Nothing here loosens R1.** `preprocess.teacher_id.require_human_confirmation` stays in the
config's `locked` list, `confirmed_teacher_track` still returns a track only when `confirmed_by`
is set, and every downstream phase still filters on that. What changes is that the abstention is
now readable and the confirmation is now reachable. R2 and R5 are unaffected: this table is not
append-only and is not in `APPEND_ONLY`.

`tests/integration/test_db_schema.py::test_columns_and_nullability_agree` holds the two new
columns and the relaxed one against `information_schema`, and
`tests/integration/test_preprocess_store.py` holds the pair of CHECKs against real inserts.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0009_teacher_proposal_evidence"
down_revision = "0008_blurred_media_pointer"
branch_labels = None
depends_on = None

# What the four rows written before this migration can honestly say about themselves. Their
# reason and their candidate ranking were computed and discarded, so the default records that
# rather than implying a ranking nobody kept. The same shape a live proposal writes, with the
# list empty and the source saying why.
UNRECORDED = '{"source": "unrecorded", "zones_available": null, "ranked": [], "truncated": 0}'
NO_REASON = ("this proposal predates the column and its reason was not recorded; re-emit it "
             "from the stored pose artifact to recover the ranking")


def upgrade() -> None:
    # A session nobody has identified a teacher in. The absence was always the meaning; until
    # now it was expressed by the row not existing, which put it out of reach of every reader.
    op.alter_column("teacher_tracks", "track_id", nullable=True)

    # No backfill, and no DML in a migration: the server defaults are what the existing four
    # rows truthfully say, and every writer after this supplies both columns. The defaults stay
    # on the columns rather than being dropped afterwards, so a row inserted by a reader who
    # has not read `as_row()` still records that it recorded nothing.
    op.add_column(
        "teacher_tracks",
        sa.Column("reason", sa.Text(), nullable=False, server_default=NO_REASON),
    )
    op.add_column(
        "teacher_tracks",
        sa.Column("candidates", postgresql.JSONB(), nullable=False,
                  server_default=sa.text(f"'{UNRECORDED}'::jsonb")),
    )

    # A person cannot confirm an absence. Without this, relaxing track_id would make
    # `confirmed_by IS NOT NULL AND track_id IS NULL` storable, and `confirmed_teacher_track`
    # would return NULL for a session a reviewer had signed off - R1 failing quietly, which is
    # the one way it must not fail.
    op.create_check_constraint(
        "nothing_is_confirmed_without_a_track",
        "teacher_tracks",
        "confirmed_by IS NULL OR track_id IS NOT NULL",
    )

    # Two origins, both now real. The vocabulary lives in praxis/vocabulary.py and
    # `TestVocabulariesMatchTheDatabase` reads this constraint's literals back out of
    # pg_constraint to hold the two together, because a vocabulary declared twice drifts (L20).
    op.create_check_constraint(
        "proposed_by_is_a_known_origin",
        "teacher_tracks",
        "proposed_by IN ('heuristic','human')",
    )

    op.execute("""
        COMMENT ON COLUMN teacher_tracks.reason IS
        'The heuristic''s own explanation, verbatim, including its caveat when the front-zone '
        'term was not measurable. Shown to the reviewer so they can disagree with a stated '
        'argument rather than with a number. See D90.'
    """)
    op.execute("""
        COMMENT ON COLUMN teacher_tracks.candidates IS
        'The ranked candidate tracks and the three fractions each was scored on, as '
        '{source, zones_available, ranked, truncated}. Instrumentation for the confirmation '
        'step; not a product feature, and not to be removed as unused. See D90.'
    """)


def downgrade() -> None:
    raise NotImplementedError(
        "re-imposing NOT NULL on track_id requires deleting every row that records an "
        "abstention, and dropping reason and candidates erases the only account a reviewer had "
        "of why the heuristic declined. Both destroy the evidence this migration exists to "
        "keep. Write a new forward migration and record why in docs/DECISIONS.md.")
