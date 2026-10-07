"""Annotation tables for the calibration and review workflow.

Revision ID: 0005_annotation
Revises: 0004_preprocess_artifacts
Create Date: 2026-09-17

The three tables here implement the two-round calibration workflow and record human
adjudication of model detections. Both `annotation_assignments` and `annotations` are
immutable by design, not by accident. See D64 for the reasoning.

`annotation_assignments` names the clips each rater is assigned to label in each round. A
rater receives exactly one assignment per clip per behaviour per round, and the UNIQUE
constraint enforces that. The assignment holds the clip's geometry to avoid a second join.

`annotations` records each label. It is **append-only**, protected by the forbid_mutation
trigger that 0001 installed. A rater cannot re-label the same clip+behaviour within a
round; the UNIQUE constraint raises 409 (Conflict). Corrections across calibration rounds
happen through new assignments in the subsequent round. This respects R5 while supporting
the two-round workflow.

`codebook_revisions` tracks when the label vocabulary changed, and every annotation
records which version it was made under. A revision cannot be overwritten, only
superseded, which preserves the history of how the codebook evolved during calibration.

Two foreign keys are deferred to the Phase 9 base-schema migration for the same reason
0001 deferred them: annotations.assignment_id references annotation_assignments before
that table is created by another process. The constraint is added when the referenced table
exists. D39.
"""
from alembic import op

revision = "0005_annotation"
down_revision = "0004_preprocess_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Clip assignment: one row per clip per behaviour per rater per round.
    op.execute("""
        CREATE TABLE annotation_assignments (
            assignment_id       CHAR(26) PRIMARY KEY,
            rater_id            CHAR(26) NOT NULL,
            session_id          CHAR(26) NOT NULL,
            clip_index          INTEGER NOT NULL,
            clip_start_s        DOUBLE PRECISION NOT NULL,
            clip_end_s          DOUBLE PRECISION NOT NULL,
            behaviour           TEXT NOT NULL,
            round_name          TEXT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (rater_id, session_id, clip_index, behaviour),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
    """)
    op.execute("CREATE INDEX idx_assignments_rater ON annotation_assignments(rater_id);")
    op.execute(
        "CREATE INDEX idx_assignments_round ON annotation_assignments(round_name);")

    # Human labels. Append-only like audit_log and adjudications.
    op.execute("""
        CREATE TABLE annotations (
            annotation_id       CHAR(26) PRIMARY KEY,
            assignment_id       CHAR(26),
            clip_id             TEXT NOT NULL,
            rater_id            CHAR(26) NOT NULL,
            behaviour           TEXT NOT NULL,
            codebook_version    TEXT NOT NULL,
            labels              JSONB NOT NULL,
            is_nonscorable      BOOLEAN NOT NULL DEFAULT false,
            note                TEXT,
            rater_confidence    TEXT NOT NULL DEFAULT 'certain',
            session_college_id  CHAR(26),
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (rater_id, clip_id, behaviour)
        );
    """)
    op.execute("CREATE INDEX idx_annotations_clip ON annotations(clip_id);")
    op.execute("CREATE INDEX idx_annotations_rater ON annotations(rater_id);")
    op.execute("CREATE INDEX idx_annotations_behaviour ON annotations(behaviour);")
    op.execute(
        "CREATE INDEX idx_annotations_version ON annotations(codebook_version);")

    # Codebook versions. The version is the primary key. Every annotation records which
    # version it was made under, so a revision can change the borderline rulings without
    # silently reinterpreting earlier labels. Supersedes is nullable because the first
    # version has no predecessor.
    op.execute("""
        CREATE TABLE codebook_revisions (
            version             TEXT PRIMARY KEY,
            document_sha256     CHAR(64) NOT NULL,
            adopted_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            supersedes          TEXT REFERENCES codebook_revisions(version),
            rationale           TEXT NOT NULL,
            UNIQUE (document_sha256)
        );
    """)
    op.execute("CREATE INDEX idx_codebook_supersedes ON codebook_revisions(supersedes);")

    # Protect annotations from mutation. The same trigger forbids UPDATE, DELETE, TRUNCATE.
    op.execute("""
        CREATE TRIGGER annotations_no_update
            BEFORE UPDATE OR DELETE ON annotations
            FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
    """)
    op.execute("""
        CREATE TRIGGER annotations_no_truncate
            BEFORE TRUNCATE ON annotations
            FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation();
    """)

    # Codebook is also append-only: versions are never changed, only superseded.
    op.execute("""
        CREATE TRIGGER codebook_revisions_no_update
            BEFORE UPDATE OR DELETE ON codebook_revisions
            FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
    """)
    op.execute("""
        CREATE TRIGGER codebook_revisions_no_truncate
            BEFORE TRUNCATE ON codebook_revisions
            FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation();
    """)

    # Assignments are immutable once issued. A rater cannot un-accept an assignment or have it
    # reassigned.
    op.execute("""
        CREATE TRIGGER annotation_assignments_no_update
            BEFORE UPDATE OR DELETE ON annotation_assignments
            FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
    """)
    op.execute("""
        CREATE TRIGGER annotation_assignments_no_truncate
            BEFORE TRUNCATE ON annotation_assignments
            FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation();
    """)

    op.execute("REVOKE UPDATE, DELETE, TRUNCATE ON annotations, codebook_revisions, "
               "annotation_assignments FROM praxis_app;")


def downgrade() -> None:
    raise RuntimeError(
        "0005_annotation cannot be reversed. Annotations, codebook revisions, and "
        "assignments are append-only. Dropping the triggers would make them mutable, "
        "and an audit record a migration can unmake was never append-only. If the schema "
        "must change, write a new forward migration and record why in docs/DECISIONS.md.")
