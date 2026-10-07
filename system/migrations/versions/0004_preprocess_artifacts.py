# -*- coding: utf-8 -*-
"""camera setups, pose artifacts, teacher tracks and learner aggregates

Revision ID: 0004_preprocess_artifacts
Revises: 0003_forbid_truncate
Create Date: 2026-09-17

SCHEMA.md section 4. Two constraints here carry rules rather than data.

`teacher_tracks.confirmed_by` is nullable and `must_be_confirmed_before_use` requires
`confirmed_at` alongside it. That is what makes the heuristic a proposal: a row exists as soon
as preprocessing runs, and it is unconfirmed until a person confirms it. Every downstream phase
filters on `confirmed_by IS NOT NULL`, so an unreviewed session cannot be trained on.

`learner_aggregates` has no identifier column and there is deliberately no `learner_tracks`
table. R1 is enforced by the absence of a place to violate it, and
`test_no_learner_tracks_persisted` reads this DDL to check that stays true.

`defined_by` and `confirmed_by` reference `users`, which no migration creates yet. The columns
are declared with the right type and the constraints are added by the migration that creates
that table, exactly as 0001 deferred the `adjudications` keys. D39.
"""
from alembic import op

revision = "0004_preprocess_artifacts"
down_revision = "0003_forbid_truncate"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE camera_setups (
            setup_id        CHAR(26) PRIMARY KEY,
            college_id      CHAR(26) NOT NULL REFERENCES colleges(college_id),
            label           TEXT NOT NULL,
            -- Normalised polygons, (0,0) top-left to (1,1) bottom-right, so one setup
            -- survives a change of recording resolution.
            zone_board      JSONB NOT NULL,
            zone_front      JSONB NOT NULL,
            zone_middle     JSONB NOT NULL,
            zone_back       JSONB NOT NULL,
            learner_region  JSONB NOT NULL,
            defined_by      CHAR(26) NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("CREATE INDEX idx_camera_setups_college ON camera_setups(college_id);")

    op.execute("ALTER TABLE sessions ADD COLUMN setup_id CHAR(26) "
               "REFERENCES camera_setups(setup_id);")

    # Pose is bulk numeric data and stays out of Postgres. The row is a pointer and a hash.
    op.execute("""
        CREATE TABLE pose_artifacts (
            session_id      CHAR(26) PRIMARY KEY REFERENCES sessions(session_id),
            relative_path   TEXT NOT NULL,
            frame_count     INTEGER NOT NULL CHECK (frame_count > 0),
            sampled_fps     DOUBLE PRECISION NOT NULL CHECK (sampled_fps > 0),
            model_version   TEXT NOT NULL,
            sha256          CHAR(64) NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    op.execute("""
        CREATE TABLE teacher_tracks (
            session_id      CHAR(26) PRIMARY KEY REFERENCES sessions(session_id),
            track_id        INTEGER NOT NULL,
            proposed_by     TEXT NOT NULL DEFAULT 'heuristic',
            heuristic_score DOUBLE PRECISION,
            confirmed_by    CHAR(26),
            confirmed_at    TIMESTAMPTZ,
            CONSTRAINT must_be_confirmed_before_use
                CHECK (confirmed_by IS NULL OR confirmed_at IS NOT NULL)
        );
    """)

    # Learner evidence: aggregate only. No per-learner row exists anywhere. R1.
    op.execute("""
        CREATE TABLE learner_aggregates (
            session_id      CHAR(26) NOT NULL REFERENCES sessions(session_id),
            t_start_s       DOUBLE PRECISION NOT NULL,
            hands_raised    INTEGER,
            gross_motion    DOUBLE PRECISION,
            person_count    INTEGER,
            PRIMARY KEY (session_id, t_start_s)
        );
    """)


def downgrade() -> None:
    # Reversible, and tested. None of these is append-only, so none inherits 0001's exemption.
    # SCHEMA.md section 12 rule 2.
    op.execute("DROP TABLE IF EXISTS learner_aggregates;")
    op.execute("DROP TABLE IF EXISTS teacher_tracks;")
    op.execute("DROP TABLE IF EXISTS pose_artifacts;")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS setup_id;")
    op.execute("DROP TABLE IF EXISTS camera_setups;")
