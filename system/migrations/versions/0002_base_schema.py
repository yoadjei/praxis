# -*- coding: utf-8 -*-
"""institutions, consent, media and sessions

Revision ID: 0002_base_schema
Revises: 0001_append_only_audit
Create Date: 2026-09-14

Everything Phase 1 needs to accept a video, and nothing else. `users`, `detections` and the rest
of SCHEMA.md arrive with the phases that use them; a table created early is a table whose shape
is guessed rather than known. That also means **the foreign keys 0001 deferred stay deferred**:
they point at `detections` and `users`, neither of which this migration creates, so D39 is
untouched by it.

Two constraints here are load-bearing and neither is obvious from the column list.

`sessions.quality_verdict` is NOT NULL with a CHECK. A session whose gate has not run has no
verdict, and the contract represents that as None, so this column is what stops an ungated
session reaching the table. D45 and D49.

`uq_sessions_media_teacher_date` is how a duplicate upload is refused. It is a constraint rather
than a `SELECT` in the service because check-then-insert is a race, and the whole point of the
rule is that two simultaneous clicks produce one session. D51.

Unlike 0001, `downgrade` here is real and is tested. 0001 refuses because reversing it would
make the audit trail mutable, which is an R5 argument; none of these tables is append-only and
none of them inherits that exemption. SCHEMA.md section 12 rule 2.
"""
from alembic import op

revision = "0002_base_schema"
down_revision = "0001_append_only_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE colleges (
            college_id      CHAR(26) PRIMARY KEY,
            code            TEXT NOT NULL UNIQUE,
            region          TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    # Pseudonymous by construction: no name, ever. The mapping to a real person is held by the
    # institution outside this database, so a dump of it stays safe to analyse.
    op.execute("""
        CREATE TABLE teachers (
            teacher_id      CHAR(26) PRIMARY KEY,
            college_id      CHAR(26) NOT NULL REFERENCES colleges(college_id),
            cohort          TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("CREATE INDEX idx_teachers_college ON teachers(college_id);")

    op.execute("""
        CREATE TABLE consent_records (
            consent_id      CHAR(26) PRIMARY KEY,
            subject_type    TEXT NOT NULL
                            CHECK (subject_type IN ('teacher','guardian_class','evaluator')),
            teacher_id      CHAR(26) REFERENCES teachers(teacher_id),
            college_id      CHAR(26) NOT NULL REFERENCES colleges(college_id),

            -- Act 843 s.27(1) requires purpose and recipients to be stated.
            purpose         TEXT NOT NULL,
            recipients      TEXT NOT NULL,
            scope           TEXT NOT NULL
                            CHECK (scope IN ('microteaching','classroom','both')),

            granted_on      DATE NOT NULL,
            expires_on      DATE,
            withdrawn_on    DATE,
            document_ref    TEXT NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    # A view rather than a helper function, so that "is this consent live" has exactly one
    # definition and it is the database's. A second copy in Python would drift.
    op.execute("""
        CREATE OR REPLACE VIEW active_consents AS
        SELECT * FROM consent_records
        WHERE withdrawn_on IS NULL
          AND (expires_on IS NULL OR expires_on >= CURRENT_DATE);
    """)

    op.execute("""
        CREATE TABLE media_objects (
            media_sha256    CHAR(64) PRIMARY KEY,
            relative_path   TEXT NOT NULL,
            bytes           BIGINT NOT NULL,
            duration_s      DOUBLE PRECISION NOT NULL,
            width           INTEGER NOT NULL,
            height          INTEGER NOT NULL,
            fps             DOUBLE PRECISION NOT NULL,
            has_audio       BOOLEAN NOT NULL,
            is_blurred      BOOLEAN NOT NULL DEFAULT false,
            reachable       BOOLEAN NOT NULL DEFAULT true,
            last_checked_at TIMESTAMPTZ,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    op.execute("""
        CREATE TABLE sessions (
            session_id      CHAR(26) PRIMARY KEY,
            teacher_id      CHAR(26) NOT NULL REFERENCES teachers(teacher_id),
            college_id      CHAR(26) NOT NULL REFERENCES colleges(college_id),
            consent_id      CHAR(26) NOT NULL REFERENCES consent_records(consent_id),
            media_sha256    CHAR(64) NOT NULL REFERENCES media_objects(media_sha256),

            domain          TEXT NOT NULL CHECK (domain IN ('microteaching','classroom')),
            subject         TEXT,
            grade_level     TEXT,
            recorded_on     DATE NOT NULL,

            -- Domain-shift covariates. Measured at capture, never imputed.
            camera_distance_m        DOUBLE PRECISION,
            room_area_m2             DOUBLE PRECISION,
            pupil_count              INTEGER,
            ambient_noise_dba        DOUBLE PRECISION,
            teacher_movement_range_m DOUBLE PRECISION,

            quality_verdict TEXT NOT NULL CHECK (quality_verdict IN ('pass','warn','fail')),
            quality_detail  JSONB NOT NULL DEFAULT '{}'::jsonb,

            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

            CONSTRAINT uq_sessions_media_teacher_date
                UNIQUE (media_sha256, teacher_id, recorded_on)
        );
    """)
    op.execute("CREATE INDEX idx_sessions_teacher ON sessions(teacher_id);")
    op.execute("CREATE INDEX idx_sessions_domain  ON sessions(domain);")
    op.execute("CREATE INDEX idx_sessions_college ON sessions(college_id);")


def downgrade() -> None:
    # SCHEMA.md section 12 rule 2: every migration has a tested downgrade, and an untested one
    # is not a downgrade. 0001 is the one exception and it earns it from R5 - an audit trail a
    # migration can un-make was never append-only. Nothing here is append-only, so nothing here
    # gets to borrow that argument. `test_0002_downgrade_is_real` runs it.
    #
    # Reverse dependency order: sessions points at all four of the others.
    op.execute("DROP TABLE IF EXISTS sessions;")
    op.execute("DROP TABLE IF EXISTS media_objects;")
    op.execute("DROP VIEW  IF EXISTS active_consents;")
    op.execute("DROP TABLE IF EXISTS consent_records;")
    op.execute("DROP TABLE IF EXISTS teachers;")
    op.execute("DROP TABLE IF EXISTS colleges;")
