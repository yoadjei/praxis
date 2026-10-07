# -*- coding: utf-8 -*-
"""The tables, as Core objects rather than ORM models.

Deliberately Core. The ORM exists to track changes to mapped objects and flush them, and two of
these tables refuse `UPDATE` and `DELETE` at the database level. Mapping them would put an
identity map and an automatic flush in front of a trigger whose whole job is to raise, and the
failure would arrive at session close, far from whatever touched the object. Core has no
identity map to accidentally flush, so the hazard is absent rather than avoided. See D52.

These definitions duplicate the DDL in `migrations/`, which is exactly the drift that has bitten
this project twice. `tests/integration/test_db_schema.py` compares them against the live
`information_schema`, so the copy is checked against the database rather than against another
document.
"""
from __future__ import annotations

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

metadata = MetaData()

ULID = CHAR(26)
SHA256 = CHAR(64)

colleges = Table(
    "colleges", metadata,
    Column("college_id", ULID, primary_key=True),
    Column("code", Text, nullable=False, unique=True),
    Column("region", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

teachers = Table(
    "teachers", metadata,
    Column("teacher_id", ULID, primary_key=True),
    Column("college_id", ULID, nullable=False),
    Column("cohort", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# A basic school hosting practicum. No `college_id`: one school hosts students from several
# Colleges of Education, so the two keys are crossed on the session rather than nested. Nesting
# would make a held-out college hold out its schools too, which is S2 collapsing into S1. 0011,
# D98.
schools = Table(
    "schools", metadata,
    Column("school_id", ULID, primary_key=True),
    Column("code", Text, nullable=False, unique=True),
    Column("district", Text),
    Column("region", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

def _consent_columns() -> list[Column]:
    """Fresh column objects each call. A Column belongs to one Table, so the view cannot borrow
    the table's; `Column.copy()` would do it and has been deprecated since 1.4."""
    return [
        Column("consent_id", ULID, primary_key=True),
        Column("subject_type", Text, nullable=False),
        Column("teacher_id", ULID),
        Column("college_id", ULID, nullable=False),
        Column("purpose", Text, nullable=False),
        Column("recipients", Text, nullable=False),
        Column("scope", Text, nullable=False),
        Column("granted_on", Date, nullable=False),
        Column("expires_on", Date),
        Column("withdrawn_on", Date),
        Column("document_ref", Text, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
    ]


consent_records = Table("consent_records", metadata, *_consent_columns())

# A view, not a table. The predicate for "this consent is live" exists once, in SQL, and ingest
# asks it rather than reimplementing it in Python where the two could drift apart.
active_consents = Table("active_consents", metadata, *_consent_columns())

media_objects = Table(
    "media_objects", metadata,
    Column("media_sha256", SHA256, primary_key=True),
    Column("relative_path", Text, nullable=False),
    Column("bytes", BigInteger, nullable=False),
    Column("duration_s", Double, nullable=False),
    Column("width", Integer, nullable=False),
    Column("height", Integer, nullable=False),
    Column("fps", Double, nullable=False),
    Column("has_audio", Boolean, nullable=False),
    # `relative_path` and `media_sha256` describe the file as ingested, and stay describing it
    # after Phase 3 destroys it: the hash is the provenance claim and repointing the path at a
    # file with different bytes would break it. `is_blurred` true means the original is gone and
    # `blurred_relative_path` is where the footage now is. 0008.
    Column("is_blurred", Boolean, nullable=False),
    Column("blurred_relative_path", Text),
    Column("reachable", Boolean, nullable=False),
    Column("last_checked_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

sessions = Table(
    "sessions", metadata,
    Column("session_id", ULID, primary_key=True),
    Column("teacher_id", ULID, nullable=False),
    Column("college_id", ULID, nullable=False),
    Column("consent_id", ULID, nullable=False),
    Column("media_sha256", SHA256, nullable=False),
    Column("domain", Text, nullable=False),
    Column("subject", Text),
    Column("grade_level", Text),
    Column("recorded_on", Date, nullable=False),
    Column("camera_distance_m", Double),
    Column("room_area_m2", Double),
    Column("pupil_count", Integer),
    Column("ambient_noise_dba", Double),
    Column("teacher_movement_range_m", Double),
    Column("quality_verdict", Text, nullable=False),
    Column("quality_detail", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("setup_id", ULID),
    # A researcher's decision not to annotate this session, with its author and its ground.
    # Distinct from `quality_verdict`, which records what the gate measured about the file:
    # a session can pass the gate and still be unusable for a stated reason. 0010, D96.
    Column("excluded_at", DateTime(timezone=True)),
    Column("excluded_by", ULID),
    Column("exclusion_reason", Text),
    # Where a practicum session was recorded. NULL for microteaching, which happens on a college
    # campus and has no basic school; `a_classroom_session_names_its_school` carries the rule
    # that does apply. S1 holds this out. 0011, D98.
    Column("school_id", ULID, ForeignKey("schools.school_id")),
)

# Append-only. Nothing in this package ever issues an UPDATE or DELETE against it, and the
# database would refuse if it did. R5.
audit_log = Table(
    "audit_log", metadata,
    Column("audit_id", BigInteger, primary_key=True, autoincrement=True),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("actor_user_id", ULID),
    Column("actor_role", Text),
    Column("event_type", Text, nullable=False),
    Column("entity_type", Text, nullable=False),
    Column("entity_id", Text, nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("prev_hash", SHA256),
    Column("row_hash", SHA256, nullable=False),
)


# ---------------------------------------------------------------------------
# Phase 3 preprocessing artifacts. SCHEMA.md section 4.
# ---------------------------------------------------------------------------

camera_setups = Table(
    "camera_setups", metadata,
    Column("setup_id", ULID, primary_key=True),
    Column("college_id", ULID, nullable=False),
    Column("label", Text, nullable=False),
    Column("zone_board", JSONB, nullable=False),
    Column("zone_front", JSONB, nullable=False),
    Column("zone_middle", JSONB, nullable=False),
    Column("zone_back", JSONB, nullable=False),
    Column("learner_region", JSONB, nullable=False),
    Column("defined_by", ULID, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

pose_artifacts = Table(
    "pose_artifacts", metadata,
    Column("session_id", ULID, primary_key=True),
    Column("relative_path", Text, nullable=False),
    Column("frame_count", Integer, nullable=False),
    Column("sampled_fps", Double, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("sha256", SHA256, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# `confirmed_by` is nullable on purpose. The heuristic writes a row with it null; a person
# fills it in. Downstream phases filter on it, so an unreviewed session cannot be trained on.
# `track_id` is nullable for the same kind of reason and became so in 0009: a session where the
# heuristic cleared nothing is a row saying nobody has identified a teacher yet, which is what a
# reviewer needs to see. `nothing_is_confirmed_without_a_track` keeps the two nullables from
# combining into a confirmation of nothing.
teacher_tracks = Table(
    "teacher_tracks", metadata,
    Column("session_id", ULID, primary_key=True),
    Column("track_id", Integer),
    Column("proposed_by", Text, nullable=False),
    Column("heuristic_score", Double),
    Column("confirmed_by", ULID),
    Column("confirmed_at", DateTime(timezone=True)),
    # The heuristic's stated argument and the ranking it argued over. 0009, D90.
    Column("reason", Text, nullable=False),
    Column("candidates", JSONB, nullable=False),
)

# No identifier column, and no per-learner table anywhere to add one to. R1 is enforced by the
# absence of a place to violate it, and the invariant test greps this package for the names
# such a table would have, so it must not appear even in a comment saying it does not exist.
learner_aggregates = Table(
    "learner_aggregates", metadata,
    Column("session_id", ULID, primary_key=True),
    Column("t_start_s", Double, primary_key=True),
    Column("hands_raised", Integer),
    Column("gross_motion", Double),
    Column("person_count", Integer),
)

# ---------------------------------------------------------------------------
# Phase 2 annotation and calibration. Two-round workflow, append-only.
# ---------------------------------------------------------------------------

annotation_assignments = Table(
    "annotation_assignments", metadata,
    Column("assignment_id", ULID, primary_key=True),
    Column("rater_id", ULID, nullable=False),
    Column("session_id", ULID, nullable=False),
    Column("clip_index", Integer, nullable=False),
    Column("clip_start_s", Double, nullable=False),
    Column("clip_end_s", Double, nullable=False),
    Column("behaviour", Text, nullable=False),
    Column("round_name", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# Append-only. Annotations are never updated or deleted; corrections are new rows. R5.
# Every annotation records the codebook version it was made under, so a revision cannot
# silently reinterpret a label made before it existed. This is a Phase 2 acceptance test.
annotations = Table(
    "annotations", metadata,
    Column("annotation_id", ULID, primary_key=True),
    Column("assignment_id", ULID),
    Column("clip_id", Text, nullable=False),
    Column("rater_id", ULID, nullable=False),
    Column("behaviour", Text, nullable=False),
    Column("codebook_version", Text, nullable=False),
    Column("labels", JSONB, nullable=False),
    Column("is_nonscorable", Boolean, nullable=False, server_default="false"),
    Column("note", Text),
    Column("rater_confidence", Text, nullable=False, server_default="'certain'"),
    Column("session_college_id", ULID),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# Codebook versions are append-only. A version is never changed, only superseded.
# The document_sha256 is the fingerprint of the prose that raters read.
codebook_revisions = Table(
    "codebook_revisions", metadata,
    Column("version", Text, primary_key=True),
    Column("document_sha256", SHA256, nullable=False),
    Column("adopted_at", DateTime(timezone=True), nullable=False),
    Column("supersedes", Text),
    Column("rationale", Text, nullable=False),
)

split_manifests = Table(
    "split_manifests", metadata,
    Column("manifest_id", ULID, primary_key=True),
    Column("config_sha256", SHA256, nullable=False),
    # TEXT[] rather than a join table. A manifest is read whole or not at all - no query asks
    # which partition one teacher is in without wanting the rest - and the array form is what
    # lets R2 be a CHECK constraint using `&&` instead of a trigger nobody can read.
    Column("train_teachers", ARRAY(Text), nullable=False),
    Column("val_teachers", ARRAY(Text), nullable=False),
    Column("test_teachers", ARRAY(Text), nullable=False),
    # S2: the college withheld entirely, teachers and schools alike. The name and the meaning
    # are unchanged; only the rung it labels moved, because the corpus is practicum throughout
    # and a held-out domain is no longer available as a level. D98.
    Column("heldout_college", ULID, ForeignKey("colleges.college_id")),
    # S1: the basic school withheld entirely, its college still in training. Crossed with
    # heldout_college, not nested under it. NULL means S1 is not measured. 0011, D98.
    Column("heldout_school", ULID, ForeignKey("schools.school_id")),
    # Which ladder built this manifest: 'domain' (S2 is all classroom footage) or 'site' (S1 is
    # the held-out school, S2 the held-out college). Without it the same shift_sessions array
    # would carry two different level definitions and a stored result could not be read. The
    # default is the older axis, so manifests written before the column read back as themselves.
    Column("shift_axis", Text, nullable=False, server_default="'domain'"),
    # `shift_sessions` here, `shift_eval_sessions` on the contract. SCHEMA.md names the column;
    # praxis/splits/store.py is the one place the two are mapped.
    Column("shift_sessions", ARRAY(Text), nullable=False, server_default="'{}'"),
    # Which split rule built this manifest. Not a setting - a property of the artefact, because
    # an S0-to-S2 drop measured under the weakened rule is a narrower claim than one measured
    # under D73's, and a reader of the stored manifest has no other way to tell them apart. The
    # default is the strict rule, so manifests written before this column existed read back as
    # what they were. D83.
    Column("classroom_teachers_trained", Boolean, nullable=False, server_default="false"),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

APPEND_ONLY = frozenset({"audit_log", "adjudications", "annotations", "codebook_revisions",
                         "annotation_assignments", "split_manifests"})
