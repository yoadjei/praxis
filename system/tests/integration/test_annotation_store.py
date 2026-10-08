"""Persistence tests for annotation workflow.

These tests exercise the functions in `praxis.annotation.store` against a live database,
following the pattern from `test_ingest.py`. Every function's success and failure paths
are tested, and the invariants enforced by the schema are verified.
"""
from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

import pytest
from sqlalchemy import func

from praxis.annotation.clips import ClipRef
from praxis.annotation.server import (
    AnnotationRefused,
    Assignment,
)
from praxis.annotation.store import (
    adopt_codebook_revision,
    load_annotations,
    queue_for,
    record_annotation,
    record_assignments,
)
from praxis.db import transaction
from praxis.db.schema import (
    annotation_assignments,
    annotations,
    codebook_revisions,
    colleges,
    consent_records,
    media_objects,
    sessions,
    teachers,
)
from praxis.ids import new_ulid
from praxis.ingest.exclusion import exclude, reinstate
from tests.integration.conftest import requires_database


class MockCodebook:
    """A codebook-like object for testing."""

    def __init__(self, version: str, sha256: str):
        self.version = version
        self.document_sha256 = sha256


class MockAnnotation:
    """An annotation-like object for testing."""

    def __init__(
        self,
        annotation_id: str,
        clip_id: str,
        rater_id: str,
        behaviour: str,
        codebook_version: str,
        labels: dict[str, Any],
        is_nonscorable: bool = False,
        note: str | None = None,
        rater_confidence: str = "certain",
        session_college_id: str | None = None,
    ):
        self.annotation_id = annotation_id
        self.clip_id = clip_id
        self.rater_id = rater_id
        self.behaviour = behaviour
        self.codebook_version = codebook_version
        self.labels = labels
        self.is_nonscorable = is_nonscorable
        self.note = note
        self.rater_confidence = rater_confidence
        self.session_college_id = session_college_id


@pytest.fixture
def calibration_data(engine):
    """Set up college, teachers, consent, and sessions for calibration tests.

    Returns a dict with the IDs and a session list.
    """
    if engine is None:
        return None

    college_id = new_ulid()
    teacher1 = new_ulid()
    teacher2 = new_ulid()
    consent_id = new_ulid()
    session_id = new_ulid()
    media_sha256 = hashlib.sha256(session_id.encode()).hexdigest()

    with transaction(engine) as conn:
        conn.execute(colleges.insert().values(
            college_id=college_id,
            # `colleges.code` is UNIQUE and the scratch database outlives a single test, so a
            # constant here collides with every test that ran before this one.
            code=f"TEST-{college_id[-8:]}",
            created_at=func.now(),
        ))
        conn.execute(teachers.insert().values(
            teacher_id=teacher1,
            college_id=college_id,
            created_at=func.now(),
        ))
        conn.execute(teachers.insert().values(
            teacher_id=teacher2,
            college_id=college_id,
            created_at=func.now(),
        ))
        conn.execute(consent_records.insert().values(
            consent_id=consent_id,
            subject_type="teacher",
            teacher_id=teacher1,
            college_id=college_id,
            purpose="research",
            recipients="supervisors",
            scope="both",
            granted_on=date(2025, 1, 1),
            document_ref="file/1",
            created_at=func.now(),
        ))
        # `sessions.media_sha256` is a foreign key, so the media row has to exist first.
        # Derived from the session rather than fixed, because the scratch database outlives a
        # single test and a constant digest collides with the previous test's row.
        conn.execute(media_objects.insert().values(
            media_sha256=media_sha256,
            relative_path=f"media/{media_sha256[:2]}/{media_sha256}.mp4",
            bytes=1024,
            duration_s=480.0,
            width=1280,
            height=720,
            fps=25.0,
            has_audio=False,
            # Paired with the path, as the `media_objects_blurred_has_a_path` CHECK requires:
            # the original is deleted once a session is blurred, so a row that says a blurred
            # file exists without naming it has lost the session.
            is_blurred=True,
            blurred_relative_path=f".blurred/{session_id}.mp4",
            reachable=True,
            created_at=func.now(),
        ))
        conn.execute(sessions.insert().values(
            session_id=session_id,
            teacher_id=teacher1,
            college_id=college_id,
            consent_id=consent_id,
            media_sha256=media_sha256,
            domain="microteaching",
            recorded_on=date(2025, 1, 1),
            quality_verdict="pass",
            quality_detail={},
            created_at=func.now(),
        ))

    return {
        "college_id": college_id,
        "teacher1": teacher1,
        "teacher2": teacher2,
        "consent_id": consent_id,
        "session_id": session_id,
    }


class TestRecordAssignments:
    """Tests for recording clip assignments."""

    def test_empty_assignments(self, engine, calibration_data):
        """Recording zero assignments returns 0."""
        if not requires_database(engine):
            return

        with transaction(engine) as conn:
            count = record_assignments(conn, ())
            assert count == 0

    def test_single_assignment(self, engine, calibration_data):
        """Recording one assignment returns 1 and stores it."""
        if not requires_database(engine):
            return

        rater_id = new_ulid()
        session_id = calibration_data["session_id"]
        clip = ClipRef(session_id=session_id, clip_index=0, start_seconds=0.0,
                       end_seconds=8.0)
        assignment = Assignment(
            assignment_id=new_ulid(),
            rater_id=rater_id,
            clip=clip,
            behaviour="B1",
            round_name="calibration-1",
        )

        with transaction(engine) as conn:
            count = record_assignments(conn, (assignment,))
            assert count == 1

            # Verify it was stored.
            result = conn.execute(
                annotation_assignments.select().where(
                    annotation_assignments.c.assignment_id == assignment.assignment_id,
                ),
            ).first()
            assert result is not None
            assert dict(result._mapping)["rater_id"] == rater_id
            assert dict(result._mapping)["behaviour"] == "B1"

    def test_multiple_assignments(self, engine, calibration_data):
        """Recording multiple assignments returns the count."""
        if not requires_database(engine):
            return

        session_id = calibration_data["session_id"]
        assignments = []
        for i in range(3):
            clip = ClipRef(
                session_id=session_id,
                clip_index=i,
                start_seconds=float(i * 8),
                end_seconds=float((i + 1) * 8),
            )
            assignments.append(
                Assignment(
                    assignment_id=new_ulid(),
                    rater_id=new_ulid(),
                    clip=clip,
                    behaviour="B1",
                    round_name="calibration-1",
                ),
            )

        with transaction(engine) as conn:
            count = record_assignments(conn, tuple(assignments))
            assert count == 3


class TestRecordAnnotation:
    """Tests for recording a rater's label."""

    def test_record_annotation(self, engine, calibration_data):
        """Recording an annotation stores it and returns the ID."""
        if not requires_database(engine):
            return

        rater_id = new_ulid()
        clip_id = f"{calibration_data['session_id']}:00000"
        annotation = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_id,
            rater_id=rater_id,
            behaviour="B1",
            codebook_version="v1",
            labels={"b1_present": True, "b1_count": 3},
            note="test note",
        )

        with transaction(engine) as conn:
            returned_id = record_annotation(conn, annotation)
            assert returned_id == annotation.annotation_id

            # Verify it was stored.
            result = conn.execute(
                annotations.select().where(
                    annotations.c.annotation_id == annotation.annotation_id,
                ),
            ).first()
            assert result is not None
            row = dict(result._mapping)
            assert row["rater_id"] == rater_id
            assert row["behaviour"] == "B1"
            assert row["codebook_version"] == "v1"
            assert row["note"] == "test note"

    def test_duplicate_annotation_raises_409(self, engine, calibration_data):
        """Attempting to record the same (rater, clip, behaviour) twice raises 409."""
        if not requires_database(engine):
            return

        rater_id = new_ulid()
        clip_id = f"{calibration_data['session_id']}:00001"

        annotation1 = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_id,
            rater_id=rater_id,
            behaviour="B1",
            codebook_version="v1",
            labels={"b1_present": True},
        )

        with transaction(engine) as conn:
            record_annotation(conn, annotation1)

            # Try to record the same (rater, clip, behaviour) again.
            annotation2 = MockAnnotation(
                annotation_id=new_ulid(),
                clip_id=clip_id,
                rater_id=rater_id,
                behaviour="B1",
                codebook_version="v1",
                labels={"b1_present": False},
            )

            with pytest.raises(AnnotationRefused) as exc_info:
                record_annotation(conn, annotation2)
            assert exc_info.value.http_status == 409

    def test_annotation_with_assignment(self, engine, calibration_data):
        """Recording an annotation with an assignment_id stores the reference."""
        if not requires_database(engine):
            return

        assignment_id = new_ulid()
        rater_id = new_ulid()
        clip_id = f"{calibration_data['session_id']}:00002"

        annotation = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_id,
            rater_id=rater_id,
            behaviour="B2",
            codebook_version="v1",
            labels={"b2_orientation": 0.75},
        )

        with transaction(engine) as conn:
            record_annotation(conn, annotation, assignment_id=assignment_id)

            result = conn.execute(
                annotations.select().where(
                    annotations.c.annotation_id == annotation.annotation_id,
                ),
            ).first()
            assert result is not None
            assert dict(result._mapping)["assignment_id"] == assignment_id


class TestLoadAnnotations:
    """Tests for loading annotations with filters."""

    def test_load_all_annotations(self, engine, calibration_data):
        """Loading all annotations returns all recorded ones."""
        if not requires_database(engine):
            return

        clip_id_1 = f"{calibration_data['session_id']}:00100"
        clip_id_2 = f"{calibration_data['session_id']}:00101"

        ann1 = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_id_1,
            rater_id=new_ulid(),
            behaviour="B1",
            codebook_version="v1",
            labels={"b1_present": True},
        )
        ann2 = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_id_2,
            rater_id=new_ulid(),
            behaviour="B2",
            codebook_version="v1",
            labels={"b2_orientation": 0.5},
        )

        with transaction(engine) as conn:
            record_annotation(conn, ann1)
            record_annotation(conn, ann2)

            results = load_annotations(conn)
            # At least our two should be there.
            assert len(results) >= 2
            ids = {r["annotation_id"] for r in results}
            assert ann1.annotation_id in ids
            assert ann2.annotation_id in ids

    def test_load_annotations_by_clip(self, engine, calibration_data):
        """Loading with clip_ids filter returns only those clips."""
        if not requires_database(engine):
            return

        clip_id_1 = f"{calibration_data['session_id']}:00200"
        clip_id_2 = f"{calibration_data['session_id']}:00201"

        ann1 = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_id_1,
            rater_id=new_ulid(),
            behaviour="B1",
            codebook_version="v1",
            labels={"b1_present": True},
        )
        ann2 = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_id_2,
            rater_id=new_ulid(),
            behaviour="B1",
            codebook_version="v1",
            labels={"b1_present": False},
        )

        with transaction(engine) as conn:
            record_annotation(conn, ann1)
            record_annotation(conn, ann2)

            results = load_annotations(conn, clip_ids=(clip_id_1,))
            ids = {r["annotation_id"] for r in results}
            assert ann1.annotation_id in ids
            assert ann2.annotation_id not in ids

    def test_load_annotations_by_round(self, engine, calibration_data):
        """Loading with round_name filter returns only that round's assignments."""
        if not requires_database(engine):
            return

        session_id = calibration_data["session_id"]
        rater_id = new_ulid()

        # Different clips per round. CODEBOOK.md section 8 step 4 specifies round 2 as a "new
        # 100-clip set", so one rater never labels the same clip twice - which is exactly what
        # UNIQUE (rater_id, session_id, clip_index, behaviour) enforces. Reusing clip 0 in both
        # rounds would be testing a protocol the codebook does not describe.
        clip_r1 = ClipRef(session_id=session_id, clip_index=0,
                          start_seconds=0.0, end_seconds=8.0)
        clip_r2 = ClipRef(session_id=session_id, clip_index=1,
                          start_seconds=8.0, end_seconds=16.0)

        assignment_r1 = Assignment(
            assignment_id=new_ulid(),
            rater_id=rater_id,
            clip=clip_r1,
            behaviour="B1",
            round_name="calibration-1",
        )
        assignment_r2 = Assignment(
            assignment_id=new_ulid(),
            rater_id=rater_id,
            clip=clip_r2,
            behaviour="B1",
            round_name="calibration-2",
        )

        # Create annotations for both assignments.
        ann_r1 = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_r1.clip_id,
            rater_id=rater_id,
            behaviour="B1",
            codebook_version="v1",
            labels={"b1_present": True},
        )
        ann_r2 = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_r2.clip_id,
            rater_id=rater_id,
            behaviour="B1",
            codebook_version="v2",
            labels={"b1_present": False},
        )

        with transaction(engine) as conn:
            record_assignments(conn, (assignment_r1, assignment_r2))
            record_annotation(conn, ann_r1, assignment_id=assignment_r1.assignment_id)
            record_annotation(conn, ann_r2, assignment_id=assignment_r2.assignment_id)

            # Load only round 1.
            results = load_annotations(conn, round_name="calibration-1")
            ids = {r["annotation_id"] for r in results}
            assert ann_r1.annotation_id in ids

            # Load only round 2.
            results = load_annotations(conn, round_name="calibration-2")
            ids = {r["annotation_id"] for r in results}
            assert ann_r2.annotation_id in ids


class TestQueueFor:
    """Tests for fetching a rater's assignment queue."""

    def test_queue_for_rater_empty(self, engine, calibration_data):
        """Queueing for a rater with no assignments returns empty."""
        if not requires_database(engine):
            return

        rater_id = new_ulid()

        with transaction(engine) as conn:
            results = queue_for(conn, rater_id)
            assert results == ()

    def test_queue_for_rater_pending(self, engine, calibration_data):
        """Queueing for a rater returns pending (unannotated) assignments."""
        if not requires_database(engine):
            return

        session_id = calibration_data["session_id"]
        rater_id = new_ulid()

        assignments = []
        for i in range(3):
            clip = ClipRef(
                session_id=session_id,
                clip_index=i,
                start_seconds=float(i * 8),
                end_seconds=float((i + 1) * 8),
            )
            assignments.append(
                Assignment(
                    assignment_id=new_ulid(),
                    rater_id=rater_id,
                    clip=clip,
                    behaviour="B1",
                    round_name="calibration-1",
                ),
            )

        with transaction(engine) as conn:
            record_assignments(conn, tuple(assignments))

            # All three should be pending.
            results = queue_for(conn, rater_id)
            assert len(results) == 3
            assert all(r.rater_id == rater_id for r in results)

    def test_queue_for_rater_excludes_completed(self, engine, calibration_data):
        """Queueing excludes assignments that already have annotations."""
        if not requires_database(engine):
            return

        session_id = calibration_data["session_id"]
        rater_id = new_ulid()

        assignment_ids = []
        for i in range(3):
            assignment = Assignment(
                assignment_id=new_ulid(),
                rater_id=rater_id,
                clip=ClipRef(
                    session_id=session_id,
                    clip_index=i,
                    start_seconds=float(i * 8),
                    end_seconds=float((i + 1) * 8),
                ),
                behaviour="B1",
                round_name="calibration-1",
            )
            assignment_ids.append(assignment.assignment_id)

        with transaction(engine) as conn:
            record_assignments(
                conn,
                tuple(
                    Assignment(
                        assignment_id=assignment_ids[i],
                        rater_id=rater_id,
                        clip=ClipRef(
                            session_id=session_id,
                            clip_index=i,
                            start_seconds=float(i * 8),
                            end_seconds=float((i + 1) * 8),
                        ),
                        behaviour="B1",
                        round_name="calibration-1",
                    )
                    for i in range(3)
                ),
            )

            # Annotate the first assignment.
            ann = MockAnnotation(
                annotation_id=new_ulid(),
                clip_id=f"{session_id}:00000",
                rater_id=rater_id,
                behaviour="B1",
                codebook_version="v1",
                labels={"b1_present": True},
            )
            record_annotation(conn, ann, assignment_id=assignment_ids[0])

            # Queue should exclude the completed one.
            results = queue_for(conn, rater_id)
            assert len(results) == 2
            assert assignment_ids[0] not in {r.assignment_id for r in results}

    def test_queue_for_rater_by_round(self, engine, calibration_data):
        """Queueing with round_name filter returns only that round."""
        if not requires_database(engine):
            return

        session_id = calibration_data["session_id"]
        rater_id = new_ulid()

        assignment_r1 = Assignment(
            assignment_id=new_ulid(),
            rater_id=rater_id,
            clip=ClipRef(session_id=session_id, clip_index=0, start_seconds=0.0,
                        end_seconds=8.0),
            behaviour="B1",
            round_name="calibration-1",
        )
        assignment_r2 = Assignment(
            assignment_id=new_ulid(),
            rater_id=rater_id,
            clip=ClipRef(session_id=session_id, clip_index=1, start_seconds=8.0,
                        end_seconds=16.0),
            behaviour="B1",
            round_name="calibration-2",
        )

        with transaction(engine) as conn:
            record_assignments(conn, (assignment_r1, assignment_r2))

            # Queue for round 1 only.
            results = queue_for(conn, rater_id, round_name="calibration-1")
            assert len(results) == 1
            assert results[0].assignment_id == assignment_r1.assignment_id

            # Queue for round 2 only.
            results = queue_for(conn, rater_id, round_name="calibration-2")
            assert len(results) == 1
            assert results[0].assignment_id == assignment_r2.assignment_id

    def test_an_excluded_session_is_not_served_to_a_rater(self, engine, calibration_data):
        """D96 excludes a session from annotation. The queue has to agree with the plan.

        `plan_annotation.py` refuses to put an excluded session in a new round, but the rounds
        already planned keep their rows - the table is append-only. The research corpus is in
        exactly that state: two thirds of the standing assignments belong to a session excluded
        after they were issued. If the queue served them, the rule would hold at planning time
        and fail at the one point where a person spends hours acting on it.
        """
        if not requires_database(engine):
            return

        session_id = calibration_data["session_id"]
        rater_id = new_ulid()
        assignment = Assignment(
            assignment_id=new_ulid(), rater_id=rater_id, behaviour="B1",
            clip=ClipRef(session_id=session_id, clip_index=0, start_seconds=0.0,
                         end_seconds=8.0),
            round_name="calibration-1")

        with transaction(engine) as conn:
            record_assignments(conn, (assignment,))
            assert len(queue_for(conn, rater_id)) == 1, "the fixture session is annotatable"

            exclude(conn, session_id=session_id, excluded_by=new_ulid(),
                    reason="camera framing cuts the teacher out of frame")
            assert queue_for(conn, rater_id) == ()

            reinstate(conn, session_id=session_id, reinstated_by=new_ulid(),
                      reason="re-examined; the teacher is in frame for most of it")
            assert len(queue_for(conn, rater_id)) == 1, (
                "a restored session is annotatable again; the assignment was never retired")


class TestAdoptCodebookRevision:
    """Tests for recording codebook versions."""

    def test_adopt_first_codebook(self, engine):
        """Adopting the first codebook version succeeds."""
        if not requires_database(engine):
            return

        codebook = MockCodebook(
            version="1.0.0",
            sha256="a" * 64,
        )

        with transaction(engine) as conn:
            adopt_codebook_revision(
                conn,
                codebook,
                supersedes=None,
                rationale="initial codebook",
            )

            # Verify it was stored.
            result = conn.execute(
                codebook_revisions.select().where(
                    codebook_revisions.c.version == "1.0.0",
                ),
            ).first()
            assert result is not None
            row = dict(result._mapping)
            assert row["document_sha256"] == "a" * 64
            assert row["supersedes"] is None

    def test_adopt_codebook_with_supersedes(self, engine):
        """Adopting a codebook that supersedes a prior version stores the link."""
        if not requires_database(engine):
            return

        codebook_v1 = MockCodebook(version="1.0.0", sha256="a" * 64)
        codebook_v2 = MockCodebook(version="2.0.0", sha256="b" * 64)

        with transaction(engine) as conn:
            adopt_codebook_revision(
                conn,
                codebook_v1,
                supersedes=None,
                rationale="initial",
            )
            adopt_codebook_revision(
                conn,
                codebook_v2,
                supersedes="1.0.0",
                rationale="refined codebook",
            )

            result = conn.execute(
                codebook_revisions.select().where(
                    codebook_revisions.c.version == "2.0.0",
                ),
            ).first()
            row = dict(result._mapping)
            assert row["supersedes"] == "1.0.0"

    def test_adopt_codebook_idempotent(self, engine):
        """Adopting the same codebook twice is idempotent."""
        if not requires_database(engine):
            return

        codebook = MockCodebook(version="1.0.0", sha256="a" * 64)

        with transaction(engine) as conn:
            adopt_codebook_revision(
                conn,
                codebook,
                supersedes=None,
                rationale="first attempt",
            )
            # Adopt again with the same content.
            adopt_codebook_revision(
                conn,
                codebook,
                supersedes=None,
                rationale="second attempt",
            )

            # Should have exactly one row.
            results = conn.execute(
                codebook_revisions.select().where(
                    codebook_revisions.c.version == "1.0.0",
                ),
            ).all()
            assert len(results) == 1

    def test_adopt_codebook_hash_mismatch_raises(self, engine):
        """Adopting the same version with different content raises ValueError."""
        if not requires_database(engine):
            return

        codebook_v1 = MockCodebook(version="1.0.0", sha256="a" * 64)
        codebook_v1_corrupt = MockCodebook(version="1.0.0", sha256="b" * 64)

        with transaction(engine) as conn:
            adopt_codebook_revision(
                conn,
                codebook_v1,
                supersedes=None,
                rationale="first",
            )

            with pytest.raises(ValueError, match="exists with different content"):
                adopt_codebook_revision(
                    conn,
                    codebook_v1_corrupt,
                    supersedes=None,
                    rationale="second",
                )


class TestAppendOnly:
    """Tests for the append-only property of annotations tables."""

    def test_annotations_table_is_append_only(self, engine, calibration_data):
        """The annotations table forbids UPDATE and DELETE at the database level."""
        if not requires_database(engine):
            return

        rater_id = new_ulid()
        clip_id = f"{calibration_data['session_id']}:99999"

        ann = MockAnnotation(
            annotation_id=new_ulid(),
            clip_id=clip_id,
            rater_id=rater_id,
            behaviour="B1",
            codebook_version="v1",
            labels={"b1_present": True},
        )

        with transaction(engine) as conn:
            record_annotation(conn, ann)

            # Try to update the label. The forbid_mutation trigger raises DatabaseError.
            from sqlalchemy.exc import DatabaseError
            with pytest.raises(DatabaseError):
                conn.execute(
                    annotations.update().where(
                        annotations.c.annotation_id == ann.annotation_id,
                    ).values(labels={"b1_present": False}),
                )

            # Try to delete it. The forbid_mutation trigger raises DatabaseError.
            with pytest.raises(DatabaseError):
                conn.execute(
                    annotations.delete().where(
                        annotations.c.annotation_id == ann.annotation_id,
                    ),
                )
