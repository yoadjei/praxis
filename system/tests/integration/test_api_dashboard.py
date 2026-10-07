# -*- coding: utf-8 -*-
"""The dashboard API, exercised through HTTP.

The dashboard surfaces corpus state, preprocessing progress, annotation status,
and recent phase runs. Every endpoint works on an empty corpus, returning 200 with
zeros and empty lists rather than a 500. Identity (teacher id, teacher code, media
filename) is never included in any response.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func

from praxis.api.app import create_app
from praxis.db import build_engine, keywords_to_url, transaction
from praxis.db.schema import (
    annotation_assignments,
    annotations,
    colleges,
    consent_records,
    media_objects,
    pose_artifacts,
    schools,
    sessions,
    teachers,
)
from praxis.ids import new_ulid
from praxis.ingest.exclusion import exclude
from praxis.tasks import RecordingQueue
from tests.integration.conftest import lenient, requires_database


def _client_for(dsn: str | None, tmp_path: Path) -> TestClient | None:
    """A client talking to `dsn`. One body, two fixtures: shared and private."""
    if dsn is None:
        return None
    media_root = tmp_path / "media"
    media_root.mkdir(exist_ok=True)
    app = create_app(
        config=lenient(),
        engine=build_engine(keywords_to_url(dsn)),
        media_root=media_root,
        enqueue=RecordingQueue(),
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(scratch_database, tmp_path) -> TestClient | None:
    """A test client against the database the whole run shares."""
    return _client_for(scratch_database, tmp_path)


@pytest.fixture
def private_client(private_database, tmp_path) -> TestClient | None:
    """A test client against an empty database nothing else writes to.

    For the endpoint tests whose subject is a corpus-wide figure. See `private_database` in
    tests/conftest.py for why those cannot use the shared one.
    """
    return _client_for(private_database, tmp_path)


@pytest.fixture
def private_corpus(private_client, private_database) -> dict | None:
    """The dashboard corpus, in a database where it is the only corpus.

    The client comes back in the same dict rather than as a second fixture, so a test cannot
    pair this corpus with the shared client and quietly assert against the wrong database.
    """
    if private_database is None:
        return None
    engine = build_engine(keywords_to_url(private_database))
    return {**_dashboard_corpus(engine), "client": private_client, "engine": engine}


@pytest.fixture
def dashboard_setup(engine) -> dict | None:
    """College, teachers, consent, media, and sessions for dashboard testing."""
    return _dashboard_corpus(engine)


def _dashboard_corpus(engine) -> dict | None:
    """Three sessions - two microteaching passes and one classroom fail - one of them
    preprocessed and another annotated, so the groupings and the joins all have something to
    find. Written once and used against both the shared and the private database."""
    if engine is None:
        return None

    college_id = new_ulid()
    school_id = new_ulid()
    teacher_id = new_ulid()
    consent_id = new_ulid()
    session_id_1 = new_ulid()
    session_id_2 = new_ulid()
    session_id_3 = new_ulid()
    rater_id = new_ulid()

    media_sha256_1 = hashlib.sha256(session_id_1.encode()).hexdigest()
    media_sha256_2 = hashlib.sha256(session_id_2.encode()).hexdigest()
    media_sha256_3 = hashlib.sha256(session_id_3.encode()).hexdigest()

    with transaction(engine) as conn:
        conn.execute(
            colleges.insert().values(
                college_id=college_id,
                code=f"TEST-{college_id[-8:]}",
                created_at=func.now(),
            )
        )
        # The classroom session below cannot be written without a school.
        # `a_classroom_session_names_its_school`, D98.
        conn.execute(
            schools.insert().values(
                school_id=school_id,
                code=f"SCH-{school_id[-8:]}",
                created_at=func.now(),
            )
        )
        conn.execute(
            teachers.insert().values(
                teacher_id=teacher_id,
                college_id=college_id,
                created_at=func.now(),
            )
        )
        conn.execute(
            consent_records.insert().values(
                consent_id=consent_id,
                subject_type="teacher",
                teacher_id=teacher_id,
                college_id=college_id,
                purpose="research",
                recipients="supervisors",
                scope="both",
                granted_on=date(2025, 1, 1),
                document_ref="file/1",
                created_at=func.now(),
            )
        )

        # Create media objects with known durations.
        for media_sha256, duration in [
            (media_sha256_1, 480.0),
            (media_sha256_2, 600.5),
            (media_sha256_3, 120.25),
        ]:
            conn.execute(
                media_objects.insert().values(
                    media_sha256=media_sha256,
                    relative_path=f"media/{media_sha256[:2]}/{media_sha256}.mp4",
                    bytes=1024,
                    duration_s=duration,
                    width=1280,
                    height=720,
                    fps=25.0,
                    has_audio=False,
                    is_blurred=False,
                    blurred_relative_path=None,
                    reachable=True,
                    created_at=func.now(),
                )
            )

        # Create sessions: 2 pass, 1 fail (for verdict grouping test).
        conn.execute(
            sessions.insert().values(
                session_id=session_id_1,
                teacher_id=teacher_id,
                college_id=college_id,
                consent_id=consent_id,
                media_sha256=media_sha256_1,
                domain="microteaching",
                recorded_on=date(2025, 1, 1),
                quality_verdict="pass",
                quality_detail={},
                created_at=func.now(),
            )
        )
        conn.execute(
            sessions.insert().values(
                session_id=session_id_2,
                teacher_id=teacher_id,
                college_id=college_id,
                consent_id=consent_id,
                media_sha256=media_sha256_2,
                domain="microteaching",
                recorded_on=date(2025, 1, 2),
                quality_verdict="pass",
                quality_detail={},
                created_at=func.now(),
            )
        )
        conn.execute(
            sessions.insert().values(
                session_id=session_id_3,
                teacher_id=teacher_id,
                college_id=college_id,
                consent_id=consent_id,
                media_sha256=media_sha256_3,
                domain="classroom",
                school_id=school_id,
                recorded_on=date(2025, 1, 3),
                quality_verdict="fail",
                quality_detail={"reason": "too short"},
                created_at=func.now(),
            )
        )

        # Create a pose artifact for session_id_1 (preprocessed).
        conn.execute(
            pose_artifacts.insert().values(
                session_id=session_id_1,
                relative_path="poses/test.json",
                frame_count=480,
                sampled_fps=25.0,
                model_version="mediapipe-v2",
                sha256="a" * 64,
                created_at=func.now(),
            )
        )

        # Create annotation assignments and annotations for session_id_2.
        assignment_id = new_ulid()
        annotation_id = new_ulid()
        conn.execute(
            annotation_assignments.insert().values(
                assignment_id=assignment_id,
                rater_id=rater_id,
                session_id=session_id_2,
                clip_index=0,
                clip_start_s=0.0,
                clip_end_s=8.0,
                behaviour="B1",
                round_name="round-1",
                created_at=func.now(),
            )
        )
        conn.execute(
            annotations.insert().values(
                annotation_id=annotation_id,
                assignment_id=assignment_id,
                clip_id=f"{session_id_2}-{0}",
                rater_id=rater_id,
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 2, "b1_amplitude": 1},
                is_nonscorable=False,
                rater_confidence="certain",
                created_at=func.now(),
            )
        )

    return {
        "college_id": college_id,
        "teacher_id": teacher_id,
        "consent_id": consent_id,
        "session_id_1": session_id_1,
        "session_id_2": session_id_2,
        "session_id_3": session_id_3,
        "rater_id": rater_id,
        "media_sha256_1": media_sha256_1,
        "media_sha256_2": media_sha256_2,
        "media_sha256_3": media_sha256_3,
    }


def test_summary_endpoint_returns_200_on_empty_corpus(private_client) -> None:
    """GET /dashboard/summary returns 200 with zeros on an empty corpus, not 500.

    A dashboard that crashes before the first session is ingested is unusable. Every
    endpoint must gracefully handle zero data.

    `private_client`, because "empty" has to be true rather than merely true so far: against
    the shared database this read `{'microteaching': {'pass': 26}}` and the only thing keeping
    it green was running before the tests that write sessions.
    """
    if not requires_database(private_client):
        return
    client = private_client

    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200

    body = response.json()
    assert "corpus_by_domain" in body
    assert isinstance(body["corpus_by_domain"], dict)
    assert len(body["corpus_by_domain"]) == 0

    assert "preprocessing" in body
    assert body["preprocessing"]["preprocessed_count"] == 0
    assert body["preprocessing"]["total_sessions"] == 0

    assert "annotation" in body
    assert body["annotation"]["assignments"] == 0
    assert body["annotation"]["annotations"] == 0

    assert "last_audit_at" in body
    assert body["last_audit_at"] is None

    assert "model" in body
    assert body["model"]["status"] == "absent"

    assert "confidence_distribution" in body
    assert body["confidence_distribution"]["status"] == "awaiting model"


def test_summary_groups_corpus_by_domain_and_verdict(private_corpus) -> None:
    """GET /dashboard/summary counts sessions by domain and quality verdict correctly.

    If the query groups incorrectly (e.g. missing GROUP BY), a wrong verdict grouping
    would be silently cached. This test asserts the counts so a broken join or group
    is caught immediately.

    The endpoint counts the whole corpus, so the exact numbers below are only the fixture's
    numbers in a database holding nothing else: against the shared one this asserted 2 and read
    28, every one of the extra twenty-six belonging to another test.
    """
    if not requires_database(private_corpus):
        return
    client = private_corpus["client"]

    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200

    body = response.json()
    corpus = body["corpus_by_domain"]

    assert "microteaching" in corpus
    assert corpus["microteaching"]["pass"] == 2
    assert corpus["microteaching"].get("fail", 0) == 0

    assert "classroom" in corpus
    assert corpus["classroom"]["fail"] == 1
    assert corpus["classroom"].get("pass", 0) == 0


def test_summary_reports_preprocessing_progress(client, dashboard_setup) -> None:
    """GET /dashboard/summary counts preprocessed sessions correctly.

    Preprocessing count comes from pose_artifacts, not derived from a timestamp or flag.
    A wrong join would silently undercount preprocessed sessions.
    """
    if not requires_database(client, dashboard_setup):
        return

    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200

    body = response.json()
    # Verify that both fields are present and non-negative.
    assert "preprocessed_count" in body["preprocessing"]
    assert "total_sessions" in body["preprocessing"]
    assert body["preprocessing"]["preprocessed_count"] >= 0
    assert body["preprocessing"]["total_sessions"] >= 0
    # At minimum, preprocessed count should not exceed total.
    assert (body["preprocessing"]["preprocessed_count"] <=
            body["preprocessing"]["total_sessions"])


def test_summary_reports_annotation_assignments_and_counts(
    client, dashboard_setup
) -> None:
    """GET /dashboard/summary reports both assignment and annotation counts.

    The assignment count and annotation count can diverge (not all assignments
    receive an annotation). Both are reported so the UI can show assignment
    queue depth.
    """
    if not requires_database(client, dashboard_setup):
        return

    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200

    body = response.json()
    # Both fields must be present and non-negative.
    assert "assignments" in body["annotation"]
    assert "annotations" in body["annotation"]
    assert body["annotation"]["assignments"] >= 0
    assert body["annotation"]["annotations"] >= 0
    # Annotations should not exceed assignments.
    assert body["annotation"]["annotations"] <= body["annotation"]["assignments"]


def test_sessions_endpoint_returns_200_with_proper_structure(client) -> None:
    """GET /dashboard/sessions returns 200 with correct response structure."""
    if not requires_database(client):
        return

    response = client.get("/api/v1/dashboard/sessions")
    assert response.status_code == 200

    body = response.json()
    assert "sessions" in body
    assert isinstance(body["sessions"], list)
    assert "total_count" in body
    assert isinstance(body["total_count"], int)
    assert body["total_count"] >= 0
    assert "skip" in body
    assert body["skip"] == 0
    assert "limit" in body
    assert body["limit"] == 25
    # The list should have at most limit items.
    assert len(body["sessions"]) <= body["limit"]


def test_sessions_reports_real_durations_from_media_objects(
    client, dashboard_setup
) -> None:
    """GET /dashboard/sessions returns actual duration_s from media_objects table.

    If the endpoint was stubbed to return 0, a test would have caught it. The duration
    comes from the probe's read of the media file at ingest, so it is never recomputed.
    """
    if not requires_database(client, dashboard_setup):
        return

    response = client.get("/api/v1/dashboard/sessions")
    assert response.status_code == 200

    body = response.json()
    sessions_list = body["sessions"]

    # Find our test sessions from dashboard_setup and verify they report correct durations.
    session_dict = {s["session_id"]: s for s in sessions_list}

    # Check that our test sessions are present and report correct durations.
    if dashboard_setup["session_id_1"] in session_dict:
        assert session_dict[dashboard_setup["session_id_1"]]["duration_s"] == 480.0
    if dashboard_setup["session_id_2"] in session_dict:
        assert session_dict[dashboard_setup["session_id_2"]]["duration_s"] == 600.5
    if dashboard_setup["session_id_3"] in session_dict:
        assert session_dict[dashboard_setup["session_id_3"]]["duration_s"] == 120.25

    # At minimum, all sessions should report non-zero durations.
    for session in sessions_list:
        assert session["duration_s"] >= 0, "duration should never be negative"


def test_sessions_reports_preprocessed_when_pose_artifact_exists(private_corpus) -> None:
    """GET /dashboard/sessions reports preprocessed=true only when pose_artifacts row
    exists for that session.

    A LEFT OUTER join is necessary so sessions without poses still appear in the list
    (to show what is outstanding). Without the left join, unpreprocessed sessions
    vanish.

    Looking the three sessions up by id is not enough on its own to make this independent of
    what else is in the database: the endpoint pages, twenty-five at a time, ordered by
    recorded_on descending, and against the shared database these three sorted off the first
    page and the lookup raised KeyError. The list endpoint is the subject here - the detail
    endpoint would not exercise the outer join - so the corpus has to be the only one.
    """
    if not requires_database(private_corpus):
        return
    client = private_corpus["client"]

    response = client.get("/api/v1/dashboard/sessions")
    assert response.status_code == 200

    body = response.json()
    sessions_dict = {s["session_id"]: s for s in body["sessions"]}

    assert sessions_dict[private_corpus["session_id_1"]]["preprocessed"] is True
    assert sessions_dict[private_corpus["session_id_2"]]["preprocessed"] is False
    assert sessions_dict[private_corpus["session_id_3"]]["preprocessed"] is False


def test_sessions_reports_annotated_when_assignment_exists(private_corpus) -> None:
    """GET /dashboard/sessions reports annotated=true only when annotation exists
    for that session via annotation_assignments.

    The link runs through annotation_assignments (session_id) and annotations
    (assignment_id), not by substring matching the session_id within clip_id. A
    substring match would depend on clip_id format and could silently break.

    Private for the same reason as the preprocessed test above: the list is paged.
    """
    if not requires_database(private_corpus):
        return
    client = private_corpus["client"]

    response = client.get("/api/v1/dashboard/sessions")
    assert response.status_code == 200

    body = response.json()
    sessions_dict = {s["session_id"]: s for s in body["sessions"]}

    assert sessions_dict[private_corpus["session_id_1"]]["annotated"] is False
    assert sessions_dict[private_corpus["session_id_2"]]["annotated"] is True
    assert sessions_dict[private_corpus["session_id_3"]]["annotated"] is False


def test_sessions_does_not_duplicate_sessions_with_many_annotations(private_corpus) -> None:
    """A session with many annotations appears ONCE in /sessions list.

    An inner join to annotations would duplicate the row. A paginated list with
    duplicates would silently drop other sessions off the page.

    Counting occurrences in the response only means anything if the session is in the response
    at all, and the list is paged: randomised ordering caught this counting zero. The irony is
    that the defect it guards against - duplicate rows pushing other sessions off a page - is
    the same arithmetic that made the test itself unreliable.
    """
    if not requires_database(private_corpus):
        return

    client, engine = private_corpus["client"], private_corpus["engine"]
    session_id = private_corpus["session_id_2"]
    rater_id = private_corpus["rater_id"]

    # Add a second annotation for the same session.
    with transaction(engine) as conn:
        assignment_id_2 = new_ulid()
        annotation_id_2 = new_ulid()
        conn.execute(
            annotation_assignments.insert().values(
                assignment_id=assignment_id_2,
                rater_id=rater_id,
                session_id=session_id,
                clip_index=1,
                clip_start_s=8.0,
                clip_end_s=16.0,
                behaviour="B2",
                round_name="round-1",
                created_at=func.now(),
            )
        )
        conn.execute(
            annotations.insert().values(
                annotation_id=annotation_id_2,
                assignment_id=assignment_id_2,
                clip_id=f"{session_id}-{1}",
                rater_id=rater_id,
                behaviour="B2",
                codebook_version="v1.0-draft",
                labels={"b2_present": True},
                is_nonscorable=False,
                rater_confidence="certain",
                created_at=func.now(),
            )
        )

    response = client.get("/api/v1/dashboard/sessions")
    body = response.json()
    sessions_list = body["sessions"]

    # Count occurrences of session_id_2.
    count = sum(1 for s in sessions_list if s["session_id"] == session_id)
    assert count == 1, "a session with many annotations should appear exactly once"


# A ULID, because audit_log.actor_user_id is CHAR(26) and a shorter value is stored
# blank-padded, which breaks every later verification of the chain (D93).
EXCLUDING_RESEARCHER = "01M2H000000000000000000000"


def test_the_session_list_says_when_a_session_was_excluded_and_why(private_corpus) -> None:
    """D96's whole point, at the surface a reviewer actually looks at.

    An excluded session is absent from the annotation plan, like one nobody has confirmed a
    teacher track for. If this screen reports them alike, the next reviewer opens the excluded
    one, tries to confirm a track, and finds no explanation. So the ground travels with the row.
    """
    if not requires_database(private_corpus):
        return

    # The private corpus, because the assertion below is about every *other* session reading as
    # not excluded. On the shared scratch database another test's exclusion would satisfy or
    # break that by accident.
    client, engine = private_corpus["client"], private_corpus["engine"]
    session_id = private_corpus["session_id_1"]
    ground = "the camera moves too much for the teacher to survive as one track"
    with transaction(engine) as conn:
        exclude(conn, session_id=session_id, reason=ground,
                excluded_by=EXCLUDING_RESEARCHER, actor_role="researcher")

    listed = {row["session_id"]: row
              for row in client.get("/api/v1/dashboard/sessions").json()["sessions"]}

    mine = listed[session_id]
    assert mine["excluded_at"] is not None
    assert mine["exclusion_reason"] == ground

    others = [row for sid, row in listed.items() if sid != session_id]
    assert others, "nothing to compare against; the fixture should seed more than one session"
    assert all(row["excluded_at"] is None and row["exclusion_reason"] is None
               for row in others), (
        "every session reads as excluded, so the field distinguishes nothing")


def test_sessions_does_not_leak_teacher_identity(client, dashboard_setup) -> None:
    """GET /dashboard/sessions response contains no teacher_id or teacher code.

    R1 and D76 require that identity (teacher id, teacher code, media filename) never
    appears in any response. The identity is required in the database for data
    provenance but must not reach the dashboard user.
    """
    if not requires_database(client, dashboard_setup):
        return

    response = client.get("/api/v1/dashboard/sessions")
    assert response.status_code == 200

    body = response.json()
    response_text = json.dumps(body)

    assert dashboard_setup["teacher_id"] not in response_text


def test_sessions_does_not_leak_media_filename(client, dashboard_setup) -> None:
    """GET /dashboard/sessions response contains no media relative_path.

    Media filenames can reveal identity. The filename is stored but never exposed
    in the API response.
    """
    if not requires_database(client, dashboard_setup):
        return

    response = client.get("/api/v1/dashboard/sessions")
    assert response.status_code == 200

    body = response.json()
    response_text = json.dumps(body)

    # The media_sha256 should not appear; neither should the relative_path.
    assert dashboard_setup["media_sha256_1"] not in response_text
    assert dashboard_setup["media_sha256_2"] not in response_text
    assert dashboard_setup["media_sha256_3"] not in response_text
    assert ".mp4" not in response_text


def test_sessions_honours_skip_parameter(private_corpus) -> None:
    """GET /dashboard/sessions?skip=N skips the first N sessions.

    Private, and asked by identity rather than by length. This compared the sizes of two
    responses and expected the skipped one to be smaller by exactly one, which holds only while
    the corpus fits inside the limit it asks for: against the shared database, once more than a
    hundred sessions had accumulated, both responses came back capped at a hundred and the test
    read 100 == 99. Randomised ordering found it.

    Comparing the identifiers is also the stronger claim. A response one shorter proves that
    something was dropped; these three sessions have known, distinct recorded_on dates, so
    naming them proves that what was dropped was the *first* one in the endpoint's own order.
    """
    if not requires_database(private_corpus):
        return
    client = private_corpus["client"]

    everything = client.get("/api/v1/dashboard/sessions?limit=100")
    assert everything.status_code == 200
    all_ids = [row["session_id"] for row in everything.json()["sessions"]]
    assert len(all_ids) == 3, (
        f"the private corpus holds three sessions; the endpoint returned {len(all_ids)}")

    skipped = client.get("/api/v1/dashboard/sessions?skip=1&limit=100")
    assert skipped.status_code == 200
    body = skipped.json()

    assert [row["session_id"] for row in body["sessions"]] == all_ids[1:]
    assert body["skip"] == 1
    assert body["total_count"] == 3, (
        "total_count reports the corpus, not the page, so skipping must not change it")


def test_sessions_honours_limit_parameter(client, dashboard_setup) -> None:
    """GET /dashboard/sessions?limit=N returns at most N sessions."""
    if not requires_database(client, dashboard_setup):
        return

    response = client.get("/api/v1/dashboard/sessions?skip=0&limit=2")
    assert response.status_code == 200

    body = response.json()
    sessions_list = body["sessions"]
    assert len(sessions_list) == 2
    assert body["limit"] == 2


def test_sessions_orders_by_recorded_on_descending(client, dashboard_setup) -> None:
    """GET /dashboard/sessions orders by recorded_on descending (newest first)."""
    if not requires_database(client, dashboard_setup):
        return

    response = client.get("/api/v1/dashboard/sessions")
    assert response.status_code == 200

    body = response.json()
    sessions_list = body["sessions"]

    # Sessions should be ordered by recorded_on descending.
    recorded_ons = [s["recorded_on"] for s in sessions_list]
    assert recorded_ons == sorted(recorded_ons, reverse=True)


def test_session_detail_returns_404_for_unknown_session(client) -> None:
    """GET /dashboard/sessions/{unknown_id} returns 404 with RFC 7807 body."""
    if not requires_database(client):
        return

    unknown_id = new_ulid()
    response = client.get(f"/api/v1/dashboard/sessions/{unknown_id}")
    assert response.status_code == 404

    body = response.json()
    assert "type" in body
    assert "detail" in body
    assert "/errors/not-found" in body["type"]
    assert unknown_id in body["detail"]


def test_session_detail_returns_session_metadata(client, dashboard_setup) -> None:
    """GET /dashboard/sessions/{id} returns session metadata and annotation count."""
    if not requires_database(client, dashboard_setup):
        return

    session_id = dashboard_setup["session_id_2"]
    response = client.get(f"/api/v1/dashboard/sessions/{session_id}")
    assert response.status_code == 200

    body = response.json()
    assert body["session_id"] == session_id
    assert body["domain"] == "microteaching"
    assert body["quality_verdict"] == "pass"
    assert body["recorded_on"] == "2025-01-02"
    assert body["duration_s"] == 600.5
    assert body["preprocessed"] is False
    assert body["annotation_count"] == 1
    assert "subject" in body
    assert "grade_level" in body
    assert "created_at" in body


def test_session_detail_does_not_leak_identity(client, dashboard_setup) -> None:
    """GET /dashboard/sessions/{id} contains no teacher_id or media filename."""
    if not requires_database(client, dashboard_setup):
        return

    session_id = dashboard_setup["session_id_2"]
    response = client.get(f"/api/v1/dashboard/sessions/{session_id}")
    assert response.status_code == 200

    body = response.json()
    response_text = json.dumps(body)

    assert dashboard_setup["teacher_id"] not in response_text
    assert dashboard_setup["media_sha256_2"] not in response_text
    assert ".mp4" not in response_text


def test_session_detail_counts_annotations_correctly(
    client, engine, dashboard_setup
) -> None:
    """GET /dashboard/sessions/{id} annotation_count reflects actual annotation count.

    The count is done through the assignment link, not by substring matching the
    session_id within the clip_id, to avoid breaking if clip_id format changes.
    """
    if not requires_database(client, engine, dashboard_setup):
        return

    session_id = dashboard_setup["session_id_2"]

    # Add another annotation for this session.
    with transaction(engine) as conn:
        rater_id = dashboard_setup["rater_id"]
        assignment_id = new_ulid()
        annotation_id = new_ulid()
        conn.execute(
            annotation_assignments.insert().values(
                assignment_id=assignment_id,
                rater_id=rater_id,
                session_id=session_id,
                clip_index=1,
                clip_start_s=8.0,
                clip_end_s=16.0,
                behaviour="B2",
                round_name="round-1",
                created_at=func.now(),
            )
        )
        conn.execute(
            annotations.insert().values(
                annotation_id=annotation_id,
                assignment_id=assignment_id,
                clip_id=f"{session_id}-{1}",
                rater_id=rater_id,
                behaviour="B2",
                codebook_version="v1.0-draft",
                labels={"b2_present": True},
                is_nonscorable=False,
                rater_confidence="certain",
                created_at=func.now(),
            )
        )

    response = client.get(f"/api/v1/dashboard/sessions/{session_id}")
    assert response.status_code == 200

    body = response.json()
    assert body["annotation_count"] == 2


def test_runs_endpoint_returns_200_on_empty_run_outputs(
    scratch_database, tmp_path
) -> None:
    """GET /dashboard/runs returns 200 with empty list when run_outputs is empty."""
    if scratch_database is None:
        return

    # Create a fresh config and app with empty run_outputs.
    empty_run_outputs = tmp_path / "run_outputs_empty"
    empty_run_outputs.mkdir()
    base_config = lenient()
    config = base_config.model_copy(update={
        "paths": base_config.paths.model_copy(update={"run_outputs": empty_run_outputs})
    })
    app = create_app(
        config=config,
        engine=build_engine(keywords_to_url(scratch_database)),
        media_root=tmp_path / "media",
        enqueue=RecordingQueue(),
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/v1/dashboard/runs")
    assert response.status_code == 200

    body = response.json()
    assert "runs" in body
    assert isinstance(body["runs"], list)
    assert len(body["runs"]) == 0


def test_runs_reports_successful_run(scratch_database, tmp_path) -> None:
    """GET /dashboard/runs includes runs that completed successfully.

    A run with manifest.json and result.json, both valid JSON, shows as ran=true.
    """
    if scratch_database is None:
        return

    # Set up a mock run directory in a fresh location.
    run_root = tmp_path / "run_outputs_success"
    run_root.mkdir()
    base_config = lenient()
    config = base_config.model_copy(update={
        "paths": base_config.paths.model_copy(update={"run_outputs": run_root})
    })
    app = create_app(
        config=config,
        engine=build_engine(keywords_to_url(scratch_database)),
        media_root=tmp_path / "media",
        enqueue=RecordingQueue(),
    )
    client = TestClient(app, raise_server_exceptions=False)

    run_id = new_ulid()
    run_dir = run_root / run_id
    run_dir.mkdir()

    manifest = {
        "run_id": run_id,
        "phase": "1",
        "finished_at": "2025-01-15T10:00:00Z",
    }
    result = {"ran": True}

    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    response = client.get("/api/v1/dashboard/runs")
    assert response.status_code == 200

    body = response.json()
    runs = body["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["phase"] == "1"
    assert runs[0]["verdict"] == "ran"
    assert runs[0]["finished_at"] == "2025-01-15T10:00:00Z"
    assert "abstention_reason" not in runs[0]


def test_runs_reports_abstained_run_with_reason(scratch_database, tmp_path) -> None:
    """GET /dashboard/runs includes runs that abstained and reports the reason.

    An abstained run (ran=false) includes the reason and any missing resources
    from the abstained object in result.json.
    """
    if scratch_database is None:
        return

    run_root = tmp_path / "run_outputs_abstain"
    run_root.mkdir()
    base_config = lenient()
    config = base_config.model_copy(update={
        "paths": base_config.paths.model_copy(update={"run_outputs": run_root})
    })
    app = create_app(
        config=config,
        engine=build_engine(keywords_to_url(scratch_database)),
        media_root=tmp_path / "media",
        enqueue=RecordingQueue(),
    )
    client = TestClient(app, raise_server_exceptions=False)

    run_id = new_ulid()
    run_dir = run_root / run_id
    run_dir.mkdir()

    manifest = {
        "run_id": run_id,
        "phase": "5",
        "finished_at": "2025-01-15T11:00:00Z",
    }
    result = {
        "ran": False,
        "abstained": {
            "reason": "model checkpoint not found",
            "missing": ["model.pth"],
        },
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    response = client.get("/api/v1/dashboard/runs")
    assert response.status_code == 200

    body = response.json()
    runs = body["runs"]
    assert len(runs) == 1
    assert runs[0]["verdict"] == "abstained"
    assert runs[0]["abstention_reason"] == "model checkpoint not found"
    assert runs[0]["abstention_missing"] == ["model.pth"]


def test_runs_skips_corrupted_directories(scratch_database, tmp_path) -> None:
    """GET /dashboard/runs skips run directories with missing or invalid JSON files.

    A corrupted run (missing manifest.json, invalid JSON, etc.) is skipped, not
    reported as an error or 500.
    """
    if scratch_database is None:
        return

    run_root = tmp_path / "run_outputs_corrupt"
    run_root.mkdir()
    base_config = lenient()
    config = base_config.model_copy(update={
        "paths": base_config.paths.model_copy(update={"run_outputs": run_root})
    })
    app = create_app(
        config=config,
        engine=build_engine(keywords_to_url(scratch_database)),
        media_root=tmp_path / "media",
        enqueue=RecordingQueue(),
    )
    client = TestClient(app, raise_server_exceptions=False)

    # Create a valid run.
    valid_run_id = new_ulid()
    valid_run_dir = run_root / valid_run_id
    valid_run_dir.mkdir()
    manifest = {"run_id": valid_run_id, "phase": "1"}
    result = {"ran": True}
    (valid_run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (valid_run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    # Create an invalid run (missing result.json).
    invalid_run_id = new_ulid()
    invalid_run_dir = run_root / invalid_run_id
    invalid_run_dir.mkdir()
    (invalid_run_dir / "manifest.json").write_text(
        json.dumps({"run_id": invalid_run_id}), encoding="utf-8"
    )

    response = client.get("/api/v1/dashboard/runs")
    assert response.status_code == 200

    body = response.json()
    runs = body["runs"]

    # Only the valid run should appear.
    assert len(runs) == 1
    assert runs[0]["run_id"] == valid_run_id


def test_runs_orders_by_run_id_descending(scratch_database, tmp_path) -> None:
    """GET /dashboard/runs orders by run_id descending (newest first).

    run_id is a ULID, so lexicographic descending is also chronologically descending.
    """
    if scratch_database is None:
        return

    run_root = tmp_path / "run_outputs_order"
    run_root.mkdir()
    base_config = lenient()
    config = base_config.model_copy(update={
        "paths": base_config.paths.model_copy(update={"run_outputs": run_root})
    })
    app = create_app(
        config=config,
        engine=build_engine(keywords_to_url(scratch_database)),
        media_root=tmp_path / "media",
        enqueue=RecordingQueue(),
    )
    client = TestClient(app, raise_server_exceptions=False)

    # Create three runs with distinct IDs.
    import time

    run_ids = []
    for i in range(3):
        time.sleep(0.01)  # Ensure distinct ULIDs.
        run_id = new_ulid()
        run_ids.append(run_id)
        run_dir = run_root / run_id
        run_dir.mkdir()
        manifest = {"run_id": run_id, "phase": str(i)}
        result = {"ran": True}
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    response = client.get("/api/v1/dashboard/runs")
    assert response.status_code == 200

    body = response.json()
    runs = body["runs"]

    # Verify they are ordered descending by run_id (which is also by time).
    returned_ids = [r["run_id"] for r in runs]
    assert returned_ids == sorted(returned_ids, reverse=True)
