"""The annotation API, exercised through HTTP.

The domain logic is tested in unit tests, and the persistence layer in integration tests.
What is checked here is the HTTP interface: status codes, response body shapes, and that
refusals reach the client as reasons rather than tracebacks.
"""
from __future__ import annotations

import hashlib
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func

from praxis.annotation.clips import clip_plan
from praxis.annotation.server import build_assignments
from praxis.api.app import create_app
from praxis.db import build_engine, keywords_to_url, transaction
from praxis.db.schema import (
    annotation_assignments,
    annotations,
    colleges,
    consent_records,
    media_objects,
    sessions,
    teachers,
)
from praxis.ids import new_ulid
from praxis.tasks import RecordingQueue
from tests.integration.conftest import lenient, requires_database


@pytest.fixture
def client(scratch_database, tmp_path):
    """A test client against a scratch database."""
    if scratch_database is None:
        return None
    media_root = tmp_path / "media"
    media_root.mkdir()
    app = create_app(
        config=lenient(),
        engine=build_engine(keywords_to_url(scratch_database)),
        media_root=media_root,
        enqueue=RecordingQueue(),
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def calibration_setup(engine):
    """College, teachers, consent, sessions, and clip assignments for testing."""
    if engine is None:
        return None

    college_id = new_ulid()
    teacher_id = new_ulid()
    consent_id = new_ulid()
    session_id = new_ulid()
    rater1_id = new_ulid()
    rater2_id = new_ulid()
    media_sha256 = hashlib.sha256(session_id.encode()).hexdigest()

    with transaction(engine) as conn:
        conn.execute(
            colleges.insert().values(
                college_id=college_id,
                # `colleges.code` is UNIQUE and the scratch database outlives a single test,
                # so a constant here collides with every test that ran before this one.
                code=f"TEST-{college_id[-8:]}",
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
        # `sessions.media_sha256` is a foreign key, so the media row has to exist first.
        conn.execute(
            media_objects.insert().values(
                media_sha256=media_sha256,
                relative_path=f"media/{media_sha256[:2]}/{media_sha256}.mp4",
                bytes=1024,
                duration_s=480.0,
                width=1280,
                height=720,
                fps=25.0,
                has_audio=False,
                # Paired with the path, as the `media_objects_blurred_has_a_path` CHECK
                # requires: the original is deleted once a session is blurred, so a row
                # claiming a blurred file without naming it has lost the session.
                is_blurred=True,
                blurred_relative_path=f".blurred/{session_id}.mp4",
                reachable=True,
                created_at=func.now(),
            )
        )
        conn.execute(
            sessions.insert().values(
                session_id=session_id,
                teacher_id=teacher_id,
                college_id=college_id,
                consent_id=consent_id,
                media_sha256=media_sha256,
                domain="microteaching",
                recorded_on=date(2025, 1, 1),
                quality_verdict="pass",
                quality_detail={},
                created_at=func.now(),
            )
        )

        # Create clips and assignments.
        #
        # The round name is uniquified for the same reason `colleges.code` is above: the
        # scratch database outlives a single test, and every round-filtered endpoint reads
        # the round by name alone. A constant here pools this test's annotations with every
        # other test that used the same name, which made the suite order-dependent - it
        # passed when these files ran first and failed when the store tests ran before them.
        # A suite whose verdict depends on file order is not evidence of anything.
        round_name = f"calibration-1-{session_id[-8:]}"
        clips = clip_plan(session_id, 32.0, clip_seconds=8.0)
        assignments = build_assignments(
            clips,
            (rater1_id, rater2_id),
            behaviours=("B1", "B2"),
            round_name=round_name,
            raters_per_clip=2,
            seed=42,
        )

        for a in assignments:
            conn.execute(
                annotation_assignments.insert().values(
                    assignment_id=a.assignment_id,
                    rater_id=a.rater_id,
                    session_id=a.clip.session_id,
                    clip_index=a.clip.clip_index,
                    clip_start_s=a.clip.start_seconds,
                    clip_end_s=a.clip.end_seconds,
                    behaviour=a.behaviour,
                    round_name=a.round_name,
                    created_at=func.now(),
                )
            )

    return {
        "college_id": college_id,
        "teacher_id": teacher_id,
        "consent_id": consent_id,
        "session_id": session_id,
        "rater1_id": rater1_id,
        "rater2_id": rater2_id,
        "assignments": assignments,
        "round_name": round_name,
    }


def test_codebook_returns_active_version(client) -> None:
    """GET /codebook returns the active codebook version and all its fields."""
    if not requires_database(client):
        return

    response = client.get("/api/v1/annotation/codebook")
    assert response.status_code == 200

    body = response.json()
    assert "version" in body
    assert "behaviours" in body
    assert "confidence_levels" in body
    assert "excluded_from_primary_irr" in body

    assert body["version"] == "v1.0-draft"
    assert len(body["behaviours"]) == 5

    b1 = next(b for b in body["behaviours"] if b["behaviour"] == "B1")
    assert b1["name"] == "Gesture production"
    assert "fields" in b1
    assert len(b1["fields"]) > 0

    b1_present = next(f for f in b1["fields"] if f["name"] == "b1_present")
    assert b1_present["scale"] == "nominal"
    assert b1_present["levels"] == [False, True]


def test_queue_for_rater_returns_pending_assignments(client, calibration_setup) -> None:
    """GET /queue?rater_id= returns assignments the rater has not yet submitted."""
    if not requires_database(client, calibration_setup):
        return

    rater1 = calibration_setup["rater1_id"]
    response = client.get(f"/api/v1/annotation/queue?rater_id={rater1}&limit=10")
    assert response.status_code == 200

    body = response.json()
    assert "assignments" in body
    assignments = body["assignments"]

    # rater1 should have assignments.
    assert len(assignments) > 0

    # Each assignment has the expected structure.
    for a in assignments:
        assert "assignment_id" in a
        assert "rater_id" in a
        assert a["rater_id"] == rater1
        assert "clip" in a
        assert "behaviour" in a
        assert "round_name" in a

        clip = a["clip"]
        assert "clip_id" in clip
        assert "session_id" in clip
        assert "clip_index" in clip
        assert "start_seconds" in clip
        assert "end_seconds" in clip


def test_queue_respects_round_name_filter(client, calibration_setup) -> None:
    """GET /queue with round_name filters to that round only."""
    if not requires_database(client, calibration_setup):
        return

    rater1 = calibration_setup["rater1_id"]
    round_name = calibration_setup["round_name"]

    response = client.get(
        f"/api/v1/annotation/queue?rater_id={rater1}&round_name={round_name}"
    )
    assert response.status_code == 200

    body = response.json()
    assignments = body["assignments"]

    for a in assignments:
        assert a["round_name"] == round_name


def test_submit_label_returns_201_with_annotation_id_and_version(
    client, calibration_setup
) -> None:
    """POST /labels with valid data returns 201 with annotation_id and codebook_version."""
    if not requires_database(client, calibration_setup):
        return

    rater1 = calibration_setup["rater1_id"]
    assignments = calibration_setup["assignments"]

    # Pick the first B1 assignment.
    b1_assignment = next(a for a in assignments if a.behaviour == "B1")
    clip_id = b1_assignment.clip.clip_id

    response = client.post(
        "/api/v1/annotation/labels",
        json={
            "clip_id": clip_id,
            "rater_id": rater1,
            "behaviour": "B1",
            "labels": {
                "b1_present": True,
                "b1_count": 3,
                "b1_amplitude": 2,
                "b1_nonscorable": False,
            },
            "is_nonscorable": False,
            "note": None,
            "rater_confidence": "certain",
            "session_college_id": calibration_setup["college_id"],
        },
    )
    assert response.status_code == 201, response.text

    body = response.json()
    assert "annotation_id" in body
    assert "codebook_version" in body
    assert body["codebook_version"] == "v1.0-draft"
    assert len(body["annotation_id"]) == 26  # ULID length


def test_submit_label_returns_422_for_unknown_field(
    client, calibration_setup
) -> None:
    """POST /labels with unknown field returns 422."""
    if not requires_database(client, calibration_setup):
        return

    rater1 = calibration_setup["rater1_id"]
    assignments = calibration_setup["assignments"]
    b1_assignment = next(a for a in assignments if a.behaviour == "B1")
    clip_id = b1_assignment.clip.clip_id

    response = client.post(
        "/api/v1/annotation/labels",
        json={
            "clip_id": clip_id,
            "rater_id": rater1,
            "behaviour": "B1",
            "labels": {
                "b1_present": True,
                "b1_count": 3,
                "b1_amplitude": 2,
                "b1_nonscorable": False,
                "b1_unknown_field": "invalid",  # Does not exist.
            },
            "is_nonscorable": False,
            "rater_confidence": "certain",
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert "type" in body
    assert "detail" in body
    assert "unknown_field" in body["detail"].lower()


def test_submit_label_returns_422_for_invalid_value(
    client, calibration_setup
) -> None:
    """POST /labels with invalid value returns 422."""
    if not requires_database(client, calibration_setup):
        return

    rater1 = calibration_setup["rater1_id"]
    assignments = calibration_setup["assignments"]
    b1_assignment = next(a for a in assignments if a.behaviour == "B1")
    clip_id = b1_assignment.clip.clip_id

    response = client.post(
        "/api/v1/annotation/labels",
        json={
            "clip_id": clip_id,
            "rater_id": rater1,
            "behaviour": "B1",
            "labels": {
                "b1_present": True,
                "b1_count": 3,
                "b1_amplitude": 99,  # Invalid: must be 1, 2 or 3.
                "b1_nonscorable": False,
            },
            "is_nonscorable": False,
            "rater_confidence": "certain",
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_submit_label_returns_409_for_duplicate(client, calibration_setup) -> None:
    """POST /labels twice for same rater+clip+behaviour returns 409 on the second."""
    if not requires_database(client, calibration_setup):
        return

    rater1 = calibration_setup["rater1_id"]
    assignments = calibration_setup["assignments"]
    b1_assignment = next(a for a in assignments if a.behaviour == "B1")
    clip_id = b1_assignment.clip.clip_id

    payload = {
        "clip_id": clip_id,
        "rater_id": rater1,
        "behaviour": "B1",
        "labels": {
            "b1_present": True,
            "b1_count": 2,
            "b1_amplitude": 1,
            "b1_nonscorable": False,
        },
        "is_nonscorable": False,
        "rater_confidence": "certain",
    }

    # First submission should succeed.
    response1 = client.post("/api/v1/annotation/labels", json=payload)
    assert response1.status_code == 201

    # Second submission with same clip+behaviour should fail.
    response2 = client.post("/api/v1/annotation/labels", json=payload)
    assert response2.status_code == 409
    body = response2.json()
    assert "detail" in body
    assert "already labelled" in body["detail"].lower()


def test_disagreements_endpoint_shows_disagreements_between_raters(
    client, calibration_setup, engine
) -> None:
    """GET /calibration/{round_name}/disagreements shows per-field disagreements."""
    if not requires_database(client, calibration_setup, engine):
        return

    rater1 = calibration_setup["rater1_id"]
    rater2 = calibration_setup["rater2_id"]
    assignments = calibration_setup["assignments"]

    # Get the first B2 assignment (both raters will annotate it).
    b2_assignments = [a for a in assignments if a.behaviour == "B2"]
    first_b2 = b2_assignments[0]
    clip_id = first_b2.clip.clip_id

    # Rater 1 labels it.
    response1 = client.post(
        "/api/v1/annotation/labels",
        json={
            "clip_id": clip_id,
            "rater_id": rater1,
            "behaviour": "B2",
            "labels": {
                "b2_facing_proportion": 0.5,
                "b2_head_torso_divergence": 1,
                "b2_dominant": "class",
                "b2_nonscorable": False,
            },
            "rater_confidence": "certain",
        },
    )
    assert response1.status_code == 201

    # Rater 2 labels it differently.
    response2 = client.post(
        "/api/v1/annotation/labels",
        json={
            "clip_id": clip_id,
            "rater_id": rater2,
            "behaviour": "B2",
            "labels": {
                "b2_facing_proportion": 0.75,  # Different.
                "b2_head_torso_divergence": 2,  # Different.
                "b2_dominant": "class",  # Same.
                "b2_nonscorable": False,
            },
            "rater_confidence": "certain",
        },
    )
    assert response2.status_code == 201

    # Get disagreements for the round.
    round_name = calibration_setup["round_name"]
    response = client.get(f"/api/v1/annotation/calibration/{round_name}/disagreements")
    assert response.status_code == 200

    body = response.json()
    assert "disagreements" in body
    disagreements = body["disagreements"]

    # Should have disagreements on b2_facing_proportion and b2_head_torso_divergence.
    facing_disagreement = next(
        (d for d in disagreements
         if d["clip_id"] == clip_id and d["field"] == "b2_facing_proportion"),
        None,
    )
    assert facing_disagreement is not None
    assert facing_disagreement["distinct_values"] == 2
    assert rater1 in facing_disagreement["by_rater"]
    assert rater2 in facing_disagreement["by_rater"]
    assert facing_disagreement["by_rater"][rater1] == 0.5
    assert facing_disagreement["by_rater"][rater2] == 0.75

    torso_disagreement = next(
        (d for d in disagreements
         if d["clip_id"] == clip_id and d["field"] == "b2_head_torso_divergence"),
        None,
    )
    assert torso_disagreement is not None


def test_agreement_endpoint_returns_statistics(
    client, calibration_setup, engine
) -> None:
    """GET /agreement returns inter-rater agreement statistics."""
    if not requires_database(client, calibration_setup, engine):
        return

    rater1 = calibration_setup["rater1_id"]
    rater2 = calibration_setup["rater2_id"]
    assignments = calibration_setup["assignments"]

    # Submit several annotations to get meaningful agreement data.
    b1_assignments = [a for a in assignments if a.behaviour == "B1"][:2]

    for i, a in enumerate(b1_assignments):
        # Rater 1.
        client.post(
            "/api/v1/annotation/labels",
            json={
                "clip_id": a.clip.clip_id,
                "rater_id": rater1,
                "behaviour": "B1",
                "labels": {
                    "b1_present": True,
                    "b1_count": 2 + i,
                    "b1_amplitude": 2,
                    "b1_nonscorable": False,
                },
                "rater_confidence": "certain",
            },
        )

        # Rater 2 (slightly different).
        client.post(
            "/api/v1/annotation/labels",
            json={
                "clip_id": a.clip.clip_id,
                "rater_id": rater2,
                "behaviour": "B1",
                "labels": {
                    "b1_present": True,
                    "b1_count": 2 + i,
                    "b1_amplitude": 2,  # Same as rater 1.
                    "b1_nonscorable": False,
                },
                "rater_confidence": "certain",
            },
        )

    response = client.get(
        f"/api/v1/annotation/agreement?round_name={calibration_setup['round_name']}")
    assert response.status_code == 200

    body = response.json()
    assert "agreement" in body
    agreement = body["agreement"]

    assert "behaviours" in agreement
    assert "gate_value" in agreement
    assert "gate_passed" in agreement

    # Should have agreement report for at least B1 (which we annotated).
    behaviours = agreement["behaviours"]
    b1_agreement = next(
        (b for b in behaviours if b["behaviour"] == "B1"),
        None,
    )
    assert b1_agreement is not None
    assert "fields" in b1_agreement
    assert len(b1_agreement["fields"]) > 0

    # Check that each field has the expected structure.
    for field in b1_agreement["fields"]:
        assert "field" in field
        assert "scale" in field
        assert "n_units" in field
        assert "n_raters" in field
        assert "alpha" in field or field["alpha"] is None
        assert "primary_statistic" in field
        assert "primary" in field or field["primary"] is None

    # And that it actually read the labels this test posted. `report` enumerates the codebook,
    # so B1 and its fields are present whether or not a single annotation was found: every
    # assertion above holds against an empty round. Without this the test passes while the
    # endpoint reports on nothing, which is how the missing assignment link stayed invisible.
    assert max(f["n_units"] for f in b1_agreement["fields"]) > 0
    assert max(f["n_raters"] for f in b1_agreement["fields"]) == 2


def test_submitted_label_is_linked_to_the_assignment_it_answers(
    client, calibration_setup, engine
) -> None:
    """A label posted through the API carries the assignment_id of the clip it answers.

    Without the link the row is invisible to every round-filtered read, because the round is
    reached through `annotation_assignments`. That is the whole calibration workflow reading
    nothing back from the tool built to feed it, and nothing fails loudly when it happens:
    the endpoints answer 200 with an empty report.
    """
    if not requires_database(client, calibration_setup, engine):
        return

    assignment = next(a for a in calibration_setup["assignments"] if a.behaviour == "B1")

    response = client.post(
        "/api/v1/annotation/labels",
        json={
            "clip_id": assignment.clip.clip_id,
            "rater_id": assignment.rater_id,
            "behaviour": "B1",
            "labels": {
                "b1_present": True,
                "b1_count": 2,
                "b1_amplitude": 2,
                "b1_nonscorable": False,
            },
            "note": "learner in the back row was out of frame",
        },
    )
    assert response.status_code == 201
    annotation_id = response.json()["annotation_id"]

    with transaction(engine) as conn:
        row = conn.execute(
            annotations.select().where(annotations.c.annotation_id == annotation_id),
        ).first()

    assert row is not None
    assert row.assignment_id == assignment.assignment_id
    # CODEBOOK.md keeps the free-text note out of quantitative analysis, which is why it is
    # not on the Annotation contract. It still has to reach its column: section 8 revises the
    # codebook on the strength of what raters wrote there, so dropping it silently destroys
    # the evidence the revision rests on.
    assert row.note == "learner in the back row was out of frame"


def test_submitted_label_is_visible_to_the_round_it_belongs_to(
    client, calibration_setup
) -> None:
    """The round-filtered reads see a label the moment it is submitted.

    This is the end-to-end form of the linkage: post through the tool, read back through the
    calibration view. It fails if the assignment link is ever dropped again.
    """
    if not requires_database(client, calibration_setup):
        return

    round_name = calibration_setup["round_name"]
    clip = next(a for a in calibration_setup["assignments"] if a.behaviour == "B2").clip

    def b2_units(response) -> list[int]:
        """n_units per B2 field. `report` enumerates the codebook, so the behaviours are
        always all five and it is the unit counts that say whether any label was read."""
        assert response.status_code == 200
        behaviours = response.json()["agreement"]["behaviours"]
        b2 = next(b for b in behaviours if b["behaviour"] == "B2")
        return [f["n_units"] for f in b2["fields"]]

    assert set(b2_units(client.get(
        f"/api/v1/annotation/agreement?round_name={round_name}"))) == {0}

    for rater_id, facing in ((calibration_setup["rater1_id"], 0.5),
                             (calibration_setup["rater2_id"], 0.75)):
        response = client.post(
            "/api/v1/annotation/labels",
            json={
                "clip_id": clip.clip_id,
                "rater_id": rater_id,
                "behaviour": "B2",
                "labels": {
                    "b2_facing_proportion": facing,
                    "b2_head_torso_divergence": 1,
                    "b2_dominant": "class",
                    "b2_nonscorable": False,
                },
            },
        )
        assert response.status_code == 201

    after = b2_units(client.get(f"/api/v1/annotation/agreement?round_name={round_name}"))
    assert max(after) > 0, (
        "the round reports no units after two raters labelled a clip in it, so the labels "
        "never reached the round-filtered read")


def test_labels_refuses_a_malformed_clip_id(client, calibration_setup) -> None:
    """A clip_id that no ClipRef could have produced is refused, not filed unlinked.

    Accepting it would write an annotation that no round can ever see, which is a silent
    loss. 422 names the problem while the rater is still looking at the screen.
    """
    if not requires_database(client, calibration_setup):
        return

    response = client.post(
        "/api/v1/annotation/labels",
        json={
            "clip_id": "not-a-clip-id",
            "rater_id": calibration_setup["rater1_id"],
            "behaviour": "B1",
            "labels": {
                "b1_present": True,
                "b1_count": 2,
                "b1_amplitude": 2,
                "b1_nonscorable": False,
            },
        },
    )
    assert response.status_code == 422
    assert "not a clip id" in response.json()["detail"]


def test_agreement_refuses_a_round_holding_two_codebook_versions(
    client, calibration_setup, engine
) -> None:
    """A mixed-version round answers 409 with the reason, not 500.

    `praxis.annotation.irr.report` refuses to pool labels made under different codebook
    versions, because a revision changes what a label means and pooling reads that change as
    rater disagreement. CODEBOOK.md section 8 revises the codebook between calibration
    rounds, so this is a state the corpus genuinely reaches. Serving it as 500 reports the
    project's own invariant as a crash and hides the one sentence that explains what to do.
    """
    if not requires_database(client, calibration_setup, engine):
        return

    round_name = calibration_setup["round_name"]
    b1 = [a for a in calibration_setup["assignments"] if a.behaviour == "B1"]
    labels = {
        "b1_present": True,
        "b1_count": 2,
        "b1_amplitude": 2,
        "b1_nonscorable": False,
    }

    response = client.post(
        "/api/v1/annotation/labels",
        json={
            "clip_id": b1[0].clip.clip_id,
            "rater_id": b1[0].rater_id,
            "behaviour": "B1",
            "labels": labels,
        },
    )
    assert response.status_code == 201

    # A second label in the same round, stamped with a superseded codebook version. Written
    # directly because the API only ever stamps the active version, which is the point.
    stale = next(a for a in b1 if a.assignment_id != b1[0].assignment_id)
    with transaction(engine) as conn:
        conn.execute(
            annotations.insert().values(
                annotation_id=new_ulid(),
                assignment_id=stale.assignment_id,
                clip_id=stale.clip.clip_id,
                rater_id=stale.rater_id,
                behaviour="B1",
                codebook_version="v0.9-superseded",
                labels=labels,
                is_nonscorable=False,
                rater_confidence="certain",
                created_at=func.now(),
            ),
        )

    response = client.get(f"/api/v1/annotation/agreement?round_name={round_name}")
    assert response.status_code == 409
    body = response.json()
    assert body["type"] == "/errors/agreement-unavailable"
    assert "v0.9-superseded" in body["detail"]
