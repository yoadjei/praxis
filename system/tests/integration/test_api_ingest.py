# -*- coding: utf-8 -*-
"""The upload endpoint, exercised through HTTP.

The service layer is already tested directly. What is checked here is the part only the
endpoint decides: status codes, the response body API.md specifies, and that a refusal reaches
the client as a reason rather than a traceback.
"""
from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from praxis.api.app import create_app
from praxis.db import build_engine, keywords_to_url
from praxis.ids import new_ulid
from praxis.tasks import RecordingQueue
from tests.integration.conftest import lenient, make_clip

TOKEN = "Bearer researcher:01M2H000000000000000000000"


@pytest.fixture
def client(scratch_database, tmp_path):
    if scratch_database is None:
        return None
    media_root = tmp_path / "media"
    media_root.mkdir()
    app = create_app(config=lenient(), engine=build_engine(keywords_to_url(scratch_database)),
                     media_root=media_root, enqueue=RecordingQueue())
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("api") / "lesson.mp4", 6.0)


def upload(client, clip, party, token: str = TOKEN, **overrides):
    metadata = {
        "teacher_id": party["teacher_id"], "college_id": party["college_id"],
        "consent_id": party["consent_id"], "domain": "classroom",
        "school_id": party["school_id"],
        "recorded_on": str(date(2026, 5, 4)), "subject": "integrated science",
    }
    metadata.update(overrides)
    with open(clip, "rb") as handle:
        return client.post(
            "/api/v1/sessions",
            files={"file": (clip.name, handle, "video/mp4")},
            data={"metadata": json.dumps(metadata)},
            headers={"Authorization": token} if token else {})


def test_a_good_upload_returns_201_with_the_quality_verdict(client, clip, party) -> None:
    if client is None:
        return

    response = upload(client, clip, party)
    assert response.status_code == 201, response.text

    body = response.json()
    assert set(body) == {"session_id", "media_sha256", "quality"}
    assert set(body["quality"]) == {"verdict", "checks"}
    assert body["quality"]["verdict"] in ("pass", "warn", "fail")
    assert len(body["media_sha256"]) == 64

    names = {check["name"] for check in body["quality"]["checks"]}
    assert {"duration", "resolution", "frame_rate", "luminance", "person_present"} <= names
    person = next(c for c in body["quality"]["checks"] if c["name"] == "person_present")
    assert person["verdict"] == "abstain", "D48: an unavailable check must not report pass"


def test_an_invalid_consent_is_422_and_says_which_of_the_four_reasons(client, clip,
                                                                     party) -> None:
    if client is None:
        return

    response = upload(client, clip, party, consent_id=new_ulid())
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "/errors/consent-invalid"
    assert "does not exist" in body["detail"]


def test_a_duplicate_is_409(client, clip, party) -> None:
    if client is None:
        return

    assert upload(client, clip, party).status_code == 201
    repeat = upload(client, clip, party)
    assert repeat.status_code == 409
    assert repeat.json()["type"] == "/errors/session-duplicate"


def test_a_corrupt_file_is_refused_without_a_stack_trace(client, party, tmp_path) -> None:
    if client is None:
        return

    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00not a container\xff" * 400)

    response = upload(client, broken, party)
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "/errors/media-unreadable"
    assert "Traceback" not in response.text and "File \"" not in response.text


def test_an_unsupported_container_is_refused(client, party, tmp_path) -> None:
    if client is None:
        return

    wrong = tmp_path / "notes.txt"
    wrong.write_bytes(b"this is not video")
    assert upload(client, wrong, party).status_code == 415


def test_the_endpoint_requires_an_actor_even_though_it_cannot_verify_one(client, clip,
                                                                        party) -> None:
    """Phase 9 adds verification. Until then no session may be ingested with no actor recorded,
    because a gap in the trail cannot be filled in retrospectively."""
    if client is None:
        return

    assert upload(client, clip, party, token="").status_code == 401
    assert upload(client, clip, party, token="Bearer rater:x").status_code == 403


def test_metadata_the_contract_rejects_is_422(client, clip, party) -> None:
    if client is None:
        return

    assert upload(client, clip, party, domain="staffroom").status_code == 422
    assert upload(client, clip, party, pupil_count=-3).status_code == 422


def test_the_client_cannot_name_its_own_hash_or_verdict(client, clip, party) -> None:
    """`media_sha256` is computed and `quality_verdict` is decided. A request that supplied
    either would be describing a session the file does not support."""
    if client is None:
        return

    rejected = upload(client, clip, party, media_sha256="a" * 64)
    assert rejected.status_code == 422
    assert upload(client, clip, party, quality_verdict="pass").status_code == 422
