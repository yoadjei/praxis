# -*- coding: utf-8 -*-
"""GET/HEAD /api/v1/media/{session_id}/video - blurred video delivery for annotation.

The video endpoint serves the blurred copy of a session's video, enforcing D18 (faces blurred,
original deleted, only the blurred is reachable) and R1 (no identity leakage). It handles HTTP
Range requests for efficient seeking and enforces path traversal protection.
"""
from __future__ import annotations

import hashlib
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func

from praxis.api.app import create_app
from praxis.db import build_engine, keywords_to_url, transaction
from praxis.db.schema import media_objects, sessions
from praxis.ids import new_ulid
from praxis.tasks import RecordingQueue
from tests.integration.conftest import lenient, requires_database


@pytest.fixture
def client(scratch_database, tmp_path) -> TestClient | None:
    """A test client against a scratch database with a temporary media root."""
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


def test_unblurred_session_returns_404(client, engine) -> None:
    """A session whose is_blurred is false is refused with 404.

    D18 requires that the original video is deleted and only the blurred copy is reachable.
    A session with is_blurred=false must never serve its media.
    """
    if not requires_database(client, engine):
        return

    # Create a session with is_blurred=false.
    college_id = new_ulid()
    teacher_id = new_ulid()
    consent_id = new_ulid()
    session_id = new_ulid()
    media_sha256 = hashlib.sha256(session_id.encode()).hexdigest()

    with transaction(engine) as conn:
        from praxis.db.schema import colleges as colleges_table
        from praxis.db.schema import consent_records
        from praxis.db.schema import teachers as teachers_table

        conn.execute(
            colleges_table.insert().values(
                college_id=college_id,
                code=f"TEST-{college_id[-8:]}",
                created_at=func.now(),
            )
        )
        conn.execute(
            teachers_table.insert().values(
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
        conn.execute(
            media_objects.insert().values(
                media_sha256=media_sha256,
                relative_path=f"media/{media_sha256}.mp4",
                bytes=512,
                duration_s=5.0,
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

    response = client.get(f"/api/v1/media/{session_id}/video")

    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "/errors/not-blurred"
    assert "not blurred" in body["detail"]


def test_path_traversal_with_double_dot_is_refused(client, engine, tmp_path) -> None:
    """A blurred_relative_path containing .. is refused rather than served.

    This is the single most valuable test: path traversal attack prevention. Write a file
    outside media_root, point the column at it with .., and assert the endpoint does NOT
    return its contents.
    """
    if not requires_database(client, engine):
        return

    # Create a file outside the media root.
    outside_file = tmp_path / "outside_secret.mp4"
    secret_content = b"SECRET_OUTSIDE_MEDIA_ROOT"
    outside_file.write_bytes(secret_content)

    # Create a session pointing to that file via path traversal.
    college_id = new_ulid()
    teacher_id = new_ulid()
    consent_id = new_ulid()
    session_id = new_ulid()
    media_sha256 = hashlib.sha256(session_id.encode()).hexdigest()

    # Construct a traversal path: "media/foo/../../outside_secret.mp4"
    traversal_path = "media/foo/../../outside_secret.mp4"

    with transaction(engine) as conn:
        from praxis.db.schema import colleges as colleges_table
        from praxis.db.schema import consent_records
        from praxis.db.schema import teachers as teachers_table

        conn.execute(
            colleges_table.insert().values(
                college_id=college_id,
                code=f"TEST-{college_id[-8:]}",
                created_at=func.now(),
            )
        )
        conn.execute(
            teachers_table.insert().values(
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
        conn.execute(
            media_objects.insert().values(
                media_sha256=media_sha256,
                relative_path=f"media/{media_sha256}.mp4",
                bytes=len(secret_content),
                duration_s=5.0,
                width=1280,
                height=720,
                fps=25.0,
                has_audio=False,
                is_blurred=True,
                blurred_relative_path=traversal_path,
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

    response = client.get(f"/api/v1/media/{session_id}/video")

    # Must return 404, NOT the secret file contents.
    assert response.status_code == 404
    assert response.content != secret_content


def test_path_traversal_with_absolute_path_is_refused(client, engine, tmp_path) -> None:
    """An absolute path in blurred_relative_path is refused rather than served."""
    if not requires_database(client, engine):
        return

    # Create a file outside the media root.
    outside_file = tmp_path / "absolute_secret.mp4"
    secret_content = b"SECRET_ABSOLUTE"
    outside_file.write_bytes(secret_content)

    # Create a session with an absolute path.
    college_id = new_ulid()
    teacher_id = new_ulid()
    consent_id = new_ulid()
    session_id = new_ulid()
    media_sha256 = hashlib.sha256(session_id.encode()).hexdigest()

    # Use the absolute path as the traversal.
    absolute_path = str(outside_file)

    with transaction(engine) as conn:
        from praxis.db.schema import colleges as colleges_table
        from praxis.db.schema import consent_records
        from praxis.db.schema import teachers as teachers_table

        conn.execute(
            colleges_table.insert().values(
                college_id=college_id,
                code=f"TEST-{college_id[-8:]}",
                created_at=func.now(),
            )
        )
        conn.execute(
            teachers_table.insert().values(
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
        conn.execute(
            media_objects.insert().values(
                media_sha256=media_sha256,
                relative_path=f"media/{media_sha256}.mp4",
                bytes=len(secret_content),
                duration_s=5.0,
                width=1280,
                height=720,
                fps=25.0,
                has_audio=False,
                is_blurred=True,
                blurred_relative_path=absolute_path,
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

    response = client.get(f"/api/v1/media/{session_id}/video")

    # Must return 404, NOT the secret file contents.
    assert response.status_code == 404
    assert response.content != secret_content


def test_unknown_session_returns_404(client) -> None:
    """An unknown session id returns 404 with /errors/not-found."""
    if not requires_database(client):
        return

    unknown_id = new_ulid()
    response = client.get(f"/api/v1/media/{unknown_id}/video")

    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "/errors/not-found"
    assert "not found" in body["detail"]


def test_malformed_session_id_returns_404(client) -> None:
    """A malformed session id (not a ULID) is refused before reaching the filesystem."""
    if not requires_database(client):
        return

    malformed_id = "not-a-ulid!!!"
    response = client.get(f"/api/v1/media/{malformed_id}/video")

    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "/errors/not-found"


def test_deleted_file_returns_404_not_500(client, engine, tmp_path) -> None:
    """A session that exists and is blurred but whose file was deleted returns 404, not 500.

    Database says the file is there, but it was deleted on disk. Must return 404, not crash.
    """
    if not requires_database(client, engine):
        return

    # Create a session with a blurred file.
    college_id = new_ulid()
    teacher_id = new_ulid()
    consent_id = new_ulid()
    session_id = new_ulid()
    media_sha256 = hashlib.sha256(session_id.encode()).hexdigest()

    blurred_relative_path = f"media/{media_sha256[:2]}/{media_sha256}_blurred.mp4"
    media_dir = tmp_path / "media" / media_sha256[:2]
    media_dir.mkdir(parents=True, exist_ok=True)
    blurred_file = media_dir / f"{media_sha256}_blurred.mp4"
    test_content = b"TEMP_CONTENT"
    blurred_file.write_bytes(test_content)

    with transaction(engine) as conn:
        from praxis.db.schema import colleges as colleges_table
        from praxis.db.schema import consent_records
        from praxis.db.schema import teachers as teachers_table

        conn.execute(
            colleges_table.insert().values(
                college_id=college_id,
                code=f"TEST-{college_id[-8:]}",
                created_at=func.now(),
            )
        )
        conn.execute(
            teachers_table.insert().values(
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
        conn.execute(
            media_objects.insert().values(
                media_sha256=media_sha256,
                relative_path=f"media/{media_sha256}.mp4",
                bytes=len(test_content),
                duration_s=5.0,
                width=1280,
                height=720,
                fps=25.0,
                has_audio=False,
                is_blurred=True,
                blurred_relative_path=blurred_relative_path,
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

    # Delete the file.
    blurred_file.unlink()

    response = client.get(f"/api/v1/media/{session_id}/video")

    # Must return 404, not 500.
    assert response.status_code == 404
    body = response.json()
    # Its own type, not the generic "not found" a missing session gets. By the time a row says
    # is_blurred the unblurred original has been deleted (D18), so a blurred file that is not on
    # disk means the footage is gone rather than merely unindexed. An operator who reads "not
    # found" goes looking for an ingest bug; this sends them to look for a backup instead, and a
    # test that accepted either message would let the two be conflated again.
    assert body["type"] == "/errors/blurred-file-missing", (
        "losing the only remaining copy of a session must not report as a lookup miss")
    assert "not on disk" in body["detail"]


# ---------------------------------------------------------------------------
# Range requests. The player seeks to each clip's start, so a session is only
# annotatable if the server can serve the middle of a file without sending the
# front of it first. These assert on real bytes.
# ---------------------------------------------------------------------------

# Deliberately not a round number and not a multiple of the 65536 byte chunk size, so an
# off-by-one in the final chunk cannot hide behind a window that ends on a boundary.
BODY = bytes(range(256)) * 701 + b"TAIL"


def _blurred_session(engine, media_root, payload: bytes) -> str:
    """A session whose blurred file holds exactly `payload`. Returns the session id."""
    from praxis.db.schema import colleges as colleges_table
    from praxis.db.schema import consent_records
    from praxis.db.schema import teachers as teachers_table

    college_id, teacher_id, consent_id = new_ulid(), new_ulid(), new_ulid()
    session_id = new_ulid()
    media_sha256 = hashlib.sha256(session_id.encode()).hexdigest()
    relative = f".blurred/{session_id}.mp4"

    destination = media_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)

    with transaction(engine) as conn:
        conn.execute(colleges_table.insert().values(
            college_id=college_id, code=f"TEST-{college_id[-8:]}", created_at=func.now()))
        conn.execute(teachers_table.insert().values(
            teacher_id=teacher_id, college_id=college_id, created_at=func.now()))
        conn.execute(consent_records.insert().values(
            consent_id=consent_id, subject_type="teacher", teacher_id=teacher_id,
            college_id=college_id, purpose="research", recipients="supervisors",
            scope="both", granted_on=date(2025, 1, 1), document_ref="file/1",
            created_at=func.now()))
        conn.execute(media_objects.insert().values(
            media_sha256=media_sha256, relative_path=f"media/{media_sha256}.mp4",
            # Deliberately disagreeing with the real length. media_objects.bytes describes the
            # ORIGINAL, which D18 deleted, and the blurred re-encode is a different size. A
            # Content-Range computed from this column would describe a file nobody has.
            bytes=len(payload) + 123456,
            duration_s=5.0, width=1280, height=720, fps=25.0, has_audio=False,
            is_blurred=True, blurred_relative_path=relative, reachable=True,
            created_at=func.now()))
        conn.execute(sessions.insert().values(
            session_id=session_id, teacher_id=teacher_id, college_id=college_id,
            consent_id=consent_id, media_sha256=media_sha256, domain="microteaching",
            recorded_on=date(2025, 1, 1), quality_verdict="pass", quality_detail={},
            created_at=func.now()))
    return session_id


def test_no_range_serves_the_whole_file_and_advertises_range_support(
    client, engine, tmp_path,
) -> None:
    """Without a Range header the whole entity comes back, with Accept-Ranges.

    That header is what tells the player it may seek at all. Without it a browser downloads from
    the start to reach the second clip, which on a thirty-minute session is the difference
    between usable and unusable.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    response = client.get(f"/api/v1/media/{session_id}/video")

    assert response.status_code == 200
    assert response.content == BODY
    assert response.headers["accept-ranges"] == "bytes"


def test_a_range_returns_exactly_those_bytes(client, engine, tmp_path) -> None:
    """206, the requested window, and a total taken from the file rather than the database.

    The expected bytes are sliced from what the test wrote, so a reader that seeked to the wrong
    offset or returned one byte too few fails here. The total is checked against the real length
    because media_objects.bytes is set to something else on purpose.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    response = client.get(
        f"/api/v1/media/{session_id}/video", headers={"Range": "bytes=100-199"})

    assert response.status_code == 206
    assert response.content == BODY[100:200]
    assert len(response.content) == 100
    assert response.headers["content-range"] == f"bytes 100-199/{len(BODY)}"


def test_a_range_spanning_the_chunk_boundary_is_exact(client, engine, tmp_path) -> None:
    """A window longer than one read chunk returns every byte once, in order.

    The body is streamed in 65536 byte chunks, so a window crossing a boundary is where a
    mistake in the remaining-bytes arithmetic shows up: it would duplicate or drop a chunk.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    start, end = 65000, 131999
    response = client.get(
        f"/api/v1/media/{session_id}/video", headers={"Range": f"bytes={start}-{end}"})

    assert response.status_code == 206
    assert response.content == BODY[start:end + 1]


def test_an_open_ended_range_runs_to_the_end_of_the_file(client, engine, tmp_path) -> None:
    """A range written "bytes=N-" means from N to the last byte, which is how a player
    resumes."""
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    response = client.get(
        f"/api/v1/media/{session_id}/video", headers={"Range": "bytes=100-"})

    assert response.status_code == 206
    assert response.content == BODY[100:]
    assert response.headers["content-range"] == f"bytes 100-{len(BODY) - 1}/{len(BODY)}"


def test_a_suffix_range_returns_the_last_bytes(client, engine, tmp_path) -> None:
    """A range written "bytes=-N" means the final N bytes, which is how a player reads the
    index at the end of an mp4.

    cv2 writes that index last, so a player that cannot fetch the tail cannot work out the
    duration and will not seek at all.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    response = client.get(
        f"/api/v1/media/{session_id}/video", headers={"Range": "bytes=-4"})

    assert response.status_code == 206
    assert response.content == b"TAIL"


def test_a_range_past_the_end_is_refused_with_the_real_size(client, engine, tmp_path) -> None:
    """An unsatisfiable range gets 416 and a Content-Range naming the true length.

    416 rather than an empty 206: the client asked for bytes that do not exist, and the size in
    that header is what lets it ask again correctly.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    response = client.get(
        f"/api/v1/media/{session_id}/video",
        headers={"Range": f"bytes={len(BODY) + 10}-{len(BODY) + 20}"})

    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(BODY)}"


def test_a_range_whose_end_overruns_is_clamped_not_refused(client, engine, tmp_path) -> None:
    """A range starting inside the file and ending past it is satisfiable, and clamped.

    RFC 7233: if the first byte is in range the request can be satisfied, so the server sends
    what it has. Refusing it would break a player that optimistically asks for a fixed-size
    window at the end of the file.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    response = client.get(
        f"/api/v1/media/{session_id}/video",
        headers={"Range": f"bytes=100-{len(BODY) + 5000}"})

    assert response.status_code == 206
    assert response.content == BODY[100:]
    assert response.headers["content-range"] == f"bytes 100-{len(BODY) - 1}/{len(BODY)}"


def test_an_unintelligible_range_is_refused_and_never_served_as_a_whole_file(
    client, engine, tmp_path,
) -> None:
    """A Range header that cannot be parsed gets 400, and crucially not a silent 200.

    This records an inherited deviation rather than an endorsement. RFC 7233 says an
    unintelligible Range should be ignored and the whole entity served; starlette answers 400.
    The behaviour is asserted as it is, because the alternative is hand-rolling range handling
    again to win a point no browser tests - and that duplicate is what previously made HEAD
    stream the entire file.

    What the test is really protecting is the pair: whatever the status, a malformed range must
    never be answered with 206 and a window the client did not ask for, which would hand a
    player the wrong bytes and leave it to render them as video.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    for header in ("items=0-10", "bytes=abc-def", "nonsense"):
        response = client.get(
            f"/api/v1/media/{session_id}/video", headers={"Range": header})
        assert response.status_code == 400, header
        assert response.status_code != 206, header


def test_a_multipart_range_is_honoured_or_refused_but_never_truncated(
    client, engine, tmp_path,
) -> None:
    """Two ranges in one header return a multipart body or a refusal, never one range silently.

    A server that answered "bytes=0-9, 20-29" with just the first window would give the player
    ten bytes while telling it the request succeeded, and the player would treat the gap as
    decodable video.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)
    response = client.get(
        f"/api/v1/media/{session_id}/video", headers={"Range": "bytes=0-9, 20-29"})

    if response.status_code == 206:
        assert "multipart/byteranges" in response.headers["content-type"], (
            "a 206 answering two ranges must say it is multipart, not pass off one window "
            "as the whole answer")
    else:
        assert response.status_code in (400, 416)


def test_head_returns_the_headers_without_the_body(client, engine, tmp_path) -> None:
    """HEAD answers with the length and no content, for both a plain and a range request.

    A HEAD that streamed the body would send the whole window in answer to a metadata question -
    gigabytes, on the sessions this system actually holds.
    """
    if not requires_database(client, engine):
        return

    session_id = _blurred_session(engine, tmp_path / "media", BODY)

    plain = client.head(f"/api/v1/media/{session_id}/video")
    assert plain.status_code == 200
    assert plain.content == b""
    assert plain.headers["content-length"] == str(len(BODY))
    assert plain.headers["accept-ranges"] == "bytes"

    ranged = client.head(
        f"/api/v1/media/{session_id}/video", headers={"Range": "bytes=10-19"})
    assert ranged.status_code == 206
    assert ranged.content == b""
    assert ranged.headers["content-range"] == f"bytes 10-19/{len(BODY)}"


def test_no_response_names_the_file_on_disk(client, engine, tmp_path) -> None:
    """R1 and D76. Neither the headers nor the body may carry the stored path.

    The session id is already known to the caller; the path is not, and it is derived from the
    media hash. A Content-Disposition naming the file would put it in every download.
    """
    if not requires_database(client, engine):
        return

    media_root = tmp_path / "media"
    session_id = _blurred_session(engine, media_root, BODY)
    relative = f".blurred/{session_id}.mp4"

    for response in (
        client.get(f"/api/v1/media/{session_id}/video"),
        client.get(f"/api/v1/media/{session_id}/video", headers={"Range": "bytes=0-9"}),
        client.head(f"/api/v1/media/{session_id}/video"),
    ):
        headers = " ".join(f"{k}: {v}" for k, v in response.headers.items())
        assert "content-disposition" not in response.headers
        assert relative not in headers
        assert str(media_root) not in headers
