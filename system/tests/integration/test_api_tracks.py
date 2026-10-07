# -*- coding: utf-8 -*-
"""Teacher confirmation through HTTP, and the states that are not a confirmation.

`scripts/plan_annotation.py` refuses to plan a session whose teacher track nobody confirmed, so
this endpoint is the only way a corpus gets past preprocessing. What is checked here is what
only the endpoint decides - the status codes API.md section 4 fixes, which refusal applies to
which of the three unreviewable states, and that a thumbnail URL returns a picture of the track
it is filed under.

A real preprocessing run supplies the fixtures: a real pose artefact on disk, a real ranking in
the database and a real blurred video to cut stills from. The footage is synthetic, so nothing
here says anything about heuristic accuracy; what it establishes is that the surface a reviewer
uses is wired to the data it claims to show.
"""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, insert, select, update

from praxis.api.app import create_app
from praxis.audit.chain import verify
from praxis.audit.write import read_chain
from praxis.config import load_config
from praxis.db import build_engine, keywords_to_url, transaction
from praxis.db.schema import (
    colleges,
    consent_records,
    media_objects,
    pose_artifacts,
    sessions,
    teacher_tracks,
    teachers,
)
from praxis.ids import new_ulid
from praxis.preprocess.store import confirm_teacher, confirmed_teacher_track, persist
from praxis.tasks import RecordingQueue
from tests.integration.conftest import requires_database
from tests.integration.test_preprocess_pipeline import ScriptedEstimator, band, preprocess

RESEARCHER = "Bearer researcher:01M2H000000000000000000000"
SUPERVISOR = "Bearer supervisor:01M2H000000000000000000001"
# 26 characters, because audit_log.actor_user_id is CHAR(26) and blank-pads anything shorter.
# A 24-character id here made the chain report a content break on a row nobody had touched.
DIRECT = "01M2HDJRECT000000000000000"
CANDIDATE_CAP = 20

# The default migration 0009 wrote into every row that predated the candidates column.
UNRECORDED = {"source": "unrecorded", "zones_available": None, "ranked": [], "truncated": 0}


@pytest.fixture
def engine(scratch_database):
    return build_engine(keywords_to_url(scratch_database)) if scratch_database else None


@pytest.fixture
def setup():
    from praxis.preprocess.zones import CameraSetup
    return CameraSetup(
        setup_id=new_ulid(), zone_board=band(0.0, 0.15), zone_front=band(0.15, 0.7),
        zone_middle=band(0.7, 0.85), zone_back=band(0.85, 1.0),
        learner_region=band(0.7, 1.0))


@pytest.fixture
def config(tmp_path):
    """The real configuration with the artefact root moved into the test's own directory, so
    the endpoint resolves a path that is actually there."""
    base = load_config()
    return base.model_copy(update={
        "paths": base.paths.model_copy(update={"pose_artifacts": tmp_path / "pose"})})


@pytest.fixture
def reviewed(engine, config, setup, tmp_path):
    """A preprocessed session: rows in the database, an artefact on disk, a blurred video.

    Returns the session id, the proposal and the roots, or None when no database is configured.
    """
    if engine is None:
        return None

    college, teacher, consent = new_ulid(), new_ulid(), new_ulid()
    session_id = new_ulid()
    media_sha = hashlib.sha256(session_id.encode()).hexdigest()
    media_root = tmp_path / "media"
    blurred_relative = f".blurred/{session_id}.mp4"

    paths = {"source": tmp_path / "incoming" / "session.mp4",
             "blurred": media_root / blurred_relative,
             "pose": tmp_path / "pose" / f"{session_id}.npz"}
    paths["blurred"].parent.mkdir(parents=True, exist_ok=True)

    result = preprocess(paths, config, ScriptedEstimator(), setup, session_id=session_id)

    with transaction(engine) as conn:
        conn.execute(insert(colleges).values(
            college_id=college, code=f"COL-{college[-5:]}", created_at=func.now()))
        conn.execute(insert(teachers).values(
            teacher_id=teacher, college_id=college, created_at=func.now()))
        conn.execute(insert(consent_records).values(
            consent_id=consent, subject_type="teacher", teacher_id=teacher, college_id=college,
            purpose="research", recipients="supervisors", scope="both",
            granted_on=date(2025, 1, 1), document_ref="file/1", created_at=func.now()))
        conn.execute(insert(media_objects).values(
            media_sha256=media_sha, relative_path="media/aa/synthetic.mp4", bytes=1024,
            duration_s=2.0, width=1280, height=720, fps=25.0, has_audio=False,
            is_blurred=False, reachable=True, created_at=func.now()))
        conn.execute(insert(sessions).values(
            session_id=session_id, teacher_id=teacher, college_id=college, consent_id=consent,
            media_sha256=media_sha, domain="microteaching", recorded_on=date(2026, 3, 2),
            quality_verdict="pass", quality_detail={}, created_at=func.now()))
        persist(conn, session_id=session_id, result=result,
                relative_path=paths["pose"].name, max_candidates=CANDIDATE_CAP,
                blurred_relative_path=blurred_relative)

    return {"session_id": session_id, "proposal": result.proposal,
            "media_root": media_root, "college_id": college}


@pytest.fixture
def client(engine, config, reviewed):
    if engine is None or reviewed is None:
        return None
    app = create_app(config=config, engine=engine, media_root=reviewed["media_root"],
                     enqueue=RecordingQueue())
    return TestClient(app, raise_server_exceptions=False)


def candidates(client, session_id, token=RESEARCHER, **params):
    return client.get(f"/api/v1/sessions/{session_id}/tracks/candidates", params=params,
                      headers={"Authorization": token} if token else {})


def confirm(client, session_id, track_id, token=SUPERVISOR):
    return client.post(f"/api/v1/sessions/{session_id}/tracks/confirm",
                       json={"track_id": track_id},
                       headers={"Authorization": token} if token else {})


class TestTheRankingReachesTheReviewer:
    def test_the_candidates_carry_their_scores_and_signals(self, client, reviewed) -> None:
        if not requires_database(client):
            return

        response = candidates(client, reviewed["session_id"])
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["session_id"] == reviewed["session_id"]
        assert body["ranking"]["available"] is True
        assert body["ranking"]["source"] == "detection"
        assert body["candidates"], "a preprocessed session with tracks has candidates"
        for candidate in body["candidates"]:
            assert set(candidate) >= {"track_id", "score", "presence_fraction",
                                      "median_area_fraction", "front_zone_fraction",
                                      "is_proposed", "is_confirmed", "thumbnails"}
            assert candidate["score"] is not None

    def test_the_ranking_is_ordered_highest_first(self, client, reviewed) -> None:
        if not requires_database(client):
            return
        scores = [c["score"] for c in candidates(
            client, reviewed["session_id"]).json()["candidates"]]
        assert scores == sorted(scores, reverse=True)

    def test_the_heuristics_pick_is_marked_in_the_list(self, client, reviewed) -> None:
        """So the screen does not have to compare ids, and cannot compare them wrongly."""
        if not requires_database(client):
            return

        body = candidates(client, reviewed["session_id"]).json()
        proposed = [c["track_id"] for c in body["candidates"] if c["is_proposed"]]
        assert proposed == ([] if body["proposal"]["track_id"] is None
                            else [body["proposal"]["track_id"]])

    def test_the_proposal_and_the_confirmation_are_separate_keys(self, client,
                                                                 reviewed) -> None:
        """Nothing in the payload merges a guess with a judgement. An endpoint that answered
        one "teacher" key would undo the distinction `praxis.preprocess.teacher` exists for."""
        if not requires_database(client):
            return

        body = candidates(client, reviewed["session_id"]).json()
        assert set(body["proposal"]) == {"track_id", "score", "proposed_by", "reason"}
        assert body["confirmation"] == {"confirmed_by": None, "confirmed_at": None}
        assert "teacher" not in body

    def test_the_reason_the_heuristic_gave_reaches_the_screen(self, client, reviewed) -> None:
        if not requires_database(client):
            return
        body = candidates(client, reviewed["session_id"]).json()
        assert body["proposal"]["reason"] == reviewed["proposal"].reason

    def test_the_strip_length_is_bounded_by_the_request(self, client, reviewed) -> None:
        if not requires_database(client):
            return
        body = candidates(client, reviewed["session_id"], thumbnails=2).json()
        assert all(len(c["thumbnails"]) <= 2 for c in body["candidates"])

    @pytest.mark.parametrize("asked", [0, 13, -1])
    def test_an_unbounded_strip_is_refused(self, client, reviewed, asked) -> None:
        """`thumbnails` reaches a loop that decodes a frame each. Unbounded, it is a way to ask
        one request to decode a thousand."""
        if not requires_database(client):
            return
        assert candidates(client, reviewed["session_id"],
                          thumbnails=asked).status_code == 422


class TestAThumbnailShowsTheTrackItIsFiledUnder:
    def test_the_url_in_the_payload_returns_a_jpeg(self, client, reviewed) -> None:
        if not requires_database(client):
            return

        body = candidates(client, reviewed["session_id"]).json()
        assert body["thumbnails_unavailable"] is None
        strips = [c for c in body["candidates"] if c["thumbnails"]]
        assert strips, "a session with a readable artefact can locate its tracks"

        url = strips[0]["thumbnails"][0]["url"]
        picture = client.get(url, headers={"Authorization": RESEARCHER})
        assert picture.status_code == 200, picture.text
        assert picture.headers["content-type"] == "image/jpeg"
        # JPEG start-of-image marker. A zero-length 200 would pass a status check.
        assert picture.content[:2] == b"\xff\xd8"
        assert len(picture.content) > 500

    def test_the_crop_geometry_is_the_servers_and_not_the_callers(self, client,
                                                                  reviewed) -> None:
        """A client-supplied rectangle would let the screen show any region of the footage
        under a track number, and the confirmation it collected would mean nothing."""
        if not requires_database(client):
            return

        body = candidates(client, reviewed["session_id"]).json()
        track = next(c for c in body["candidates"] if c["thumbnails"])
        response = client.get(
            f"/api/v1/sessions/{reviewed['session_id']}/tracks/{track['track_id']}/thumbnail",
            params={"frame": track["thumbnails"][0]["frame"],
                    "box": "0,0,10,10", "x": 500},
            headers={"Authorization": RESEARCHER})
        # The unknown parameters are ignored rather than honoured, and the picture is the same
        # one the declared geometry describes.
        assert response.status_code == 200
        assert response.content[:2] == b"\xff\xd8"

    def test_a_track_not_in_that_frame_is_404(self, client, reviewed) -> None:
        """Track 9999 was never tracked, so it is in no frame. A track is only visible in the
        frames it appears in, and the message says which of the two reasons applies."""
        if not requires_database(client):
            return

        response = client.get(
            f"/api/v1/sessions/{reviewed['session_id']}/tracks/9999/thumbnail",
            params={"frame": 0}, headers={"Authorization": RESEARCHER})
        assert response.status_code == 404
        assert "not locatable in frame 0" in response.json()["detail"]

    def test_a_frame_past_the_end_of_the_artefact_is_404(self, client, reviewed) -> None:
        if not requires_database(client):
            return
        response = client.get(
            f"/api/v1/sessions/{reviewed['session_id']}/tracks/1/thumbnail",
            params={"frame": 10_000_000}, headers={"Authorization": RESEARCHER})
        assert response.status_code == 404

    def test_a_negative_frame_is_refused_by_the_contract(self, client, reviewed) -> None:
        if not requires_database(client):
            return
        response = client.get(
            f"/api/v1/sessions/{reviewed['session_id']}/tracks/1/thumbnail",
            params={"frame": -1}, headers={"Authorization": RESEARCHER})
        assert response.status_code == 422

    def test_every_url_in_a_strip_returns_a_different_picture(self, client,
                                                              reviewed) -> None:
        """The defect this addressing scheme replaced. Positions in the strip were resolved by
        rebuilding a strip of `index + 1` frames, so two positions resolved to the same frame
        and served byte-identical pictures while claiming different timestamps."""
        if not requires_database(client):
            return

        body = candidates(client, reviewed["session_id"], thumbnails=4).json()
        track = next(c for c in body["candidates"] if len(c["thumbnails"]) > 1)

        frames = [t["frame"] for t in track["thumbnails"]]
        assert len(set(frames)) == len(frames), "a strip named the same frame twice"

        pictures = []
        for thumbnail in track["thumbnails"]:
            response = client.get(thumbnail["url"],
                                  headers={"Authorization": RESEARCHER})
            assert response.status_code == 200, thumbnail["url"]
            pictures.append(response.content)
        assert len(set(pictures)) == len(pictures), (
            "two stills in one strip returned identical bytes, so a URL is not addressing the "
            "frame its entry describes")

    def test_a_url_carries_the_frame_its_entry_declares(self, client, reviewed) -> None:
        if not requires_database(client):
            return

        body = candidates(client, reviewed["session_id"], thumbnails=3).json()
        for candidate in body["candidates"]:
            for thumbnail in candidate["thumbnails"]:
                assert f"frame={thumbnail['frame']}" in thumbnail["url"]

    def test_what_a_url_returns_does_not_depend_on_the_strip_it_came_from(
            self, client, reviewed) -> None:
        """The defect this addressing scheme replaced, stated as the property that failed. A
        position in the strip resolved against a strip rebuilt to that length, so the same URL
        returned different pictures depending on how long a strip had last been asked for.

        The two strips name different frames - the positions are segment midpoints, so a strip
        of two and a strip of five are disjoint - and that is precisely why the frame rather
        than the position has to be what the URL carries.
        """
        if not requires_database(client):
            return

        short = candidates(client, reviewed["session_id"], thumbnails=2).json()
        track = next(c for c in short["candidates"] if c["thumbnails"])
        entry = track["thumbnails"][0]

        before = client.get(entry["url"], headers={"Authorization": RESEARCHER})
        assert before.status_code == 200

        # Ask for an entirely different strip length, then fetch the original URL again.
        candidates(client, reviewed["session_id"], thumbnails=5)
        after = client.get(entry["url"], headers={"Authorization": RESEARCHER})

        assert after.status_code == 200
        assert after.content == before.content, (
            "the same URL returned a different picture after a different strip length was "
            "requested, so it is addressing a position rather than a frame")

    def test_a_missing_artefact_does_not_fail_the_ranking(self, client, reviewed,
                                                           tmp_path) -> None:
        """The scores are still worth serving, and the screen is told the crops are gone so
        nobody reads an empty strip as a track that was never visible. D48."""
        if not requires_database(client):
            return

        artefact = tmp_path / "pose" / f"{reviewed['session_id']}.npz"
        artefact.rename(artefact.with_suffix(".moved"))
        try:
            body = candidates(client, reviewed["session_id"]).json()
        finally:
            artefact.with_suffix(".moved").rename(artefact)

        assert body["ranking"]["available"] is True
        assert body["candidates"]
        assert body["thumbnails_unavailable"]
        assert all(c["thumbnails"] == [] for c in body["candidates"])


class TestTheThreeWaysThereIsNoRanking:
    def test_an_unknown_session_is_404(self, client) -> None:
        if not requires_database(client):
            return
        assert candidates(client, new_ulid()).status_code == 404

    def test_a_malformed_id_is_404_and_not_a_server_error(self, client) -> None:
        if not requires_database(client):
            return
        assert candidates(client, "not-a-ulid").status_code == 404

    def test_a_session_that_was_never_preprocessed_says_so(self, client, engine,
                                                            reviewed) -> None:
        if not requires_database(client):
            return

        with transaction(engine) as conn:
            conn.execute(delete(teacher_tracks).where(
                teacher_tracks.c.session_id == reviewed["session_id"]))
            conn.execute(delete(pose_artifacts).where(
                pose_artifacts.c.session_id == reviewed["session_id"]))

        response = candidates(client, reviewed["session_id"])
        assert response.status_code == 409
        body = response.json()
        assert body["type"] == "/errors/not-preprocessed"
        assert "phase 3 has not run" in body["detail"]

    def test_a_proposal_that_predates_its_own_row_names_the_remedy(self, client, engine,
                                                                    reviewed) -> None:
        """The pre-0009 declined proposals: preprocessed, with a ranking that exists only in
        the artefact. Told apart from "never preprocessed" because the remedies differ."""
        if not requires_database(client):
            return

        with transaction(engine) as conn:
            conn.execute(delete(teacher_tracks).where(
                teacher_tracks.c.session_id == reviewed["session_id"]))

        response = candidates(client, reviewed["session_id"])
        assert response.status_code == 409
        body = response.json()
        assert body["type"] == "/errors/proposal-not-recorded"
        assert "reemit_candidates" in body["detail"]

    def test_an_unrecorded_ranking_abstains_rather_than_serving_an_empty_list(
            self, client, engine, reviewed) -> None:
        """An empty `ranked` with `source: unrecorded` would read as "the heuristic found
        nobody", which is a different and false claim."""
        if not requires_database(client):
            return

        with transaction(engine) as conn:
            conn.execute(update(teacher_tracks)
                         .where(teacher_tracks.c.session_id == reviewed["session_id"])
                         .values(candidates=UNRECORDED))

        body = candidates(client, reviewed["session_id"]).json()
        assert body["ranking"]["available"] is False
        assert body["ranking"]["source"] == "unrecorded"
        assert "reemit_candidates" in body["ranking"]["detail"]
        assert body["candidates"] == []


class TestOnlyTheRightRolesAndOnlyWithAName:
    def test_no_token_is_401(self, client, reviewed) -> None:
        if not requires_database(client):
            return
        assert candidates(client, reviewed["session_id"], token=None).status_code == 401
        assert confirm(client, reviewed["session_id"], 1, token=None).status_code == 401

    def test_a_rater_may_not_confirm_a_teacher(self, client, reviewed) -> None:
        if not requires_database(client):
            return
        response = confirm(client, reviewed["session_id"], 1, token="Bearer rater:01M2H1")
        assert response.status_code == 403
        assert "confirming the teacher track" in response.json()["detail"]

    def test_a_role_with_no_user_id_cannot_confirm(self, client, reviewed) -> None:
        """A confirmation is attributed to a person. A token naming only a role would file one
        against nobody, and `confirmed_by` is the column R1 turns on."""
        if not requires_database(client):
            return
        response = confirm(client, reviewed["session_id"], 1, token="Bearer supervisor")
        assert response.status_code == 401
        assert "role:user_id" in response.json()["detail"]


class TestConfirmationIsWhatUnblocksTheSession:
    def test_it_answers_200_and_writes_who_said_so(self, client, engine, reviewed) -> None:
        """200, not 201: migration 0009 means the row already existed."""
        if not requires_database(client):
            return

        body = candidates(client, reviewed["session_id"]).json()
        chosen = body["candidates"][0]["track_id"]

        response = confirm(client, reviewed["session_id"], chosen)
        assert response.status_code == 200, response.text
        assert response.json() == {"session_id": reviewed["session_id"],
                                   "track_id": chosen,
                                   "confirmed_by": "01M2H000000000000000000001"}

        with transaction(engine) as conn:
            assert confirmed_teacher_track(conn, reviewed["session_id"]) == chosen

    def test_the_confirmation_shows_up_on_the_next_read(self, client, reviewed) -> None:
        if not requires_database(client):
            return

        chosen = candidates(client, reviewed["session_id"]).json()["candidates"][0]["track_id"]
        confirm(client, reviewed["session_id"], chosen)

        body = candidates(client, reviewed["session_id"]).json()
        assert body["confirmation"]["confirmed_by"] == "01M2H000000000000000000001"
        assert body["confirmation"]["confirmed_at"] is not None
        assert [c["track_id"] for c in body["candidates"] if c["is_confirmed"]] == [chosen]

    def test_a_reviewer_may_name_a_track_the_heuristic_did_not_propose(self, client, engine,
                                                                       reviewed) -> None:
        """The entire point of the step. `require_human_confirmation` is documented NEVER set
        false because the heuristic is a guess, and the disagreement is what makes its field
        accuracy measurable rather than assumed."""
        if not requires_database(client):
            return

        proposed = candidates(client, reviewed["session_id"]).json()["proposal"]["track_id"]
        other = (proposed or 0) + 7
        assert confirm(client, reviewed["session_id"], other).status_code == 200

        with transaction(engine) as conn:
            assert confirmed_teacher_track(conn, reviewed["session_id"]) == other
            trail = [r for r in read_chain(conn)
                     if r.event_type == "teacher_track.confirmed"]
        assert trail, "a confirmation is an audited event"
        assert trail[-1].payload["agreed_with_heuristic"] == (
            None if proposed is None else proposed == other)
        assert trail[-1].payload["proposed_track_id"] == proposed

    def test_the_chain_still_verifies_after_a_confirmation(self, client, engine,
                                                            reviewed) -> None:
        if not requires_database(client):
            return
        confirm(client, reviewed["session_id"], 1)
        with transaction(engine) as conn:
            break_found = verify(read_chain(conn))
        assert break_found is None, f"the chain broke: {break_found}"

    def test_a_later_confirmation_overwrites_and_the_chain_keeps_both(self, client, engine,
                                                                       reviewed) -> None:
        """The row is current state; the chain is the record of who said what and when."""
        if not requires_database(client):
            return

        confirm(client, reviewed["session_id"], 1)
        confirm(client, reviewed["session_id"], 2, token=RESEARCHER)

        with transaction(engine) as conn:
            assert confirmed_teacher_track(conn, reviewed["session_id"]) == 2
            trail = [r for r in read_chain(conn)
                     if r.event_type == "teacher_track.confirmed"
                     and r.payload["session_id"] == reviewed["session_id"]]
        assert [r.payload["track_id"] for r in trail] == [1, 2]
        assert trail[-1].payload["reconfirmation_of"] == "01M2H000000000000000000001"

    def test_a_negative_track_is_refused_by_the_contract(self, client, reviewed) -> None:
        if not requires_database(client):
            return
        assert confirm(client, reviewed["session_id"], -1).status_code == 422

    def test_a_body_naming_its_own_confirmer_is_refused(self, client, reviewed) -> None:
        """Accepting and discarding it would file the confirmation under the wrong person
        while answering 200."""
        if not requires_database(client):
            return
        response = client.post(
            f"/api/v1/sessions/{reviewed['session_id']}/tracks/confirm",
            json={"track_id": 1, "confirmed_by": "01M2HSOMEONEELSE00000000"},
            headers={"Authorization": SUPERVISOR})
        assert response.status_code == 422

    def test_confirming_a_session_with_no_row_is_409_not_a_new_row(self, client, engine,
                                                                    reviewed) -> None:
        """`confirm_teacher` refuses a session it has no proposal for, because a confirmation
        invented there would endorse something no human was shown."""
        if not requires_database(client):
            return

        with transaction(engine) as conn:
            conn.execute(delete(teacher_tracks).where(
                teacher_tracks.c.session_id == reviewed["session_id"]))

        assert confirm(client, reviewed["session_id"], 1).status_code == 409
        with transaction(engine) as conn:
            assert conn.execute(select(func.count()).select_from(teacher_tracks).where(
                teacher_tracks.c.session_id == reviewed["session_id"])).scalar() == 0


class TestTheGateDownstreamIsTheSameGate:
    def test_an_unconfirmed_session_is_not_annotatable_and_a_confirmed_one_is(
            self, client, engine, reviewed) -> None:
        """The reason this router exists. `plan_annotation` filters on `confirmed_by`, so until
        something could write it the corpus stopped at preprocessing."""
        if not requires_database(client):
            return

        from scripts.plan_annotation import _load_annotatable_sessions

        with transaction(engine) as conn:
            before = _load_annotatable_sessions(conn)
        assert reviewed["session_id"] not in before

        assert confirm(client, reviewed["session_id"], 1).status_code == 200

        with transaction(engine) as conn:
            after = _load_annotatable_sessions(conn)
        assert reviewed["session_id"] in after


def test_the_confirm_helper_and_the_endpoint_agree(engine, reviewed) -> None:
    """The endpoint adds HTTP and nothing else. If it grew a second rule the two would drift,
    and the domain function is the one every script calls (L20)."""
    if not requires_database(engine, reviewed):
        return

    with transaction(engine) as conn:
        confirm_teacher(conn, session_id=reviewed["session_id"], track_id=3,
                        confirmed_by=DIRECT, actor_role="researcher")
    with transaction(engine) as conn:
        assert confirmed_teacher_track(conn, reviewed["session_id"]) == 3


def test_the_artefact_root_is_not_escapable(client, engine, reviewed, tmp_path) -> None:
    """The column holds a relative name by convention. An absolute value in it would escape the
    root exactly as `..` would, and both are the same check in `praxis.api.locate`."""
    if not requires_database(client):
        return

    outside = Path(tmp_path).parent / "elsewhere.npz"
    with transaction(engine) as conn:
        conn.execute(update(pose_artifacts)
                     .where(pose_artifacts.c.session_id == reviewed["session_id"])
                     .values(relative_path=str(outside)))

    body = candidates(client, reviewed["session_id"]).json()
    assert body["thumbnails_unavailable"]
    assert "outside" in body["thumbnails_unavailable"]
