# -*- coding: utf-8 -*-
"""What preprocessing leaves in the database, and what it must never leave there.

BUILD-SPEC Phase 3's first acceptance test is about database state after a real preprocessing
run, not about source code, so these run a session through the pipeline and then look at the
tables. Without `PRAXIS_TEST_DSN` there is no database to look at; each test then degrades to a
stated weaker check that still runs and still asserts, because "the database test did not run"
and "the database test passed" must never look alike.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import DBAPIError

from praxis.audit.chain import verify
from praxis.audit.write import read_chain
from praxis.db import build_engine, keywords_to_url, transaction
from praxis.db.schema import (
    colleges,
    consent_records,
    learner_aggregates,
    media_objects,
    pose_artifacts,
    sessions,
    teacher_tracks,
    teachers,
)
from praxis.ids import new_ulid
from praxis.preprocess.pose import Detection
from praxis.preprocess.store import (
    PersistError,
    confirm_teacher,
    confirmed_teacher_track,
    persist,
    save_camera_setup,
)
from praxis.vocabulary import TRACK_ORIGINS
from tests.integration.conftest import HEIGHT, requires_database
from tests.integration.test_preprocess_pipeline import (
    ScriptedEstimator,
    _keypoints,
    band,
    preprocess,
)

# Names a per-learner table would have if anybody ever added one. R1.
FORBIDDEN_TABLES = ("learner_tracks", "pupil_tracks", "student_tracks")
IDENTIFYING_COLUMNS = ("learner_id", "pupil_id", "student_id", "learner_track_id")

# How many ranked candidates these tests ask to be stored. A literal rather than the configured
# value, because what is under test here is that the cap is applied and its remainder counted,
# not what the deployment sets it to.
CANDIDATE_CAP = 20


@pytest.fixture
def engine(scratch_database):
    return build_engine(keywords_to_url(scratch_database)) if scratch_database else None


@pytest.fixture
def session_row(engine):
    """A committed session with its foreign keys, so the preprocessing rows have something to
    hang off. Returns None when no database is configured."""
    if engine is None:
        return None

    college, teacher, consent = new_ulid(), new_ulid(), new_ulid()
    session_id = new_ulid()
    # Derived from the session, because the scratch database outlives a single test and a fixed
    # digest would collide with the previous test's media row.
    media_sha = hashlib.sha256(session_id.encode()).hexdigest()
    with transaction(engine) as conn:
        conn.execute(insert(colleges).values(
            college_id=college, code=f"COL-{college[-5:]}", created_at=func.now()))
        conn.execute(insert(teachers).values(
            teacher_id=teacher, college_id=college, created_at=func.now()))
        conn.execute(insert(consent_records).values(
            consent_id=consent, subject_type="teacher", teacher_id=teacher, college_id=college,
            purpose="research", recipients="supervisors", scope="both",
            granted_on=date(2025, 1, 1), document_ref="file/1", created_at=func.now()))
        # `is_blurred` true has to come with the path to the blurred file, which the CHECK
        # constraint `media_objects_blurred_has_a_path` enforces. A row claiming a blurred file
        # exists without saying where leaves the only surviving copy unreachable: the original
        # is deleted by then (D18), so a dangling pointer loses the session.
        conn.execute(insert(media_objects).values(
            media_sha256=media_sha, relative_path="media/aa/synthetic.mp4", bytes=1024,
            duration_s=2.0, width=1280, height=720, fps=25.0, has_audio=False,
            is_blurred=True, blurred_relative_path=f".blurred/{session_id}.mp4",
            reachable=True, created_at=func.now()))
        conn.execute(insert(sessions).values(
            session_id=session_id, teacher_id=teacher, college_id=college, consent_id=consent,
            media_sha256=media_sha, domain="microteaching", recorded_on=date(2026, 3, 2),
            quality_verdict="pass", quality_detail={}, created_at=func.now()))
    return {"session_id": session_id, "college_id": college, "teacher_id": teacher,
            "user_id": new_ulid()}


@pytest.fixture
def setup():
    from praxis.preprocess.zones import CameraSetup
    return CameraSetup(
        setup_id=new_ulid(), zone_board=band(0.0, 0.15), zone_front=band(0.15, 0.7),
        zone_middle=band(0.7, 0.85), zone_back=band(0.85, 1.0),
        learner_region=band(0.7, 1.0))


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return {"source": tmp_path / "incoming" / "session.mp4",
            "blurred": tmp_path / "media" / "session.blurred.mp4",
            "pose": tmp_path / "pose" / "session.npz"}


def preprocessed(paths, config, setup, session_row=None):
    session_id = session_row["session_id"] if session_row else None
    if session_id is None:
        return preprocess(paths, config, ScriptedEstimator(), setup)
    return preprocess(paths, config, ScriptedEstimator(), setup, session_id=session_id)


class FaintEstimator:
    """One small person, seated at the back, visible for a fifth of the session.

    Built to be declined rather than to be wrong: low presence, small area, and no time in the
    front zone, so the weighted score lands under the floor on every term it has. This is the
    case the corpus actually contains - three of its seven sessions clear nothing - and until
    0009 it produced no row at all, so none of it could be asserted against a database.
    """

    model_version = "scripted-faint-v1"

    def __init__(self, visible_in_one_of: int = 5) -> None:
        self.calls = 0
        self.visible_in_one_of = visible_in_one_of

    def detect(self, frame: np.ndarray) -> list[Detection]:
        index = self.calls
        self.calls += 1
        if index % self.visible_in_one_of:
            return []
        # In the back band, which the `setup` fixture puts below 0.85 of the frame height.
        x, y, w, h = 500.0, 0.88 * HEIGHT, 40.0, 60.0
        return [Detection(bbox=(x, y, x + w, y + h),
                          keypoints=_keypoints(x, y, w, h, hand_up=False), confidence=0.9)]


def declined(paths, config, setup, session_row):
    return preprocess(paths, config, FaintEstimator(), setup,
                      session_id=session_row["session_id"])


# ---------------------------------------------------------------------------
# R1: there is nowhere to persist a learner
# ---------------------------------------------------------------------------

def test_no_learner_tracks_are_persisted_after_preprocessing(engine, session_row, paths,
                                                             config, setup) -> None:
    """BUILD-SPEC Phase 3 acceptance test 1, asked of the live database rather than the source.

    The assertion is stronger than "zero rows": there is no table that could hold one, and no
    column on any table that could identify a learner. Zero rows is a state somebody can change;
    no column is not.
    """
    if engine is None:
        from praxis.preprocess.aggregates import LearnerBin
        # Degraded, and still an assertion: the only learner object the pipeline produces has
        # no field an identifier could occupy.
        assert set(LearnerBin(0.0, 0, None, 0).as_row("x")) == {
            "session_id", "t_start_s", "hands_raised", "gross_motion", "person_count"}
        return

    result = preprocessed(paths, config, setup, session_row)
    with transaction(engine) as conn:
        save_camera_setup(conn, setup, college_id=session_row["college_id"],
                          label="back of room", defined_by=session_row["user_id"])
        persist(conn, session_id=session_row["session_id"], result=result,
                relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP,
                setup_id=setup.setup_id,
                actor_user_id=session_row["user_id"], actor_role="operator")

    with engine.connect() as conn:
        present = {row[0] for row in conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))}
        assert not present & set(FORBIDDEN_TABLES)

        columns = {row[0] for row in conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public'"))}
        assert not columns & set(IDENTIFYING_COLUMNS)

        stored = conn.execute(select(learner_aggregates).where(
            learner_aggregates.c.session_id == session_row["session_id"])).mappings().all()
        assert stored, "the learner evidence was written"
        assert all(set(row) == {"session_id", "t_start_s", "hands_raised", "gross_motion",
                                "person_count"} for row in stored)


# ---------------------------------------------------------------------------
# The proposal reaches the database unconfirmed, and only a person changes that
# ---------------------------------------------------------------------------

def test_the_teacher_track_is_stored_unconfirmed(engine, session_row, paths, config,
                                                 setup) -> None:
    if engine is None:
        result = preprocessed(paths, config, setup, session_row)
        assert result.proposal.as_row(
            source="detection", limit=CANDIDATE_CAP)["confirmed_by"] is None
        return

    result = preprocessed(paths, config, setup, session_row)
    with transaction(engine) as conn:
        persist(conn, session_id=session_row["session_id"], result=result,
                relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)

    with engine.connect() as conn:
        row = conn.execute(select(teacher_tracks).where(
            teacher_tracks.c.session_id == session_row["session_id"])).mappings().one()
        assert row["proposed_by"] == "heuristic"
        assert row["confirmed_by"] is None and row["confirmed_at"] is None
        assert confirmed_teacher_track(conn, session_row["session_id"]) is None, (
            "an unconfirmed proposal must not be reachable through a function that says "
            "confirmed; every downstream phase calls it")


def test_a_person_can_confirm_a_track_the_heuristic_did_not_propose(engine, session_row, paths,
                                                                    config, setup) -> None:
    """The reviewer overruling the heuristic is the step working, not failing. Both numbers go
    to the audit trail, which is what makes field accuracy measurable rather than assumed."""
    if not requires_database(engine):
        return

    result = preprocessed(paths, config, setup, session_row)
    proposed = result.proposal.track_id
    assert proposed is not None
    other = proposed + 1

    with transaction(engine) as conn:
        persist(conn, session_id=session_row["session_id"], result=result,
                relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)
    with transaction(engine) as conn:
        confirm_teacher(conn, session_id=session_row["session_id"], track_id=other,
                        confirmed_by=session_row["user_id"], actor_role="reviewer")

    with engine.connect() as conn:
        row = conn.execute(select(teacher_tracks).where(
            teacher_tracks.c.session_id == session_row["session_id"])).mappings().one()
        assert row["track_id"] == other
        assert row["confirmed_by"] == session_row["user_id"]
        assert row["confirmed_at"] is not None
        assert confirmed_teacher_track(conn, session_row["session_id"]) == other

        entries = [r for r in read_chain(conn) if r.event_type == "teacher_track.confirmed"]
        assert entries[-1].payload["proposed_track_id"] == proposed
        assert entries[-1].payload["agreed_with_heuristic"] is False


class TestADeclinedProposalIsARow:
    """The case the corpus actually contains, which before 0009 could not be stored, read or
    confirmed. Three of seven real sessions clear nothing above the floor."""

    def test_the_row_exists_with_a_null_track(self, engine, session_row, paths, config,
                                              setup) -> None:
        if not requires_database(engine):
            return

        result = declined(paths, config, setup, session_row)
        assert result.proposal.track_id is None, "the fixture is meant to be declined"

        with transaction(engine) as conn:
            persist(conn, session_id=session_row["session_id"], result=result,
                    relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)

        with engine.connect() as conn:
            row = conn.execute(select(teacher_tracks).where(
                teacher_tracks.c.session_id == session_row["session_id"])).mappings().one()
            assert row["track_id"] is None
            assert row["confirmed_by"] is None
            assert confirmed_teacher_track(conn, session_row["session_id"]) is None

    def test_the_row_carries_the_score_that_failed_and_the_reason(self, engine, session_row,
                                                                  paths, config, setup) -> None:
        """A null with no evidence beside it is not a verdict. D48."""
        if not requires_database(engine):
            return

        result = declined(paths, config, setup, session_row)
        with transaction(engine) as conn:
            persist(conn, session_id=session_row["session_id"], result=result,
                    relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)

        with engine.connect() as conn:
            row = conn.execute(select(teacher_tracks).where(
                teacher_tracks.c.session_id == session_row["session_id"])).mappings().one()
            assert row["heuristic_score"] == pytest.approx(result.proposal.score)
            assert "below the" in row["reason"]
            assert row["candidates"]["source"] == "detection"
            assert row["candidates"]["ranked"], "the reviewer is shown something to choose from"

    def test_a_reviewer_can_identify_a_track_the_heuristic_never_proposed(
            self, engine, session_row, paths, config, setup) -> None:
        """The case `confirm_teacher`'s own docstring calls the entire point of the step, and
        the case it refused outright until 0009."""
        if not requires_database(engine):
            return

        result = declined(paths, config, setup, session_row)
        chosen = result.proposal.signals[0].track_id

        with transaction(engine) as conn:
            persist(conn, session_id=session_row["session_id"], result=result,
                    relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)
        with transaction(engine) as conn:
            confirm_teacher(conn, session_id=session_row["session_id"], track_id=chosen,
                            confirmed_by=session_row["user_id"], actor_role="reviewer")

        with engine.connect() as conn:
            assert confirmed_teacher_track(conn, session_row["session_id"]) == chosen
            row = conn.execute(select(teacher_tracks).where(
                teacher_tracks.c.session_id == session_row["session_id"])).mappings().one()
            assert row["proposed_by"] == "human"

            entry = [r for r in read_chain(conn)
                     if r.event_type == "teacher_track.confirmed"][-1]
            assert entry.payload["proposed_track_id"] is None
            # Null, not false. There was nothing to agree with, and counting an abstention as a
            # disagreement would make a cautious heuristic look wrong.
            assert entry.payload["agreed_with_heuristic"] is None
            # A null payload value hashes and verifies like any other. R5.
            assert verify(read_chain(conn)) is None

    def test_agreeing_with_a_real_proposal_leaves_the_origin_alone(
            self, engine, session_row, paths, config, setup) -> None:
        """`proposed_by` records where the track number came from, not who looked at it."""
        if not requires_database(engine):
            return

        result = preprocessed(paths, config, setup, session_row)
        proposed = result.proposal.track_id
        assert proposed is not None

        with transaction(engine) as conn:
            persist(conn, session_id=session_row["session_id"], result=result,
                    relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)
        with transaction(engine) as conn:
            confirm_teacher(conn, session_id=session_row["session_id"], track_id=proposed,
                            confirmed_by=session_row["user_id"], actor_role="reviewer")

        with engine.connect() as conn:
            row = conn.execute(select(teacher_tracks).where(
                teacher_tracks.c.session_id == session_row["session_id"])).mappings().one()
            assert row["proposed_by"] == "heuristic"
            entry = [r for r in read_chain(conn)
                     if r.event_type == "teacher_track.confirmed"][-1]
            assert entry.payload["agreed_with_heuristic"] is True


class TestTheDatabaseRefusesWhatTheCommentUsedToPromise:
    """Both CHECKs added by 0009, asked of the live database rather than of a docstring."""

    def test_a_confirmation_without_a_track_is_refused(self, engine, session_row, paths,
                                                       config, setup) -> None:
        """Otherwise `confirmed_teacher_track` returns null for a session a reviewer signed off,
        which is R1 failing quietly - the one way it must not fail."""
        if not requires_database(engine):
            return

        result = declined(paths, config, setup, session_row)
        with transaction(engine) as conn:
            persist(conn, session_id=session_row["session_id"], result=result,
                    relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)

        with pytest.raises(DBAPIError, match="nothing_is_confirmed_without_a_track"), \
                transaction(engine) as conn:
            conn.execute(teacher_tracks.update()
                         .where(teacher_tracks.c.session_id == session_row["session_id"])
                         .values(confirmed_by=session_row["user_id"],
                                 confirmed_at=func.now()))

    def test_an_undeclared_origin_is_refused(self, engine, session_row, paths, config,
                                             setup) -> None:
        if not requires_database(engine):
            return

        result = preprocessed(paths, config, setup, session_row)
        with transaction(engine) as conn:
            persist(conn, session_id=session_row["session_id"], result=result,
                    relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)

        with pytest.raises(DBAPIError, match="proposed_by_is_a_known_origin"), \
                transaction(engine) as conn:
            conn.execute(teacher_tracks.update()
                         .where(teacher_tracks.c.session_id == session_row["session_id"])
                         .values(proposed_by="model"))


class TestVocabulariesMatchTheDatabase:
    """The declared tuple and the CHECK's literals, read back out of the catalogue.

    L20: the two subject-type vocabularies disagreed for exactly as long as nobody compared
    them. `proposed_by` had no CHECK at all until 0009, so this is the first thing holding the
    constraint and `praxis.vocabulary` together.
    """

    def test_the_check_literals_are_the_declared_origins(self, engine) -> None:
        if not requires_database(engine):
            return

        with engine.connect() as conn:
            definition = conn.execute(text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'proposed_by_is_a_known_origin'")).scalar_one()

        found = set(re.findall(r"'([a-z_]+)'::text", definition))
        assert found == set(TRACK_ORIGINS)

    def test_every_declared_origin_is_storable(self, engine, session_row, paths, config,
                                               setup) -> None:
        """The other direction. A value the tuple declares and the table refuses is the same
        defect as a value the table allows and the tuple omits."""
        if not requires_database(engine):
            return

        result = preprocessed(paths, config, setup, session_row)
        with transaction(engine) as conn:
            persist(conn, session_id=session_row["session_id"], result=result,
                    relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)

        for origin in TRACK_ORIGINS:
            with transaction(engine) as conn:
                conn.execute(teacher_tracks.update()
                             .where(teacher_tracks.c.session_id == session_row["session_id"])
                             .values(proposed_by=origin))


def test_confirming_a_session_that_was_never_preprocessed_is_refused(engine,
                                                                     session_row) -> None:
    """Still refused, and for the one reason that survives 0009: a session with no row has not
    been preprocessed, so there is no ranking anybody could have been shown. The refusal used to
    cover a second case as well - the heuristic declining - and that case is now the point."""
    if not requires_database(engine):
        return

    with pytest.raises(PersistError, match="has not been preprocessed"), \
            transaction(engine) as conn:
        confirm_teacher(conn, session_id=session_row["session_id"], track_id=1,
                        confirmed_by=session_row["user_id"])


# ---------------------------------------------------------------------------
# The pointer, the audit row, and what must not be written twice
# ---------------------------------------------------------------------------

def test_the_pose_pointer_and_its_hash_are_stored(engine, session_row, paths, config,
                                                  setup) -> None:
    if not requires_database(engine):
        return

    result = preprocessed(paths, config, setup, session_row)
    with transaction(engine) as conn:
        save_camera_setup(conn, setup, college_id=session_row["college_id"],
                          label="back of room", defined_by=session_row["user_id"])
        persist(conn, session_id=session_row["session_id"], result=result,
                relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP,
                setup_id=setup.setup_id)

    with engine.connect() as conn:
        row = conn.execute(select(pose_artifacts).where(
            pose_artifacts.c.session_id == session_row["session_id"])).mappings().one()
        assert row["sha256"] == result.artifact.sha256
        assert row["frame_count"] == result.frames_processed
        assert row["sampled_fps"] == config.preprocess.sample_fps
        assert row["relative_path"] == "pose/session.npz"


def test_preprocessing_is_recorded_on_the_audit_chain(engine, session_row, paths, config,
                                                      setup) -> None:
    """R5. The run destroyed the unblurred original, so the trail has to say so."""
    if not requires_database(engine):
        return

    result = preprocessed(paths, config, setup, session_row)
    with transaction(engine) as conn:
        persist(conn, session_id=session_row["session_id"], result=result,
                relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP,
                actor_user_id=session_row["user_id"], actor_role="operator")

    with engine.connect() as conn:
        records = read_chain(conn)
        assert verify(records) is None, "the chain still verifies from genesis"
        entry = [r for r in records if r.event_type == "session.preprocessed"][-1]
        assert entry.payload["original_deleted"] is True
        assert entry.payload["pose_sha256"] == result.artifact.sha256
        assert entry.payload["config_sha256"] == result.artifact.config_sha256


def test_a_session_cannot_be_preprocessed_twice(engine, session_row, paths, config,
                                                setup) -> None:
    """Re-running is not a repair. The original was deleted by the first run, so a second would
    read the blurred video and overwrite the hash that records what the first one saw."""
    if not requires_database(engine):
        return

    result = preprocessed(paths, config, setup, session_row)
    with transaction(engine) as conn:
        persist(conn, session_id=session_row["session_id"], result=result,
                relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)

    with pytest.raises(PersistError, match="already has a pose artefact"), \
            transaction(engine) as conn:
        persist(conn, session_id=session_row["session_id"], result=result,
                relative_path="pose/session.npz", max_candidates=CANDIDATE_CAP)


def test_overlapping_zones_are_refused_at_the_point_they_are_defined(engine, session_row,
                                                                     setup) -> None:
    """An operator error caught where it can still be corrected, rather than silently deciding
    a band of the room by check order for every session recorded from this camera."""
    if not requires_database(engine):
        return

    from praxis.preprocess.zones import CameraSetup
    overlapping = CameraSetup(
        setup_id=new_ulid(), zone_board=setup.zone_board, zone_front=band(0.15, 0.8),
        zone_middle=band(0.7, 0.85), zone_back=band(0.85, 1.0),
        learner_region=setup.learner_region)

    with pytest.raises(PersistError, match="overlapping seating zones"), \
            transaction(engine) as conn:
        save_camera_setup(conn, overlapping, college_id=session_row["college_id"],
                          label="bad", defined_by=session_row["user_id"])
