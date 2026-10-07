# -*- coding: utf-8 -*-
"""Excluding a session from annotation, and the states that are not an exclusion.

The thing being prevented is silence. A session nobody has confirmed a teacher track for and a
session a researcher has decided against are both simply absent from the annotation plan, and
reporting them alike turns a decision into a backlog item that never clears. So the assertions
that matter here are the ones about what gets *said*: that a ground is required, that it reaches
the row and the audit chain, and that the planner names an exclusion as a decision rather than
as something still waiting.

Synthetic fixtures throughout. They establish that the mechanism records what it claims to
record; nothing here speaks to whether any particular exclusion was a good research judgement.
"""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.exc import DBAPIError

from praxis.audit.chain import verify
from praxis.audit.write import read_chain
from praxis.db import build_engine, keywords_to_url, transaction
from praxis.db.schema import (
    colleges,
    consent_records,
    media_objects,
    sessions,
    teachers,
)
from praxis.ids import new_ulid
from praxis.ingest.exclusion import ExclusionError, exclude, excluded, reinstate
from tests.integration.conftest import requires_database

ACTOR = "01M2H000000000000000000000"
OTHER = "01M2H000000000000000000001"
GROUND = "the camera moves too much for the teacher to survive as one track"


@pytest.fixture
def engine(scratch_database):
    return build_engine(keywords_to_url(scratch_database)) if scratch_database else None


@pytest.fixture
def session_id(engine):
    """A committed session with its foreign keys, so the update has something to land on."""
    if engine is None:
        return None

    college, teacher, consent = new_ulid(), new_ulid(), new_ulid()
    session, media = new_ulid(), new_ulid().lower().ljust(64, "a")[:64]
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
            media_sha256=media, relative_path="test/original.mp4", bytes=1024,
            duration_s=120.0, width=1280, height=720, fps=25.0, has_audio=False,
            is_blurred=True, blurred_relative_path="test/blurred.mp4",
            reachable=True, created_at=func.now()))
        conn.execute(insert(sessions).values(
            session_id=session, teacher_id=teacher, college_id=college, consent_id=consent,
            media_sha256=media, domain="microteaching", recorded_on=date(2026, 3, 2),
            quality_verdict="pass", quality_detail={}, created_at=func.now()))
    return session


def state(engine, session_id):
    with transaction(engine) as conn:
        return conn.execute(
            select(sessions.c.excluded_at, sessions.c.excluded_by,
                   sessions.c.exclusion_reason)
            .where(sessions.c.session_id == session_id)).first()


class TestAnExclusionIsRecordedWithItsGround:
    def test_the_three_columns_are_written_together(self, engine, session_id) -> None:
        if not requires_database(engine, session_id):
            return

        with transaction(engine) as conn:
            exclude(conn, session_id=session_id, reason=GROUND, excluded_by=ACTOR)

        row = state(engine, session_id)
        assert row.excluded_at is not None
        assert row.excluded_by.strip() == ACTOR
        assert row.exclusion_reason == GROUND

    def test_it_reaches_the_audit_chain(self, engine, session_id) -> None:
        """The row is current state. The chain is the record that somebody decided, and it is
        the half a later reader can question."""
        if not requires_database(engine, session_id):
            return

        with transaction(engine) as conn:
            exclude(conn, session_id=session_id, reason=GROUND, excluded_by=ACTOR,
                    actor_role="researcher")

        with transaction(engine) as conn:
            trail = [r for r in read_chain(conn) if r.event_type == "session.excluded"]
            broke = verify(read_chain(conn))
        assert broke is None, f"the chain broke: {broke}"
        assert trail, "an exclusion is an audited decision"
        assert trail[-1].payload["reason"] == GROUND
        assert trail[-1].payload["excluded_by"] == ACTOR

    def test_a_blank_ground_is_refused(self, engine, session_id) -> None:
        """An exclusion recorded as a bare flag reads as unfinished work to everyone after."""
        if not requires_database(engine, session_id):
            return

        for blank in ("", "   ", "\n\t"):
            with transaction(engine) as conn, pytest.raises(ExclusionError,
                                                             match="stated ground"):
                exclude(conn, session_id=session_id, reason=blank, excluded_by=ACTOR)
        assert state(engine, session_id).excluded_at is None

    def test_the_ground_is_stripped_not_stored_padded(self, engine, session_id) -> None:
        if not requires_database(engine, session_id):
            return
        with transaction(engine) as conn:
            exclude(conn, session_id=session_id, reason=f"  {GROUND}  ", excluded_by=ACTOR)
        assert state(engine, session_id).exclusion_reason == GROUND

    def test_a_session_that_does_not_exist_is_refused(self, engine) -> None:
        if not requires_database(engine):
            return
        with transaction(engine) as conn, pytest.raises(ExclusionError, match="does not exist"):
            exclude(conn, session_id=new_ulid(), reason=GROUND, excluded_by=ACTOR)

    def test_excluding_twice_is_refused_rather_than_overwriting(self, engine,
                                                                 session_id) -> None:
        """The second ground might differ from the first, and quietly replacing it would lose
        the reasoning somebody acted on."""
        if not requires_database(engine, session_id):
            return

        with transaction(engine) as conn:
            exclude(conn, session_id=session_id, reason=GROUND, excluded_by=ACTOR)
        with transaction(engine) as conn, pytest.raises(ExclusionError,
                                                        match="already excluded"):
            exclude(conn, session_id=session_id, reason="a different ground",
                    excluded_by=OTHER)
        assert state(engine, session_id).exclusion_reason == GROUND


class TestTheDatabaseRefusesAHalfRecordedDecision:
    def test_a_ground_with_no_author_and_no_time(self, engine, session_id) -> None:
        """`an_exclusion_names_a_person_a_time_and_a_ground`. Enforced at the database because
        a second writer would be a second place to forget it."""
        if not requires_database(engine, session_id):
            return

        from sqlalchemy import update
        with transaction(engine) as conn, pytest.raises(DBAPIError) as raised:
            conn.execute(update(sessions)
                         .where(sessions.c.session_id == session_id)
                         .values(exclusion_reason=GROUND))
        assert "an_exclusion_names_a_person_a_time_and_a_ground" in str(raised.value)

    def test_an_author_with_no_ground(self, engine, session_id) -> None:
        if not requires_database(engine, session_id):
            return

        from sqlalchemy import update
        with transaction(engine) as conn, pytest.raises(DBAPIError) as raised:
            conn.execute(update(sessions)
                         .where(sessions.c.session_id == session_id)
                         .values(excluded_at=func.now(), excluded_by=ACTOR))
        assert "an_exclusion_names_a_person_a_time_and_a_ground" in str(raised.value)

    def test_a_blank_ground_at_the_database(self, engine, session_id) -> None:
        """`an_exclusion_ground_is_not_blank`. A blank string satisfies the first constraint
        and says nothing, which is the failure this pair exists to prevent."""
        if not requires_database(engine, session_id):
            return

        from sqlalchemy import update
        with transaction(engine) as conn, pytest.raises(DBAPIError) as raised:
            conn.execute(update(sessions)
                         .where(sessions.c.session_id == session_id)
                         .values(excluded_at=func.now(), excluded_by=ACTOR,
                                 exclusion_reason="   "))
        assert "an_exclusion_ground_is_not_blank" in str(raised.value)


class TestReinstatementIsItsOwnDecision:
    def test_it_clears_the_row_and_keeps_both_events(self, engine, session_id) -> None:
        if not requires_database(engine, session_id):
            return

        with transaction(engine) as conn:
            exclude(conn, session_id=session_id, reason=GROUND, excluded_by=ACTOR)
        with transaction(engine) as conn:
            reinstate(conn, session_id=session_id, reason="re-shot with a tripod",
                      reinstated_by=OTHER)

        row = state(engine, session_id)
        assert row.excluded_at is None and row.excluded_by is None
        assert row.exclusion_reason is None

        # Scoped to this session: the scratch database outlives one test, so the chain also
        # carries every other test's decisions.
        with transaction(engine) as conn:
            trail = [r for r in read_chain(conn)
                     if r.event_type == "session.excluded" and r.entity_id == session_id]
        assert len(trail) == 2, "the chain records that somebody changed their mind"
        assert trail[-1].payload["reason"].startswith("REINSTATED:")
        assert trail[-1].payload["reinstates"] == GROUND

    def test_reinstating_what_was_never_excluded_is_refused(self, engine,
                                                             session_id) -> None:
        """It would put a decision in the trail that was never made."""
        if not requires_database(engine, session_id):
            return
        with transaction(engine) as conn, pytest.raises(ExclusionError, match="not excluded"):
            reinstate(conn, session_id=session_id, reason="x", reinstated_by=ACTOR)


class TestTheReportCannotSilentlyOmitThem:
    def test_excluded_lists_every_one_with_its_ground(self, engine, session_id) -> None:
        if not requires_database(engine, session_id):
            return

        # Scoped to this session; the scratch database carries other tests' exclusions too.
        with transaction(engine) as conn:
            assert session_id not in excluded(conn)
            exclude(conn, session_id=session_id, reason=GROUND, excluded_by=ACTOR)
        with transaction(engine) as conn:
            listed = excluded(conn)
        assert listed[session_id] == GROUND

    def test_the_planner_reports_an_exclusion_as_a_decision_not_as_waiting(
            self, engine, session_id) -> None:
        """The whole point. An excluded session must not read as one nobody has got to."""
        if not requires_database(engine, session_id):
            return

        from scripts.plan_annotation import _load_annotatable_sessions

        with transaction(engine) as conn:
            exclude(conn, session_id=session_id, reason=GROUND, excluded_by=ACTOR)

        with transaction(engine) as conn, pytest.warns(UserWarning) as caught:
            annotatable = _load_annotatable_sessions(conn)

        assert session_id not in annotatable

        # Only what was said about *this* session. Other tests leave their own sessions in the
        # scratch database, and some of those are legitimately awaiting confirmation.
        mine = [str(w.message) for w in caught if session_id in str(w.message)]
        assert mine, "the planner said nothing at all about an excluded session"
        assert any("excluded from annotation" in line for line in mine)
        assert any(GROUND in line for line in mine)
        assert not any("awaiting teacher confirmation" in line for line in mine), (
            "an exclusion was reported as something nobody has got to yet")
