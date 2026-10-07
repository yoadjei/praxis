# -*- coding: utf-8 -*-
"""Integration tests for the annotation assignment planning script.

The script bridges preprocessed sessions to raters by cutting clips, sampling for
calibration if needed, and building a deterministic assignment roster. These tests
verify that the roster is built correctly, spread across sessions, reproducible from
the seed, and persisted idempotently.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import func, select

from praxis.annotation.store import queue_for
from praxis.config import load_config
from praxis.db import build_engine, keywords_to_url, transaction
from praxis.db.schema import (
    annotation_assignments,
    colleges,
    consent_records,
    media_objects,
    sessions,
    teacher_tracks,
    teachers,
)
from praxis.ids import new_ulid
from praxis.vocabulary import BEHAVIOUR_IDS
from tests.integration.conftest import (
    requires_database,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def insert_media_and_session(
    conn,
    session_id: str,
    teacher_id: str,
    college_id: str,
    consent_id: str,
    duration_s: float,
    is_blurred: bool = True,
    blurred_relative_path: str | None = None,
    quality_verdict: str | None = None,
    teacher_confirmed: bool = True,
) -> None:
    """Insert a media object and session for testing.

    Quality verdict must be one of: 'pass', 'warn', 'fail'.

    `teacher_confirmed` writes the confirmed `teacher_tracks` row that docs/API.md section 4
    requires before a session may enter annotation, and defaults true because a session without
    one is not annotatable and almost every test here is about what happens to one that is.
    Pass false to build the waiting case.
    """
    media_sha = new_ulid()
    conn.execute(
        media_objects.insert().values(
            media_sha256=media_sha,
            relative_path="test/original.mp4",
            bytes=1000000,
            blurred_relative_path=blurred_relative_path,
            is_blurred=is_blurred,
            duration_s=duration_s,
            width=1280,
            height=720,
            fps=25.0,
            has_audio=True,
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
            media_sha256=media_sha,
            domain="microteaching",
            recorded_on=date(2025, 1, 1),
            quality_verdict=quality_verdict or "pass",
            quality_detail={},
            created_at=func.now(),
        )
    )
    if teacher_confirmed:
        conn.execute(
            teacher_tracks.insert().values(
                session_id=session_id,
                track_id=1,
                proposed_by="heuristic",
                heuristic_score=0.9,
                confirmed_by=new_ulid(),
                confirmed_at=func.now(),
                reason="fixture: a confirmed teacher, which is what makes this annotatable",
                candidates={"source": "detection", "zones_available": True, "truncated": 0,
                            "ranked": [{"track_id": 1, "presence_fraction": 1.0,
                                        "median_area_fraction": 0.1,
                                        "front_zone_fraction": 1.0, "score": 0.9}]},
            )
        )


@pytest.fixture
def annotation_engine(engine):
    """Engine with college and teacher for annotation tests."""
    return _annotation_parties(engine)


@pytest.fixture
def private_annotation_engine(private_database):
    """The same parties, in a database that no other test writes to.

    The planner reads the whole corpus to decide what can be annotated, so a test asserting
    that it refuses - because nothing is annotatable, or because the sample covers too few
    teachers - is asserting something about every row in the database. Both such tests here
    passed run by file and failed run as a suite, for that reason and no other.
    """
    if private_database is None:
        return None
    return _annotation_parties(build_engine(keywords_to_url(private_database)))


def _annotation_parties(engine) -> dict | None:
    """One college, two teachers, and a live consent for each. One body, two fixtures."""
    if engine is None:
        return None

    college_id = new_ulid()
    teacher_id_1 = new_ulid()
    teacher_id_2 = new_ulid()
    consent_id_1 = new_ulid()
    consent_id_2 = new_ulid()

    with transaction(engine) as conn:
        conn.execute(
            colleges.insert().values(
                college_id=college_id,
                code=f"COL-{college_id[-5:]}",
                created_at=func.now(),
            )
        )
        conn.execute(
            teachers.insert().values(
                teacher_id=teacher_id_1,
                college_id=college_id,
                created_at=func.now(),
            )
        )
        conn.execute(
            teachers.insert().values(
                teacher_id=teacher_id_2,
                college_id=college_id,
                created_at=func.now(),
            )
        )
        conn.execute(
            consent_records.insert().values(
                consent_id=consent_id_1,
                subject_type="teacher",
                teacher_id=teacher_id_1,
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
            consent_records.insert().values(
                consent_id=consent_id_2,
                subject_type="teacher",
                teacher_id=teacher_id_2,
                college_id=college_id,
                purpose="research",
                recipients="supervisors",
                scope="both",
                granted_on=date(2025, 1, 1),
                document_ref="file/1",
                created_at=func.now(),
            )
        )

    return {
        "engine": engine,
        "college_id": college_id,
        "teacher_1": teacher_id_1,
        "consent_1": consent_id_1,
        "teacher_2": teacher_id_2,
        "consent_2": consent_id_2,
    }


@pytest.fixture
def config(repo_root) -> dict:
    """Load config and return both the config object and key values."""
    cfg = load_config(repo_root / "configs" / "default.yaml")
    return {
        "config": cfg,
        "clip_length_s": cfg.behaviour.clip.length_s,
        "calibration_clips": cfg.annotation.calibration_clips_per_round,
        "min_teachers": cfg.annotation.calibration_min_teachers,
        "seed": cfg.run.seed,
        "behaviours": list(BEHAVIOUR_IDS),
    }


def run_plan_annotation(
    argv: list[str], annotation_engine: dict
) -> subprocess.CompletedProcess:
    """Run the plan_annotation script against the scratch database.

    Returns CompletedProcess with stdout, stderr, and returncode populated.

    Two things here are load-bearing and were both wrong when this file was first written.

    `DATABASE_URL` is set from the scratch engine. The script builds its own engine from the
    environment, and the environment running the suite points at the development cluster, so
    without this the test would seed a scratch database and the script under test would read -
    and write assignments into - the real corpus. It is not a test if it measures a different
    database from the one it set up.

    `stdin=subprocess.DEVNULL`, because pytest replaces stdin with an object that has no
    inheritable handle, and a child process on Windows then fails to start at all with
    "OSError: [WinError 6] The handle is invalid". `tests/conftest.py` does the same when it
    shells out to alembic.
    """
    url = annotation_engine["engine"].url.render_as_string(hide_password=False)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "plan_annotation.py"), *argv],
        cwd=REPO_ROOT, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=300,
        env={**os.environ, "DATABASE_URL": url})


class TestUnblurredSessionsReportedAsWaiting:
    """A session that is not blurred should be reported as waiting, not silently dropped."""

    def test_unblurred_session_yields_no_clips_and_warns(
        self, private_annotation_engine, config
    ) -> None:
        """An unblurred session (awaiting preprocessing) is reported as a warning.

        The one session this writes is unblurred, so the planner has nothing to plan and must
        refuse. That is only true of a database where it is the only session: against the shared
        one the planner found other tests' blurred sessions, planned those, and exited 0.
        """
        if not requires_database(private_annotation_engine):
            return

        annotation_engine = private_annotation_engine
        engine = annotation_engine["engine"]
        teacher_id = annotation_engine["teacher_1"]
        college_id = annotation_engine["college_id"]
        consent_id = annotation_engine["consent_1"]
        session_id = new_ulid()

        # Insert a session that is NOT blurred
        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s=60.0,
                is_blurred=False,
                blurred_relative_path=None,
                quality_verdict="pass",
            )

        # Run plan_annotation - it should fail because no annotatable sessions exist
        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "calibration-1",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )
        # Should fail because there are no annotatable sessions
        assert result.returncode != 0

    def test_a_session_awaiting_teacher_confirmation_is_not_annotatable(
        self, private_annotation_engine, config
    ) -> None:
        """docs/API.md section 4: confirmation is mandatory before a session enters annotation.

        Nothing enforced it until now. `confirmed_teacher_track`, whose docstring calls itself
        what every downstream phase must call, had no caller anywhere outside its own tests, and
        the research database held 4932 assignments across sessions with zero confirmed tracks.
        A clip cut from an unconfirmed session can show somebody who is not the teacher, and the
        rater's label is then attached to that person's conduct. R1.
        """
        if not requires_database(private_annotation_engine):
            return

        annotation_engine = private_annotation_engine
        engine = annotation_engine["engine"]
        session_id = new_ulid()

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                annotation_engine["teacher_1"],
                annotation_engine["college_id"],
                annotation_engine["consent_1"],
                duration_s=60.0,
                blurred_relative_path="test/blurred.mp4",
                quality_verdict="pass",
                teacher_confirmed=False,
            )

        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "calibration-1",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        assert result.returncode != 0
        assert "confirmed teacher track" in result.stderr
        with engine.connect() as conn:
            planned = conn.execute(
                select(func.count()).select_from(annotation_assignments)
                .where(annotation_assignments.c.session_id == session_id)).scalar_one()
        assert planned == 0

    def test_an_unconfirmed_proposal_is_not_enough(
        self, private_annotation_engine, config
    ) -> None:
        """A proposal is a guess. The row existing is not the same as a person having agreed,
        which is the whole reason `confirmed_by` is separate from `track_id`."""
        if not requires_database(private_annotation_engine):
            return

        annotation_engine = private_annotation_engine
        engine = annotation_engine["engine"]
        session_id = new_ulid()

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                annotation_engine["teacher_1"],
                annotation_engine["college_id"],
                annotation_engine["consent_1"],
                duration_s=60.0,
                blurred_relative_path="test/blurred.mp4",
                quality_verdict="pass",
                teacher_confirmed=False,
            )
            # The heuristic proposed, and nobody has agreed.
            conn.execute(teacher_tracks.insert().values(
                session_id=session_id, track_id=3, proposed_by="heuristic",
                heuristic_score=0.42, reason="track 3 scored 0.420",
                candidates={"source": "detection", "zones_available": False,
                            "truncated": 0, "ranked": []}))

        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "calibration-1",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        assert result.returncode != 0
        assert "nobody has confirmed it" in result.stderr

    def test_multiple_sessions_with_one_unblurred_reports_waiting(
        self, annotation_engine, config
    ) -> None:
        """With mixed sessions, the unblurred one is reported but doesn't block."""
        if not requires_database(annotation_engine):
            return

        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_1 = annotation_engine["teacher_1"]
        consent_1 = annotation_engine["consent_1"]
        teacher_2 = annotation_engine["teacher_2"]
        consent_2 = annotation_engine["consent_2"]

        # Create 2 more teachers to meet minimum requirement (4 total)
        teacher_3 = new_ulid()
        consent_3 = new_ulid()
        teacher_4 = new_ulid()
        consent_4 = new_ulid()

        session_blurred = new_ulid()
        session_unblurred = new_ulid()

        with transaction(engine) as conn:
            # Add 2 more teachers with consent
            conn.execute(
                teachers.insert().values(
                    teacher_id=teacher_3,
                    college_id=college_id,
                    created_at=func.now(),
                )
            )
            conn.execute(
                consent_records.insert().values(
                    consent_id=consent_3,
                    subject_type="teacher",
                    teacher_id=teacher_3,
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
                teachers.insert().values(
                    teacher_id=teacher_4,
                    college_id=college_id,
                    created_at=func.now(),
                )
            )
            conn.execute(
                consent_records.insert().values(
                    consent_id=consent_4,
                    subject_type="teacher",
                    teacher_id=teacher_4,
                    college_id=college_id,
                    purpose="research",
                    recipients="supervisors",
                    scope="both",
                    granted_on=date(2025, 1, 1),
                    document_ref="file/1",
                    created_at=func.now(),
                )
            )

            # Blurred sessions (annotatable) - one per teacher to meet minimum
            insert_media_and_session(
                conn,
                session_blurred,
                teacher_1,
                college_id,
                consent_1,
                duration_s=60.0,
                is_blurred=True,
                blurred_relative_path="test/blurred1.mp4",
                quality_verdict="pass",
            )
            insert_media_and_session(
                conn,
                new_ulid(),
                teacher_2,
                college_id,
                consent_2,
                duration_s=60.0,
                is_blurred=True,
                blurred_relative_path="test/blurred2.mp4",
                quality_verdict="pass",
            )
            insert_media_and_session(
                conn,
                new_ulid(),
                teacher_3,
                college_id,
                consent_3,
                duration_s=60.0,
                is_blurred=True,
                blurred_relative_path="test/blurred3.mp4",
                quality_verdict="pass",
            )
            insert_media_and_session(
                conn,
                new_ulid(),
                teacher_4,
                college_id,
                consent_4,
                duration_s=60.0,
                is_blurred=True,
                blurred_relative_path="test/blurred4.mp4",
                quality_verdict="pass",
            )

            # Unblurred session (waiting) from teacher_2
            session_unblurred = new_ulid()
            insert_media_and_session(
                conn,
                session_unblurred,
                teacher_2,
                college_id,
                consent_2,
                duration_s=60.0,
                is_blurred=False,
                blurred_relative_path=None,
                quality_verdict="pass",
            )

        # A round name of its own. "calibration-1" is shared with other tests in this file, and
        # the unique constraint on (rater, session, clip_index, behaviour) spans rounds, so a
        # reused name lets one test consume the pairings another is about to plan.
        round_name = f"calibration-mixed-{new_ulid()[:8]}"
        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", round_name,
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )
        assert result.returncode == 0, f"planning refused: {result.stderr}{result.stdout}"

        # Per session, within this round. The comment above this said "for the blurred session
        # only" while the query counted every assignment in the database with no WHERE clause
        # at all, so `count > 0` was satisfied by any row any earlier test had written and the
        # test passed whether or not this round planned anything. Counting the unblurred session
        # as well is what makes it an assertion about exclusion rather than about mere presence.
        with transaction(engine) as conn:
            def planned(session_id: str) -> int:
                return conn.execute(
                    select(func.count())
                    .select_from(annotation_assignments)
                    .where(annotation_assignments.c.round_name == round_name,
                           annotation_assignments.c.session_id == session_id)
                ).scalar()

            assert planned(session_blurred) > 0, (
                "the blurred session is annotatable and this round planned none of it")
            assert planned(session_unblurred) == 0, (
                "the unblurred session is still awaiting preprocessing and must not be "
                "planned; D18 blurs before anything is annotated")


class TestQualityVerdictRefusal:
    """A session that failed its quality gate is excluded from annotation."""

    def test_refused_session_is_excluded(self, private_annotation_engine, config) -> None:
        """A session with quality_verdict='fail' is not offered for annotation.

        Three things were wrong here, and the test could not have reported any of them.

        It read `exit_code = run_plan_annotation(...)` and asserted `exit_code != 0`, but that
        helper returns a `subprocess.CompletedProcess`. An object compared to an integer with
        `!=` is unequal by definition, so the assertion was true for every possible outcome,
        including the script exiting 0.

        Its comment said "the only session is refused", which was true only of a database this
        test had to itself; against the shared one, the planner found other tests' annotatable
        sessions and would have succeeded.

        And both docstrings described the script as filtering `quality_verdict != 'refuse'`.
        There is no such verdict - the vocabulary is pass, warn, fail, enforced by a CHECK - and
        the script was corrected to 'fail' earlier; the prose outlived the code. Asserting the
        refusal names the failed verdict now, so a script that refused for some unrelated reason
        would not satisfy it.
        """
        if not requires_database(private_annotation_engine):
            return

        annotation_engine = private_annotation_engine
        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]
        session_id = new_ulid()

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s=60.0,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
                quality_verdict="fail",  # Should be excluded
            )

        # The only session in this database, and it failed its quality gate, so there is
        # nothing to plan.
        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", f"calibration-refused-{new_ulid()[:8]}",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        assert result.returncode != 0, (
            f"a corpus whose only session failed quality was planned anyway: "
            f"{result.stderr}{result.stdout}")
        output = (result.stderr + result.stdout).lower()
        assert "annotatable" in output or "no clips" in output or "nothing" in output, (
            f"the refusal does not say that nothing was annotatable, so it may have refused "
            f"for an unrelated reason: {result.stderr}{result.stdout}")

        with transaction(engine) as conn:
            planned = conn.execute(
                select(func.count())
                .select_from(annotation_assignments)
                .where(annotation_assignments.c.session_id == session_id)
            ).scalar()
        assert planned == 0, "a session that failed quality was assigned to a rater"


class TestDeterministicSampling:
    """The sample is deterministic from the seed."""

    def test_same_seed_gives_same_sample(self, annotation_engine, config) -> None:
        """Running twice with the same seed produces identical sampled clips.

        Runs the script back-to-back with no database changes between runs, then
        compares the manifest files (minus created_at). Mutation caught: script
        changes the clip selection logic based on something non-deterministic.
        """
        if not requires_database(annotation_engine):
            return

        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]

        # Create a session long enough for multiple clips
        session_id = new_ulid()
        duration_s = 100.0  # ~12 clips at 8 seconds each

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
                quality_verdict="pass",
            )

        # run twice back-to-back with nothing inserted in between
        cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
        round_1 = f"det-run-1-{new_ulid()[:8]}"
        round_2 = f"det-run-2-{new_ulid()[:8]}"

        result_1 = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", round_1,
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )
        assert result_1.returncode == 0

        # immediately run again without any database changes
        result_2 = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", round_2,
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )
        assert result_2.returncode == 0

        # read both manifests and compare deterministic fields only
        path_1 = cfg.paths.run_outputs / f"annotation-plan-{round_1}.json"
        path_2 = cfg.paths.run_outputs / f"annotation-plan-{round_2}.json"

        with open(path_1) as f:
            manifest_1 = json.load(f)
        with open(path_2) as f:
            manifest_2 = json.load(f)

        # compare fields that should be identical (seed, clip selection, etc)
        # ignore round, created_at, and assignment counts (which depend on
        # whether assignments already existed)
        deterministic_fields = [
            "seed", "clip_seconds", "min_tail_seconds", "behaviours",
            "clips_available_per_session", "clips_sampled_per_session",
            "sampled_clip_ids", "total_clips_available",
            "total_clips_sampled", "raters_per_clip"
        ]

        for field in deterministic_fields:
            assert manifest_1[field] == manifest_2[field], \
                f"field {field} differs between runs"

    def test_different_seed_gives_different_sample(self, private_annotation_engine, config,
                                                     tmp_path) -> None:
        """A different seed in the config makes the script build a different roster.

        Mutation caught: the script ignores `run.seed` or uses a hardcoded value.

        What is measured, and why it is not the assignment rows. `build_assignments` is a pure
        function of its clips, raters, parameters and seed; the script then drops any assignment
        whose `(rater, session, clip_index, behaviour)` already exists, and that key spans
        rounds. So a second round can never reproduce the first one's rows even when it chooses
        exactly the same roster - which makes comparing two rounds' rows unable to distinguish
        the seed from the deduplication. The first version of this test compared them anyway,
        and dropped `behaviour` from the comparison as well, leaving two copies of the full
        rater-by-clip cross product that were equal for any seed.

        The deduplication is the instrument rather than the obstacle. Replanning with the same
        seed rebuilds the identical roster, every row of which is already present, so the round
        creates nothing and reports all of it as already existing. Replanning with a different
        seed picks pairings the first roster did not, so the round creates some. Those two
        numbers come straight off the manifest and say exactly what the seed is responsible for.
        """
        if not requires_database(private_annotation_engine):
            return

        annotation_engine = private_annotation_engine
        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]

        session_id = new_ulid()
        duration_s = 100.0

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
                quality_verdict="pass",
            )

        cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
        default_config_path = REPO_ROOT / "configs" / "default.yaml"

        import yaml
        cfg_dict = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
        cfg_dict["run"]["seed"] = cfg_dict["run"]["seed"] + 1000
        alt_config_path = tmp_path / "alt_config.yaml"
        alt_config_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

        def plan(round_name: str, config_path: Path) -> dict:
            result = run_plan_annotation(
                [
                    "--raters", "R1,R2,R3,R4,R5",
                    "--round", round_name,
                    "--all-clips",
                    "--config", str(config_path),
                ],
                annotation_engine,
            )
            assert result.returncode == 0, (
                f"{round_name} failed: {result.stderr}{result.stdout}")
            manifest_path = cfg.paths.run_outputs / f"annotation-plan-{round_name}.json"
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        first = plan(f"seed-first-{new_ulid()[:8]}", default_config_path)
        assert first["assignments_created"] > 0, (
            "the first round planned nothing, so neither of the comparisons below means "
            "anything")

        # Same seed, same corpus, a new round name. The roster is rebuilt identically and every
        # row of it is already there.
        same = plan(f"seed-same-{new_ulid()[:8]}", default_config_path)
        assert same["seed"] == first["seed"]
        assert same["assignments_created"] == 0, (
            f"replanning with seed {same['seed']} created "
            f"{same['assignments_created']} assignments; the same seed over the same corpus "
            f"must rebuild the same roster, so there should have been nothing left to create")
        assert same["assignments_already_existed"] == first["assignments_created"]

        # A different seed, everything else held. Pairings the first roster did not choose.
        other = plan(f"seed-other-{new_ulid()[:8]}", alt_config_path)
        assert other["seed"] == first["seed"] + 1000
        assert other["assignments_created"] > 0, (
            f"seed {other['seed']} produced a roster already covered by seed {first['seed']}, "
            f"so the seed is not reaching the assignment")


class TestSpreadAcrossSessions:
    """The sample spreads across sessions rather than concentrating in one."""

    def test_sample_draws_from_multiple_sessions(
        self, private_annotation_engine, tmp_path
    ) -> None:
        """D55: a sample smaller than one session's clips still reaches both sessions.

        Three things this asserts that it did not before.

        It counts the sessions in *its own round*. The count was unfiltered, so in the shared
        database it was over every assignment any test had ever planned, and that was more than
        one whatever the sampler did.

        It checks that the run succeeded. The round name has to begin with "calibration" for
        sampling to happen at all, which also means the minimum-teacher gate applies - and
        against a corpus holding fewer than the four teachers the default config asks for, the
        script refused, exited non-zero, created nothing, and the unfiltered count read other
        tests' rows and passed anyway. The return value was discarded, so there was nothing to
        notice.

        And it draws fewer clips than one session holds. Taking 100 clips from two sessions of
        twelve takes all twenty-four and reaches both sessions however the sampler is written;
        only a sample that has to choose can show it choosing round-robin.
        """
        if not requires_database(private_annotation_engine):
            return

        annotation_engine = private_annotation_engine
        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_1 = annotation_engine["teacher_1"]
        consent_1 = annotation_engine["consent_1"]
        teacher_2 = annotation_engine["teacher_2"]
        consent_2 = annotation_engine["consent_2"]

        # 100 seconds at 8 seconds a clip is twelve clips in each session, twenty-four in all.
        session_1 = new_ulid()
        session_2 = new_ulid()
        duration_s = 100.0

        with transaction(engine) as conn:
            insert_media_and_session(
                conn, session_1, teacher_1, college_id, consent_1, duration_s,
                is_blurred=True, blurred_relative_path="test/blurred1.mp4")
            insert_media_and_session(
                conn, session_2, teacher_2, college_id, consent_2, duration_s,
                is_blurred=True, blurred_relative_path="test/blurred2.mp4")

        # Eight clips, so the sampler has to leave sixteen behind, and a floor of two teachers,
        # which is what this corpus has. The defaults are 100 and 4: the first takes everything
        # and the second refuses the round.
        import yaml
        cfg_dict = yaml.safe_load(
            (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
        cfg_dict["annotation"]["calibration_clips_per_round"] = 8
        cfg_dict["annotation"]["calibration_min_teachers"] = 2
        config_path = tmp_path / "spread_config.yaml"
        config_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

        round_name = f"calibration-spread-{new_ulid()[:8]}"
        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", round_name,
                "--config", str(config_path),
            ],
            annotation_engine,
        )
        assert result.returncode == 0, (
            f"planning refused: {result.stderr}{result.stdout}")

        with transaction(engine) as conn:
            sessions_in_sample = conn.execute(
                select(func.count(func.distinct(annotation_assignments.c.session_id)))
                .where(annotation_assignments.c.round_name == round_name)
            ).scalar()

        assert sessions_in_sample == 2, (
            f"eight clips drawn from two sessions of twelve reached {sessions_in_sample} of "
            f"them; a round-robin over sessions reaches both (D55)")


class TestCalibrationMinTeachersEnforced:
    """calibration_min_teachers is enforced."""

    def test_too_few_teachers_refuses_with_message(
        self, private_annotation_engine, tmp_path
    ) -> None:
        """Refusal when the sample spans fewer distinct teachers than calibration requires.

        Sets the minimum above anything the corpus could satisfy and asserts the script exits
        non-zero saying both what it found and what it needed. Mutation caught: the script
        ignores the constraint, or refuses without reporting the count.
        """
        if not requires_database(private_annotation_engine):
            return

        annotation_engine = private_annotation_engine
        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]

        # Create a session for this teacher
        session_id = new_ulid()
        duration_s = 200.0

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
            )

            # Every teacher in the database. An upper bound on what any sample could span, and
            # that is all it is used for: the gate reports how many teachers the *sample*
            # covers, which is a different and smaller number, because a teacher with no
            # blurred session has no clip to be sampled. This test used to assert the corpus
            # figure appeared in the gate's message, which held only while the two happened to
            # coincide.
            all_teachers = conn.execute(
                select(func.count(func.distinct(teachers.c.teacher_id)))
            ).scalar()

        # create a temp config with min_teachers set above what exists
        default_config_path = REPO_ROOT / "configs" / "default.yaml"
        with open(default_config_path) as f:
            config_text = f.read()

        import yaml
        cfg_dict = yaml.safe_load(config_text)
        # set minimum far above what could possibly exist
        min_teachers_impossible = all_teachers + 100
        cfg_dict["annotation"]["calibration_min_teachers"] = \
            min_teachers_impossible

        alt_config_path = tmp_path / "impossible_config.yaml"
        with open(alt_config_path, "w") as f:
            yaml.dump(cfg_dict, f)

        # The round name has to begin with "calibration". The gate lives inside the sampling
        # branch, which the script only takes for a calibration round - a production round takes
        # every clip and has no sample to check the coverage of. A name like "min-teacher-test"
        # skips the branch entirely and the script succeeds, which this test previously read as
        # the gate being broken. The gate is fine; the round name was not.
        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "calibration-min-teacher-test",
                "--config", str(alt_config_path),
            ],
            annotation_engine,
        )

        # should fail because we set minimum above what exists
        assert result.returncode != 0, (
            f"the gate did not fire with calibration_min_teachers="
            f"{min_teachers_impossible}: {result.stderr}{result.stdout}")

        # The message must name both numbers, so an operator reads what was found and what was
        # wanted rather than being told only that something was too few. The found number is
        # read back out rather than predicted: predicting it would mean reimplementing the
        # sampler here, and a second implementation of the thing under test proves nothing.
        output = result.stderr + result.stdout
        assert "teacher" in output.lower()
        assert str(min_teachers_impossible) in output

        reported = re.search(r"only (\d+) distinct teachers", output)
        assert reported, f"the refusal did not say how many teachers it found: {output}"
        found = int(reported.group(1))
        assert found < min_teachers_impossible, (
            f"the gate refused while reporting {found} teachers, which is not below the "
            f"{min_teachers_impossible} it asked for")
        assert found <= all_teachers, (
            f"the sample is reported as spanning {found} teachers, more than the "
            f"{all_teachers} that exist")


class TestIdempotentPlanning:
    """Running twice creates nothing the second time; adding a rater creates for that rater."""

    def test_running_twice_with_same_raters_creates_nothing_second_time(
        self, annotation_engine, config
    ) -> None:
        """Running the planner twice with identical parameters is idempotent."""
        if not requires_database(annotation_engine):
            return

        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]
        session_id = new_ulid()

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s=100.0,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
            )

        # First run
        run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "production-1",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        with transaction(engine) as conn:
            count_after_first = conn.execute(
                select(func.count()).select_from(annotation_assignments)
            ).scalar()

        # Second run with identical parameters
        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "production-1",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        with transaction(engine) as conn:
            count_after_second = conn.execute(
                select(func.count()).select_from(annotation_assignments)
            ).scalar()

        # No new rows should be created
        assert count_after_second == count_after_first
        # The output should say "already existed"
        assert "already existed" in result.stdout

    def test_adding_rater_creates_assignments_only_for_new_rater(
        self, annotation_engine, config
    ) -> None:
        """Adding a rater to a second run creates assignments only for the new rater."""
        if not requires_database(annotation_engine):
            return

        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]
        session_id = new_ulid()

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s=100.0,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
            )

        # First run with R1, R2
        run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "production-expand",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        with transaction(engine) as conn:
            r1_r2_count = conn.execute(
                select(func.count()).select_from(annotation_assignments)
            ).scalar()

        # Second run with R1, R2, R3
        run_plan_annotation(
            [
                "--raters", "R1,R2,R3",
                "--round", "production-expand",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        with transaction(engine) as conn:
            r1_r2_r3_count = conn.execute(
                select(func.count()).select_from(annotation_assignments)
            ).scalar()
            all_raters = set(
                r[0].strip()
                for r in conn.execute(
                    select(func.distinct(annotation_assignments.c.rater_id))
                ).fetchall()
            )

        # New assignments should be created for R3 only
        assert r1_r2_r3_count > r1_r2_count
        # R3 should now be in the rater set
        assert "R3" in all_raters


class TestDryRunWritesNothing:
    """--dry-run reports what would be created but writes nothing."""

    def test_dry_run_creates_no_assignments(self, annotation_engine, config) -> None:
        """--dry-run exits cleanly but does not write to the database."""
        if not requires_database(annotation_engine):
            return

        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]
        session_id = new_ulid()

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s=100.0,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
            )
            count_before = conn.execute(
                select(func.count()).select_from(annotation_assignments)
            ).scalar()

        # Run with --dry-run
        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "dry-run-test",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
                "--dry-run",
            ],
            annotation_engine,
        )

        with transaction(engine) as conn:
            count_after = conn.execute(
                select(func.count()).select_from(annotation_assignments)
            ).scalar()

        # Exit code should be 0 (success)
        assert result.returncode == 0
        # Database should not have changed
        assert count_after == count_before
        # Output should indicate it was a dry run
        assert "dry run" in result.stdout


class TestAssignmentCount:
    """The assignment count matches the formula."""

    def test_assignment_count_matches_formula(
        self, annotation_engine
    ) -> None:
        """Total assignments match formula from manifest.

        Reads the manifest and asserts:
        assignments_created + assignments_already_existed
          == total_clips_sampled * len(behaviours) * raters_per_clip

        Mutation caught: script multiplies wrong terms, drops a behaviour from
        assignments, or ignores raters_per_clip.
        """
        if not requires_database(annotation_engine):
            return

        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]
        session_id = new_ulid()

        duration_s = 100.0

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
            )

        cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
        raters_per_clip = 2
        round_name = f"formula-check-{new_ulid()[:8]}"

        result = run_plan_annotation(
            [
                "--raters", "R1,R2,R3",
                "--round", round_name,
                "--raters-per-clip", str(raters_per_clip),
                "--all-clips",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )
        assert result.returncode == 0

        # read manifest
        manifest_path = cfg.paths.run_outputs / f"annotation-plan-{round_name}.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        # verify the formula
        assignments_created = manifest["assignments_created"]
        assignments_already_existed = manifest["assignments_already_existed"]
        total_clips_sampled = manifest["total_clips_sampled"]
        num_behaviours = len(manifest["behaviours"])
        raters_per_clip_from_manifest = manifest["raters_per_clip"]

        total_assignments = assignments_created + assignments_already_existed
        expected = (total_clips_sampled * num_behaviours *
                    raters_per_clip_from_manifest)

        assert total_assignments == expected


class TestQueueFor:
    """Every created assignment is reachable by queue_for()."""

    def test_assignments_reachable_by_queue_for(self, annotation_engine, config) -> None:
        """After planning, queue_for() returns all assigned work for each rater."""
        if not requires_database(annotation_engine):
            return

        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]
        session_id = new_ulid()

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s=100.0,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
            )

        raters = ["R1", "R2"]
        run_plan_annotation(
            [
                "--raters", ",".join(raters),
                "--round", "queue-test",
                "--all-clips",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        with transaction(engine) as conn:
            # For each rater, get their queue with a high limit to include all assignments
            for rater in raters:
                # Pad rater ID to match database storage
                rater_padded = rater.ljust(26)
                queued = queue_for(
                    conn, rater_padded, round_name="queue-test", limit=10000
                )

                # The number of queued items should match their assignments
                rater_assignment_count = conn.execute(
                    select(func.count()).select_from(annotation_assignments).where(
                        (annotation_assignments.c.rater_id == rater_padded) &
                        (annotation_assignments.c.round_name == "queue-test")
                    )
                ).scalar()

                assert len(queued) == rater_assignment_count


class TestManifestCreation:
    """The manifest file is created with correct metadata."""

    def test_manifest_file_created(self, annotation_engine, config, tmp_path) -> None:
        """After planning, a manifest file is created with correct structure."""
        if not requires_database(annotation_engine):
            return

        engine = annotation_engine["engine"]
        college_id = annotation_engine["college_id"]
        teacher_id = annotation_engine["teacher_1"]
        consent_id = annotation_engine["consent_1"]
        session_id = new_ulid()

        with transaction(engine) as conn:
            insert_media_and_session(
                conn,
                session_id,
                teacher_id,
                college_id,
                consent_id,
                duration_s=100.0,
                is_blurred=True,
                blurred_relative_path="test/blurred.mp4",
            )

        result = run_plan_annotation(
            [
                "--raters", "R1,R2",
                "--round", "manifest-test",
                "--all-clips",
                "--config", str(REPO_ROOT / "configs" / "default.yaml"),
            ],
            annotation_engine,
        )

        # Extract manifest path from output
        assert "manifest:" in result.stdout
        manifest_line = next(
            line for line in result.stdout.split("\n")
            if "manifest:" in line
        )
        manifest_path = Path(manifest_line.split("manifest:")[1].strip())

        # Verify manifest file exists and is valid JSON
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Check required fields
        assert "round" in manifest
        assert "raters" in manifest
        assert "seed" in manifest
        assert "behaviours" in manifest
        assert "total_clips_sampled" in manifest
        assert "assignments_created" in manifest
        assert manifest["round"] == "manifest-test"
        assert set(manifest["raters"]) >= {"R1", "R2"}
