# -*- coding: utf-8 -*-
"""The bulk corpus loader, against a real database and a real ffprobe.

This script is the one that will touch the actual recordings, so what is tested is the
behaviour that protects them: that a manifest is validated whole before any file is read, that
consent is resolved and never invented, that a rerun does not duplicate a session, and that one
bad file does not stop the batch.

The pseudonym key file is tested too. It is the artefact that re-identifies the corpus, and the
reason it exists outside the database is that nothing inside the database may.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import func, select

from praxis.db import transaction
from praxis.db.schema import (
    colleges,
    consent_records,
    media_objects,
    schools,
    sessions,
    teachers,
)
from praxis.ids import new_ulid
from praxis.ingest.corpus import (
    ManifestError,
    group_parts,
    read_keymap,
    read_manifest,
    resolve_identifiers,
    run_batch,
    write_keymap,
)
from tests.integration.conftest import (
    FFMPEG,
    FFPROBE,
    lenient,
    make_clip,
    requires_database,
)

HEADER = ("filename,teacher_code,college_code,domain,school_code,recorded_on,subject,"
          "grade_level,camera_distance_m,room_area_m2,pupil_count,ambient_noise_dba,"
          "teacher_movement_range_m\n")
FULL = "3.5,48,31,52,4.2"
# A classroom row must name its school and a microteaching row must not. D98.
SCHOOL = "SCH-ONPAPER"


@pytest.fixture
def party(engine):
    """A college, a school, a teacher and an active consent covering both domains."""
    if engine is None:
        return None

    college_id, teacher_id, consent_id = new_ulid(), new_ulid(), new_ulid()
    school_id = new_ulid()
    with transaction(engine) as conn:
        conn.execute(colleges.insert().values(
            college_id=college_id, code=f"CORP-{college_id[-8:]}", created_at=func.now()))
        # A classroom row names a school and `resolve_identifiers(create_missing=False)` has to
        # find it. Registered here rather than minted on the fly, so these tests exercise the
        # resolution path the real loader takes rather than the one that invents an id. D98.
        conn.execute(schools.insert().values(
            school_id=school_id, code=f"SCH-{school_id[-8:]}", created_at=func.now()))
        conn.execute(teachers.insert().values(
            teacher_id=teacher_id, college_id=college_id, created_at=func.now()))
        conn.execute(consent_records.insert().values(
            consent_id=consent_id, subject_type="teacher", teacher_id=teacher_id,
            college_id=college_id, purpose="research", recipients="supervisors",
            scope="both", granted_on=date(2026, 1, 1), document_ref="forms/1",
            created_at=func.now()))
    return {"college_id": college_id, "teacher_id": teacher_id, "consent_id": consent_id,
            "college_code": f"CORP-{college_id[-8:]}", "teacher_code": "T-01",
            "school_id": school_id, "school_code": f"SCH-{school_id[-8:]}"}


# Two codes for the tests that only exercise `group_parts`, a pure function over manifest rows
# which resolves no identifier against anything. Borrowing them from the `party` fixture made
# those tests fail wherever no database was configured, while testing nothing that needed one.
UNRESOLVED = {"teacher_code": "T-01", "college_code": "CORP-ONPAPER"}


def manifest_at(path: Path, rows: str) -> Path:
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def keymap_at(path: Path, party: dict) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["teacher_code", "teacher_id", "college_code", "college_id",
                         "school_code", "school_id"])
        writer.writerow([party["teacher_code"], party["teacher_id"],
                         party["college_code"], party["college_id"], "", ""])
        writer.writerow(["", "", "", "", party["school_code"], party["school_id"]])
    return path


class TestManifestValidation:
    """Everything wrong is reported at once, because the fix happens in a spreadsheet."""

    def test_a_missing_required_column_is_refused_by_name(self, tmp_path) -> None:
        bad = tmp_path / "m.csv"
        bad.write_text("filename,teacher_code\na.mp4,T-01\n", encoding="utf-8")
        with pytest.raises(ManifestError, match="missing required columns"):
            read_manifest(bad, tmp_path)

    def test_every_problem_is_collected_not_just_the_first(self, tmp_path) -> None:
        manifest = manifest_at(tmp_path / "m.csv",
                               f"missing.mp4,T-01,ACC,microteaching,,2026-03-04,,,{FULL}\n"
                               f"bad.mp4,T-02,ACC,staffroom,,04/03/2026,,,{FULL}\n")
        _, problems = read_manifest(manifest, tmp_path)

        assert len(problems) >= 3
        assert any("no file at" in p for p in problems)
        assert any("not an ISO date" in p for p in problems)

    def test_a_domain_outside_the_two_is_refused(self, tmp_path) -> None:
        """The domain decides the shift level, so one that is neither cannot be placed."""
        manifest = manifest_at(tmp_path / "m.csv",
                               f"a.mp4,T-01,ACC,staffroom,,2026-03-04,,,{FULL}\n")
        _, problems = read_manifest(manifest, tmp_path)
        assert any("shift level" in p for p in problems)

    def test_a_duplicate_filename_is_refused(self, tmp_path) -> None:
        make_clip(tmp_path / "a.mp4", 2.0)
        manifest = manifest_at(tmp_path / "m.csv",
                               f"a.mp4,T-01,ACC,microteaching,,2026-03-04,,,{FULL}\n"
                               f"a.mp4,T-02,ACC,microteaching,,2026-03-05,,,{FULL}\n")
        _, problems = read_manifest(manifest, tmp_path)
        assert any("already appears on line" in p for p in problems)

    def test_missing_covariates_are_reported_but_not_refused(self, tmp_path) -> None:
        """Phase 6's attribution needs them; ingest does not. A session without them is
        storable and cannot appear in the shift analysis, and the operator is told which."""
        make_clip(tmp_path / "a.mp4", 2.0)
        manifest = manifest_at(tmp_path / "m.csv",
                               "a.mp4,T-01,ACC,microteaching,,2026-03-04,,,,,,,\n")
        rows, problems = read_manifest(manifest, tmp_path)

        assert problems == []
        assert rows[0].missing_covariates == [
            "camera_distance_m", "room_area_m2", "pupil_count",
            "ambient_noise_dba", "teacher_movement_range_m"]


class TestConsentResolution:
    def test_consent_is_resolved_from_the_teacher_not_the_manifest(self, engine, party,
                                                                    tmp_path) -> None:
        if not requires_database(engine):
            return

        make_clip(tmp_path / "a.mp4", 2.0)
        manifest = manifest_at(
            tmp_path / "m.csv",
            f"a.mp4,{party['teacher_code']},{party['college_code']},"
            f"microteaching,,2026-03-04,,,{FULL}\n")
        rows, _ = read_manifest(manifest, tmp_path)
        mapping = read_keymap(keymap_at(tmp_path / "keys.csv", party))

        with transaction(engine) as conn:
            problems = resolve_identifiers(conn, rows, mapping, create_missing=False)

        assert problems == []
        assert rows[0].consent_id == party["consent_id"]

    def test_a_teacher_without_consent_is_refused_and_none_is_created(self, engine,
                                                                       party,
                                                                       tmp_path) -> None:
        """A consent record stands for a signed form. Minting one from a spreadsheet row
        would be manufacturing the legal basis for processing the footage."""
        if not requires_database(engine):
            return

        orphan_id = new_ulid()
        with transaction(engine) as conn:
            conn.execute(teachers.insert().values(
                teacher_id=orphan_id, college_id=party["college_id"],
                created_at=func.now()))
            before = conn.execute(
                select(func.count()).select_from(consent_records)).scalar()

        make_clip(tmp_path / "a.mp4", 2.0)
        manifest = manifest_at(
            tmp_path / "m.csv",
            f"a.mp4,T-ORPHAN,{party['college_code']},microteaching,,2026-03-04,,,{FULL}\n")
        rows, _ = read_manifest(manifest, tmp_path)

        keys = tmp_path / "keys.csv"
        with open(keys, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["teacher_code", "teacher_id", "college_code", "college_id"])
            writer.writerow(["T-ORPHAN", orphan_id, party["college_code"],
                             party["college_id"]])

        with transaction(engine) as conn:
            problems = resolve_identifiers(conn, rows, keymap_from(keys),
                                           create_missing=False)
            after = conn.execute(
                select(func.count()).select_from(consent_records)).scalar()

        assert any("no active consent" in p for p in problems)
        assert after == before, "no consent record may be created by this script"

    def test_a_key_file_pointing_at_an_unknown_teacher_is_refused(self, engine, party,
                                                                   tmp_path) -> None:
        """The key file and the database disagreeing about who someone is must stop the run,
        not be resolved in one direction."""
        if not requires_database(engine):
            return

        make_clip(tmp_path / "a.mp4", 2.0)
        manifest = manifest_at(
            tmp_path / "m.csv",
            f"a.mp4,T-GHOST,{party['college_code']},microteaching,,2026-03-04,,,{FULL}\n")
        rows, _ = read_manifest(manifest, tmp_path)

        keys = tmp_path / "keys.csv"
        with open(keys, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["teacher_code", "teacher_id", "college_code", "college_id"])
            writer.writerow(["T-GHOST", new_ulid(), party["college_code"],
                             party["college_id"]])

        with transaction(engine) as conn:
            problems = resolve_identifiers(conn, rows, keymap_from(keys),
                                           create_missing=False)
        assert any("disagree about who this is" in p for p in problems)


def keymap_from(path: Path):
    return read_keymap(path)


class TestBatch:
    def test_a_session_is_ingested_with_its_covariates(self, engine, party,
                                                        media_root, tmp_path) -> None:
        if not requires_database(engine):
            return

        make_clip(tmp_path / "a.mp4", 2.0)
        manifest = manifest_at(
            tmp_path / "m.csv",
            f"a.mp4,{party['teacher_code']},{party['college_code']},"
            f"classroom,{party['school_code']},2026-03-04,science,JHS2,{FULL}\n")
        rows, _ = read_manifest(manifest, tmp_path)
        mapping = read_keymap(keymap_at(tmp_path / "keys.csv", party))
        with transaction(engine) as conn:
            assert resolve_identifiers(conn, rows, mapping, create_missing=False) == []

        ingested, skipped, failed = run_batch(
            group_parts(rows)[0], engine=engine, config=lenient(), ffprobe=FFPROBE,
            ffmpeg=FFMPEG,
            media_root=media_root, report=lambda _m: None)

        assert (ingested, skipped, failed) == (1, 0, 0)
        with transaction(engine) as conn:
            row = conn.execute(select(sessions).where(
                sessions.c.teacher_id == party["teacher_id"])).mappings().first()
        assert row["domain"] == "classroom"
        assert float(row["camera_distance_m"]) == 3.5
        assert row["pupil_count"] == 31

    def test_running_it_twice_does_not_duplicate_the_session(self, engine, party,
                                                              media_root,
                                                              tmp_path) -> None:
        """An interrupted run is resumed by running it again, which is only safe if the
        loader is idempotent by content.

        The duration is unique to this test on purpose. `make_clip` is deterministic, so two
        tests building a clip with the same parameters produce byte-identical files, and the
        scratch database outlives a single test - the loader would skip this one as already
        stored by another test, and the assertion would fail for a reason unrelated to it.
        """
        if not requires_database(engine):
            return

        make_clip(tmp_path / "a.mp4", 3.0)
        manifest = manifest_at(
            tmp_path / "m.csv",
            f"a.mp4,{party['teacher_code']},{party['college_code']},"
            f"microteaching,,2026-03-04,,,{FULL}\n")
        keys = keymap_at(tmp_path / "keys.csv", party)

        def load():
            rows, _ = read_manifest(manifest, tmp_path)
            with transaction(engine) as conn:
                resolve_identifiers(conn, rows, read_keymap(keys), create_missing=False)
            return run_batch(group_parts(rows)[0], engine=engine, config=lenient(),
                             ffprobe=FFPROBE,
                             ffmpeg=FFMPEG, media_root=media_root,
                             report=lambda _m: None)

        assert load() == (1, 0, 0)
        assert load() == (0, 1, 0), "the second run must skip, not re-ingest"

        with transaction(engine) as conn:
            count = conn.execute(select(func.count()).select_from(sessions).where(
                sessions.c.teacher_id == party["teacher_id"])).scalar()
        assert count == 1

    def test_one_bad_file_does_not_stop_the_batch(self, engine, party, media_root,
                                                   tmp_path) -> None:
        """A corpus of a hundred sessions must not be held up by the one that failed."""
        if not requires_database(engine):
            return

        # A duration unique to this test, for the reason given above.
        make_clip(tmp_path / "good.mp4", 4.0)
        (tmp_path / "broken.mp4").write_bytes(b"not a video at all" * 256)
        manifest = manifest_at(
            tmp_path / "m.csv",
            f"broken.mp4,{party['teacher_code']},{party['college_code']},"
            f"microteaching,,2026-03-04,,,{FULL}\n"
            f"good.mp4,{party['teacher_code']},{party['college_code']},"
            f"microteaching,,2026-03-05,,,{FULL}\n")
        rows, _ = read_manifest(manifest, tmp_path)
        with transaction(engine) as conn:
            resolve_identifiers(conn, rows, read_keymap(keymap_at(tmp_path / "k.csv", party)),
                                create_missing=False)

        ingested, _, failed = run_batch(
            group_parts(rows)[0], engine=engine, config=lenient(), ffprobe=FFPROBE,
            ffmpeg=FFMPEG,
            media_root=media_root, report=lambda _m: None)

        assert (ingested, failed) == (1, 1)


class TestKeyFile:
    def test_the_key_file_round_trips(self, tmp_path) -> None:
        from praxis.ingest.corpus import Row

        row = Row(line=2, filename="a.mp4", path=tmp_path / "a.mp4",
                  teacher_code="T-01", college_code="ACC", domain="microteaching",
                  recorded_on=date(2026, 3, 4))
        row.teacher_id, row.college_id = new_ulid(), new_ulid()

        path = tmp_path / "keys.csv"
        write_keymap(path, {"teachers": {}, "colleges": {}}, [row])
        mapping = read_keymap(path)

        assert mapping["teachers"]["T-01"] == row.teacher_id
        assert mapping["colleges"]["ACC"] == row.college_id

    def test_an_absent_key_file_is_empty_rather_than_an_error(self, tmp_path) -> None:
        """The first run has no key file yet, and --create-missing writes one."""
        assert read_keymap(tmp_path / "nope.csv") == {
            "teachers": {}, "colleges": {}, "schools": {}}


def test_the_key_file_is_gitignored(repo_root: Path) -> None:
    """It is the artefact that re-identifies the corpus. The database deliberately cannot,
    and committing the map would undo that in one line. D76."""
    ignored = (repo_root.parent / ".gitignore").read_text(encoding="utf-8")
    assert "keys/" in ignored or "pseudonyms" in ignored


# ---------------------------------------------------------------------------
# Split recordings
#
# One lesson in the real corpus arrived as seven files of exactly 208 seconds, cut by a transfer
# tool that caps file size. Ingested as seven sessions it would put one teacher in several
# partitions, which is R2 broken by a filename convention. D82.
# ---------------------------------------------------------------------------

PARTS_HEADER = ("filename,teacher_code,college_code,domain,school_code,recorded_on,subject,"
                "grade_level,part_group,camera_distance_m,room_area_m2,pupil_count,"
                "ambient_noise_dba,teacher_movement_range_m\n")


def parts_manifest(path: Path, rows: str) -> Path:
    path.write_text(PARTS_HEADER + rows, encoding="utf-8")
    return path


class TestSplitRecordings:
    def part_rows(self, party, group: str, names: list[str], **overrides) -> str:
        fields = {"domain": "classroom", "recorded_on": "2026-03-04",
                  "school": party.get("school_code", SCHOOL),
                  "subject": "science", "grade_level": "JHS2", "teacher": party["teacher_code"]}
        fields.update(overrides)
        return "".join(
            f"{name},{fields['teacher']},{party['college_code']},{fields['domain']},"
            f"{fields['school']},{fields['recorded_on']},{fields['subject']},"
            f"{fields['grade_level']},{group},{FULL}\n"
            for name in names)

    def test_parts_of_one_recording_become_one_session(self, engine, party, media_root,
                                                        tmp_path) -> None:
        if not requires_database(engine):
            return

        for name in ("p1.mp4", "p2.mp4", "p3.mp4"):
            make_clip(tmp_path / name, 2.0)
        manifest = parts_manifest(
            tmp_path / "m.csv",
            self.part_rows(party, "lesson-7", ["p1.mp4", "p2.mp4", "p3.mp4"]))

        rows, problems = read_manifest(manifest, tmp_path)
        assert problems == []
        groups, grouping = group_parts(rows)
        assert grouping == []
        assert len(groups) == 1 and len(groups[0]) == 3

        with transaction(engine) as conn:
            resolve_identifiers(conn, rows, read_keymap(keymap_at(tmp_path / "k.csv", party)),
                                create_missing=False)

        ingested, skipped, failed = run_batch(
            groups, engine=engine, config=lenient(), ffprobe=FFPROBE, ffmpeg=FFMPEG,
            media_root=media_root, report=lambda _m: None)

        assert (ingested, skipped, failed) == (1, 0, 0)

        with transaction(engine) as conn:
            stored = conn.execute(select(sessions).where(
                sessions.c.teacher_id == party["teacher_id"])).mappings().all()
        assert len(stored) == 1, "three parts must not become three sessions"

    def test_the_joined_session_carries_the_whole_recording(self, engine, party, media_root,
                                                             tmp_path) -> None:
        """The duration is the sum of the parts. A session shorter than its own footage would be
        scored as if the missing minutes were never recorded."""
        if not requires_database(engine):
            return

        for name in ("q1.mp4", "q2.mp4"):
            make_clip(tmp_path / name, 3.0)
        manifest = parts_manifest(tmp_path / "m.csv",
                                  self.part_rows(party, "lesson-8", ["q1.mp4", "q2.mp4"]))
        rows, _ = read_manifest(manifest, tmp_path)
        groups, _ = group_parts(rows)
        with transaction(engine) as conn:
            resolve_identifiers(conn, rows, read_keymap(keymap_at(tmp_path / "k.csv", party)),
                                create_missing=False)

        run_batch(groups, engine=engine, config=lenient(), ffprobe=FFPROBE, ffmpeg=FFMPEG,
                  media_root=media_root, report=lambda _m: None)

        with transaction(engine) as conn:
            stored = conn.execute(select(media_objects)).mappings().all()
        joined = [m for m in stored if m["duration_s"] > 5.0]
        assert joined, f"no media object spans both parts: {[m['duration_s'] for m in stored]}"

    def test_a_group_whose_parts_disagree_about_the_session_is_refused(self, tmp_path) -> None:
        """Two parts naming two teachers is a mistake in the field log. Only one value can
        survive into `sessions`, and silently taking the first would hide it."""
        for name in ("r1.mp4", "r2.mp4"):
            make_clip(tmp_path / name, 2.0)
        manifest = parts_manifest(
            tmp_path / "m.csv",
            self.part_rows(UNRESOLVED, "lesson-9", ["r1.mp4"])
            + self.part_rows(UNRESOLVED, "lesson-9", ["r2.mp4"], domain="microteaching",
                             school=""))

        rows, _ = read_manifest(manifest, tmp_path)
        _, problems = group_parts(rows)

        assert any("disagrees on domain" in p for p in problems), problems

    def test_a_group_of_one_is_refused_rather_than_ingested_quietly(self, tmp_path) -> None:
        make_clip(tmp_path / "s1.mp4", 2.0)
        manifest = parts_manifest(tmp_path / "m.csv",
                                  self.part_rows(UNRESOLVED, "lesson-10", ["s1.mp4"]))
        rows, _ = read_manifest(manifest, tmp_path)
        _, problems = group_parts(rows)

        assert any("only one row" in p for p in problems), problems

    def test_an_ordinary_session_needs_no_part_group(self, tmp_path) -> None:
        """The column is optional and a blank one means what it says."""
        make_clip(tmp_path / "t1.mp4", 2.0)
        manifest = parts_manifest(
            tmp_path / "m.csv",
            f"t1.mp4,{UNRESOLVED['teacher_code']},{UNRESOLVED['college_code']},classroom,"
            f"{SCHOOL},2026-03-04,science,JHS2,,{FULL}\n")
        rows, problems = read_manifest(manifest, tmp_path)

        assert problems == [] and rows[0].part_group is None
        groups, grouping = group_parts(rows)
        assert grouping == [] and len(groups) == 1 and len(groups[0]) == 1
