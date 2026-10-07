# -*- coding: utf-8 -*-
"""Drafting a manifest from a directory of footage.

What this decides is which files become one session, and that is the decision R2 rests on. A
drafter that merged two candidates into one, or split one across two, would plant a leakage bug
no downstream check can see - so every rule it applies is tested here against real files rather
than against a description of them.

It also must never put a person's name in the manifest. The directories in a hand-assembled
corpus are named after people, and those names are the re-identifying information D76 keeps out
of the database.
"""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.draft_manifest import (
    build,
    mark_duplicates,
    mark_part_groups,
    read_clips,
    write_codes,
    write_manifest,
)
from tests.integration.conftest import CONFIG, FFPROBE, make_clip

COLOURS = ("red", "green", "blue", "yellow", "magenta", "cyan")


def tiny(path: Path, seconds: float = 2.0, size: str = "320x240",
         colour: str | None = None) -> Path:
    """Small and fast. The drafter reads containers; it does not judge quality.

    `colour` exists because two clips generated from identical parameters are
    byte-identical, and the drafter would correctly deduplicate them before any grouping
    rule ran. Parts of a real recording share a duration and differ in content.
    """
    luma = (f"color=c={colour}:size={size}:rate=5") if colour else None
    return make_clip(path, seconds, with_audio=False, size=size, rate=5, luma=luma)


def drafted(media_dir: Path, **overrides):
    clips = read_clips(media_dir, FFPROBE)
    mark_duplicates(clips)
    notes = mark_part_groups(clips)
    options = dict(domain="microteaching", college="C-01", ffprobe=FFPROBE)
    options.update(overrides)
    return build(clips, **options), notes


class TestWhatBecomesASession:
    def test_each_file_is_its_own_session_by_default(self, tmp_path) -> None:
        tiny(tmp_path / "a.mp4", colour="red")
        tiny(tmp_path / "b.mp4", seconds=3.0, colour="green")

        draft, notes = drafted(tmp_path)

        assert len(draft.rows) == 2
        assert notes == []
        assert all(row["part_group"] == "" for row in draft.rows)

    def test_parts_cut_to_one_length_are_grouped(self, tmp_path) -> None:
        """The signature of a size-capped splitter: one stem, consecutive indices, identical
        duration and geometry."""
        for index in range(1, 4):
            tiny(tmp_path / f"lesson.{index}.mp4", seconds=2.0,
                 colour=COLOURS[index - 1])

        draft, notes = drafted(tmp_path)

        groups = {row["part_group"] for row in draft.rows}
        assert groups == {"lesson"}
        assert len(draft.rows) == 3
        assert any("3 parts" in note for note in notes)

    def test_files_of_different_lengths_are_not_grouped(self, tmp_path) -> None:
        """The check that keeps separate sessions separate. Two lessons that happen to share a
        stem are not one recording, and merging them would put one teacher's footage under
        another's name."""
        tiny(tmp_path / "lesson.1.mp4", seconds=2.0)
        tiny(tmp_path / "lesson.2.mp4", seconds=6.0)

        draft, notes = drafted(tmp_path)

        assert all(row["part_group"] == "" for row in draft.rows)
        assert any("differ by more than" in note for note in notes)

    def test_a_trailing_digit_without_a_dot_is_not_a_part(self, tmp_path) -> None:
        """`teaching@4` and `teaching@5` in the real corpus are separate sessions that end in
        consecutive digits. A splitter appends `.N`; requiring the dot is what tells them
        apart."""
        tiny(tmp_path / "teaching@4.mp4", seconds=2.0, colour="red")
        tiny(tmp_path / "teaching@5.mp4", seconds=2.0, colour="green")

        draft, notes = drafted(tmp_path)

        assert all(row["part_group"] == "" for row in draft.rows)
        assert notes == []

    def test_non_consecutive_indices_are_reported_rather_than_joined(self, tmp_path) -> None:
        """A gap means a part is missing from the directory, and joining what is present would
        produce a session shorter than the recording it claims to be."""
        for index in (1, 2, 4):
            tiny(tmp_path / f"gap.{index}.mp4", seconds=2.0, colour=COLOURS[index - 1])

        _, notes = drafted(tmp_path)

        assert any("not consecutive" in note for note in notes)

    def test_parts_of_different_sizes_are_not_grouped(self, tmp_path) -> None:
        tiny(tmp_path / "mix.1.mp4", seconds=2.0, size="320x240", colour="red")
        tiny(tmp_path / "mix.2.mp4", seconds=2.0, size="640x480", colour="green")

        _, notes = drafted(tmp_path)

        assert any("frame sizes" in note for note in notes)


class TestDuplicatesAndUnreadables:
    def test_the_second_name_for_one_recording_is_excluded(self, tmp_path) -> None:
        """Content addressing would skip it at ingest with no explanation."""
        original = tiny(tmp_path / "one.mp4")
        copy = tmp_path / "two.mp4"
        copy.write_bytes(original.read_bytes())

        draft, _ = drafted(tmp_path)

        assert len(draft.rows) == 1
        assert any("byte-identical" in why for _, why in draft.excluded)

    def test_the_copy_inside_a_named_directory_is_the_one_kept(self, tmp_path) -> None:
        """It carries a teacher; the loose copy does not."""
        (tmp_path / "Kwame").mkdir()
        inside = tiny(tmp_path / "Kwame" / "lesson.mp4")
        (tmp_path / "loose.mp4").write_bytes(inside.read_bytes())

        draft, _ = drafted(tmp_path)

        assert len(draft.rows) == 1
        assert draft.rows[0]["filename"].endswith("lesson.mp4")
        assert draft.rows[0]["teacher_code"] != ""

    def test_an_unreadable_file_is_excluded_with_its_reason(self, tmp_path) -> None:
        tiny(tmp_path / "good.mp4")
        (tmp_path / "truncated.mp4").write_bytes(b"not an mp4" * 200)

        draft, _ = drafted(tmp_path)

        assert len(draft.rows) == 1
        assert any("unreadable" in why for _, why in draft.excluded)


class TestIdentityNeverReachesTheManifest:
    def test_a_directory_name_becomes_a_neutral_code(self, tmp_path) -> None:
        """D76. The directories are named after people and the manifest is not gitignored."""
        for index, name in enumerate(("Adelaide", "patricia")):
            (tmp_path / name).mkdir()
            tiny(tmp_path / name / "lesson.mp4", colour=COLOURS[index])

        draft, _ = drafted(tmp_path)

        codes = {row["teacher_code"] for row in draft.rows}
        assert codes == {"T-01", "T-02"}
        for row in draft.rows:
            assert "Adelaide" not in row["teacher_code"]
            assert "patricia" not in row["teacher_code"]

    def test_no_identifier_column_carries_the_directory_name(self, tmp_path) -> None:
        """`filename` has to keep the real path or the loader cannot find the file, so the
        directory name does survive into the manifest. What must not happen is a name becoming
        an *identifier* - a `teacher_code` the loader would then carry into the key file and,
        through it, into the shape of the corpus."""
        (tmp_path / "Adelaide").mkdir()
        tiny(tmp_path / "Adelaide" / "lesson.mp4")

        draft, _ = drafted(tmp_path)

        for row in draft.rows:
            assert "Adelaide" not in row["teacher_code"]
            assert "Adelaide" not in row["college_code"]
            assert "Adelaide" in row["filename"], "the path must stay resolvable"

    def test_the_directory_name_reaches_the_key_file(self, tmp_path) -> None:
        """Which is where the correspondence belongs: gitignored, and kept away from the
        video it re-identifies. D76."""
        (tmp_path / "Adelaide").mkdir()
        tiny(tmp_path / "Adelaide" / "lesson.mp4")

        draft, _ = drafted(tmp_path)
        keys = tmp_path / "keys.csv"
        write_codes(keys, draft.codes)

        assert "Adelaide" in keys.read_text(encoding="utf-8")
        assert "T-01" in keys.read_text(encoding="utf-8")

    def test_the_filename_is_not_what_the_database_stores(self) -> None:
        """The guarantee that actually matters. `media_objects.relative_path` is derived from
        the content hash, and the uploaded filename supplies only the extension, so a directory
        named after a person cannot reach the database through this path."""
        from praxis.ingest.storage import normalised_extension, relative_path_for

        stored = relative_path_for("a" * 64, normalised_extension(
            "Adelaide/lesson.mp4", CONFIG.ingest.accepted_containers))

        assert "Adelaide" not in stored
        assert "lesson" not in stored
        assert stored.endswith(".mp4")

    def test_a_file_outside_any_directory_gets_no_teacher(self, tmp_path) -> None:
        """The decisive one. teacher_id is the split key, so an unattributable file is left
        blank rather than guessed: inventing one merges or splits a candidate, and the first is
        a leakage bug nothing downstream can detect."""
        tiny(tmp_path / "unattributed.mp4")

        draft, _ = drafted(tmp_path)

        assert draft.rows[0]["teacher_code"] == ""


class TestWhatIsDerivedAndWhatIsNot:
    def test_a_date_in_the_filename_is_read(self, tmp_path) -> None:
        tiny(tmp_path / "WhatsApp Video 2026-09-23 at 3.00.19 PM.mp4")
        draft, _ = drafted(tmp_path)
        assert draft.rows[0]["recorded_on"] == "2026-09-23"

    def test_a_compact_date_is_read_too(self, tmp_path) -> None:
        tiny(tmp_path / "20260925_122420-002.mp4")
        draft, _ = drafted(tmp_path)
        assert draft.rows[0]["recorded_on"] == "2026-09-25"

    def test_a_file_stating_no_date_is_left_blank(self, tmp_path) -> None:
        tiny(tmp_path / "teaching.mp4")
        draft, _ = drafted(tmp_path)
        assert draft.rows[0]["recorded_on"] == ""

    def test_the_domain_is_the_one_asked_for(self, tmp_path) -> None:
        tiny(tmp_path / "a.mp4")
        draft, _ = drafted(tmp_path, domain="classroom")
        assert draft.rows[0]["domain"] == "classroom"

    def test_covariates_are_never_invented(self, tmp_path) -> None:
        """They are measured at recording time and cannot be recovered from the file."""
        tiny(tmp_path / "a.mp4")
        draft, _ = drafted(tmp_path)
        for column in ("camera_distance_m", "room_area_m2", "pupil_count",
                       "ambient_noise_dba", "teacher_movement_range_m"):
            assert draft.rows[0][column] == ""


def test_the_draft_is_readable_by_the_loader(tmp_path) -> None:
    """The two scripts have to agree on the columns, and a drifted header is the kind of thing
    that is only noticed when a batch refuses every row."""
    from praxis.ingest.corpus import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

    (tmp_path / "Kwame").mkdir()
    tiny(tmp_path / "Kwame" / "WhatsApp Video 2026-09-23 at 1.00.00 PM.mp4")
    draft, _ = drafted(tmp_path)
    manifest = tmp_path / "out.csv"
    write_manifest(manifest, draft.rows)

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))

    assert set(REQUIRED_COLUMNS) <= set(header)
    assert set(header) <= set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)


def test_a_drafted_manifest_is_gitignored(repo_root: Path) -> None:
    """Its `filename` column holds real paths, and a hand-assembled corpus has directories
    named after the people in the footage. Committing one re-identifies the corpus as surely as
    committing the key file would. The template under docs/ carries no paths and stays. D87."""
    ignored = (repo_root.parent / ".gitignore").read_text(encoding="utf-8")

    assert "sessions-draft.csv" in ignored
    assert "sessions-template.csv" not in ignored, (
        "the committed template has no paths in it and must not be ignored")
