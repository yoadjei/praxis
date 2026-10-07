# -*- coding: utf-8 -*-
"""Loading a corpus from a field-log manifest. One session per row.

Separated from `scripts/ingest_corpus.py` so that the loader is a library and the script is
only a command line over it. `praxis.phases.corpus` needs the same code to run phase 1, and a
library reaching into a script to get it would invert the dependency - the package would then
be unimportable wherever `scripts/` was not copied alongside it.

**Why a key file and not a column.** `teachers` carries a ULID, a college and a cohort, and
nothing else - no name, no staff number, no code. That is deliberate: R1 keeps identity out of
the system, and a `teacher_code` column would put the field log's re-identifying key inside
the database the thesis ships. So the mapping from the field log's codes to ULIDs lives in a
separate key file that the candidate holds, listed in `.gitignore`, and belongs on neither the
media volume nor the working volume alongside the data it would re-identify. See D76.

**Consent is resolved, never created.** A consent record is the database's account of a signed
paper form, and code that minted one from a spreadsheet row would be manufacturing the legal
basis for processing the footage. The loader finds the active consent for that teacher whose
scope covers the session's domain, and refuses the row if there is not exactly one.
`scripts/import_consents.py` is where a consent enters the system, from a file naming the
signed forms.

**Every row is validated before any file is touched.** A batch that ingests forty sessions and
then stops on a typo in the forty-first leaves the operator to work out what happened; this
reads the whole manifest, reports everything wrong with it, and only then starts writing.

**Idempotent by content.** A file whose SHA-256 is already in `media_objects` is skipped
rather than re-ingested, so an interrupted run is resumed by running it again.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import func, select

from praxis.db import transaction
from praxis.db.schema import (
    active_consents,
    colleges,
    media_objects,
    schools,
    teachers,
)
from praxis.ids import new_ulid
from praxis.ingest.consent import COVERAGE
from praxis.ingest.parts import PartGroup, join_parts
from praxis.ingest.service import IngestRequest, ingest

# What the field log must supply. The five covariates are optional to ingest and required by
# Phase 6: the attribution model regresses the per-session performance drop on exactly these,
# so a session missing them is ingestable and cannot appear in the shift analysis.
MICROTEACHING = "microteaching"
CLASSROOM = "classroom"
DOMAINS = (MICROTEACHING, CLASSROOM)

REQUIRED_COLUMNS = ("filename", "teacher_code", "college_code", "domain", "recorded_on")
COVARIATE_COLUMNS = ("camera_distance_m", "room_area_m2", "pupil_count",
                     "ambient_noise_dba", "teacher_movement_range_m")
OPTIONAL_COLUMNS = ("school_code", "subject", "grade_level", "part_group", *COVARIATE_COLUMNS)

# Rows sharing a `part_group` are one session that a transfer tool cut into pieces, and they
# are rejoined before anything is ingested. Without it each piece becomes its own session and
# one teacher lands in several partitions, which is R2 broken by a filename convention. The
# column is optional: an ordinary session leaves it blank. D82.

KEYMAP_COLUMNS = ("teacher_code", "teacher_id", "college_code", "college_id")
# The school half of the key file. Separate rows rather than more columns on the teacher rows:
# a school is not a property of a teacher - several colleges send students to one school - so
# pairing them would write the same school once per teacher and let two rows disagree. D98.
SCHOOL_KEYMAP_COLUMNS = ("school_code", "school_id")


class ManifestError(RuntimeError):
    """The manifest cannot be used, with every reason listed rather than the first."""


@dataclass
class Row:
    """One validated manifest row, with the identifiers it resolved to."""

    line: int
    filename: str
    path: Path
    teacher_code: str
    college_code: str
    domain: str
    recorded_on: date
    school_code: str | None = None
    subject: str | None = None
    grade_level: str | None = None
    part_group: str | None = None
    covariates: dict[str, float | int | None] = field(default_factory=dict)
    teacher_id: str | None = None
    college_id: str | None = None
    school_id: str | None = None
    consent_id: str | None = None

    @property
    def missing_covariates(self) -> list[str]:
        return [name for name in COVARIATE_COLUMNS if self.covariates.get(name) is None]


def _number(value: str, name: str, line: int, *, integer: bool = False):
    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(text) if integer else float(text)
    except ValueError as bad:
        raise ManifestError(f"line {line}: {name}={value!r} is not a number") from bad


def read_manifest(path: Path, media_dir: Path) -> tuple[list[Row], list[str]]:
    """Parse and check the manifest, returning the rows and every problem found.

    Problems are collected rather than raised one at a time, because the operator is going to
    fix them in a spreadsheet and wants the whole list.
    """
    if not path.is_file():
        raise ManifestError(f"no manifest at {path}")

    rows: list[Row] = []
    problems: list[str] = []
    seen: dict[str, int] = {}

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise ManifestError(
                f"{path.name} is missing required columns {missing}. Required: "
                f"{list(REQUIRED_COLUMNS)}; optional: {list(OPTIONAL_COLUMNS)}")

        for line, raw in enumerate(reader, start=2):
            filename = (raw.get("filename") or "").strip()
            if not filename:
                problems.append(f"line {line}: filename is empty")
                continue
            if filename in seen:
                problems.append(
                    f"line {line}: filename {filename!r} already appears on line "
                    f"{seen[filename]}")
                continue
            seen[filename] = line

            media_path = media_dir / filename
            if not media_path.is_file():
                problems.append(f"line {line}: no file at {media_path}")

            recorded = (raw.get("recorded_on") or "").strip()
            try:
                recorded_on = datetime.strptime(recorded, "%Y-%m-%d").date()
            except ValueError:
                problems.append(
                    f"line {line}: recorded_on={recorded!r} is not an ISO date (YYYY-MM-DD)")
                continue

            domain = (raw.get("domain") or "").strip()
            if domain not in DOMAINS:
                problems.append(
                    f"line {line}: domain={domain!r} must be 'microteaching' or 'classroom'; "
                    f"it decides the shift level the session belongs to")
                continue

            try:
                covariates = {
                    "camera_distance_m": _number(raw.get("camera_distance_m", ""),
                                                 "camera_distance_m", line),
                    "room_area_m2": _number(raw.get("room_area_m2", ""),
                                            "room_area_m2", line),
                    "pupil_count": _number(raw.get("pupil_count", ""), "pupil_count", line,
                                           integer=True),
                    "ambient_noise_dba": _number(raw.get("ambient_noise_dba", ""),
                                                 "ambient_noise_dba", line),
                    "teacher_movement_range_m": _number(
                        raw.get("teacher_movement_range_m", ""),
                        "teacher_movement_range_m", line),
                }
            except ManifestError as bad:
                problems.append(str(bad))
                continue

            teacher_code = (raw.get("teacher_code") or "").strip()
            college_code = (raw.get("college_code") or "").strip()
            if not teacher_code or not college_code:
                problems.append(f"line {line}: teacher_code and college_code are both required")
                continue

            # The school is what places a practicum session on the shift ladder, so a classroom
            # row without one is refused here rather than ingested and discovered later by
            # `a_classroom_session_names_its_school`. Microteaching happens on a college campus
            # and has no basic school, so a code there is the error. D98.
            school_code = (raw.get("school_code") or "").strip() or None
            if domain == CLASSROOM and school_code is None:
                problems.append(
                    f"line {line}: a classroom recording must name its school_code. S1 holds a "
                    f"school out, and a session with none would join the training pool instead "
                    f"of being held out.")
                continue
            if domain != CLASSROOM and school_code is not None:
                problems.append(
                    f"line {line}: school_code {school_code!r} on a {domain} recording. "
                    f"Microteaching happens on a college campus and has no basic school, so "
                    f"this either names the wrong domain or the wrong school.")
                continue

            rows.append(Row(
                line=line, filename=filename, path=media_path,
                teacher_code=teacher_code, college_code=college_code,
                school_code=school_code,
                domain=domain, recorded_on=recorded_on,
                subject=(raw.get("subject") or "").strip() or None,
                grade_level=(raw.get("grade_level") or "").strip() or None,
                part_group=(raw.get("part_group") or "").strip() or None,
                covariates=covariates))

    return rows, problems


def group_parts(rows: list[Row]) -> tuple[list[list[Row]], list[str]]:
    """Collapse rows sharing a `part_group` into one session each, in manifest order.

    Returns a list of row-groups - a plain session is a group of one - and every problem found.

    Rows of one group must agree on everything that identifies the session, because they are
    about to become a single row in `sessions` and only one value of each can survive. Letting
    them differ would mean the loader silently picked the first, and a group whose parts carry
    two different teachers is a mistake in the field log that must be seen, not resolved. D82.
    """
    order: list[str] = []
    grouped: dict[str, list[Row]] = {}
    singles: list[list[Row]] = []
    problems: list[str] = []

    for row in rows:
        if row.part_group is None:
            singles.append([row])
            order.append(f"\x00single:{row.filename}")
            grouped[order[-1]] = singles[-1]
            continue
        if row.part_group not in grouped:
            order.append(row.part_group)
            grouped[row.part_group] = []
        grouped[row.part_group].append(row)

    for key in order:
        members = grouped[key]
        if key.startswith("\x00single:") or len(members) == 1:
            if not key.startswith("\x00single:"):
                problems.append(
                    f"line {members[0].line}: part_group {members[0].part_group!r} has only "
                    f"one row. A group is two or more files; leave the column blank for an "
                    f"ordinary session.")
            continue

        first = members[0]
        for other in members[1:]:
            for label, mine, theirs in (
                ("teacher_code", first.teacher_code, other.teacher_code),
                ("college_code", first.college_code, other.college_code),
                ("school_code", first.school_code, other.school_code),
                ("domain", first.domain, other.domain),
                ("recorded_on", first.recorded_on, other.recorded_on),
                ("subject", first.subject, other.subject),
                ("grade_level", first.grade_level, other.grade_level),
            ):
                if mine != theirs:
                    problems.append(
                        f"line {other.line}: part_group {first.part_group!r} disagrees on "
                        f"{label} ({mine!r} on line {first.line} against {theirs!r}). The "
                        f"parts of one recording describe one session.")
            if other.covariates != first.covariates:
                problems.append(
                    f"line {other.line}: part_group {first.part_group!r} disagrees on its "
                    f"covariates with line {first.line}. They describe the same room.")

    return [grouped[key] for key in order], problems


def read_keymap(path: Path) -> dict[str, dict[str, str]]:
    """The field log's codes to the ULIDs they stand for.

    Held outside the database on purpose: it is the file that re-identifies the corpus, and the
    system is built so that nothing inside it can. Keep it somewhere the video is not.
    """
    empty: dict[str, dict[str, str]] = {"teachers": {}, "colleges": {}, "schools": {}}
    if not path.is_file():
        return empty

    mapping = empty
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line, raw in enumerate(csv.DictReader(handle), start=2):
            teacher_code = (raw.get("teacher_code") or "").strip()
            college_code = (raw.get("college_code") or "").strip()
            school_code = (raw.get("school_code") or "").strip()
            if college_code and (raw.get("college_id") or "").strip():
                mapping["colleges"][college_code] = raw["college_id"].strip()
            if teacher_code and (raw.get("teacher_id") or "").strip():
                mapping["teachers"][teacher_code] = raw["teacher_id"].strip()
            if school_code and (raw.get("school_id") or "").strip():
                mapping["schools"][school_code] = raw["school_id"].strip()
            if not teacher_code and not college_code and not school_code:
                raise ManifestError(f"{path.name} line {line}: no code on this row")
    return mapping


def write_keymap(path: Path, mapping: dict[str, dict[str, str]],
                 rows: list[Row]) -> None:
    """Rewrite the key file, preserving every pairing the rows established."""
    pairs: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        if row.teacher_id and row.college_id:
            pairs[row.teacher_code] = (row.teacher_id, row.college_code, row.college_id)

    schools_seen = {row.school_code: row.school_id for row in rows
                    if row.school_code and row.school_id}
    schools_seen.update({code: found for code, found in mapping.get("schools", {}).items()
                         if code not in schools_seen})

    # One header carrying both shapes. A teacher row leaves the school columns empty and a
    # school row leaves the teacher columns empty, which is what `read_keymap` expects: it takes
    # whatever pairing a row completes and refuses a row that completes none.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*KEYMAP_COLUMNS, *SCHOOL_KEYMAP_COLUMNS])
        for teacher_code, (teacher_id, college_code, college_id) in sorted(pairs.items()):
            writer.writerow([teacher_code, teacher_id, college_code, college_id, "", ""])
        for school_code, school_id in sorted(schools_seen.items()):
            writer.writerow(["", "", "", "", school_code, school_id])


def resolve_identifiers(connection, rows: list[Row], mapping: dict[str, dict[str, str]],
                        *, create_missing: bool) -> list[str]:
    """Attach college_id, school_id, teacher_id and consent_id to every row, or say why not."""
    problems: list[str] = []

    for row in rows:
        if row.school_code is not None:
            school_id = mapping.get("schools", {}).get(row.school_code)
            if school_id is None:
                school_id = connection.execute(
                    select(schools.c.school_id).where(schools.c.code == row.school_code)
                ).scalar_one_or_none()
            if school_id is None:
                if not create_missing:
                    problems.append(
                        f"line {row.line}: school {row.school_code!r} is not in the key file "
                        f"or the database; pass --create-missing to register it")
                    continue
                school_id = new_ulid()
                connection.execute(schools.insert().values(
                    school_id=school_id, code=row.school_code, created_at=func.now()))
                mapping.setdefault("schools", {})[row.school_code] = school_id
            row.school_id = school_id

        college_id = mapping["colleges"].get(row.college_code)
        if college_id is None:
            found = connection.execute(
                select(colleges.c.college_id).where(colleges.c.code == row.college_code)
            ).scalar_one_or_none()
            college_id = found
        if college_id is None:
            if not create_missing:
                problems.append(
                    f"line {row.line}: college {row.college_code!r} is not in the key file or "
                    f"the database; pass --create-missing to register it")
                continue
            college_id = new_ulid()
            connection.execute(colleges.insert().values(
                college_id=college_id, code=row.college_code, created_at=func.now()))
            mapping["colleges"][row.college_code] = college_id
        row.college_id = college_id

        teacher_id = mapping["teachers"].get(row.teacher_code)
        if teacher_id is not None:
            exists = connection.execute(
                select(teachers.c.teacher_id).where(teachers.c.teacher_id == teacher_id)
            ).scalar_one_or_none()
            if exists is None:
                problems.append(
                    f"line {row.line}: the key file maps {row.teacher_code!r} to {teacher_id}, "
                    f"which is not in the database. The key file and the database disagree "
                    f"about who this is; resolve that before ingesting anything.")
                continue
        else:
            if not create_missing:
                problems.append(
                    f"line {row.line}: teacher {row.teacher_code!r} is not in the key file; "
                    f"pass --create-missing to mint a pseudonym for them")
                continue
            teacher_id = new_ulid()
            connection.execute(teachers.insert().values(
                teacher_id=teacher_id, college_id=college_id, created_at=func.now()))
            mapping["teachers"][row.teacher_code] = teacher_id
        row.teacher_id = teacher_id

        # Resolved, never created. A consent record stands for a signed form.
        candidates = connection.execute(
            select(active_consents.c.consent_id, active_consents.c.scope)
            .where(active_consents.c.teacher_id == teacher_id)
        ).mappings().all()
        # COVERAGE, not a local rule. `consent.resolve` will apply it again at ingest time,
        # and two copies of "does this scope cover this domain" would let the loader accept a
        # row that ingest then refuses, one file at a time, halfway through a batch.
        covering = [c for c in candidates
                    if row.domain in COVERAGE.get(c["scope"], frozenset())]
        if not covering:
            problems.append(
                f"line {row.line}: teacher {row.teacher_code!r} has no active consent covering "
                f"domain {row.domain!r}. Ingest stores nothing without one, and this script "
                f"will not create it: a consent record stands for a signed form.")
            continue
        if len(covering) > 1:
            problems.append(
                f"line {row.line}: teacher {row.teacher_code!r} has {len(covering)} active "
                f"consents covering {row.domain!r}; the row must name which one applies")
            continue
        row.consent_id = covering[0]["consent_id"]

    return problems


def already_ingested(connection, path: Path, chunk_bytes: int) -> str | None:
    """The SHA-256 if this file is already stored, so a resumed run skips it.

    Hashing costs a full read of a multi-gigabyte file, which is the price of an idempotent
    loader and is cheaper than a duplicate session nobody notices.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    checksum = digest.hexdigest()

    found = connection.execute(
        select(media_objects.c.media_sha256)
        .where(media_objects.c.media_sha256 == checksum)
    ).scalar_one_or_none()
    return checksum if found else None


def chunks_of(path: Path, size: int):
    with open(path, "rb") as handle:
        while True:
            block = handle.read(size)
            if not block:
                return
            yield block


def _source_for(group: list[Row], *, scratch: Path, ffmpeg, ffprobe,
                report) -> tuple[Path, str, Path | None]:
    """The single file this group ingests as, joining its parts if there are several.

    Returns the path to ingest, the name to record, and the temporary joined file if one was
    made, so the caller can remove it once its bytes are stored. The joined file is scratch: the
    copy that survives is the one `ingest` writes under the media root, and the parts stay where
    they are until `preprocess.blur` deletes the originals after blurring.
    """
    first = group[0]
    if len(group) == 1:
        return first.path, first.filename, None

    joined = scratch / f"{first.part_group}.mp4"
    report(f"  join     {len(group)} parts -> {first.part_group}")
    metadata = join_parts(
        PartGroup(group=first.part_group, paths=tuple(r.path for r in group)),
        joined, ffmpeg=ffmpeg, ffprobe=ffprobe)
    report(f"           {metadata.duration_s:.1f}s, {metadata.bytes / 1e6:.1f} MB, "
           f"{metadata.width}x{metadata.height}")
    return joined, f"{first.part_group}.mp4", joined


def run_batch(groups: list[list[Row]], *, engine, config, ffprobe, ffmpeg,
              media_root: Path, limit: int | None = None,
              report=print) -> tuple[int, int, int]:
    """Ingest the resolved row-groups, returning how many were stored, skipped and refused.

    One group is one session. A group of several rows is a recording a transfer tool cut into
    parts; it is rejoined first and ingested as the single file it was before the split, so the
    stored content hash is the hash of the real session. D82.

    Separate from `main` so it can be exercised against a real database and a real ffprobe
    without going through argument parsing, and so a test can hand it the lenient quality gate
    short fixtures need. An operator has no way to weaken that gate from the command line,
    which is why there is no flag for it.
    """
    ingested = skipped = failed = 0
    scratch = media_root / ".joining"
    scratch.mkdir(parents=True, exist_ok=True)

    for group in groups[:limit]:
        row = group[0]
        try:
            source, name, temporary = _source_for(
                group, scratch=scratch, ffmpeg=ffmpeg, ffprobe=ffprobe, report=report)
        except Exception as refused:
            report(f"  refused  {row.part_group or row.filename}: {refused}")
            failed += 1
            continue

        try:
            ingested, skipped, failed = _ingest_one(
                row, source, name, engine=engine, config=config, ffprobe=ffprobe,
                ffmpeg=ffmpeg, media_root=media_root, report=report,
                tally=(ingested, skipped, failed))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    scratch.rmdir() if not any(scratch.iterdir()) else None
    return ingested, skipped, failed


def _ingest_one(row: Row, source: Path, name: str, *, engine, config, ffprobe, ffmpeg,
                media_root: Path, report, tally: tuple[int, int, int]) -> tuple[int, int, int]:
    """Ingest one already-assembled file. Returns the running tally, updated."""
    ingested, skipped, failed = tally

    with transaction(engine) as connection:
        existing = already_ingested(connection, source, config.ingest.hash_chunk_bytes)
    if existing:
        report(f"  skip     {name} is already stored as {existing[:12]}")
        return ingested, skipped + 1, failed

    request = IngestRequest(
        filename=name,
        chunks=chunks_of(source, config.ingest.hash_chunk_bytes),
        teacher_id=row.teacher_id, college_id=row.college_id,
        consent_id=row.consent_id, domain=row.domain, school_id=row.school_id,
        recorded_on=row.recorded_on, subject=row.subject,
        grade_level=row.grade_level,
        **{column: row.covariates[column] for column in COVARIATE_COLUMNS},
        actor_user_id=None, actor_role="operator")

    try:
        result = ingest(request, engine=engine, config=config, ffprobe=ffprobe,
                        ffmpeg=ffmpeg, media_root=media_root)
    except Exception as refused:
        # One bad file does not end the batch. A corpus of a hundred sessions must not be
        # held up by the one that failed, and the refusal is reported, not swallowed.
        report(f"  refused  {name}: {refused}")
        return ingested, skipped, failed + 1

    report(f"  ok       {name} -> session {result.session.session_id} "
           f"({result.quality.verdict})")
    return ingested + 1, skipped, failed
