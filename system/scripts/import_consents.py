# -*- coding: utf-8 -*-
"""Create consent records from the signed forms.

    python scripts/import_consents.py --consents consents.csv --keymap keys/pseudonyms.csv
    python scripts/import_consents.py --consents consents.csv --dry-run

A `consent_records` row stands for a signed form, and this script will not invent one. Every
row must carry a `document_ref` naming where that form can be found, and a row without one is
refused rather than defaulted - a consent record whose provenance is a default value is not
evidence of anything, and it is the one row in this database that a research ethics committee
would ask to see.

Act 843 s.27(1) requires the purpose and the recipients to be stated, so both are required
too. They are the same for every subject in this study, which is why they can come from the
command line, but neither has a default.

**Scope is not a formality.** `praxis/ingest/consent.py` refuses a session whose domain the
consent does not cover, so a form signed for microteaching only will correctly refuse a
classroom recording. Record what the subject agreed to, not what would make ingest succeed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import func, select

from praxis.db import build_engine, transaction
from praxis.db.schema import active_consents, colleges, consent_records, teachers
from praxis.ids import new_ulid
from praxis.vocabulary import CONSENT_SCOPES, SUBJECT_TYPES
from scripts.ingest_corpus import read_keymap

REQUIRED_COLUMNS = ("teacher_code", "college_code", "scope", "granted_on", "document_ref")
OPTIONAL_COLUMNS = ("subject_type", "purpose", "recipients", "expires_on")


class ConsentError(RuntimeError):
    """The consent file cannot be read, or a row in it is not a consent."""


class _DryRun(Exception):
    """Raised to abandon the transaction, which is how --dry-run writes nothing."""


def _date(value: str, field: str, line: int) -> date:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ConsentError(f"line {line}: {field} {value!r} is not an ISO date") from exc


def read_consents(path: Path, *, purpose: str | None,
                  recipients: str | None) -> tuple[list[dict[str, object]], list[str]]:
    """Parse the file into rows ready to insert, plus the reasons any row was refused.

    Refusals are collected rather than raised on the first one, so an operator correcting a
    hand-written file sees every problem in one pass instead of one per run.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise ConsentError(f"{path} is missing columns: {', '.join(missing)}")
        unknown = header - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS)
        if unknown:
            raise ConsentError(f"{path} has unknown columns: {', '.join(sorted(unknown))}")
        raw = list(reader)

    rows: list[dict[str, object]] = []
    problems: list[str] = []
    for index, record in enumerate(raw, start=2):
        values = {key: (value or "").strip() for key, value in record.items() if key}
        blank = [column for column in REQUIRED_COLUMNS if not values.get(column)]
        if blank:
            problems.append(f"line {index}: {', '.join(blank)} is blank")
            continue

        scope = values["scope"]
        if scope not in CONSENT_SCOPES:
            problems.append(
                f"line {index}: scope {scope!r} is not one of {', '.join(CONSENT_SCOPES)}")
            continue

        subject_type = values.get("subject_type") or "teacher"
        if subject_type not in SUBJECT_TYPES:
            problems.append(f"line {index}: subject_type {subject_type!r} is not one of "
                            f"{', '.join(SUBJECT_TYPES)}")
            continue

        stated_purpose = values.get("purpose") or purpose
        stated_recipients = values.get("recipients") or recipients
        if not stated_purpose or not stated_recipients:
            problems.append(
                f"line {index}: Act 843 s.27(1) requires purpose and recipients to be stated; "
                f"give them per row or with --purpose and --recipients")
            continue

        try:
            granted = _date(values["granted_on"], "granted_on", index)
            expires = _date(values["expires_on"], "expires_on", index) \
                if values.get("expires_on") else None
        except ConsentError as exc:
            problems.append(str(exc))
            continue

        if expires is not None and expires <= granted:
            problems.append(f"line {index}: expires_on {expires} is not after granted_on "
                            f"{granted}")
            continue

        rows.append({
            "line": index,
            "teacher_code": values["teacher_code"],
            "college_code": values["college_code"],
            "subject_type": subject_type,
            "purpose": stated_purpose,
            "recipients": stated_recipients,
            "scope": scope,
            "granted_on": granted,
            "expires_on": expires,
            "document_ref": values["document_ref"],
        })

    return rows, problems


def _existing_active(connection, teacher_id: str) -> list[str]:
    """Scopes this teacher already has a live consent for."""
    return [row[0] for row in connection.execute(
        select(active_consents.c.scope).where(active_consents.c.teacher_id == teacher_id))]


def insert(connection, rows: list[dict[str, object]], mapping: dict[str, dict[str, str]],
           *, allow_duplicate: bool) -> tuple[int, list[str]]:
    """Insert each row, skipping any teacher who already has a live consent for that scope.

    The skip is the default because running this script twice is the normal way an operator adds
    one late form, and a second identical consent would make `resolve` ambiguous about which
    record a session was taken under.
    """
    created, problems = 0, []
    for row in rows:
        teacher_id = mapping["teachers"].get(str(row["teacher_code"]))
        college_id = mapping["colleges"].get(str(row["college_code"]))
        if teacher_id is None:
            problems.append(f"line {row['line']}: teacher_code {row['teacher_code']!r} is not "
                            f"in the key file; ingest the sessions first or add it there")
            continue
        if college_id is None:
            problems.append(f"line {row['line']}: college_code {row['college_code']!r} is not "
                            f"in the key file")
            continue

        known_teacher = connection.execute(
            select(func.count()).select_from(teachers)
            .where(teachers.c.teacher_id == teacher_id)).scalar_one()
        known_college = connection.execute(
            select(func.count()).select_from(colleges)
            .where(colleges.c.college_id == college_id)).scalar_one()
        if not known_teacher or not known_college:
            problems.append(f"line {row['line']}: the key file maps "
                            f"{row['teacher_code']!r} to an identifier the database does not "
                            f"have; the two have drifted apart")
            continue

        live = _existing_active(connection, teacher_id)
        if not allow_duplicate and str(row["scope"]) in live:
            problems.append(f"line {row['line']}: {row['teacher_code']} already has a live "
                            f"consent for scope {row['scope']!r}; skipped")
            continue

        connection.execute(consent_records.insert().values(
            consent_id=new_ulid(), subject_type=row["subject_type"], teacher_id=teacher_id,
            college_id=college_id, purpose=row["purpose"], recipients=row["recipients"],
            scope=row["scope"], granted_on=row["granted_on"], expires_on=row["expires_on"],
            document_ref=row["document_ref"], created_at=func.now()))
        created += 1

    return created, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--consents", required=True, type=Path,
                        help="CSV of signed forms")
    parser.add_argument("--keymap", default=Path("keys/pseudonyms.csv"), type=Path,
                        help="code to identifier map written by ingest_corpus.py")
    parser.add_argument("--purpose", help="stated purpose, if not given per row")
    parser.add_argument("--recipients", help="stated recipients, if not given per row")
    parser.add_argument("--allow-duplicate", action="store_true",
                        help="insert even when a live consent for that scope exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be created and write nothing")
    args = parser.parse_args(argv)

    try:
        rows, problems = read_consents(
            args.consents, purpose=args.purpose, recipients=args.recipients)
    except (ConsentError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.keymap.is_file():
        print(f"error: no key file at {args.keymap}. It maps teacher and college codes to the "
              f"identifiers in the database, and ingest_corpus.py writes it.", file=sys.stderr)
        return 2
    mapping = read_keymap(args.keymap)

    print(f"{len(rows)} consent rows readable, {len(problems)} refused before the database")
    for problem in problems:
        print(f"  refused: {problem}")

    if not rows:
        return 1 if problems else 0

    engine = build_engine()
    created, insert_problems = 0, []
    try:
        with transaction(engine) as connection:
            created, insert_problems = insert(
                connection, rows, mapping, allow_duplicate=args.allow_duplicate)
            if args.dry_run:
                raise _DryRun
    except _DryRun:
        # `transaction` rolls back on any exception, so raising is how a dry run inspects what
        # would happen while writing nothing. Checking a flag at commit time would mean the
        # write path and the rehearsal path were different code, and only one of them tested.
        pass

    for problem in insert_problems:
        print(f"  skipped: {problem}")

    if args.dry_run:
        print(f"dry run: would create {created} consent records; nothing was written")
        return 0

    print(f"created {created} consent records")
    return 0 if created or not insert_problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
