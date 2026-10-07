# -*- coding: utf-8 -*-
"""Tests for scripts/import_consents.py, the consent record importer.

Consent records stand for signed forms, never default values. The script enforces that every
row carries a document_ref (its provenance), that purpose and recipients are stated, and that
dates are valid. It also links each row back to a teacher in the database via a key file.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import and_, select, text

from praxis.db import transaction
from praxis.db.schema import consent_records
from praxis.ids import new_ulid
from praxis.ingest.consent import COVERAGE, resolve
from praxis.ingest.errors import IngestRefused
from praxis.vocabulary import CONSENT_SCOPES, SUBJECT_TYPES
from scripts.import_consents import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    ConsentError,
    insert,
    read_consents,
)
from tests.integration.conftest import requires_database


def csv_at(path: Path, header: str, *rows: str) -> Path:
    """Write a CSV to path with the given header and data rows."""
    lines = [header, *rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestReadConsents:
    """Parse consent files into structured rows, collecting all problems at once."""

    def test_a_well_formed_file_parses(self, tmp_path) -> None:
        """A minimal valid file, with all required columns and one data row."""
        header = ",".join(REQUIRED_COLUMNS)
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(
            path, purpose="research", recipients="supervisors")

        assert len(rows) == 1
        assert len(problems) == 0
        assert rows[0]["teacher_code"] == "T-001"
        assert rows[0]["college_code"] == "COL-A"
        assert rows[0]["scope"] == "microteaching"
        assert rows[0]["granted_on"] == date(2026, 1, 15)
        assert rows[0]["expires_on"] is None
        assert rows[0]["document_ref"] == "forms/1"

    def test_purpose_comes_from_row_when_present(self, tmp_path) -> None:
        """If the row has a purpose, use it even when --purpose is given."""
        header = ",".join((*REQUIRED_COLUMNS, "purpose"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,row-purpose"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, _ = read_consents(path, purpose="cli-purpose", recipients="supervisors")

        assert len(rows) == 1
        assert rows[0]["purpose"] == "row-purpose"

    def test_purpose_comes_from_cli_when_row_is_empty(self, tmp_path) -> None:
        """If the row's purpose is blank or missing, use --purpose."""
        header = ",".join((*REQUIRED_COLUMNS, "purpose"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, _ = read_consents(path, purpose="cli-purpose", recipients="supervisors")

        assert len(rows) == 1
        assert rows[0]["purpose"] == "cli-purpose"

    def test_recipients_comes_from_row_when_present(self, tmp_path) -> None:
        """If the row has recipients, use it even when --recipients is given."""
        header = ",".join((*REQUIRED_COLUMNS, "recipients"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,row-recipients"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, _ = read_consents(path, purpose="research", recipients="cli-recipients")

        assert len(rows) == 1
        assert rows[0]["recipients"] == "row-recipients"

    def test_recipients_comes_from_cli_when_row_is_empty(self, tmp_path) -> None:
        """If the row's recipients is blank or missing, use --recipients."""
        header = ",".join((*REQUIRED_COLUMNS, "recipients"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, _ = read_consents(path, purpose="research", recipients="cli-recipients")

        assert len(rows) == 1
        assert rows[0]["recipients"] == "cli-recipients"

    def test_a_blank_document_ref_is_refused(self, tmp_path) -> None:
        """A row without a document_ref is refused. It is evidence of nothing."""
        header = ",".join(REQUIRED_COLUMNS)
        row1 = "T-001,COL-A,microteaching,2026-01-15,forms/1"
        row2 = "T-002,COL-A,microteaching,2026-01-15,"
        path = csv_at(tmp_path / "consents.csv", header, row1, row2)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 1
        assert len(problems) == 1
        assert "document_ref" in problems[0]
        assert "line 3" in problems[0]

    def test_a_missing_required_column_raises_consent_error(self, tmp_path) -> None:
        """The file must have every required column before it is read."""
        header = "teacher_code,college_code,scope"
        path = csv_at(tmp_path / "consents.csv", header)

        with pytest.raises(ConsentError, match="missing columns"):
            read_consents(path, purpose="research", recipients="supervisors")

    def test_an_unknown_column_raises_consent_error(self, tmp_path) -> None:
        """Unknown columns are refused, so a header typo is not silently ignored."""
        header = ",".join((*REQUIRED_COLUMNS, "typo_column"))
        path = csv_at(tmp_path / "consents.csv", header)

        with pytest.raises(ConsentError, match="unknown columns"):
            read_consents(path, purpose="research", recipients="supervisors")

    def test_an_invalid_scope_is_refused_with_the_valid_list(self, tmp_path) -> None:
        """Invalid scope names are listed, with the valid scopes shown in the error."""
        header = ",".join(REQUIRED_COLUMNS)
        row = "T-001,COL-A,invalid_scope,2026-01-15,forms/1"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 0
        assert len(problems) == 1
        assert "invalid_scope" in problems[0]
        for valid_scope in CONSENT_SCOPES:
            assert valid_scope in problems[0]

    def test_an_invalid_subject_type_is_refused_with_the_valid_list(self, tmp_path) -> None:
        """Invalid subject_type values are refused with valid options listed."""
        header = ",".join((*REQUIRED_COLUMNS, "subject_type"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,invalid_type"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 0
        assert len(problems) == 1
        assert "invalid_type" in problems[0]
        for valid_type in SUBJECT_TYPES:
            assert valid_type in problems[0]

    def test_a_valid_subject_type_is_accepted(self, tmp_path) -> None:
        """Valid subject_type values in SUBJECT_TYPES are accepted."""
        for subject_type in SUBJECT_TYPES:
            header = ",".join((*REQUIRED_COLUMNS, "subject_type"))
            row = f"T-001,COL-A,microteaching,2026-01-15,forms/1,{subject_type}"
            path = csv_at(tmp_path / "consents.csv", header, row)

            rows, problems = read_consents(path, purpose="research", recipients="supervisors")

            assert len(rows) == 1, f"subject_type {subject_type!r} should be valid"
            assert len(problems) == 0

    def test_a_non_iso_granted_on_is_refused(self, tmp_path) -> None:
        """granted_on must be ISO format (YYYY-MM-DD), not other date formats."""
        header = ",".join(REQUIRED_COLUMNS)
        row = "T-001,COL-A,microteaching,15/01/2026,forms/1"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 0
        assert len(problems) == 1
        assert "granted_on" in problems[0]
        assert "not an ISO date" in problems[0]

    def test_a_non_iso_expires_on_is_refused(self, tmp_path) -> None:
        """expires_on must be ISO format (YYYY-MM-DD) when present."""
        header = ",".join((*REQUIRED_COLUMNS, "expires_on"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,31-01-2027"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 0
        assert len(problems) == 1
        assert "expires_on" in problems[0]
        assert "not an ISO date" in problems[0]

    def test_expires_on_must_be_after_granted_on(self, tmp_path) -> None:
        """expires_on must be strictly after granted_on, not equal or before."""
        header = ",".join((*REQUIRED_COLUMNS, "expires_on"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,2026-01-15"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 0
        assert len(problems) == 1
        assert "expires_on" in problems[0]
        assert "not after granted_on" in problems[0]

    def test_expires_on_before_granted_on_is_refused(self, tmp_path) -> None:
        """expires_on before granted_on is refused."""
        header = ",".join((*REQUIRED_COLUMNS, "expires_on"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,2026-01-14"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 0
        assert len(problems) == 1

    def test_expires_on_after_granted_on_is_accepted(self, tmp_path) -> None:
        """expires_on that is strictly after granted_on is accepted."""
        header = ",".join((*REQUIRED_COLUMNS, "expires_on"))
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,2026-01-16"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 1
        assert len(problems) == 0
        assert rows[0]["expires_on"] == date(2026, 1, 16)

    def test_purpose_and_recipients_missing_from_both_row_and_arguments_is_refused(
        self, tmp_path
    ) -> None:
        """Act 843 s.27(1): purpose and recipients must be stated, from row or CLI."""
        header = ",".join(REQUIRED_COLUMNS)
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose=None, recipients=None)

        assert len(rows) == 0
        assert len(problems) == 1
        assert "Act 843" in problems[0]
        assert "--purpose" in problems[0]
        assert "--recipients" in problems[0]

    def test_purpose_missing_but_recipients_present_is_refused(self, tmp_path) -> None:
        """Act 843 requires both; missing one is refused."""
        header = ",".join(REQUIRED_COLUMNS)
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose=None, recipients="supervisors")

        assert len(rows) == 0
        assert len(problems) == 1

    def test_recipients_missing_but_purpose_present_is_refused(self, tmp_path) -> None:
        """Act 843 requires both; missing one is refused."""
        header = ",".join(REQUIRED_COLUMNS)
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients=None)

        assert len(rows) == 0
        assert len(problems) == 1

    def test_problems_are_collected_not_raised_on_first(self, tmp_path) -> None:
        """All problems in a file are reported at once, not one per run."""
        header = ",".join(REQUIRED_COLUMNS)
        row1 = "T-001,COL-A,,2026-01-15,forms/1"
        row2 = "T-002,COL-A,microteaching,invalid_date,forms/2"
        row3 = "T-003,COL-A,invalid_scope,2026-01-15,forms/3"
        path = csv_at(tmp_path / "consents.csv", header, row1, row2, row3)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 0
        assert len(problems) == 3

    def test_a_scope_check_stops_processing_that_row(self, tmp_path) -> None:
        """When scope is invalid, the row stops processing and is not added."""
        header = ",".join((*REQUIRED_COLUMNS, "expires_on"))
        row = "T-001,COL-A,invalid_scope,2026-01-15,forms/1,2026-01-14"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 0
        # One problem: invalid scope stops the row (doesn't reach expires_on check)
        assert len(problems) == 1
        assert "invalid_scope" in problems[0]

    def test_whitespace_is_stripped_from_values(self, tmp_path) -> None:
        """Leading and trailing whitespace in values is stripped."""
        header = ",".join(REQUIRED_COLUMNS)
        row = " T-001 , COL-A , microteaching , 2026-01-15 , forms/1 "
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, _ = read_consents(path, purpose="research", recipients="supervisors")

        assert len(rows) == 1
        assert rows[0]["teacher_code"] == "T-001"
        assert rows[0]["college_code"] == "COL-A"

    def test_optional_columns_are_allowed(self, tmp_path) -> None:
        """Optional columns are accepted without error."""
        header = ",".join(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
        row = "T-001,COL-A,microteaching,2026-01-15,forms/1,teacher,research,supervisors,"
        path = csv_at(tmp_path / "consents.csv", header, row)

        rows, problems = read_consents(path, purpose=None, recipients=None)

        assert len(rows) == 1
        assert len(problems) == 0

    def test_all_scopes_are_valid(self, tmp_path) -> None:
        """Every scope in CONSENT_SCOPES is accepted."""
        for scope in CONSENT_SCOPES:
            header = ",".join(REQUIRED_COLUMNS)
            row = f"T-001,COL-A,{scope},2026-01-15,forms/1"
            path = csv_at(tmp_path / "consents.csv", header, row)

            rows, problems = read_consents(path, purpose="research", recipients="supervisors")

            assert len(rows) == 1, f"scope {scope!r} should be valid"
            assert len(problems) == 0


class TestInsertConsents:
    """Insert consent records into the database, linking to teachers via a key file."""

    def test_a_row_with_unmapped_teacher_code_is_refused_and_nothing_inserted(
        self, engine
    ) -> None:
        """If teacher_code is not in the key file, the row is refused."""
        if not requires_database(engine):
            return

        mapping = {"teachers": {"T-OTHER": new_ulid()}, "colleges": {"COL-A": new_ulid()}}
        rows = [
            {
                "line": 2,
                "teacher_code": "T-UNKNOWN",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            }
        ]

        with transaction(engine) as conn:
            created, problems = insert(conn, rows, mapping, allow_duplicate=False)

        assert created == 0
        assert len(problems) == 1
        assert "T-UNKNOWN" in problems[0]
        assert "not in the key file" in problems[0]

    def test_a_row_with_unmapped_college_code_is_refused_and_nothing_inserted(
        self, engine
    ) -> None:
        """If college_code is not in the key file, the row is refused."""
        if not requires_database(engine):
            return

        mapping = {"teachers": {"T-001": new_ulid()}, "colleges": {"COL-OTHER": new_ulid()}}
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-UNKNOWN",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            }
        ]

        with transaction(engine) as conn:
            created, problems = insert(conn, rows, mapping, allow_duplicate=False)

        assert created == 0
        assert len(problems) == 1
        assert "COL-UNKNOWN" in problems[0]

    def test_a_mapping_pointing_to_nonexistent_teacher_is_refused(
        self, engine, party
    ) -> None:
        """If the key file maps to a teacher_id not in the database, the row is refused."""
        if not requires_database(engine):
            return

        fake_teacher_id = new_ulid()
        mapping = {
            "teachers": {"T-001": fake_teacher_id},
            "colleges": {"COL-A": party["college_id"]},
        }
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            }
        ]

        with transaction(engine) as conn:
            created, problems = insert(conn, rows, mapping, allow_duplicate=False)

        assert created == 0
        assert len(problems) == 1
        assert "drifted apart" in problems[0]

    def test_a_mapping_pointing_to_nonexistent_college_is_refused(
        self, engine, party
    ) -> None:
        """If the key file maps to a college_id not in the database, the row is refused."""
        if not requires_database(engine):
            return

        fake_college_id = new_ulid()
        mapping = {
            "teachers": {"T-001": party["teacher_id"]},
            "colleges": {"COL-A": fake_college_id},
        }
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            }
        ]

        with transaction(engine) as conn:
            created, problems = insert(conn, rows, mapping, allow_duplicate=False)

        assert created == 0
        assert len(problems) == 1

    def test_a_successful_insert_creates_a_record_resolve_can_find(
        self, engine, party
    ) -> None:
        """A successfully inserted consent is discoverable by resolve() for sessions."""
        if not requires_database(engine):
            return

        mapping = {
            "teachers": {"T-001": party["teacher_id"]},
            "colleges": {"COL-A": party["college_id"]},
        }
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            }
        ]

        with transaction(engine) as conn:
            created, _ = insert(conn, rows, mapping, allow_duplicate=False)
            # Get the newly created consent and resolve it
            new_record = conn.execute(
                select(consent_records).where(
                    and_(
                        consent_records.c.teacher_id == party["teacher_id"],
                        consent_records.c.scope == "microteaching",
                    )
                )
            ).first()
            resolved = resolve(conn, new_record.consent_id, "microteaching")

        assert created == 1
        assert resolved is not None
        assert resolved.scope == "microteaching"

    def test_resolve_refuses_a_scope_not_covered_by_consent(self, engine, party) -> None:
        """When a consent covers 'microteaching', resolve() refuses 'classroom'."""
        if not requires_database(engine):
            return

        mapping = {
            "teachers": {"T-001": party["teacher_id"]},
            "colleges": {"COL-A": party["college_id"]},
        }
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            }
        ]

        with transaction(engine) as conn:
            created, _ = insert(conn, rows, mapping, allow_duplicate=False)
            # Get the newly created consent with scope 'microteaching'
            new_record = conn.execute(
                select(consent_records).where(
                    and_(
                        consent_records.c.teacher_id == party["teacher_id"],
                        consent_records.c.scope == "microteaching",
                    )
                )
            ).first()
            # Try to resolve with a different scope - should raise because 'microteaching'
            # doesn't cover 'classroom'
            # `consent_invalid` is a factory that builds the exception, not the class, so
            # naming it here raised TypeError inside pytest.raises before resolve() was ever
            # called - the test could not have failed for the reason it was written for.
            # `match` is load-bearing too: resolve() refuses four distinguishable ways, and the
            # bare class would also accept "consent does not exist", which is not this.
            with pytest.raises(IngestRefused, match="does not cover session domain"):
                resolve(conn, new_record.consent_id, "classroom")

        assert created == 1

    def test_a_second_import_for_the_same_teacher_and_scope_is_skipped_by_default(
        self, engine, party
    ) -> None:
        """Running the script twice with the same teacher and scope creates only one."""
        if not requires_database(engine):
            return

        mapping = {
            "teachers": {"T-001": party["teacher_id"]},
            "colleges": {"COL-A": party["college_id"]},
        }
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            }
        ]

        with transaction(engine) as conn:
            created1, problems1 = insert(conn, rows, mapping, allow_duplicate=False)

        with transaction(engine) as conn:
            created2, problems2 = insert(conn, rows, mapping, allow_duplicate=False)

        assert created1 == 1
        assert len(problems1) == 0
        assert created2 == 0
        assert len(problems2) == 1
        assert "already has a live consent" in problems2[0]

    def test_allow_duplicate_true_inserts_even_when_live_consent_exists(
        self, engine, party
    ) -> None:
        """With allow_duplicate=True, a second import for the same scope is inserted."""
        if not requires_database(engine):
            return

        mapping = {
            "teachers": {"T-001": party["teacher_id"]},
            "colleges": {"COL-A": party["college_id"]},
        }
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 16),
                "expires_on": None,
                "document_ref": "forms/2",
            }
        ]

        with transaction(engine) as conn:
            created1, _ = insert(conn, rows, mapping, allow_duplicate=False)

        with transaction(engine) as conn:
            created2, problems2 = insert(conn, rows, mapping, allow_duplicate=True)

        assert created1 == 1
        assert created2 == 1
        assert len(problems2) == 0

    def test_different_scopes_for_the_same_teacher_are_both_inserted(
        self, engine, party
    ) -> None:
        """A teacher can have consents for multiple different scopes."""
        if not requires_database(engine):
            return

        mapping = {
            "teachers": {"T-001": party["teacher_id"]},
            "colleges": {"COL-A": party["college_id"]},
        }
        rows1 = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            }
        ]
        rows2 = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "classroom",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/2",
            }
        ]

        with transaction(engine) as conn:
            created1, _ = insert(conn, rows1, mapping, allow_duplicate=False)
            created2, _ = insert(conn, rows2, mapping, allow_duplicate=False)

        assert created1 == 1
        assert created2 == 1

    def test_insert_sets_all_fields_including_optional_ones(
        self, engine, party
    ) -> None:
        """All fields from the row are set in the database record."""
        if not requires_database(engine):
            return

        expires = date(2027, 1, 15)
        mapping = {
            "teachers": {"T-001": party["teacher_id"]},
            "colleges": {"COL-A": party["college_id"]},
        }
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                # Not "teacher", which is the default the script falls back to: a value the row
                # has to have carried is what proves the column was written from the row.
                "subject_type": "guardian_class",
                "purpose": "evaluation",
                "recipients": "ethics committee",
                "scope": "classroom",
                "granted_on": date(2026, 1, 15),
                "expires_on": expires,
                "document_ref": "forms/1.pdf",
            }
        ]

        with transaction(engine) as conn:
            created, problems = insert(conn, rows, mapping, allow_duplicate=False)
            # Read it back to verify
            records = conn.execute(
                select(consent_records).where(
                    consent_records.c.teacher_id == party["teacher_id"]
                )
            ).fetchall()

        assert created == 1
        assert len(problems) == 0
        assert len(records) >= 1
        record = records[-1]
        assert record.subject_type == "guardian_class"
        assert record.purpose == "evaluation"
        assert record.recipients == "ethics committee"
        assert record.scope == "classroom"
        assert record.granted_on == date(2026, 1, 15)
        assert record.expires_on == expires
        assert record.document_ref == "forms/1.pdf"

    def test_one_bad_row_does_not_stop_good_rows_from_being_inserted(
        self, engine, party
    ) -> None:
        """Multiple rows are processed; bad ones are skipped, good ones are inserted."""
        if not requires_database(engine):
            return

        mapping = {
            "teachers": {"T-001": party["teacher_id"], "T-UNKNOWN": new_ulid()},
            "colleges": {"COL-A": party["college_id"]},
        }
        rows = [
            {
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/1",
            },
            {
                "line": 3,
                "teacher_code": "T-UNKNOWN",
                "college_code": "COL-A",
                "subject_type": "teacher",
                "purpose": "research",
                "recipients": "supervisors",
                "scope": "microteaching",
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": "forms/2",
            },
        ]

        with transaction(engine) as conn:
            created, problems = insert(conn, rows, mapping, allow_duplicate=False)

        assert created == 1
        assert len(problems) == 1


def check_literals(connection, table: str, column: str) -> set[str]:
    """The values a `CHECK (<column> IN (...))` on `table` accepts, read from the catalogue.

    Read rather than restated. The point of the tests below is that two declarations of one
    vocabulary agree, so the second one has to come from the database itself; a list written out
    here would be a third copy and could drift from both.
    """
    definitions = connection.execute(
        text("SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c "
             "JOIN pg_class t ON t.oid = c.conrelid "
             "WHERE t.relname = :table AND c.contype = 'c'"),
        {"table": table},
    ).scalars().all()

    matching = [one for one in definitions if column in one]
    assert len(matching) == 1, (
        f"expected exactly one CHECK on {table}.{column}, found {len(matching)}: {matching}")
    return set(re.findall(r"'([^']*)'::text", matching[0]))


class TestVocabulariesMatchTheDatabase:
    """`praxis.vocabulary` declares what `consent_records` CHECKs, and nothing else does.

    This is the test that was missing, and its absence is why the defect it now covers survived
    a passing suite. `test_a_valid_subject_type_is_accepted` above writes a CSV and asserts the
    reader accepted it, which checks the script against its own constant: while the script
    listed 'teacher', 'learner', 'guardian' and the table listed 'teacher', 'guardian_class',
    'evaluator', that test was green. What the disagreement actually did was report a row as
    valid and then abort the whole import on a CHECK violation when writing it, and refuse two
    values the table allows. Only a test that reads both declarations can see it.
    """

    def test_the_check_constraints_list_exactly_the_declared_vocabularies(self, engine) -> None:
        """Neither declaration is wider than the other, in either direction."""
        if not requires_database(engine):
            return

        with engine.connect() as conn:
            assert check_literals(conn, "consent_records", "subject_type") == set(SUBJECT_TYPES)
            assert check_literals(conn, "consent_records", "scope") == set(CONSENT_SCOPES)

    def test_every_declared_subject_type_can_actually_be_imported(self, engine, party) -> None:
        """One row per value, written through `insert`, so the path a real file takes is the
        path under test. A value the script accepts and the table refuses fails here."""
        if not requires_database(engine):
            return

        mapping = {
            "teachers": {"T-001": party["teacher_id"]},
            "colleges": {"COL-A": party["college_id"]},
        }
        for index, subject_type in enumerate(SUBJECT_TYPES):
            rows = [{
                "line": 2,
                "teacher_code": "T-001",
                "college_code": "COL-A",
                "subject_type": subject_type,
                "purpose": "research",
                "recipients": "supervisors",
                # A different scope each time: the same teacher, scope and document would be
                # refused as a duplicate by the second iteration rather than by the CHECK.
                "scope": CONSENT_SCOPES[index % len(CONSENT_SCOPES)],
                "granted_on": date(2026, 1, 15),
                "expires_on": None,
                "document_ref": f"forms/subject-{subject_type}",
            }]
            with transaction(engine) as conn:
                created, problems = insert(conn, rows, mapping, allow_duplicate=True)
            assert problems == [], f"subject_type {subject_type!r} was refused: {problems}"
            assert created == 1, f"subject_type {subject_type!r} wrote no row"

    def test_coverage_decides_every_declared_scope(self) -> None:
        """`COVERAGE` in praxis.ingest.consent maps scope to the domains it permits, and
        `resolve` reads it with a `frozenset()` default. A scope missing from it would refuse
        every domain - the safe direction, but silently, and the operator would be told a live
        consent does not cover a session it was signed for."""
        assert set(COVERAGE) == set(CONSENT_SCOPES)
