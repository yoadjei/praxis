# -*- coding: utf-8 -*-
"""The Session contract against the sessions table.

Three documents describe a session and no two of them agreed: BUILD-SPEC section 4.1, SCHEMA.md
section 3, and API.md section 3, which makes the contract the upload request body. They differed
on seven fields. Nothing compared them, so nothing failed.

This is the same shape as the `Detection.value` defect, and the lesson from that one is that
fixing the instance is worth less than installing the comparison. See D49.
"""
from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from praxis.contracts.session import Session
from praxis.ids import new_ulid

NOW = datetime(2026, 9, 13, 9, 0, tzinfo=UTC)

# Columns the database keeps for its own bookkeeping, which no contract should carry.
#
# The three exclusion columns are here on purpose. `Session` is the upload request body - what
# an operator sends when a recording arrives - and an exclusion is a researcher's decision taken
# long afterwards, on a session already ingested, preprocessed and reviewed. Putting them on the
# contract would make them look settable at upload, and the one thing worse than an unrecorded
# exclusion is one recorded by whoever uploaded the file. Written by `praxis.ingest.exclusion`
# only. D96.
DDL_ONLY = {"created_at", "excluded_at", "excluded_by", "exclusion_reason"}

# The one place the two are allowed to differ, and why. A session object exists from the moment
# the operator's metadata is parsed, which is before the file has been probed, so the contract
# has to represent "not gated yet". The row does not: the column is NOT NULL precisely so that
# an ungated session cannot be written. Listing it here rather than loosening the check keeps
# the exception visible. D49.
UNGATED_UNTIL_INGEST = {"quality_verdict"}


# Table-level clauses sit at the same indent as columns and are not columns. Without this the
# parser read `CONSTRAINT` as a column name the moment section 3 gained its first inline CHECK,
# and reported it as a field the contract was missing.
NOT_A_COLUMN = {"CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN", "EXCLUDE"}


def ddl_columns(repo_root: Path) -> dict[str, bool]:
    """Every column of `sessions`, mapped to whether it is nullable."""
    schema = (repo_root / "docs" / "SCHEMA.md").read_text(encoding="utf-8")
    block = re.search(r"CREATE TABLE sessions\s*\((.*?)\n\);", schema, re.DOTALL)
    assert block, "SCHEMA.md no longer defines sessions; the contract has nothing to agree with"

    columns: dict[str, bool] = {}
    for line in block.group(1).splitlines():
        match = re.match(r"\s{4}(\w+)\s+\S", line)
        if match and match.group(1) not in NOT_A_COLUMN:
            columns[match.group(1)] = "NOT NULL" not in line and "PRIMARY KEY" not in line
    return columns


def build(**overrides) -> Session:
    base = dict(
        session_id=new_ulid(), teacher_id="T-041", college_id=new_ulid(),
        consent_id=new_ulid(), media_sha256="b" * 64, domain="classroom",
        school_id=new_ulid(),
        subject="integrated science", grade_level="JHS 2", recorded_on=date(2026, 5, 4),
        created_at=NOW)
    base.update(overrides)
    return Session(**base)


def test_the_contract_and_the_table_describe_the_same_object(repo_root: Path) -> None:
    columns = ddl_columns(repo_root)
    fields = set(Session.model_fields)

    # Stated as two directed rules rather than set equality. The table may carry what the
    # contract does not, but only what `DDL_ONLY` names and for the reason written there; the
    # contract may never carry a field the table has no column for. Plain equality against
    # `columns - DDL_ONLY` is wrong in the other direction - `created_at` is in DDL_ONLY and on
    # the contract both - and reports a divergence with nothing in either diff.
    assert not fields - set(columns), (
        f"D49: the contract has fields the sessions DDL has no column for: "
        f"{sorted(fields - set(columns))}")
    assert not set(columns) - fields - DDL_ONLY, (
        f"D49: the sessions DDL has columns the contract does not carry: "
        f"{sorted(set(columns) - fields - DDL_ONLY)}. Add them to the contract, or to DDL_ONLY "
        f"with the reason they belong to the table alone.")


def test_every_exception_names_a_column_that_exists(repo_root: Path) -> None:
    """The exception lists are the one part of this comparison nothing else checks.

    A name left in `DDL_ONLY` after its column is dropped silently widens the exemption, and the
    next field added under that name would be exempt without anyone deciding so. Same shape as
    L18: an assertion about the data has to be checked against the data.
    """
    columns = set(ddl_columns(repo_root))
    stale = sorted((DDL_ONLY | UNGATED_UNTIL_INGEST) - columns)
    assert not stale, (
        f"these are exempted from the contract comparison but are not columns of sessions: "
        f"{stale}. Remove them, or the exemption covers a name nothing defines.")


def admits_none(annotation: object) -> bool:
    """Whether the field can actually hold None.

    Not the same question as whether it has a default. `quality_detail` defaults to `{}` and can
    never be None, which matches its NOT NULL column exactly; asking about defaults would have
    reported that as a divergence and taught everyone to widen the exception list.
    """
    return annotation is type(None) or type(None) in get_args(annotation)


def test_they_agree_on_what_may_be_missing(repo_root: Path) -> None:
    """Names matching is not enough. The first pass compared only names and missed `subject`,
    which the table leaves optional and the contract demanded."""
    columns = ddl_columns(repo_root)
    disagreements = []
    for name, nullable in columns.items():
        if name in DDL_ONLY | UNGATED_UNTIL_INGEST:
            continue
        optional = admits_none(Session.model_fields[name].annotation)
        if optional != nullable:
            disagreements.append(
                f"{name}: table {'allows' if nullable else 'forbids'} null, "
                f"contract {'allows' if optional else 'refuses'} it")

    assert not disagreements, "D49: nullability has diverged:\n  " + "\n  ".join(disagreements)


def test_the_documented_exception_is_still_the_only_one(repo_root: Path) -> None:
    """If the column ever became nullable, the exception above would be silently protecting
    nothing, and an ungated session could reach the table."""
    columns = ddl_columns(repo_root)
    for name in UNGATED_UNTIL_INGEST:
        assert not columns[name], (
            f"D49: sessions.{name} is now nullable, so the contract carrying None is no longer "
            f"an exception the database is correcting. Remove it from UNGATED_UNTIL_INGEST.")
        assert admits_none(Session.model_fields[name].annotation)


# ---------------------------------------------------------------------------
# The fields the reconciliation changed
# ---------------------------------------------------------------------------

def test_a_session_is_ungated_until_the_gate_has_run() -> None:
    """quality_verdict is NOT NULL in the table and None here, and that gap is the point:
    the object exists before the probe, the row must not."""
    session = build()
    assert session.quality_verdict is None and not session.is_gated
    assert session.quality_detail == {}

    gated = build(quality_verdict="fail", quality_detail={"duration_s": {"verdict": "fail"}})
    assert gated.is_gated and gated.quality_verdict == "fail"


def test_a_verdict_the_table_forbids_is_refused() -> None:
    """The CHECK constraint permits pass, warn and fail. `abstain` is a per-check verdict
    under D48 and must never reach the session's overall column."""
    for bad in ("abstain", "pending", "ok", ""):
        with pytest.raises(ValidationError):
            build(quality_verdict=bad)


def test_consent_is_a_ulid_now_that_it_is_a_foreign_key() -> None:
    assert build().consent_id
    with pytest.raises(ValidationError, match="expected a ULID"):
        build(consent_id="consent-form-14")


def test_recorded_on_is_a_date_not_a_duration() -> None:
    """BUILD-SPEC carried recorded_duration_s; the duration lives on media_objects, where it is
    measured. What the session needs is when the teaching happened."""
    assert "recorded_duration_s" not in Session.model_fields
    assert build().recorded_on == date(2026, 5, 4)
    with pytest.raises(ValidationError):
        build(recorded_on="not a date")


def test_the_covariates_survived_the_reconciliation() -> None:
    """The five covariates are the reason Phase 6 can attribute degradation at all."""
    assert not build().is_attributable

    full = build(camera_distance_m=3.5, room_area_m2=48.0, pupil_count=52,
                 ambient_noise_dba=61.0, teacher_movement_range_m=4.2)
    assert full.is_attributable
    assert set(full.covariates) == {
        "camera_distance_m", "room_area_m2", "pupil_count", "ambient_noise_dba",
        "teacher_movement_range_m"}


def test_a_session_cannot_be_edited_in_place() -> None:
    session = build()
    with pytest.raises(ValidationError):
        session.domain = "microteaching"
