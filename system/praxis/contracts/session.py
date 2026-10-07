# -*- coding: utf-8 -*-
"""Session: one recording, and the covariates that make domain shift attributable.

`domain` is the single most important field in the system. Every evaluation is stratified by
it, and `teacher_id` is the split key R2 turns on, so neither may ever be null.

The five covariates are measured at capture and never imputed. A session recorded without them
can still be used to train and evaluate, but it cannot contribute to the Phase 6 regression
that attributes degradation to a named cause, and that attribution is what turns a negative
result into design knowledge. There is no way to add them afterwards.

Field for field this is the `sessions` table in SCHEMA.md section 3, and API.md section 3 makes
it the body of the upload request as well. `test_session_contract_matches_ddl` holds the two
together. See D49 for what went wrong while they were allowed to drift.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from praxis.ids import is_ulid

Domain = Literal["microteaching", "classroom"]
QualityVerdict = Literal["pass", "warn", "fail"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def missing_school(domain: str, school_id: str | None) -> str | None:
    """The reason this domain/school pair is refused, or None if it is allowed.

    One implementation, called by `Session` and by the API's `SessionMetadata`, because the
    request body has to be refused *before* the upload is stored - `Session` is constructed
    after the file is on disk and probed - and a rule written at both points gets relaxed at one
    (L20). The database carries it a third time as `a_classroom_session_names_its_school`, which
    is deliberate: that layer catches a writer that went around the contract entirely. D98.
    """
    if domain == "classroom" and school_id is None:
        return ("a classroom session must name the basic school it was recorded in. The school "
                "is what places it on the shift ladder: S1 holds one out, and a session with "
                "none would land in the training pool instead of being held out, which is the "
                "contamination the ladder exists to measure against.")
    return None


class Session(BaseModel):
    """One recorded teaching session, pseudonymous by construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    teacher_id: str = Field(min_length=1, description="pseudonymous, stable across sessions")
    college_id: str
    consent_id: str = Field(min_length=1, description="ingest refuses without a live one")
    media_sha256: str

    domain: Domain = Field(description="source vs target; every evaluation stratifies on it")
    school_id: str | None = Field(
        default=None,
        description=("the basic school, for a classroom session. None for microteaching, which "
                     "happens on a college campus and has no basic school. S1 holds this out, "
                     "so a classroom session without it cannot be placed on the ladder. D98"))
    subject: str | None = None
    grade_level: str | None = None
    recorded_on: date = Field(
        description="when the teaching happened, not when it was uploaded")

    # Domain-shift covariates. All measured, none assumed.
    camera_distance_m: float | None = Field(default=None, gt=0)
    room_area_m2: float | None = Field(default=None, gt=0)
    pupil_count: int | None = Field(default=None, ge=0)
    ambient_noise_dba: float | None = Field(default=None, ge=0)
    teacher_movement_range_m: float | None = Field(default=None, ge=0)

    # None until the gate has run. The column is NOT NULL, so an ungated session cannot be
    # written; the state exists in the object because the object is built before the probe.
    quality_verdict: QualityVerdict | None = None
    quality_detail: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime

    @field_validator("session_id", "college_id", "consent_id")
    @classmethod
    def identifiers_are_ulids(cls, value: str) -> str:
        if not is_ulid(value):
            raise ValueError(f"expected a ULID, got {value!r}")
        return value

    @field_validator("media_sha256")
    @classmethod
    def hash_is_lowercase_hex(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("media_sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("school_id")
    @classmethod
    def school_is_a_ulid_when_given(cls, value: str | None) -> str | None:
        if value is not None and not is_ulid(value):
            raise ValueError(f"school_id must be a ULID, got {value!r}")
        return value

    @model_validator(mode="after")
    def a_classroom_session_names_its_school(self) -> Session:
        """Mirrors the CHECK of the same name. See `missing_school`."""
        refusal = missing_school(self.domain, self.school_id)
        if refusal:
            raise ValueError(refusal)
        return self

    @property
    def covariates(self) -> dict[str, float | int | None]:
        """The five covariates, for the Phase 6 attribution model."""
        return {
            "camera_distance_m": self.camera_distance_m,
            "room_area_m2": self.room_area_m2,
            "pupil_count": self.pupil_count,
            "ambient_noise_dba": self.ambient_noise_dba,
            "teacher_movement_range_m": self.teacher_movement_range_m,
        }

    @property
    def is_attributable(self) -> bool:
        """Whether this session can contribute to degradation attribution.

        Reported rather than enforced: a session missing a covariate is still valid training
        and evaluation data. It simply cannot answer why performance dropped, and Phase 6
        needs to count how many such sessions there are rather than discover it late.
        """
        return all(value is not None for value in self.covariates.values())

    @property
    def is_gated(self) -> bool:
        return self.quality_verdict is not None
