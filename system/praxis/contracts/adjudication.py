# -*- coding: utf-8 -*-
"""Adjudication: the human decision, and the instrumentation the reliance study needs.

`seconds_on_item`, `evidence_replays`, `revealed_indication` and `was_seeded_error` exist for
the research, not the product. §4.4 says so explicitly: **do not remove them as unused.** They
are how H6 and H7 are answered, and an adjudication recorded without them is a data point the
study cannot use.

Adjudications are append-only. There is no edit and no delete; a correction is a new row
referencing the same detection, and the API returns the latest while the history stays
queryable. That is enforced at the database by a trigger, and mirrored here by the model being
frozen.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Action = Literal["confirm", "edit", "reject"]
Arm = Literal["dossier", "no_dossier"]

MIN_RATIONALE_CHARS = 10


class Adjudication(BaseModel):
    """One reviewer's decision on one detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adjudication_id: str
    detection_id: str
    reviewer_id: str
    action: Action
    edited_value: dict[str, Any] | None = Field(
        default=None,
        description="the corrected codebook fields, same shape as Detection.predicted")
    rationale: str | None = Field(default=None, description="required for edit and reject")

    # Reliance-study instrumentation. Not product features.
    seconds_on_item: float = Field(ge=0)
    evidence_replays: int = Field(default=0, ge=0)
    revealed_indication: bool = Field(
        description="whether the evidence-first gate was opened before deciding")
    arm: Arm | None = None
    was_seeded_error: bool = Field(
        default=False, description="probe flag; hidden from the reviewer"
    )

    created_at: datetime

    @model_validator(mode="after")
    def changes_are_justified(self) -> Adjudication:
        # A reviewer who overrides the model says why. Without this the reliance study cannot
        # distinguish a considered correction from a reflexive one, and the audit trail cannot
        # reconstruct the reasoning behind a final state.
        if (self.action in ("edit", "reject") and
                (self.rationale is None or
                 len(self.rationale.strip()) < MIN_RATIONALE_CHARS)):
            raise ValueError(
                f"action={self.action!r} requires a rationale of at least "
                f"{MIN_RATIONALE_CHARS} characters")
        return self

    @model_validator(mode="after")
    def edits_carry_a_value(self) -> Adjudication:
        if self.action == "edit" and not self.edited_value:
            raise ValueError("action='edit' requires edited_value")
        if self.action != "edit" and self.edited_value is not None:
            raise ValueError(f"edited_value is set but action is {self.action!r}")
        return self

    def validate_against(self, detection) -> None:
        """Check the edit is a label the detection's codebook version defines.

        Not a model validator, because the shape a correction may take depends on the
        behaviour of the detection being corrected and this model deliberately does not hold a
        reference to it: the adjudications table keys on `detection_id` and nothing else.
        The review layer calls this with both in scope, which is the only place both exist.

        A reviewer may correct one field or several, so a partial edit is accepted here where a
        partial *prediction* is not. Every field named must still be one the codebook defines,
        and every value must sit inside its declared domain.
        """
        from praxis.annotation.codebook import get_codebook

        if self.action != "edit":
            return
        if detection.detection_id != self.detection_id:
            raise ValueError(
                f"adjudication {self.adjudication_id} is for detection "
                f"{self.detection_id}, not {detection.detection_id}")
        get_codebook(detection.codebook_version).validate_labels(
            detection.behaviour, self.edited_value)
