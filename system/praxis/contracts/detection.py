# -*- coding: utf-8 -*-
"""Detection: one behaviour, one clip, and the confidence that must accompany it.

Rule R3 is enforced structurally here. `confidence` is a required field of a frozen model
with `extra="forbid"`, so there is no construction path that yields a bare label, and no
later code can attach one as an afterthought.

**A prediction is a label, and it is checked against the same codebook a rater is checked
against.** `predicted` holds the behaviour's codebook fields, not a single number: the config
declares both `presence_behaviours` and `intensity_behaviours` for all five, and B1 alone emits
presence, count, amplitude and non-scorable. One float could not carry that. Validating it
through `Codebook.validate_labels` — the same function the annotation tool uses — is what makes
a model output and a human label comparable at all, which is the entire basis of Phase 2's
human ceiling being a meaningful comparator in Phase 6.

**The version travels with the detection.** A codebook revision that adds a field would
otherwise retroactively invalidate every detection recorded before it, so `codebook_version` is
recorded here exactly as `annotations.codebook_version` is recorded for a rater's label.

The routing outcome is computed at write time and stored, not derived at read time. SCHEMA.md
§7 gives the reason: a suppressed value must never be serialised, and putting that decision in
the view layer puts it in the place most likely to be refactored wrongly.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from praxis.annotation.codebook import ACTIVE_CODEBOOK, get_codebook
from praxis.contracts.confidence import ConfidenceState
from praxis.vocabulary import (
    BEHAVIOUR_NAMES,
    BehaviourId,
    GateOutcome,
    SuppressionReason,
)

__all__ = ["BEHAVIOUR_NAMES", "BehaviourId", "Detection", "GateOutcome", "SuppressionReason"]


class Detection(BaseModel):
    """One model output. Never leaves the inference layer without its confidence state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_id: str
    session_id: str
    behaviour: BehaviourId
    t_start_s: float = Field(ge=0)
    t_end_s: float = Field(gt=0)

    predicted: dict[str, Any] = Field(
        description="the behaviour's codebook fields, the same shape as annotations.labels")
    # default_factory, not default: a plain default is read once when the class is defined,
    # so a detection built after a newer codebook was activated would still claim the old one.
    codebook_version: str = Field(default_factory=lambda: ACTIVE_CODEBOOK.version,
                                  description="the vocabulary this prediction is written in")

    confidence: ConfidenceState          # REQUIRED. R3 is enforced by this line
    evidence_ref: str = Field(min_length=1, description="the supporting clip and region")
    model_version: str = Field(
        min_length=1, description='pinned, e.g. "b-resnet50-tcn-v3-ens5"'
    )

    gate_outcome: GateOutcome
    suppression_reason: SuppressionReason | None = None
    explanation_ref: str | None = None

    @model_validator(mode="after")
    def clip_bounds_are_ordered(self) -> Detection:
        if self.t_end_s <= self.t_start_s:
            raise ValueError(f"t_end_s={self.t_end_s} must exceed t_start_s={self.t_start_s}")
        return self

    @model_validator(mode="after")
    def prediction_is_a_label_the_codebook_defines(self) -> Detection:
        """Refuse a prediction this behaviour has no field for, or a value out of its domain.

        Without this a B1 detection could carry a B3-shaped dict and nothing would notice until
        a reviewer was shown a zone where a gesture count belonged.
        """
        codebook = get_codebook(self.codebook_version)
        codebook.validate_labels(self.behaviour, self.predicted)

        missing = codebook.behaviour(self.behaviour).field_names - set(self.predicted)
        if missing:
            raise ValueError(
                f"{self.behaviour} prediction omits {sorted(missing)}. A partial prediction "
                f"would be shown to a reviewer as a complete one.")
        return self

    @model_validator(mode="after")
    def suppression_states_its_reason(self) -> Detection:
        if self.gate_outcome == "suppress" and self.suppression_reason is None:
            raise ValueError("a suppressed detection must say why; see API.md §6")
        if self.gate_outcome != "suppress" and self.suppression_reason is not None:
            raise ValueError(
                f"suppression_reason is set but gate_outcome is {self.gate_outcome!r}")
        return self

    @property
    def is_nonscorable(self) -> bool:
        """Whether the model judged this clip unscoreable for this behaviour.

        Every behaviour carries a `bN_nonscorable` field, and the model asserting it is the
        model declining to answer. `routing.gate` turns that into `model_abstained`.
        """
        return bool(self.predicted.get(f"{self.behaviour.lower()}_nonscorable", False))

    def to_payload(self) -> dict[str, Any]:
        """The serialised shape, with the suppression rule applied at the source.

        For a suppressed detection there is **no `predicted` key at all**. Not null, not
        present and hidden, not greyed out in the interface: absent. API.md §6 states the rule
        and `tests/integration/test_no_bare_predictions.py` asserts it against the raw JSON, so
        the omission has to happen here rather than in a template.
        """
        confidence: dict[str, Any] = {
            "ood_flag": self.confidence.ood_flag,
            "validated_domain": self.confidence.in_validated_domain,
            "plain_language": self.confidence.plain_language(),
        }
        payload: dict[str, Any] = {
            "detection_id": self.detection_id,
            "session_id": self.session_id,
            "behaviour": self.behaviour,
            "gate_outcome": self.gate_outcome,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "evidence_ref": self.evidence_ref,
        }

        if self.gate_outcome == "suppress":
            payload["suppression_reason"] = self.suppression_reason
            payload["confidence"] = confidence
            return payload

        confidence.update({
            "calibrated_prob": self.confidence.calibrated_prob,
            "method": self.confidence.method,
            "epistemic": self.confidence.epistemic,
        })
        payload["predicted"] = dict(self.predicted)
        payload["codebook_version"] = self.codebook_version
        payload["confidence"] = confidence
        payload["model_version"] = self.model_version
        if self.explanation_ref is not None:
            payload["explanation_ref"] = self.explanation_ref
        if self.gate_outcome == "escalate":
            payload["requires_second_reviewer"] = True
        return payload
