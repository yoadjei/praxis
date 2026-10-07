# -*- coding: utf-8 -*-
"""ConfidenceState: the construct that separates this system from the ones it is compared to.

Rule R3 says a bare label may never leave the inference layer. That rule is enforced here and
in `detection.py`, by making the confidence a required field rather than an optional
enrichment: there is no way to construct a Detection without one.

Two points §4.3 left open are settled here rather than left to the caller. Both are recorded
as D16.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CalibrationMethod = Literal["temperature", "ensemble", "mc_dropout"]
ValidatedDomain = Literal["microteaching", "classroom", "unknown"]


class ConfidenceState(BaseModel):
    """What the model believes, how it came to believe it, and whether it should be trusted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_prob: float = Field(ge=0.0, le=1.0, description="uncalibrated model output")
    calibrated_prob: float = Field(ge=0.0, le=1.0, description="after temperature or ensemble")
    method: CalibrationMethod

    # Mutual information between the predictive distribution and the ensemble members.
    epistemic: float | None = Field(default=None, ge=0.0)

    ood_score: float
    ood_flag: bool = Field(description="True when outside the validated distribution")
    in_validated_domain: ValidatedDomain

    @model_validator(mode="after")
    def ensemble_reports_its_disagreement(self) -> ConfidenceState:
        # D16. For an ensemble, epistemic uncertainty is the method's whole output: it is the
        # signal the Phase 6 OOD argument and the escalate branch of the routing gate both
        # read. An ensemble detection without it is not an ensemble detection.
        if self.method == "ensemble" and self.epistemic is None:
            raise ValueError(
                "method='ensemble' requires epistemic; it is the ensemble disagreement that "
                "routing and OOD detection both read. See D16.")
        return self

    @model_validator(mode="after")
    def flagged_means_unknown_domain(self) -> ConfidenceState:
        # D16. API.md §6 already pairs these in the suppressed payload it specifies. Claiming
        # a detection is out of distribution while also naming the distribution it belongs to
        # is a contradiction, and it would reach a reviewer as a confident-looking sentence.
        if self.ood_flag and self.in_validated_domain != "unknown":
            raise ValueError(
                f"ood_flag is True but in_validated_domain is "
                f"{self.in_validated_domain!r}; a flagged detection is outside every "
                f"validated domain, so it must be 'unknown'. See D16.")
        return self

    def plain_language(self) -> str:
        """The sentence the interface leads with.

        Teacher educators in this setting cannot be assumed fluent with calibrated
        probability, so API.md §6 requires a sentence and keeps the number available on
        demand. This is that sentence, and it is generated here so every surface says the
        same thing about the same state.
        """
        if self.ood_flag:
            return ("This recording is unlike those the model was checked on. "
                    "No suggestion is offered.")
        if self.calibrated_prob >= 0.85:
            return ("High confidence. This recording resembles what the model was "
                    "checked on.")
        if self.calibrated_prob >= 0.70:
            return "Moderate confidence. Worth checking against the evidence."
        return "Low confidence. Treat the evidence as the only reliable guide."
