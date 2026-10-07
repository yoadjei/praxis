# -*- coding: utf-8 -*-
"""Attribution: turning an observed drop into something a college could act on.

BUILD-SPEC Phase 6 is emphatic that "attribution is what turns a negative result into design
knowledge". A number saying performance fell by eleven points tells a teacher-training college
nothing it can do differently. A coefficient saying it fell with camera distance and not with
class size tells them where to put the camera.

The model is a mixed-effects regression with a random intercept per teacher, and the three
choices that make it honest -- dropping incomplete sessions rather than imputing them,
absorbing the teacher rather than pooling over one, and standardising so the coefficients rank
by influence rather than by units -- are recorded as D29.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from praxis.evaluation.shift import ShiftError


@dataclass(frozen=True)
class Coefficient:
    """One covariate's estimated effect, with an interval and never without one."""

    name: str
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float

    @property
    def excludes_zero(self) -> bool:
        return (self.ci_lower > 0.0) or (self.ci_upper < 0.0)

    def __str__(self) -> str:
        return (f"{self.name}: {self.estimate:+.4f} "
                f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  p={self.p_value:.3f}")


@dataclass(frozen=True)
class AttributionResult:
    """What the degradation is attributable to, with the caveats attached."""

    coefficients: tuple[Coefficient, ...]
    n_sessions: int
    n_teachers: int
    n_dropped_missing_covariates: int
    model: str
    converged: bool
    warnings: tuple[str, ...]

    def named(self, name: str) -> Coefficient:
        for coefficient in self.coefficients:
            if coefficient.name == name:
                return coefficient
        raise KeyError(name)

    @property
    def attributable_to(self) -> list[str]:
        """Covariates whose interval excludes zero. The ones a college could act on."""
        return [c.name for c in self.coefficients
                if c.excludes_zero and c.name not in ("Intercept", "Group Var")]


def attribution_feasibility(per_session: Sequence[Mapping[str, object]],
                            covariates: Sequence[str],
                            outcome: str = "drop",
                            teacher_key: str = "teacher_id") -> str | None:
    """Why attribution cannot run on this corpus, or None if it can.

    `attribute_degradation` raises in the same circumstances, and raising is right there: it is
    asked to produce a regression and will not produce a meaningless one. What was missing is a
    way to ask first. A caller assembling a Phase 6 report needs to say "attribution abstained,
    and here is why" in the table, and getting that from an exception means writing the one
    `except` that also swallows the real failures.

    D48: an unavailable check abstains, and abstention is its own verdict. The corpus this was
    written against has around eleven sessions and lesson plans whose covariate fields are
    inconsistently filled, so this is the expected path rather than the exceptional one. D86.
    """
    covariates = list(covariates)
    if not covariates:
        return "no covariates were supplied, so there is nothing to attribute the drop to"

    complete = [row for row in per_session
                if not any(row.get(name) is None
                           for name in (*covariates, outcome, teacher_key))]
    needed = len(covariates) + 2
    if len(complete) < needed:
        dropped = len(per_session) - len(complete)
        return (f"{len(complete)} of {len(per_session)} sessions carry every covariate and "
                f"{needed} are needed for {len(covariates)} of them; {dropped} were dropped "
                f"for missing measurements. Covariates are captured at recording time and "
                f"cannot be recovered afterwards")
    return None


def attribute_degradation(per_session: Sequence[Mapping[str, object]],
                          covariates: Sequence[str],
                          outcome: str = "drop",
                          teacher_key: str = "teacher_id",
                          ci_level: float = 0.95) -> AttributionResult:
    """Regress per-session performance drop on the measured covariates.

    A mixed-effects model with a random intercept per teacher, because a teacher contributes
    several sessions and treating those as independent would understate the standard errors
    and manufacture significance. The random intercept absorbs whatever is constant about a
    candidate, so what the fixed effects pick up is the recording conditions.

    **Sessions missing any covariate are dropped, never imputed.** The `Session` contract calls
    the covariates "all measured, none assumed", and imputing one would invent the quantity the
    regression exists to estimate. The number dropped is returned, because a result computed on
    half the corpus should be read as one.
    """
    covariates = list(covariates)
    if not covariates:
        raise ShiftError("no covariates supplied; there is nothing to attribute the drop to")

    complete, dropped = [], 0
    for row in per_session:
        if any(row.get(name) is None for name in (*covariates, outcome, teacher_key)):
            dropped += 1
            continue
        complete.append(row)

    if len(complete) < len(covariates) + 2:
        raise ShiftError(
            f"{len(complete)} sessions carry every covariate, which cannot support "
            f"{len(covariates)} of them. {dropped} were dropped for missing measurements. "
            f"Covariates must be captured at recording time; a session recorded without them "
            f"cannot contribute to attribution afterwards.")

    import pandas as pd

    frame = pd.DataFrame(complete)
    teachers = frame[teacher_key].nunique()
    formula = f"{outcome} ~ " + " + ".join(covariates)
    notes: list[str] = []

    # Standardising puts the coefficients on comparable footing: camera distance in metres and
    # room area in square metres differ by an order of magnitude, so raw coefficients would
    # rank the covariates by their units rather than by their influence.
    for name in covariates:
        spread = frame[name].std()
        if spread == 0 or np.isnan(spread):
            notes.append(f"{name} does not vary across sessions and cannot be attributed to")
            continue
        frame[name] = (frame[name] - frame[name].mean()) / spread

    usable = [c for c in covariates if frame[c].std() > 0]
    formula = f"{outcome} ~ " + " + ".join(usable)

    import statsmodels.formula.api as smf

    converged, model_name = True, "mixed_effects"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if teachers < 2:
                raise ValueError("fewer than two teachers")
            fitted = smf.mixedlm(formula, frame, groups=frame[teacher_key]).fit()
            converged = bool(getattr(fitted, "converged", True))
            if not converged:
                notes.append(
                    "the mixed model did not converge; coefficients are reported but their "
                    "standard errors should not be trusted")
        except Exception as exc:
            notes.append(
                f"a mixed model could not be fitted ({exc}); fell back to ordinary least "
                f"squares, which treats a teacher's sessions as independent and so "
                f"understates the standard errors. Read the intervals as optimistic.")
            fitted = smf.ols(formula, frame).fit()
            model_name = "ols_fallback"

    intervals = fitted.conf_int(alpha=1.0 - ci_level)
    coefficients = tuple(
        Coefficient(name=str(name), estimate=float(fitted.params[name]),
                    std_error=float(fitted.bse[name]),
                    ci_lower=float(intervals.loc[name][0]),
                    ci_upper=float(intervals.loc[name][1]),
                    p_value=float(fitted.pvalues[name]))
        for name in fitted.params.index if str(name) != "Group Var")

    return AttributionResult(
        coefficients=coefficients, n_sessions=len(complete), n_teachers=int(teachers),
        n_dropped_missing_covariates=dropped, model=model_name, converged=converged,
        warnings=tuple(notes))


