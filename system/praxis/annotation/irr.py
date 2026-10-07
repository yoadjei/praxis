# -*- coding: utf-8 -*-
"""Inter-rater agreement: the human band that H1 is tested against and H2 is expressed against.

This is a reported thesis result, not a preprocessing step. BUILD-SPEC puts Phase 2 before any
modelling for that reason: without it there is no ground truth and no baseline to state model
performance against, so a model number would have nothing to mean.

**What is different here from the predecessor project.** `imported/__init__.py` lists four
changes required before that code serves PRAXIS, and the fourth is the one this module exists
to make: the predecessor assumed a five-point ordinal rubric throughout, while B1 to B5 emit
booleans, counts, proportions and unordered categories. The statistic is therefore selected per
field from the scale type declared in `codebook.py`, and a single instrument-wide statistic is
never computed.

**Krippendorff's alpha is reported for every field regardless of scale.** The per-scale
statistics that `configs/default.yaml` names — weighted kappa for ordinal, ICC(2,k) for
continuous — are reported beside it, but alpha is the only measure defined across all four
scale types, so it is the one that makes B1 comparable with B4. Reporting only the per-scale
statistic would leave the five behaviours on incommensurable footings, which is precisely what
§7's registered prediction about their ordering requires.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from praxis.annotation.codebook import ACTIVE_CODEBOOK, Codebook, ScaleType
from praxis.annotation.statistics import (
    IccResult,
    Interval,
    bootstrap_interval,
    icc_2k,
    krippendorff_alpha,
    mean_pairwise_kappa,
)
from praxis.vocabulary import BehaviourId

# §8: production annotation is blocked until alpha reaches this on every categorical field.
# Mirrors annotation.alpha_gate in configs/default.yaml.
DEFAULT_ALPHA_GATE = 0.667


@dataclass(frozen=True)
class Rater:
    """A coder, with the two attributes the artefact checks turn on."""

    rater_id: str
    role: str | None = None
    college_id: str | None = None


@dataclass(frozen=True)
class Annotation:
    """One rater's labels for one behaviour on one clip.

    `codebook_version` is required and carried on every row, so a later revision of the
    borderline rulings never silently reinterprets a label made under an earlier one.
    """

    clip_id: str
    rater_id: str
    behaviour: BehaviourId
    codebook_version: str
    labels: dict[str, Any]
    is_nonscorable: bool = False
    rater_confidence: str = "certain"
    session_college_id: str | None = None


@dataclass
class FieldAgreement:
    """Agreement on one codeable field, with everything needed to read it honestly."""

    behaviour: BehaviourId
    field: str
    scale: ScaleType
    n_units: int
    n_raters: int
    alpha: Interval | None
    primary_statistic: str
    primary: Interval | None
    icc: IccResult | None = None
    notes: list[str] = dataclass_field(default_factory=list)

    @property
    def meets_gate(self) -> bool | None:
        """Whether alpha clears §8's gate. None where alpha is undefined."""
        if self.alpha is None:
            return None
        return self.alpha.point >= DEFAULT_ALPHA_GATE


@dataclass
class BehaviourAgreement:
    """One behaviour's fields, and the weakest of them."""

    behaviour: BehaviourId
    name: str
    fields: list[FieldAgreement]

    @property
    def mean_alpha(self) -> float | None:
        values = [f.alpha.point for f in self.fields if f.alpha is not None]
        return sum(values) / len(values) if values else None

    @property
    def weakest_field(self) -> FieldAgreement | None:
        scored = [f for f in self.fields if f.alpha is not None]
        return min(scored, key=lambda f: f.alpha.point) if scored else None

    @property
    def blocked_fields(self) -> list[str]:
        """Fields below the gate. §8: these are redefined or dropped, not carried forward."""
        return [f.field for f in self.fields if f.meets_gate is False]


@dataclass(frozen=True)
class RangeCheck:
    """How concentrated one field's labels are, and by which measure.

    The measure is carried with the number because it differs by scale type; see
    `_restriction_of_range`. A proportion without its measure would not be interpretable.
    """

    field: str
    proportion: float
    measure: str
    n: int

    def __str__(self) -> str:
        return f"{self.proportion:.0%} ({self.measure}, n={self.n})"


@dataclass
class ArtefactChecks:
    """Checks that report whether or not they flatter, per BUILD-SPEC Phase 2."""

    restriction_of_range: dict[str, RangeCheck]
    rater_role_effects: dict[str, float | None]
    familiarity_effects: dict[str, float | None]
    warnings: list[str]


@dataclass
class AgreementReport:
    """The per-behaviour agreement table with confidence intervals that H1 is tested against."""

    codebook_version: str
    n_annotations: int
    n_raters: int
    n_clips: int
    behaviours: list[BehaviourAgreement]
    artefacts: ArtefactChecks
    excluded_guesses: int
    design_fully_crossed: bool
    warnings: list[str]

    def behaviour(self, behaviour: BehaviourId) -> BehaviourAgreement:
        for entry in self.behaviours:
            if entry.behaviour == behaviour:
                return entry
        raise KeyError(behaviour)

    def observed_order(self) -> list[BehaviourId]:
        """Behaviours ranked by mean alpha, best first.

        §7 registers the predicted order B1, B3, B5, B2, B4 before annotation begins, with
        B4 lowest. Comparing this against that prediction is what makes it a test. "If B4
        does not come out lowest, that is a finding and it is reported as one."
        """
        scored = [(b.behaviour, b.mean_alpha) for b in self.behaviours
                  if b.mean_alpha is not None]
        return [behaviour for behaviour, _ in sorted(scored, key=lambda kv: -kv[1])]

    def as_table(self) -> list[dict[str, Any]]:
        """Flat rows, one per field, for writing out or rendering."""
        rows = []
        for entry in self.behaviours:
            for item in entry.fields:
                rows.append({
                    "behaviour": item.behaviour,
                    "field": item.field,
                    "scale": item.scale.value,
                    "n_units": item.n_units,
                    "n_raters": item.n_raters,
                    "alpha": None if item.alpha is None else round(item.alpha.point, 4),
                    "alpha_ci_low": None if item.alpha is None else round(item.alpha.lower, 4),
                    "alpha_ci_high": None if item.alpha is None else round(item.alpha.upper, 4),
                    "primary_statistic": item.primary_statistic,
                    "primary": None if item.primary is None else round(item.primary.point, 4),
                    "icc_2k": None if item.icc is None or item.icc.icc is None
                              else round(item.icc.icc, 4),
                    "meets_alpha_gate": item.meets_gate,
                    "notes": "; ".join(item.notes),
                })
        return rows


# ---------------------------------------------------------------------------
# Assembling units
# ---------------------------------------------------------------------------

def _units_for(annotations: Sequence[Annotation], field: str,
               raters: Sequence[str]) -> list[list[Any]]:
    """One row per clip, one column per rater, None where that rater did not code it.

    The rater ordering is fixed across every field so that a pairwise statistic compares the
    same two people from one field to the next.
    """
    position = {rater: index for index, rater in enumerate(raters)}
    by_clip: dict[str, list[Any]] = defaultdict(lambda: [None] * len(raters))

    for annotation in annotations:
        if field not in annotation.labels:
            continue
        by_clip[annotation.clip_id][position[annotation.rater_id]] = annotation.labels[field]

    return [by_clip[clip] for clip in sorted(by_clip)]


def _statistic_for(scale: ScaleType) -> str:
    """The primary statistic for a scale, matching annotation.irr in default.yaml."""
    return {
        ScaleType.NOMINAL: "krippendorff_alpha",
        ScaleType.ORDINAL: "weighted_kappa_quadratic",
        ScaleType.INTERVAL: "icc_2k",
        ScaleType.RATIO: "krippendorff_alpha",
    }[scale]


# ---------------------------------------------------------------------------
# Artefact checks
# ---------------------------------------------------------------------------

def _restriction_of_range(annotations: Sequence[Annotation],
                          codebook: Codebook) -> dict[str, RangeCheck]:
    """How concentrated a field's labels are, and therefore how much room to disagree existed.

    BUILD-SPEC Phase 2 asks for "proportion of labels in extreme categories". That phrase
    only means something on a ranked scale with an interior: on a boolean, *both* categories
    are extremes, so the proportion is always 1.0 and the check says nothing. On an unordered
    category there are no extremes at all.

    So the measure is chosen to match the scale, and the report says which was used:

    * **ordinal with three or more bands** - the share in the top and bottom bands, which is
      the measure as written, and the case it was written for.
    * **everything else** - the share held by the single most-used category. A field where
      95 per cent of labels are one value leaves almost nothing to disagree about, which is
      the same concern the original phrasing was reaching for.

    Reported whether or not it flatters.
    """
    tallies: dict[str, Counter] = defaultdict(Counter)
    scales: dict[str, tuple[ScaleType, tuple]] = {}

    for annotation in annotations:
        spec = codebook.behaviour(annotation.behaviour)
        for name, value in annotation.labels.items():
            if name not in spec.field_names:
                continue
            field = spec.field(name)
            if not field.levels:
                continue
            key = f"{annotation.behaviour}.{name}"
            tallies[key][value] += 1
            scales[key] = (field.scale, field.levels)

    checks: dict[str, RangeCheck] = {}
    for key, counts in tallies.items():
        total = sum(counts.values())
        if not total:
            continue
        scale, levels = scales[key]
        if scale is ScaleType.ORDINAL and len(levels) >= 3:
            share = (counts[levels[0]] + counts[levels[-1]]) / total
            measure = "share in the top and bottom bands"
        else:
            share = max(counts.values()) / total
            measure = "share held by the most-used category"
        checks[key] = RangeCheck(field=key, proportion=share, measure=measure, n=total)
    return checks


def _agreement_within(annotations: Sequence[Annotation], codebook: Codebook) -> float | None:
    """Mean nominal alpha across every categorical field, for subgroup comparisons."""
    raters = sorted({a.rater_id for a in annotations})
    if len(raters) < 2:
        return None

    values: list[float] = []
    for behaviour_spec in codebook.behaviours:
        subset = [a for a in annotations if a.behaviour == behaviour_spec.behaviour]
        if not subset:
            continue
        for field in behaviour_spec.fields:
            if field.scale is not ScaleType.NOMINAL:
                continue
            alpha = krippendorff_alpha(_units_for(subset, field.name, raters),
                                       ScaleType.NOMINAL)
            if alpha is not None:
                values.append(alpha)
    return sum(values) / len(values) if values else None


def _artefact_checks(annotations: Sequence[Annotation], codebook: Codebook,
                     raters: dict[str, Rater]) -> ArtefactChecks:
    warnings: list[str] = []

    restriction = _restriction_of_range(annotations, codebook)
    for key, check in sorted(restriction.items()):
        if check.proportion >= 0.90:
            warnings.append(
                f"{key}: {check.proportion:.0%} {check.measure}. Agreement on this field is "
                f"close to uninformative, because there was almost nothing to disagree "
                f"about. Read its alpha with that in mind.")

    by_role: dict[str, float | None] = {}
    roles = {raters[r].role for r in raters if raters[r].role}
    if len(roles) >= 2:
        for role in sorted(roles):
            members = {r for r in raters if raters[r].role == role}
            if len(members) < 2:
                by_role[role] = None
                warnings.append(
                    f"only one rater holds the role {role!r}, so agreement within it is not "
                    f"defined. Reported as not computable rather than as absent, because "
                    f"'we could not measure it' and 'we measured it and it was low' are "
                    f"different findings.")
                continue
            by_role[role] = _agreement_within(
                [a for a in annotations if a.rater_id in members], codebook)
    elif roles:
        warnings.append(
            f"every rater has role {roles.pop()!r}, so a rater-role effect cannot be "
            f"estimated. Reported as not computable rather than as absent.")

    familiarity: dict[str, float | None] = {}
    labelled = [a for a in annotations if a.session_college_id]
    if labelled and any(raters[a.rater_id].college_id for a in labelled):
        own = [a for a in labelled
               if raters[a.rater_id].college_id == a.session_college_id]
        other = [a for a in labelled
                 if raters[a.rater_id].college_id != a.session_college_id]
        familiarity["own_college"] = _agreement_within(own, codebook) if own else None
        familiarity["other_college"] = _agreement_within(other, codebook) if other else None
        if familiarity.get("own_college") is not None \
                and familiarity.get("other_college") is not None:
            gap = familiarity["own_college"] - familiarity["other_college"]
            if abs(gap) >= 0.10:
                warnings.append(
                    f"raters agree {gap:+.3f} differently on their own college's sessions. "
                    f"A familiarity effect of this size belongs in the limitations, not a "
                    f"footnote.")

    return ArtefactChecks(restriction, by_role, familiarity, warnings)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def report(annotations: Iterable[Annotation],
           codebook: Codebook = ACTIVE_CODEBOOK,
           raters: Sequence[Rater] | None = None,
           bootstrap: int = 1000,
           ci_level: float = 0.95,
           seed: int = 20260911,
           exclude_guesses: bool = True) -> AgreementReport:
    """The per-behaviour agreement table with confidence intervals, per field.

    Guesses are excluded from the primary analysis by default and counted separately, per §1:
    a ground truth built partly from guesses is not a ground truth. Set `exclude_guesses`
    false to see what including them does, which is a robustness check and not the headline.

    Annotations made under a codebook version other than the one supplied are refused rather
    than pooled. Pooling them would compare labels whose definitions differ, which looks like
    disagreement between raters and is actually disagreement between documents.
    """
    annotations = list(annotations)
    warnings: list[str] = []

    wrong_version = sorted({a.codebook_version for a in annotations
                            if a.codebook_version != codebook.version})
    if wrong_version:
        raise ValueError(
            f"annotations were made under codebook version(s) {wrong_version} but the report "
            f"was asked for {codebook.version}. Labels are not comparable across versions: a "
            f"revision changes what a label means, so pooling them would read a change of "
            f"definition as a disagreement between raters.")

    # One pass each rather than a membership test against a list: an annotation carries a
    # dict and so is unhashable, which would make `a not in guesses` a linear scan per row.
    excluded_levels = set(codebook.excluded_from_primary_irr)
    n_guesses = sum(1 for a in annotations if a.rater_confidence in excluded_levels)
    if exclude_guesses and n_guesses:
        annotations = [a for a in annotations
                       if a.rater_confidence not in excluded_levels]

    roster = {r.rater_id: r for r in (raters or [])}
    for rater_id in {a.rater_id for a in annotations}:
        roster.setdefault(rater_id, Rater(rater_id))
    rater_ids = sorted(roster)

    clips = {a.clip_id for a in annotations}
    expected_cells = len(clips) * len(rater_ids)
    fully_crossed = expected_cells > 0 and len(
        {(a.clip_id, a.rater_id) for a in annotations}) == expected_cells
    if not fully_crossed and expected_cells:
        warnings.append(
            "the design is not fully crossed: not every rater coded every clip. ICC "
            "underestimates reliability in this case and is reported as a lower bound; "
            "Krippendorff's alpha tolerates the missingness and is the primary measure.")

    behaviours: list[BehaviourAgreement] = []
    for spec in codebook.behaviours:
        subset = [a for a in annotations if a.behaviour == spec.behaviour]
        entries: list[FieldAgreement] = []

        for field in spec.fields:
            units = _units_for(subset, field.name, rater_ids)
            notes: list[str] = []

            alpha = bootstrap_interval(
                units, lambda u, s=field.scale: krippendorff_alpha(u, s),
                draws=bootstrap, level=ci_level, seed=seed)

            icc = None
            if field.scale is ScaleType.INTERVAL:
                icc = icc_2k(units)
                if icc.warning:
                    notes.append(icc.warning)

            if field.scale is ScaleType.ORDINAL:
                primary = bootstrap_interval(
                    units, lambda u, lv=field.levels: mean_pairwise_kappa(u, lv),
                    draws=bootstrap, level=ci_level, seed=seed)
            elif field.scale is ScaleType.INTERVAL:
                primary = bootstrap_interval(
                    units, lambda u: icc_2k(u).icc,
                    draws=bootstrap, level=ci_level, seed=seed)
            else:
                primary = alpha

            if alpha is None:
                notes.append(
                    "alpha is undefined here: either fewer than two raters coded any clip, "
                    "or every rater used one category throughout, which is a real state of "
                    "the data rather than perfect agreement.")

            entries.append(FieldAgreement(
                behaviour=spec.behaviour, field=field.name, scale=field.scale,
                n_units=len(units), n_raters=len(rater_ids), alpha=alpha,
                primary_statistic=_statistic_for(field.scale), primary=primary,
                icc=icc, notes=notes))

        behaviours.append(BehaviourAgreement(spec.behaviour, spec.name, entries))

    return AgreementReport(
        codebook_version=codebook.version,
        n_annotations=len(annotations),
        n_raters=len(rater_ids),
        n_clips=len(clips),
        behaviours=behaviours,
        artefacts=_artefact_checks(annotations, codebook, roster),
        excluded_guesses=n_guesses if exclude_guesses else 0,
        design_fully_crossed=fully_crossed,
        warnings=warnings,
    )
