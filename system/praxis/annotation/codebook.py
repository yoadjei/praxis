# -*- coding: utf-8 -*-
"""The label vocabulary, machine-readable, and the scale type of every field.

`docs/CODEBOOK.md` is the authority a rater reads. This module is the same thing in a form
the tool and the agreement analysis can check against, and the two are tied together by
`Codebook.document_sha256`: a codebook whose prose has changed without its version changing is
detected rather than assumed away.

**The scale type of each field is the load-bearing part.** `imported/__init__.py` names it as
one of the four changes that must happen before the predecessor's statistics mean anything
here: that code assumed a five-point ordinal rubric throughout, while B1 to B5 emit booleans,
counts, proportions, angles-as-bands and unordered categories. A quadratic weighted kappa on a
gesture count is not so much wrong as meaningless, so the statistic is selected per field from
the scale recorded here rather than chosen once for the whole instrument.

Two rules the annotation tool enforces from this module:

* a label the loaded version does not define is refused, naming the offending key
* every annotation records the version it was made under, so a later revision never silently
  reinterprets an earlier label
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from praxis.vocabulary import BehaviourId


class ScaleType(str, Enum):
    """How a field's values relate to one another, which fixes the statistic.

    The four are Krippendorff's own difference functions. `NOMINAL` covers booleans and
    unordered categories; `ORDINAL` covers ranked bands whose spacing is not claimed to be
    equal; `INTERVAL` covers proportions, where a gap of 0.25 means the same anywhere on the
    range; `RATIO` covers counts, which have a true zero so a difference between 1 and 2
    matters more than the same difference between 11 and 12.
    """

    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


class FieldSpec(BaseModel):
    """One codeable field: what it is called, what it may hold, and how it is compared."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    scale: ScaleType
    levels: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    description: str

    @model_validator(mode="after")
    def domain_is_stated(self) -> FieldSpec:
        if self.scale in (ScaleType.NOMINAL, ScaleType.ORDINAL):
            if not self.levels:
                raise ValueError(f"{self.name}: a {self.scale.value} field must list levels")
        elif self.minimum is None or self.maximum is None:
            raise ValueError(f"{self.name}: a {self.scale.value} field must state its range")
        return self

    def accepts(self, value: Any) -> bool:
        """Whether `value` is inside this field's declared domain.

        Boolean-ness has to match before equality is consulted, because Python's `bool` is a
        subclass of `int` and plain `in` is therefore too permissive in both directions:
        `0 in (False, True)` is true, so a count could pass as a presence flag, and
        `True in (1, 2, 3)` is true, so a presence flag could pass as an amplitude band. Either
        would reach a reviewer as a legal-looking label and serialise as the wrong JSON type.
        """
        if self.levels is not None:
            return any(isinstance(value, bool) == isinstance(level, bool) and value == level
                       for level in self.levels)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        return self.minimum <= float(value) <= self.maximum


class BehaviourSpec(BaseModel):
    """One of the five behaviours, with its fields and its non-scorable conditions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    behaviour: BehaviourId
    name: str
    fields: tuple[FieldSpec, ...]
    nonscorable_when: str

    @property
    def field_names(self) -> set[str]:
        return {field.name for field in self.fields}

    def field(self, name: str) -> FieldSpec:
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise KeyError(f"{self.behaviour} defines no field {name!r}")


class CodebookError(ValueError):
    """Raised when a label does not match the loaded codebook version."""


class Codebook(BaseModel):
    """A versioned label vocabulary. Annotations are made under exactly one of these."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    document_sha256: str | None = None
    behaviours: tuple[BehaviourSpec, ...]
    # §1: every label carries one of these, and guesses are excluded from the primary
    # agreement analysis, because a ground truth built partly from guesses is not one.
    confidence_levels: tuple[str, ...] = ("certain", "probable", "guess")
    excluded_from_primary_irr: tuple[str, ...] = ("guess",)

    def behaviour(self, behaviour: BehaviourId) -> BehaviourSpec:
        for spec in self.behaviours:
            if spec.behaviour == behaviour:
                return spec
        raise KeyError(f"codebook {self.version} defines no behaviour {behaviour!r}")

    def validate_labels(self, behaviour: BehaviourId, labels: Mapping[str, Any]) -> None:
        """Refuse anything this version does not define, naming what was wrong.

        The tool cannot emit a label absent from the loaded version, which is how the
        codebook stays authoritative rather than advisory. API.md §5 returns 422 naming the
        offending key, so the message here is written to be shown to a person.
        """
        spec = self.behaviour(behaviour)

        unknown = sorted(set(labels) - spec.field_names)
        if unknown:
            raise CodebookError(
                f"{behaviour} has no field {unknown[0]!r} in codebook {self.version}. "
                f"Defined fields are {sorted(spec.field_names)}.")

        for name, value in labels.items():
            field = spec.field(name)
            if not field.accepts(value):
                permitted = (f"one of {list(field.levels)}" if field.levels is not None
                             else f"between {field.minimum} and {field.maximum}")
                raise CodebookError(
                    f"{behaviour}.{name} does not accept {value!r} in codebook "
                    f"{self.version}; it must be {permitted}.")

    def with_document_hash(self, path: str | Path) -> Codebook:
        """Bind this vocabulary to the prose a rater was trained on.

        SCHEMA.md stores `document_sha256` beside the version so that a revision to the
        borderline rulings, which changes what a label *means* without changing the field
        set, cannot pass unnoticed.
        """
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return self.model_copy(update={"document_sha256": digest})


def _boolean(name: str, description: str) -> FieldSpec:
    return FieldSpec(name=name, scale=ScaleType.NOMINAL, levels=(False, True),
                     description=description)


# ---------------------------------------------------------------------------
# v1.0-draft, transcribed from docs/CODEBOOK.md.
#
# Literature-derived and NOT yet expert-validated: §9 of that document lists five open items
# for the Delphi panel, including whether B4 is worth retaining at all. The panel's decisions
# replace the corresponding sections and increment the version to v2.0.
# ---------------------------------------------------------------------------

CODEBOOK_V1 = Codebook(
    version="v1.0-draft",
    behaviours=(
        BehaviourSpec(
            behaviour="B1",
            name="Gesture production",
            nonscorable_when=(
                "the teacher is out of frame for more than 50 per cent of the clip; both "
                "hands are occluded for more than 50 per cent; motion blur prevents "
                "locating the hands"),
            fields=(
                _boolean("b1_present", "at least one gesture unit occurred in the clip"),
                FieldSpec(
                    name="b1_count", scale=ScaleType.RATIO, minimum=0, maximum=20,
                    description="number of discrete gesture units"),
                FieldSpec(
                    name="b1_amplitude", scale=ScaleType.ORDINAL, levels=(1, 2, 3),
                    description=("largest gesture in the clip: 1 within torso width, "
                                 "2 to shoulder width, 3 beyond shoulder width or above head")),
                _boolean("b1_nonscorable", "this behaviour cannot be judged in this clip"),
            ),
        ),
        BehaviourSpec(
            behaviour="B2",
            name="Body orientation to class",
            nonscorable_when=(
                "fewer than five upper-body keypoints are trackable for more than 50 per "
                "cent of the clip; the learner region was not defined for this camera "
                "setup; no learner is in frame for more than 50 per cent of the clip; "
                "the camera moved mid-clip"),
            fields=(
                # §3 restricts raters to quarters: "Finer judgment is not reliable by eye and
                # pretending otherwise inflates apparent precision." The domain is therefore
                # five points, but the spacing is genuinely equal, so it is interval and not
                # ordinal, and ICC(2,k) applies to it as configs/default.yaml specifies.
                FieldSpec(
                    name="b2_facing_proportion", scale=ScaleType.INTERVAL,
                    minimum=0.0, maximum=1.0,
                    description=("proportion of the clip with the torso within 45 degrees "
                                 "of the learner-region centroid, estimated in quarters")),
                FieldSpec(
                    name="b2_head_torso_divergence", scale=ScaleType.ORDINAL,
                    levels=(1, 2, 3),
                    description=("median absolute head-torso difference: 1 under 30 "
                                 "degrees, 2 from 30 to 60, 3 over 60")),
                FieldSpec(
                    name="b2_dominant", scale=ScaleType.NOMINAL,
                    levels=("class", "board", "materials", "away"),
                    description="what the teacher predominantly faces across the clip"),
                _boolean("b2_nonscorable", "this behaviour cannot be judged in this clip"),
            ),
        ),
        BehaviourSpec(
            behaviour="B3",
            name="Spatial position and mobility",
            nonscorable_when=(
                "the teacher is out of frame for more than 30 per cent of the clip, a "
                "stricter threshold than B1 because position is the measure itself; zones "
                "are undefined for the setup; no learner is in frame for more than 30 per "
                "cent of the clip, which leaves b3_proximity without a referent; the "
                "camera moved"),
            fields=(
                FieldSpec(
                    name="b3_zone", scale=ScaleType.NOMINAL,
                    levels=("board", "front", "middle", "back"),
                    description="zone occupied longest in the clip"),
                _boolean("b3_zone_changed", "the teacher crossed a zone boundary"),
                FieldSpec(
                    name="b3_transitions", scale=ScaleType.RATIO, minimum=0, maximum=6,
                    description="number of boundary crossings"),
                FieldSpec(
                    name="b3_movement", scale=ScaleType.ORDINAL, levels=(1, 2, 3, 4),
                    description=("1 stationary, 2 shifting in place, 3 walking within a "
                                 "zone, 4 traversing zones")),
                FieldSpec(
                    name="b3_proximity", scale=ScaleType.ORDINAL, levels=(1, 2, 3, 4),
                    description=("distance to nearest learner: 1 within arm's reach, "
                                 "2 one to two metres, 3 two to four, 4 over four")),
                _boolean("b3_nonscorable", "this behaviour cannot be judged in this clip"),
            ),
        ),
        BehaviourSpec(
            behaviour="B4",
            name="Postural stance",
            nonscorable_when=(
                "the torso is occluded for more than 50 per cent of the clip; the teacher "
                "is out of frame for more than 50 per cent; only the head and shoulders "
                "are visible"),
            fields=(
                FieldSpec(
                    name="b4_posture", scale=ScaleType.NOMINAL,
                    levels=("standing_upright", "standing_supported", "seated",
                            "crouching", "other"),
                    description="dominant posture across the clip"),
                FieldSpec(
                    name="b4_lean", scale=ScaleType.ORDINAL, levels=(1, 2, 3),
                    description=("torso deviation from vertical: 1 under 10 degrees, "
                                 "2 from 10 to 30, 3 over 30")),
                # D7: there is no openness scale. Openness interprets what a posture
                # signifies, which is an inferred construct and excluded by §0.
                FieldSpec(
                    name="b4_arms", scale=ScaleType.NOMINAL,
                    levels=("crossed", "at_sides", "hands_clasped", "one_raised",
                            "both_raised", "holding_object", "indeterminate"),
                    description="dominant arm configuration, not what it signifies"),
                _boolean("b4_nonscorable", "this behaviour cannot be judged in this clip"),
            ),
        ),
        BehaviourSpec(
            behaviour="B5",
            name="Board and material use",
            nonscorable_when=(
                "the board is outside the frame and the clip's activity appears "
                "board-directed; the teacher's hands are occluded for more than 50 per cent"),
            fields=(
                _boolean("b5_board", "writing, drawing, erasing, or pointing at the board"),
                _boolean("b5_material", "handling an instructional artifact"),
                FieldSpec(
                    name="b5_type", scale=ScaleType.NOMINAL,
                    levels=("writing", "pointing_at_board", "erasing", "displaying_object",
                            "demonstrating_with_object", "distributing", "none"),
                    description="dominant activity"),
                FieldSpec(
                    name="b5_duration", scale=ScaleType.ORDINAL, levels=(0, 1, 2, 3),
                    description=("seconds engaged: 0 none, 1 under two, 2 from two to "
                                 "five, 3 over five")),
                _boolean("b5_nonscorable", "this behaviour cannot be judged in this clip"),
            ),
        ),
    ),
)

# §7 registers these before annotation begins, so that H1 is a test and not a description.
# B4 is predicted lowest. "If B4 does not come out lowest, that is a finding and it is
# reported as one."
EXPECTED_AGREEMENT_ORDER: tuple[BehaviourId, ...] = ("B1", "B3", "B5", "B2", "B4")

ACTIVE_CODEBOOK = CODEBOOK_V1

# Every version ever activated, kept so that a label or a prediction made under an earlier one
# can still be checked against the vocabulary it was actually made under. A revision that adds
# a field would otherwise retroactively invalidate everything recorded before it.
CODEBOOKS: dict[str, Codebook] = {CODEBOOK_V1.version: CODEBOOK_V1}


def get_codebook(version: str) -> Codebook:
    """The vocabulary a given version defines. Unknown versions are refused, never guessed."""
    try:
        return CODEBOOKS[version]
    except KeyError:
        raise CodebookError(
            f"codebook version {version!r} is not registered; known versions are "
            f"{sorted(CODEBOOKS)}. A label or detection recorded under an unregistered "
            f"version cannot be checked against anything.") from None
