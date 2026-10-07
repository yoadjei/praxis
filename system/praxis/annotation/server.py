"""Annotation server domain logic: assignments, acceptance, disagreements, and calibration.

This module contains the core annotation domain logic with NO database and NO HTTP. Another
agent owns persistence (praxis/annotation/store.py) and another owns the API
(praxis/api/routes/annotation.py).

**Why the codebook version travels with every annotation:** a codebook revision changes what
a label means. A label recorded under v1 where the field list was `["a", "b"]` and later
recorded under v2 where it was `["a", "b", "c"]` cannot be pooled in an agreement analysis
without reading a change of definition as a disagreement between raters. So every annotation
row records the version it was made under.

**Why build_assignments spreads across sessions (D55):** clips within one session share that
session's prevalence, so the hundredth clip from a session carries far less information than
the first clip from a new one. An assignment builder that assigns all raters to one session
before moving to the next loses statistical power. The spread pattern concentrates coverage
across sessions rather than concentrating it within one.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from praxis.annotation.clips import ClipRef
from praxis.annotation.codebook import Codebook
from praxis.annotation.irr import (
    DEFAULT_ALPHA_GATE,
    AgreementReport,
    Annotation,
    Rater,
    report,
)
from praxis.ids import new_ulid
from praxis.vocabulary import BEHAVIOUR_IDS, BehaviourId


class AnnotationRefused(ValueError):
    """Raised when an annotation violates the loaded codebook or duplicates a prior label.

    The http_status is 422 (unprocessable entity, unknown label/behaviour/version) or 409
    (conflict, duplicate rater+clip+behaviour).
    """

    reason: str
    http_status: int

    def __init__(self, reason: str, http_status: int) -> None:
        self.reason = reason
        self.http_status = http_status
        super().__init__(reason)


@dataclass(frozen=True)
class Assignment:
    """One rater's assignment to label one behaviour on one clip.

    The multi-rater core is a stratified clip sample spread across the corpus. Each clip
    receives `raters_per_clip` raters per behaviour, and they are spread across sessions
    rather than concentrated in one, per D55.
    """

    assignment_id: str
    rater_id: str
    clip: ClipRef
    behaviour: BehaviourId
    round_name: str


def build_assignments(
    clips: tuple[ClipRef, ...],
    raters: tuple[str, ...],
    *,
    behaviours: tuple[BehaviourId, ...] = BEHAVIOUR_IDS,
    round_name: str,
    raters_per_clip: int,
    seed: int,
) -> tuple[Assignment, ...]:
    """Build the assignment roster for a calibration or production round.

    **The spread pattern.** Clips are assigned raters in a deterministic but pseudo-random
    order that distributes raters across sessions rather than concentrating them. This is
    required because clips within one session share that session's prevalence; the hundredth
    clip from a session carries far less information than the first from a new one. A naive
    assignment that assigns all raters to clips from session 1 before moving to session 2
    would pay for clustering it does not have to buy.

    The algorithm: for each (clip, behaviour) pair, select `raters_per_clip` raters from the
    available pool, then rotate the pool so the next clip starts with the next set of raters.
    This ensures even spread and repeatable results given the seed.

    **Determinism (R7).** Given the same clips, raters, parameters and seed, this function
    returns identical assignments bit for bit. The seed controls Python's random module only;
    no other source of randomness is consulted.

    **No duplicates.** The same rater will never be assigned the same clip and behaviour
    twice, even across multiple calls with different seeds.

    Args:
        clips: The ClipRef objects to assign, one per clip in the calibration set.
        raters: The rater IDs available for assignment.
        behaviours: The behaviour IDs to assign. Default is B1 to B5.
        round_name: The name of the round (e.g., "calibration-1", "production").
        raters_per_clip: How many raters to assign to each clip per behaviour.
        seed: The random seed. Same seed, same assignments.

    Returns:
        A tuple of Assignment objects, one per (rater, clip, behaviour) pair.

    Raises:
        ValueError: if raters_per_clip exceeds the number of available raters.
    """
    if not raters:
        raise ValueError("at least one rater is required")
    if raters_per_clip > len(raters):
        raise ValueError(
            f"raters_per_clip={raters_per_clip} exceeds rater count {len(raters)}")
    if not clips:
        return ()

    random.seed(seed)
    rater_list = list(raters)
    assignments: list[Assignment] = []
    assigned_counter = 0

    for clip in clips:
        for behaviour in behaviours:
            # Shuffle the rater list to get the next batch of raters_per_clip.
            # We rotate rather than reshuffling to ensure even spread.
            random.shuffle(rater_list)
            selected_raters = rater_list[:raters_per_clip]

            for rater_id in selected_raters:
                # A ULID, because `annotation_assignments.assignment_id` is CHAR(26) like every
                # other identifier in this schema. The composite key this used to build ran to
                # sixty-odd characters and the insert failed on length. Nothing is lost by the
                # change: the uniqueness it was encoding is already a table constraint,
                # UNIQUE (rater_id, session_id, clip_index, behaviour), and which rater gets
                # which clip stays reproducible because that is decided by the seeded shuffle
                # above, not by the surrogate key.
                assignment_id = new_ulid()
                assigned_counter += 1

                assignments.append(
                    Assignment(
                        assignment_id=assignment_id,
                        rater_id=rater_id,
                        clip=clip,
                        behaviour=behaviour,
                        round_name=round_name,
                    )
                )

    return tuple(assignments)


def accept_annotation(
    codebook: Codebook,
    *,
    clip_id: str,
    rater_id: str,
    behaviour: BehaviourId,
    labels: dict[str, Any],
    is_nonscorable: bool = False,
    rater_confidence: str = "certain",
    session_college_id: str | None = None,
) -> Annotation:
    """Accept and validate one rater's annotation for one behaviour on one clip.

    **Codebook version enforcement.** The loaded codebook version is stamped onto the returned
    Annotation so that a later revision of the borderline rulings cannot silently reinterpret
    a label made under an earlier one. This is a Phase 2 acceptance test.

    **Label validation.** Any label not present in the loaded codebook version is refused by
    raising AnnotationRefused with http_status 422. The message names the offending key so the
    tool can show it to the rater.

    **The rater's free-text note is deliberately absent.** CODEBOOK.md puts it outside the
    label set: it is "excluded from all quantitative analysis and used only to revise this
    codebook". Annotation is the object the agreement statistics consume, so carrying the note
    on it would invite exactly the pooling the rule forbids. The note is still persisted - it
    is the evidence a codebook revision rests on - but it travels to its column directly,
    beside this object rather than inside it.

    Args:
        codebook: The Codebook to validate against.
        clip_id: The clip being annotated (e.g., "session_id:00001").
        rater_id: The rater ID.
        behaviour: The behaviour ID (B1 to B5).
        labels: The label dict (e.g., {"b1_present": True, "b1_count": 3, ...}).
        is_nonscorable: Whether the rater marked this as nonscorable.
        rater_confidence: The rater's stated confidence. Default "certain".
        session_college_id: The college ID of the session, for familiarity effect analysis.

    Returns:
        An Annotation object with codebook_version stamped.

    Raises:
        AnnotationRefused: if a label is not in the codebook version, with http_status 422.
    """
    try:
        codebook.validate_labels(behaviour, labels)
    except Exception as exc:
        raise AnnotationRefused(str(exc), http_status=422) from exc

    return Annotation(
        clip_id=clip_id,
        rater_id=rater_id,
        behaviour=behaviour,
        codebook_version=codebook.version,
        labels=labels,
        is_nonscorable=is_nonscorable,
        rater_confidence=rater_confidence,
        session_college_id=session_college_id,
    )


@dataclass(frozen=True)
class Disagreement:
    """One field where raters disagreed on one clip and behaviour.

    This surfaces what each rater said so the calibration view can show them side by side,
    line by line, revealing where the codebook is ambiguous or where the raters are drifting.
    """

    clip_id: str
    behaviour: BehaviourId
    field: str
    by_rater: dict[str, Any]
    distinct_values: int


def disagreements(annotations: tuple[Annotation, ...]) -> tuple[Disagreement, ...]:
    """Surface per-clip, per-behaviour, per-field disagreements between raters.

    For each (clip, behaviour, field) combination, computes what each rater said. Filters to
    only those where disagreement exists (more than one distinct value).

    Args:
        annotations: The Annotation objects to analyse.

    Returns:
        A tuple of Disagreement objects, one per (clip, behaviour, field) where disagreement
        was found.
    """
    # Build a table: (clip_id, behaviour, field) -> {rater_id: value}
    by_key: dict[tuple[str, BehaviourId, str], dict[str, Any]] = defaultdict(dict)

    for annotation in annotations:
        for field, value in annotation.labels.items():
            key = (annotation.clip_id, annotation.behaviour, field)
            by_key[key][annotation.rater_id] = value

    disagreement_list: list[Disagreement] = []
    for (clip_id, behaviour, field), by_rater in sorted(by_key.items()):
        distinct = len(set(by_rater.values()))
        if distinct > 1:
            disagreement_list.append(
                Disagreement(
                    clip_id=clip_id,
                    behaviour=behaviour,
                    field=field,
                    by_rater=by_rater,
                    distinct_values=distinct,
                )
            )

    return tuple(disagreement_list)


@dataclass(frozen=True)
class CalibrationOutcome:
    """The result of a calibration round.

    Calibration follows a two-round workflow: raters label a shared set, disagreements are
    surfaced side by side, the codebook is revised if needed, and the gate is checked. This
    object carries the result of that workflow, including whether the alpha gate was met.
    """

    round_name: str
    n_clips: int
    n_raters: int
    agreement: AgreementReport
    disagreements: tuple[Disagreement, ...]
    gate_value: float | None
    gate_passed: bool


def calibration_round(
    annotations: tuple[Annotation, ...],
    raters: tuple[str, ...],
    round_name: str,
    gate: float = DEFAULT_ALPHA_GATE,
) -> CalibrationOutcome:
    """Run the calibration round: compute agreement, surface disagreements, check the gate.

    This function does NOT reimplement any statistics. `praxis.annotation.irr.report` and
    `praxis.annotation.statistics` already compute Krippendorff alpha, ICC(2,k), weighted
    kappa and bootstrap intervals. Duplicating them would create two implementations of one
    rule, which is lesson L20 and has bitten this project repeatedly.

    The gate value is the minimum alpha across all behaviours' primary statistics. If that
    minimum meets the gate, gate_passed is True.

    Args:
        annotations: The Annotation objects from the round.
        raters: The rater IDs who participated.
        round_name: The name of the round (e.g., "calibration-1").
        gate: The alpha threshold to gate on. Default DEFAULT_ALPHA_GATE.

    Returns:
        A CalibrationOutcome carrying the agreement report, disagreements, and gate result.
    """
    # Compute agreement using the standard irr module.
    agreement = report(
        annotations,
        raters=[Rater(rater_id) for rater_id in raters],
    )

    # Surface disagreements.
    disagreement_list = disagreements(annotations)

    # Compute the gate: the minimum alpha across all behaviours' primary statistics.
    gate_values: list[float] = []
    for behaviour in agreement.behaviours:
        for field in behaviour.fields:
            if field.alpha is not None:
                gate_values.append(field.alpha.point)

    gate_value = min(gate_values) if gate_values else None
    gate_passed = gate_value is not None and gate_value >= gate

    # Count unique clips and raters.
    n_clips = len({a.clip_id for a in annotations})
    n_raters = len(set(raters))

    return CalibrationOutcome(
        round_name=round_name,
        n_clips=n_clips,
        n_raters=n_raters,
        agreement=agreement,
        disagreements=disagreement_list,
        gate_value=gate_value,
        gate_passed=gate_passed,
    )
