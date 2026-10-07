# -*- coding: utf-8 -*-
"""The domain-shift experiment. This phase produces the thesis's primary finding.

Three levels, each adding exactly one source of shift to the one above it:

| level | held out | what it isolates |
|---|---|---|
| S0 | teachers only; their sites appear in training | baseline: unseen person, seen site |
| S1 | a whole basic school, its college in training | site shift |
| S2 | a whole college, every school under it | population and site shift |

**The shift is the site, not the domain.** The earlier ladder put S2 in a different domain -
microteaching trained, classroom practicum held out - and rested on the rule that classroom
footage is never trained on. The revised scope makes the corpus practicum throughout, so
classroom footage is all of the data; that rule would refuse every manifest, and the thing
nobody should do is relax it. A school and a college are **crossed, not nested**: one basic
school hosts students from several Colleges of Education, so holding out a college does not hold
out a school, and S2 does not collapse into S1. D98.

**H4 predicts that calibration degrades faster than accuracy from S0 to S2.** Both are therefore
reported at every level and neither is reported without the other, which is R4 enforced by the
shape of the table rather than by care.

**The leakage check runs before anything is computed.** A contaminated S2 set would produce a
smaller degradation, which is the direction that flatters the system, and nothing in any metric
would reveal it.

**Observing the drop is not the finding; attributing it is.** That is
`evaluation/attribution.py`, and it is not optional reporting. `evaluation/abstention.py` then
turns the OOD scores into the one threshold the Phase 8 routing gate needs. Both depend on this
module and neither is imported back into it, so the gate cannot be bypassed by importing a
downstream piece directly.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from praxis.contracts.manifest import SplitManifest
from praxis.evaluation.harness import EvaluationResult


class ShiftError(RuntimeError):
    """Raised when the shift experiment cannot be run honestly."""


class ShiftLevel(str, Enum):
    """The three levels, in the order degradation is expected to grow."""

    S0 = "S0"
    S1 = "S1"
    S2 = "S2"

    @property
    def description(self) -> str:
        return {
            ShiftLevel.S0: "i.i.d. practicum test, teachers held out, their sites seen",
            ShiftLevel.S1: "held-out basic school, its college seen in training",
            ShiftLevel.S2: "held-out college, every school and teacher under it",
        }[self]

    @property
    def isolates(self) -> str:
        return {ShiftLevel.S0: "baseline: unseen teacher, seen site",
                ShiftLevel.S1: "site shift: room, camera placement, pupil cohort, acoustics",
                ShiftLevel.S2: "population and site shift together"}[self]

    def __str__(self) -> str:
        # without this a str-mixin enum formats as "ShiftLevel.S2", which lands in error
        # messages and in anything that keys a dict by the printed level
        return self.value


def _level(value: ShiftLevel | str) -> ShiftLevel:
    """Accept either the enum or the bare string, which is what a config or a CSV carries."""
    if isinstance(value, ShiftLevel):
        return value
    try:
        return ShiftLevel(str(value))
    except ValueError as exc:
        raise ShiftError(
            f"{value!r} is not a shift level; expected one of "
            f"{[level.value for level in ShiftLevel]}") from exc


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

# S1 and S2 hold out a site the manifest names, so the manifest is also where the held-out
# sessions are declared. S0 is defined by `test_teachers` instead and carries no session list,
# which is why the declaration check below applies to these two and not to it.
SITE_LEVELS = (ShiftLevel.S1, ShiftLevel.S2)


def _stand_down_available(manifest: SplitManifest) -> bool:
    """Whether D83's relaxation of the teacher rule applies to this manifest.

    It applies only to the design it was written for: one where the shift is the *domain* and
    the corpus may leave no microteaching-only teachers to train on. A manifest that names a
    held-out school or college has made the site the shift, and there a teacher of a held-out
    session appearing in training is not a weaker rule - it is the leak itself. So the
    relaxation is read off the manifest together with the two site fields rather than from the
    flag alone.

    Stated as one function because the alternative is the same condition written at each check,
    and a rule written more than once gets relaxed in one place and not the others (L20). D98.
    """
    return (manifest.classroom_teachers_trained
            and manifest.heldout_school is None
            and manifest.heldout_college is None)


def assert_no_leakage(manifest: SplitManifest,
                      sessions_by_level: Mapping[ShiftLevel | str, Sequence[str]],
                      session_teacher: Mapping[str, str],
                      training_sessions: Sequence[str] = ()) -> None:
    """Refuse to run when evaluation data touched training. Raise, never warn.

    Four checks, each catching a different way the same contamination arrives, and **every one
    of them runs at every level supplied**:

    1. a held-out session appearing directly in a training partition, which BUILD-SPEC names;
    2. a held-out session with no teacher recorded, so disjointness cannot be checked at all;
    3. a held-out session whose *teacher* trained, which a session-level check alone would miss;
    4. an S1 or S2 session absent from `shift_eval_sessions`, meaning the manifest never
       declared it held out and nothing was protecting it.

    **Until D98 the first, second and fourth checks looked only at S2.** An S1 session could sit
    in a training partition, arrive with no teacher, or go undeclared, and the harness would
    run. Under the old ladder S1 was microteaching like the training set and the hole was
    survivable. Under this one S1 is a held-out school, so the hole was the leak: a site trained
    on cannot measure site shift, and the S0-to-S1 drop would simply be small.

    **Check 3 is D73, not R2, and the manifest decides whether it applies.** R2 forbids a
    teacher appearing in more than one *partition*, and `SplitManifest.teachers_are_disjoint`
    enforces that unconditionally, here and everywhere else. What check 3 adds is stronger: that
    nobody who contributes a held-out session trained at all. A corpus with no
    microteaching-only teachers cannot satisfy that and still have anything to train on, so a
    manifest from the microteaching design may declare `classroom_teachers_trained` and the
    check stands down for it - see `_stand_down_available`, which is also where the limit of
    that relaxation lives. It reads the declaration off the manifest rather than taking a
    parameter, because a flag passed in separately could disagree with the artefact that gets
    published.

    Check 1 never stands down, at any level. Under the weakened rule a teacher's other session
    may train while this one is held out; the held-out session itself being trained on is the
    leak the whole harness exists to refuse. D83, D98.
    """
    problems: list[str] = []
    levels = {_level(level): list(ids) for level, ids in sessions_by_level.items()}
    trained = set(training_sessions)
    trained_teachers = set(manifest.train_teachers) | set(manifest.val_teachers)
    declared = set(manifest.shift_eval_sessions)
    relaxed = _stand_down_available(manifest)

    for level in sorted(levels, key=lambda each: each.value):
        ids = levels[level]

        overlapping = sorted(set(ids) & trained)
        if overlapping:
            problems.append(
                f"{level} sessions appear in a training partition: {overlapping}. A session "
                f"held "
                f"out to measure {level.isolates} is never trained on.")

        unknown = sorted(s for s in ids if s not in session_teacher)
        if unknown:
            problems.append(
                f"{level} sessions have no teacher recorded: {unknown}. teacher_id is the "
                f"split "
                f"key and may never be null, so disjointness cannot be checked for these.")

        leaked = sorted({session_teacher[s] for s in ids
                         if s in session_teacher and session_teacher[s] in trained_teachers})
        if leaked and not relaxed:
            problems.append(
                f"teachers {leaked} contribute {level} sessions and also appear in a training "
                f"or validation partition. A model that has seen this candidate teaching will "
                f"shift "
                f"less than one that has not, so the drop would understate itself. This is D73 "
                f"rather than R2: if the corpus leaves no microteaching-only teachers to train "
                f"on, set splits.classroom_teachers_may_train, which records the weaker rule "
                f"on the manifest instead of hiding it here - and note that it does not apply "
                f"to a "
                f"manifest naming a held-out school or college, where the site is the shift.")

        if level in SITE_LEVELS:
            undeclared = sorted(set(ids) - declared)
            if undeclared:
                problems.append(
                    f"{level} sessions {undeclared} are not listed in the manifest's "
                    f"shift_eval_sessions, so nothing recorded that they were held out.")

    if problems:
        raise ShiftError("the shift harness refuses to run:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# The headline table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShiftRow:
    """One level, for one calibration method."""

    level: ShiftLevel
    method: str
    n: int
    accuracy: float
    ece: float
    accuracy_drop: float | None
    ece_rise: float | None

    @property
    def relative_accuracy_drop(self) -> float | None:
        return None if self.accuracy_drop is None else self.accuracy_drop / max(
            self.accuracy + self.accuracy_drop, 1e-12)

    @property
    def relative_ece_rise(self) -> float | None:
        baseline = self.ece - (self.ece_rise or 0.0)
        return None if self.ece_rise is None else self.ece_rise / max(baseline, 1e-12)


# below this baseline ECE the ratio is dominated by its denominator rather than by the shift,
# so it is reported as unstable. 0.01 is the same floor D22 uses for "already calibrated".
STABLE_BASELINE_ECE = 0.01


@dataclass(frozen=True)
class Degradation:
    """H4 stated with its parts, so the ratio cannot be quoted without them.

    The ratio is the headline, and on its own it is misleading. Its denominator is the
    *relative* accuracy fall, which is small whenever accuracy holds up, and its numerator is
    the relative ECE rise, whose baseline is small whenever the method calibrates well
    in-distribution. A method that is good at both therefore produces the largest ratio, which
    is the right direction but the wrong magnitude to quote. The absolute rise is carried
    alongside for that reason, and `stable` says whether the baseline could support a ratio.
    """

    method: str
    accuracy_fall: float
    relative_accuracy_fall: float
    ece_rise: float
    relative_ece_rise: float
    ratio: float | None
    stable: bool

    def __str__(self) -> str:
        if self.ratio is None:
            return (
                f"{self.method}: ECE rose {self.ece_rise:+.4f} while accuracy did not fall; "
                f"no ratio is defined, which is the stronger statement")
        caveat = "" if self.stable else "  (unstable: baseline ECE below the floor)"
        return (f"{self.method}: accuracy -{self.relative_accuracy_fall:.1%}, "
                f"ECE +{self.relative_ece_rise:.1%} ({self.ece_rise:+.4f} absolute), "
                f"ratio {self.ratio:.1f}x{caveat}")


@dataclass(frozen=True)
class ShiftTable:
    """Accuracy and ECE at every level, for every method. The thesis's headline figure."""

    rows: tuple[ShiftRow, ...]

    def for_method(self, method: str) -> list[ShiftRow]:
        return [r for r in self.rows if r.method == method]

    def degradation(self, method: str) -> Degradation | None:
        """The S0 to S2 change for one method, in both absolute and relative terms.

        Returned as None only when the method lacks either end of the comparison.
        """
        rows = {r.level: r for r in self.for_method(method)}
        if ShiftLevel.S0 not in rows or ShiftLevel.S2 not in rows:
            return None

        s0, s2 = rows[ShiftLevel.S0], rows[ShiftLevel.S2]
        accuracy_fall = s0.accuracy - s2.accuracy
        ece_rise = s2.ece - s0.ece
        relative_accuracy_fall = accuracy_fall / max(s0.accuracy, 1e-12)
        relative_ece_rise = ece_rise / max(s0.ece, 1e-12)
        # None rather than infinity where accuracy did not fall: "calibration degraded and
        # accuracy did not" is a stronger statement than any ratio, and a divide-by-zero
        # would bury it.
        ratio = (None if relative_accuracy_fall <= 0
                 else float(relative_ece_rise / relative_accuracy_fall))
        return Degradation(method=method, accuracy_fall=float(accuracy_fall),
                           relative_accuracy_fall=float(relative_accuracy_fall),
                           ece_rise=float(ece_rise),
                           relative_ece_rise=float(relative_ece_rise), ratio=ratio,
                           stable=s0.ece >= STABLE_BASELINE_ECE)

    def degradation_ratio(self, method: str) -> float | None:
        """How many times faster calibration degrades than accuracy, S0 to S2. H4 in one number.

        Greater than one means ECE rose proportionally more than accuracy fell, which is the
        prediction. Read `degradation` rather than this wherever the number is being reported:
        the ratio alone overstates how large the effect is.
        """
        result = self.degradation(method)
        return None if result is None else result.ratio

    def as_table(self) -> list[dict[str, object]]:
        return [
            {
                "level": r.level.value,
                "isolates": r.level.isolates,
                "method": r.method,
                "n": r.n,
                "accuracy": round(r.accuracy, 4),
                "ece": round(r.ece, 4),
                "accuracy_drop": (
                    None if r.accuracy_drop is None else round(r.accuracy_drop, 4)
                ),
                "ece_rise": None if r.ece_rise is None else round(r.ece_rise, 4),
            }
            for r in self.rows
        ]


def build_shift_table(
        results: Mapping[str, Mapping[ShiftLevel | str, EvaluationResult]]) -> ShiftTable:
    """Assemble the table from per-method, per-level evaluations.

    Drops are measured against that method's own S0, so each method is compared with itself.
    Comparing every method against a single shared baseline would confound how well a method
    calibrates in-distribution with how well it survives leaving it, and those are different
    claims: temperature scaling is expected to do the first well and the second badly.
    """
    rows: list[ShiftRow] = []
    for method in sorted(results):
        by_level = {_level(level): result for level, result in results[method].items()}
        if ShiftLevel.S0 not in by_level:
            raise ShiftError(f"method {method!r} has no S0 baseline to measure drops against")

        baseline = by_level[ShiftLevel.S0]
        for level in (ShiftLevel.S0, ShiftLevel.S1, ShiftLevel.S2):
            result = by_level.get(level)
            if result is None:
                continue
            is_baseline = level is ShiftLevel.S0
            rows.append(ShiftRow(
                level=level, method=method,
                n=sum(b.n for b in result.behaviours),
                accuracy=result.macro_accuracy, ece=result.macro_ece,
                accuracy_drop=None if is_baseline
                else baseline.macro_accuracy - result.macro_accuracy,
                ece_rise=None if is_baseline else result.macro_ece - baseline.macro_ece))
    return ShiftTable(tuple(rows))


