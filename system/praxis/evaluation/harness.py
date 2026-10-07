# -*- coding: utf-8 -*-
"""Evaluation, with calibration attached to accuracy by the return type.

Rule R4 says calibration is reported wherever accuracy is reported. That is enforced here
structurally rather than by convention: `BehaviourEvaluation` has a required `calibration`
field, `evaluate` is the only way to build one, and there is no accessor that returns accuracy
without it. `tests/test_invariants.py::test_eval_returns_calibration` asserts that this module
mentions ECE at all, and it now does real work rather than passing vacuously.

**Everything is per behaviour and never only in aggregate.** B1 is expected to be the easiest of
the five and B4 the hardest, so a pooled figure would average a strong result with a weak one
and describe neither. A macro figure is offered alongside, never instead.

**Intervals, not point estimates.** Per-class recall carries a bootstrap interval because the
per-behaviour test sets are small and a recall of 0.80 on forty clips is a different claim from
the same number on four hundred.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from praxis.confidence.metrics import (
    DEFAULT_BINS,
    BinningMode,
    CalibrationReport,
    calibration_report,
)


class EvaluationError(RuntimeError):
    """Raised when an evaluation is asked for that the data cannot support."""


def require_band_statistic(subject: str, band: tuple[float, float] | None,
                          statistic: str | None) -> None:
    """A human band must arrive with the name of the coefficient that produced it.

    One implementation, called from every dataclass that carries the pair, because a rule
    written twice is a rule that will be enforced in one place and forgotten in the other (L20).
    """
    if band is not None and not statistic:
        raise ValueError(
            f"{subject} was given a human band {band} with no statement of which coefficient "
            f"produced it. An interval whose statistic is unnamed cannot be read, and the one "
            f"thing a reader would do with it - compare it to accuracy - is the comparison "
            f"naming it exists to prevent (D97).")


def human_comparison(accuracy: float, band: tuple[float, float] | None,
                     statistic: str | None) -> dict[str, object] | None:
    """Accuracy and the human band together, with no claim that they are comparable.

    Returns None when Phase 2 measured no band, which is a real state - the corpus may not have
    been double-coded for every field - and is reported as an abstention rather than as a
    failure (D48).

    Deliberately not a boolean. Accuracy is a raw proportion on [0, 1]; the band is a
    chance-corrected agreement coefficient, and Krippendorff's alpha, quadratic kappa and
    ICC(2,k) can all be negative. `accuracy >= band_low` is arithmetic between incommensurable
    quantities, and both of this project's score types used to expose exactly that as a boolean
    called `within_human_band`. The caller gets both numbers and the statistic's name, and
    states them side by side. D97.
    """
    if band is None:
        return None
    return {
        "accuracy": accuracy,
        "human_band": list(band),
        "statistic": statistic,
        "commensurable": False,
        "note": (f"accuracy is a proportion on [0, 1]; the band is {statistic}, which is "
                 f"chance-corrected and can be negative. Reported together, not compared."),
    }


@dataclass(frozen=True)
class ConfusionStructure:
    """The four cells. Reported rather than summarised, because the errors differ in kind.

    A missed gesture and a hallucinated one are not the same mistake for a reviewer: the first
    leaves them nothing to check, the second gives them something wrong to agree with.
    """

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def n(self) -> int:
        return (self.true_positive + self.false_positive
                + self.true_negative + self.false_negative)

    @property
    def recall_positive(self) -> float | None:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else None

    @property
    def recall_negative(self) -> float | None:
        actual = self.true_negative + self.false_positive
        return self.true_negative / actual if actual else None

    @property
    def precision_positive(self) -> float | None:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else None

    def f1(self, positive: bool = True) -> float | None:
        recall = self.recall_positive if positive else self.recall_negative
        if positive:
            precision = self.precision_positive
        else:
            predicted = self.true_negative + self.false_negative
            precision = self.true_negative / predicted if predicted else None
        if recall is None or precision is None or (recall + precision) == 0:
            return None
        return 2.0 * recall * precision / (recall + precision)


@dataclass(frozen=True)
class RecallInterval:
    """One class's recall with its bootstrap interval."""

    label: str
    point: float | None
    lower: float | None
    upper: float | None
    n: int

    def __str__(self) -> str:
        if self.point is None:
            return f"{self.label}: undefined (n={self.n})"
        return f"{self.label}: {self.point:.3f} [{self.lower:.3f}, {self.upper:.3f}] n={self.n}"


@dataclass(frozen=True)
class BehaviourEvaluation:
    """One behaviour's result. Accuracy cannot leave this module without calibration.

    `human_band` is the agreement interval Phase 2 measured for this behaviour. It is optional
    because the corpus may not have been double-coded for every behaviour, but where it exists
    the model figure is stated against it rather than against an absolute target: D6 withdrew
    the 85 per cent accuracy goal precisely because a number the candidate does not control
    converts a research objective into a failure condition.

    **The two numbers are reported side by side and no verdict is derived from them.** The band
    is a chance-corrected agreement coefficient - Krippendorff's alpha, quadratic kappa or
    ICC(2,k), per `annotation.irr` in the configuration - and every one of those can be
    negative. Accuracy is a raw proportion on [0, 1]. `accuracy >= band_low` is arithmetic
    between incommensurable quantities, and this class used to expose exactly that as a boolean
    called `within_human_band`. `human_band_statistic` is required alongside the pair so a
    reader is never shown an interval without being told what produced it. D97.
    """

    behaviour: str
    n: int
    accuracy: float
    macro_f1: float
    confusion: ConfusionStructure
    recalls: tuple[RecallInterval, ...]
    calibration: CalibrationReport            # REQUIRED. R4 is enforced by this line
    human_band: tuple[float, float] | None = None
    human_band_statistic: str | None = None

    def __post_init__(self) -> None:
        require_band_statistic(self.behaviour, self.human_band, self.human_band_statistic)

    @property
    def ece(self) -> float:
        return self.calibration.ece

    @property
    def against_humans(self) -> dict[str, object] | None:
        """Both numbers and the statistic that produced the band. No verdict. D97."""
        return human_comparison(self.accuracy, self.human_band, self.human_band_statistic)

    def summary(self) -> str:
        return (f"{self.behaviour}: acc={self.accuracy:.3f}  macroF1={self.macro_f1:.3f}  "
                f"ECE={self.calibration.ece:.4f}  n={self.n}")


@dataclass(frozen=True)
class EvaluationResult:
    """Every behaviour, with a macro figure alongside and never instead."""

    behaviours: tuple[BehaviourEvaluation, ...]
    label: str = ""

    def behaviour(self, name: str) -> BehaviourEvaluation:
        for entry in self.behaviours:
            if entry.behaviour == name:
                return entry
        raise KeyError(name)

    @property
    def macro_accuracy(self) -> float:
        return float(np.mean([b.accuracy for b in self.behaviours]))

    @property
    def macro_ece(self) -> float:
        """Mean ECE across behaviours. Always available wherever macro_accuracy is. R4."""
        return float(np.mean([b.ece for b in self.behaviours]))

    def as_table(self) -> list[dict[str, object]]:
        """One row per behaviour. The human band appears as its endpoints and the statistic
        that produced them, and there is no column holding a verdict on the two (D97)."""
        return [{"behaviour": b.behaviour, "n": b.n,
                 "accuracy": round(b.accuracy, 4), "macro_f1": round(b.macro_f1, 4),
                 "ece": round(b.calibration.ece, 4), "mce": round(b.calibration.mce, 4),
                 "brier": round(b.calibration.brier, 4), "nll": round(b.calibration.nll, 4),
                 "human_band_low": None if b.human_band is None else round(b.human_band[0], 4),
                 "human_band_high": None if b.human_band is None else round(b.human_band[1], 4),
                 "human_band_statistic": b.human_band_statistic}
                for b in self.behaviours]


def confusion_structure(predicted: np.ndarray, truth: np.ndarray) -> ConfusionStructure:
    return ConfusionStructure(
        true_positive=int(np.sum((predicted == 1) & (truth == 1))),
        false_positive=int(np.sum((predicted == 1) & (truth == 0))),
        true_negative=int(np.sum((predicted == 0) & (truth == 0))),
        false_negative=int(np.sum((predicted == 0) & (truth == 1))))


def _recall_interval(probabilities: np.ndarray, truth: np.ndarray, positive: bool,
                     draws: int, seed: int, level: float) -> RecallInterval:
    """Bootstrap over clips, which are the sampling unit."""
    target = 1.0 if positive else 0.0
    label = "present" if positive else "absent"
    selected = truth == target
    n = int(selected.sum())
    if n == 0:
        return RecallInterval(label, None, None, None, 0)

    predicted = (probabilities >= 0.5).astype(float)[selected]
    point = float((predicted == target).mean())
    if draws < 1 or n < 2:
        return RecallInterval(label, point, None, None, n)

    generator = np.random.default_rng(seed)
    resampled = [float((predicted[generator.integers(0, n, n)] == target).mean())
                 for _ in range(draws)]
    tail = (1.0 - level) / 2.0 * 100.0
    return RecallInterval(label, point, float(np.percentile(resampled, tail)),
                          float(np.percentile(resampled, 100.0 - tail)), n)


def evaluate_behaviour(behaviour: str, probabilities: Sequence[float],
                       labels: Sequence[int | bool],
                       human_band: tuple[float, float] | None = None,
                       human_band_statistic: str | None = None,
                       n_bins: int = DEFAULT_BINS,
                       mode: BinningMode = "positive_class",
                       bootstrap: int = 1000, ci_level: float = 0.95,
                       seed: int = 20260911) -> BehaviourEvaluation:
    """Accuracy, macro-F1, per-class recall with intervals, confusion, and ECE, together.

    `human_band_statistic` names the coefficient the band came from and is required whenever a
    band is given. It is one argument rather than one per behaviour because Phase 2 computes the
    whole set with a single coefficient, chosen in `annotation.irr`.
    """
    probs = np.asarray(probabilities, dtype=float).ravel()
    truth = np.asarray(labels, dtype=float).ravel()
    if probs.size != truth.size:
        raise EvaluationError(f"{probs.size} probabilities against {truth.size} labels")
    if probs.size == 0:
        raise EvaluationError(f"{behaviour}: nothing to evaluate")

    predicted = (probs >= 0.5).astype(float)
    confusion = confusion_structure(predicted, truth)
    f1_positive, f1_negative = confusion.f1(True), confusion.f1(False)
    defined = [f for f in (f1_positive, f1_negative) if f is not None]

    return BehaviourEvaluation(
        behaviour=behaviour,
        n=int(probs.size),
        accuracy=float((predicted == truth).mean()),
        macro_f1=float(np.mean(defined)) if defined else 0.0,
        confusion=confusion,
        recalls=(_recall_interval(probs, truth, True, bootstrap, seed, ci_level),
                 _recall_interval(probs, truth, False, bootstrap, seed + 1, ci_level)),
        calibration=calibration_report(probs, truth, n_bins, mode),
        human_band=human_band,
        human_band_statistic=human_band_statistic)


def evaluate(predictions: dict[str, tuple[Sequence[float], Sequence[int | bool]]],
             human_bands: dict[str, tuple[float, float]] | None = None,
             human_band_statistic: str | None = None,
             label: str = "", **kwargs) -> EvaluationResult:
    """Evaluate every behaviour. There is no path through here that omits calibration.

    Supplying `human_bands` without `human_band_statistic` raises, from `BehaviourEvaluation`.
    """
    human_bands = human_bands or {}
    if not predictions:
        raise EvaluationError("no behaviours supplied")
    return EvaluationResult(
        behaviours=tuple(
            evaluate_behaviour(behaviour, probs, labels,
                               human_band=human_bands.get(behaviour),
                               human_band_statistic=human_band_statistic, **kwargs)
            for behaviour, (probs, labels) in sorted(predictions.items())),
        label=label)
