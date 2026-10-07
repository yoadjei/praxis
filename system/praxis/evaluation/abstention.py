# -*- coding: utf-8 -*-
"""Choosing where to abstain, by a stated rule rather than by eye.

The routing gate in Phase 8 needs one number: the OOD score above which a detection is
suppressed rather than shown. This module produces the table that number is read from, and
applies the criterion the config names to read it.

Two things are reported for every candidate threshold, never one. A threshold that suppresses
nothing and a threshold that suppresses everything both look excellent on one of the two
numbers alone, and the trade between them is the whole decision. D30 records why the stated
criterion is applied unchanged even where it chooses badly, and what is reported beside it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from praxis.confidence.metrics import expected_calibration_error
from praxis.evaluation.harness import confusion_structure
from praxis.evaluation.shift import ShiftError

# A gap in suppression rate wide enough that spending it on a negligible F1 gain is worth
# saying out loud. Five per cent of a practicum corpus is roughly one session.
FLAT_CURVE_GAP = 0.05


@dataclass(frozen=True)
class SweepRow:
    """One candidate threshold, and what it costs."""

    threshold: float
    suppression_rate: float
    n_presented: int
    accuracy_on_presented: float | None
    ece_on_presented: float | None
    f1_on_presented: float | None

    def as_dict(self) -> dict[str, object]:
        return {"threshold": round(self.threshold, 4),
                "suppression_rate": round(self.suppression_rate, 4),
                "n_presented": self.n_presented,
                "accuracy": None if self.accuracy_on_presented is None
                else round(self.accuracy_on_presented, 4),
                "ece": None if self.ece_on_presented is None
                else round(self.ece_on_presented, 4),
                "f1": None if self.f1_on_presented is None
                else round(self.f1_on_presented, 4)}


@dataclass(frozen=True)
class ThresholdChoice:
    """The chosen operating point and the stated reason for it."""

    threshold: float | None
    criterion: str
    row: SweepRow | None
    justification: str
    sweep: tuple[SweepRow, ...]
    frugal: SweepRow | None = None

    @property
    def suppression_bought_little(self) -> bool:
        """True when the criterion paid a lot of suppression for almost no quality.

        Not an error. It is the signal that the ceiling, not the data, is choosing the
        threshold, and someone should look at the table rather than at the chosen number.
        """
        if self.row is None or self.frugal is None:
            return False
        return self.frugal.suppression_rate < self.row.suppression_rate - FLAT_CURVE_GAP

    def as_table(self) -> list[dict[str, object]]:
        return [row.as_dict() for row in self.sweep]


def threshold_sweep(ood_scores: Sequence[float], probabilities: Sequence[float],
                    labels: Sequence[int | bool],
                    sweep_from: float = 0.0, sweep_to: float = 1.0,
                    steps: int = 101) -> list[SweepRow]:
    """What each threshold costs: how much is suppressed, and how good what remains is.

    A detection is suppressed when its OOD score reaches the threshold, so a low threshold
    suppresses more. Both the suppression rate and the quality of what survives are reported,
    because a threshold that suppresses nothing and one that suppresses everything both look
    excellent on one of the two numbers alone.
    """
    scores = np.asarray(ood_scores, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=float)
    if not scores.size == probs.size == truth.size:
        raise ShiftError(
            f"{scores.size} scores, {probs.size} probabilities and {truth.size} labels")

    rows: list[SweepRow] = []
    for threshold in np.linspace(sweep_from, sweep_to, steps):
        presented = scores < threshold
        n_presented = int(presented.sum())
        suppression = 1.0 - n_presented / scores.size

        if n_presented == 0 or len(np.unique(truth[presented])) < 1:
            rows.append(SweepRow(float(threshold), suppression, 0, None, None, None))
            continue

        kept_probs, kept_truth = probs[presented], truth[presented]
        predicted = (kept_probs >= 0.5).astype(float)
        confusion = confusion_structure(predicted, kept_truth)
        f1_scores = [f for f in (confusion.f1(True), confusion.f1(False)) if f is not None]

        ece = (expected_calibration_error(kept_probs, kept_truth)
               if len(np.unique(kept_truth)) >= 1 else None)
        rows.append(SweepRow(
            threshold=float(threshold), suppression_rate=float(suppression),
            n_presented=n_presented,
            accuracy_on_presented=float((predicted == kept_truth).mean()),
            ece_on_presented=ece,
            f1_on_presented=float(np.mean(f1_scores)) if f1_scores else None))
    return rows


def choose_threshold(sweep: Sequence[SweepRow],
                     criterion: str = "max_f1_at_suppression_rate_below_0.30",
                     max_suppression: float = 0.30,
                     frugality_tolerance: float = 0.01) -> ThresholdChoice:
    """Pick the operating point by a stated rule, never by eye.

    Parasuraman and Riley: neglect of automation is driven by false alarms when base rates are
    ignored. A threshold chosen by looking at a curve is a threshold chosen by whichever
    picture was drawn, and it cannot be defended in a viva or reproduced by a reader. The rule
    is named in the config, applied here, and the chosen value recorded in
    `docs/DECISIONS.md`.

    **The criterion has a known weakness and the result names it rather than hiding it.** Where
    F1 is flat across thresholds — which the first synthetic run showed it can be — "maximise
    F1 subject to a suppression ceiling" will happily spend twenty per cent of the corpus to
    buy half an F1 point, because nothing in the rule says suppression is a cost. The chosen
    threshold is still the one the stated criterion gives, since changing the rule after seeing
    the curve is the practice the rule exists to prevent. Alongside it, `frugal` reports the
    least-suppressing threshold within `frugality_tolerance` F1 of the best, so the trade is
    visible to whoever has to defend the number.
    """
    eligible = [row for row in sweep
                if row.suppression_rate <= max_suppression and row.f1_on_presented is not None]
    if not eligible:
        return ThresholdChoice(
            None, criterion, None,
            f"no threshold keeps the suppression rate at or below {max_suppression:.0%} while "
            f"leaving anything to score. Either the detector flags almost everything, which "
            f"is a finding about the detector, or the ceiling needs revising with a reason.",
            tuple(sweep))

    best = max(eligible, key=lambda row: (row.f1_on_presented, -row.suppression_rate))
    frugal = min((row for row in eligible
                  if row.f1_on_presented >= best.f1_on_presented - frugality_tolerance),
                 key=lambda row: (row.suppression_rate, -row.f1_on_presented))

    justification = (
        f"criterion {criterion!r}: of the {len(eligible)} thresholds holding suppression "
        f"at or below {max_suppression:.0%}, {best.threshold:.4f} gives the highest "
        f"macro-F1 on what is presented ({best.f1_on_presented:.4f}) while suppressing "
        f"{best.suppression_rate:.1%} and leaving {best.n_presented} detections.")
    if frugal.suppression_rate < best.suppression_rate - FLAT_CURVE_GAP:
        justification += (
            f" F1 is nearly flat here: {frugal.threshold:.4f} suppresses only "
            f"{frugal.suppression_rate:.1%} for {frugal.f1_on_presented:.4f}, within "
            f"{frugality_tolerance:.2f} of the best. The ceiling, not the data, is doing the "
            f"choosing, and the table should be read before this number is quoted.")

    return ThresholdChoice(threshold=best.threshold, criterion=criterion, row=best,
                           justification=justification, sweep=tuple(sweep), frugal=frugal)
