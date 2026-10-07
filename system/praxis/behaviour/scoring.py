# -*- coding: utf-8 -*-
"""Macro-F1 for a codebook field, in one place.

Both the training monitor and the frozen comparison report need it, and a statistic with two
implementations is lesson L20: the one that drifts is always the one used for early stopping,
because nothing reports it beside the other.

Neither function defines F1. `ConfusionStructure.f1` in `evaluation/harness.py` does, and these
build the counts and call it.
"""
from __future__ import annotations

import numpy as np

from praxis.evaluation.harness import ConfusionStructure


def confusion_for(predicted: np.ndarray, truth: np.ndarray,
                  positive_level: int = 1) -> ConfusionStructure:
    """One-vs-rest counts for a single level."""
    predicted_positive = predicted == positive_level
    actually_positive = truth == positive_level
    return ConfusionStructure(
        true_positive=int((predicted_positive & actually_positive).sum()),
        false_positive=int((predicted_positive & ~actually_positive).sum()),
        true_negative=int((~predicted_positive & ~actually_positive).sum()),
        false_negative=int((~predicted_positive & actually_positive).sum()))


def macro_f1_binary(predicted: np.ndarray, truth: np.ndarray) -> float | None:
    """The mean of positive-class and negative-class F1.

    The same definition `evaluate_behaviour` reports, so a field's macro-F1 in the frozen
    report and the same field's contribution to the early-stopping monitor are the one number.
    """
    if truth.size == 0:
        return None
    counts = confusion_for(predicted, truth, positive_level=1)
    defined = [f for f in (counts.f1(True), counts.f1(False)) if f is not None]
    return float(np.mean(defined)) if defined else None


def macro_f1_multiclass(predicted: np.ndarray, truth: np.ndarray,
                        n_classes: int) -> float | None:
    """One-vs-rest F1 averaged over the levels that actually occur.

    A level nobody labelled and nobody predicted is undefined, not zero. Scoring it zero would
    punish the model for a level the evaluation split happens not to contain, and at S1 and S2 -
    where a held-out college or a classroom may genuinely never show a posture - that would read
    as degradation rather than as absence.
    """
    if truth.size == 0:
        return None
    per_level = []
    for level in range(n_classes):
        counts = confusion_for(predicted, truth, positive_level=level)
        if counts.true_positive + counts.false_negative == 0:
            continue
        value = counts.f1(True)
        if value is not None:
            per_level.append(value)
    return float(np.mean(per_level)) if per_level else None
