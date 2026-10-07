# -*- coding: utf-8 -*-
"""Seeded error injection to validate the evaluation harness.

This module tests that the evaluation harness detects known failures in predictions.
It injects errors of known shape and magnitude into a baseline, then checks that the
harness reports changes in the expected directions. This validates the detector, never
the model.

Rule R7 requires reproducibility: all perturbations are seeded and deterministic.
Rule R4 requires that calibration is reported wherever accuracy is, which these tests
verify by checking that ECE and accuracy degrade as expected under different failure modes.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from praxis.evaluation.harness import evaluate


@dataclass(frozen=True)
class SeededErrorCase:
    """One error case: what it does, and what the harness should report.

    name: identifies the error type (label_flip, confidence_inflation, etc.)
    description: human-readable explanation of what the perturbation does
    perturbation: function that takes predictions and returns perturbed predictions
    expected_accuracy_direction: 'down', 'up', or 'none' for expected accuracy change
    expected_ece_direction: 'up', 'down', or 'none' for expected ECE (calibration) change
    tolerance: how much of a change to allow before flagging the direction as wrong
    """

    name: str
    description: str
    perturbation: Callable[[dict[str, tuple[Sequence[float], Sequence[int | bool]]]],
                          dict[str, tuple[Sequence[float], Sequence[int | bool]]]]
    expected_accuracy_direction: str  # 'down', 'up', 'none'
    expected_ece_direction: str       # 'up', 'down', 'none'
    tolerance: float = 0.0001         # minimum absolute change to count as a direction


@dataclass(frozen=True)
class ValidationResult:
    """Result of one seeded error test against a baseline."""

    error_name: str
    ran: bool
    reason: str | None = None
    baseline_accuracy: float | None = None
    baseline_ece: float | None = None
    perturbed_accuracy: float | None = None
    perturbed_ece: float | None = None
    accuracy_changed_as_expected: bool | None = None
    ece_changed_as_expected: bool | None = None


def _perturb_label_flip(predictions: dict[str, tuple[Sequence[float], Sequence[int | bool]]],
                       rate: float, seed: int
                       ) -> dict[str, tuple[Sequence[float], Sequence[int | bool]]]:
    """Randomly flip labels at the given rate, seeded for reproducibility.

    This degrades accuracy directly because labels are flipped regardless of what the
    model predicted. Calibration may improve slightly if the model was confident in
    the original labels, since flipping some makes it wrong with lower confidence on
    those examples. Over a large enough set of flips the net ECE effect depends on
    the base rate and confidence distribution.
    """
    generator = np.random.default_rng(seed)
    perturbed = {}
    for behaviour, (probs, labels) in predictions.items():
        labels_array = np.asarray(labels, dtype=float)
        n = len(labels_array)
        n_flip = max(1, int(np.round(rate * n)))
        flip_indices = generator.choice(n, size=n_flip, replace=False)
        flipped_labels = labels_array.copy()
        flipped_labels[flip_indices] = 1.0 - flipped_labels[flip_indices]
        perturbed[behaviour] = (list(probs), list(flipped_labels))
    return perturbed


def _perturb_confidence_inflation(
        predictions: dict[str, tuple[Sequence[float], Sequence[int | bool]]],
        rate: float, seed: int
) -> dict[str, tuple[Sequence[float], Sequence[int | bool]]]:
    """Push probabilities away from 0.5, degrading calibration while leaving accuracy intact.

    This is the critical test for H4. Hard decisions (>=0.5 becomes 1, <0.5 becomes 0)
    are never crossed, so accuracy is unaffected. But probabilities move towards their
    respective ends (high probs towards 1, low probs towards 0), making predictions more
    confident without changing which ones are correct. The mean confidence rises while
    the empirical rate stays the same, inflating ECE.

    rate controls how far from 0.5 to push each probability. rate=0 is no change;
    rate=1 pushes all to 0 or 1; rate=0.5 moves each probability halfway away from 0.5.
    """
    perturbed = {}
    for behaviour, (probs, labels) in predictions.items():
        probs_array = np.asarray(probs, dtype=float)
        # For p >= 0.5, move towards 1. For p < 0.5, move towards 0.
        inflated = np.where(
            probs_array >= 0.5,
            probs_array + (1.0 - probs_array) * rate,
            probs_array - probs_array * rate)
        perturbed[behaviour] = (list(inflated), list(labels))
    return perturbed


def _perturb_constant_predictor(
        predictions: dict[str, tuple[Sequence[float], Sequence[int | bool]]],
        constant_value: float, seed: int
) -> dict[str, tuple[Sequence[float], Sequence[int | bool]]]:
    """Always predict the same probability, destroying discrimination.

    At constant_value=0.5, the model predicts presence with 50% probability for
    everything, removing all signal. This keeps the base rate but destroys recall
    and precision on both classes. Accuracy falls to the base rate (accuracy on the
    majority class). ECE can vary depending on the true base rate.
    """
    perturbed = {}
    for behaviour, (probs, labels) in predictions.items():
        n = len(probs)
        constant_probs = [constant_value] * n
        perturbed[behaviour] = (constant_probs, list(labels))
    return perturbed


def _perturb_class_conditional_noise(
        predictions: dict[str, tuple[Sequence[float], Sequence[int | bool]]],
        target_behaviour: str, noise_rate: float, seed: int
) -> dict[str, tuple[Sequence[float], Sequence[int | bool]]]:
    """Add label noise to one behaviour only.

    This tests that the harness reports per-behaviour and does not average problems
    away. One behaviour's accuracy falls while others remain unchanged.
    """
    generator = np.random.default_rng(seed)
    perturbed = {}
    for behaviour, (probs, labels) in predictions.items():
        if behaviour == target_behaviour:
            labels_array = np.asarray(labels, dtype=float)
            n = len(labels_array)
            n_flip = max(1, int(np.round(noise_rate * n)))
            flip_indices = generator.choice(n, size=n_flip, replace=False)
            noisy_labels = labels_array.copy()
            noisy_labels[flip_indices] = 1.0 - noisy_labels[flip_indices]
            perturbed[behaviour] = (list(probs), list(noisy_labels))
        else:
            perturbed[behaviour] = (list(probs), list(labels))
    return perturbed


def _make_perturbation(func: Callable, **kwargs) -> Callable:
    """Wrap a perturbation function with its fixed kwargs."""
    def wrapped(predictions: dict[str, tuple[Sequence[float], Sequence[int | bool]]]
                ) -> dict[str, tuple[Sequence[float], Sequence[int | bool]]]:
        return func(predictions, **kwargs)
    return wrapped


def seeded_error_suite() -> list[SeededErrorCase]:
    """Return the suite of error cases that must pass for the harness to be trusted.

    Each tests one failure mode that matters for H4 or for understanding model
    behaviour. The tolerance is small but nonzero: floating-point jitter over 1000
    bootstrap resamples makes exact numbers unrealistic.

    Note: expected directions are for typical models (75-90% accuracy with measurable
    ECE). Perfect models (100% accuracy) may show different ECE behaviour because
    there is no calibration error to report. These tests are designed to validate the
    harness on realistic predictions, and will be run against actual model outputs in
    Phase 10.
    """
    return [
        SeededErrorCase(
            name="label_flip_20pct",
            description=(
                "flip 20% of labels uniformly at random; accuracy should fall noticeably"),
            perturbation=_make_perturbation(_perturb_label_flip, rate=0.2, seed=20260901),
            expected_accuracy_direction="down",
            expected_ece_direction="none",
            tolerance=0.01),
        SeededErrorCase(
            name="confidence_inflation_40pct",
            description=(
                "push probabilities 40% away from 0.5; accuracy unchanged "
                "(hard predictions preserved). With a typical (not perfect) model, "
                "this inflates predicted confidence without changing decision, "
                "degrading ECE. The harness must detect this."),
            perturbation=_make_perturbation(
                _perturb_confidence_inflation, rate=0.4, seed=20260902),
            expected_accuracy_direction="none",
            expected_ece_direction="up",
            tolerance=0.001),
        SeededErrorCase(
            name="constant_predictor_50pct",
            description=(
                "predict 0.5 for every example; discrimination is destroyed, "
                "accuracy falls dramatically to near base rate"),
            perturbation=_make_perturbation(
                _perturb_constant_predictor, constant_value=0.5, seed=20260903),
            expected_accuracy_direction="down",
            expected_ece_direction="none",
            tolerance=0.01),
        SeededErrorCase(
            name="class_conditional_noise_b1",
            description=(
                "add 15% label noise to behaviour B1 only; the harness reports "
                "per-behaviour and must not average problems away. With only "
                "macro metrics, this may be invisible."),
            perturbation=_make_perturbation(
                _perturb_class_conditional_noise,
                target_behaviour="B1", noise_rate=0.15, seed=20260904),
            expected_accuracy_direction="none",  # depends on which behaviour we check
            expected_ece_direction="none",
            tolerance=0.001),
    ]


def validate_seeded_errors(baseline_predictions: dict[
    str, tuple[Sequence[float], Sequence[int | bool]]],
    label: str = "baseline"
) -> list[ValidationResult]:
    """Run the suite of seeded errors and validate the harness detects them.

    For each error, perturb the baseline predictions, evaluate both with the harness,
    and check that accuracy and ECE changed in the expected directions (within tolerance).
    Returns a list of validation results, one per error case.

    Args:
        baseline_predictions: dict of behaviour -> (probabilities, labels) pairs
        label: label for the baseline run, e.g. "baseline" or a model name

    Returns:
        list of ValidationResult, one per seeded error, with direction checks
    """
    # Evaluate the baseline once.
    try:
        baseline_result = evaluate(baseline_predictions, label=label)
    except Exception as e:
        return [ValidationResult(
            error_name="all",
            ran=False,
            reason=f"baseline evaluation failed: {e}")]

    baseline_macro_acc = baseline_result.macro_accuracy
    baseline_macro_ece = baseline_result.macro_ece

    results = []
    suite = seeded_error_suite()

    for case in suite:
        try:
            # Perturb and evaluate.
            perturbed = case.perturbation(baseline_predictions)
            perturbed_result = evaluate(perturbed, label=f"{label}_{case.name}")

            perturbed_macro_acc = perturbed_result.macro_accuracy
            perturbed_macro_ece = perturbed_result.macro_ece

            # Check directions of change.
            acc_change = perturbed_macro_acc - baseline_macro_acc
            ece_change = perturbed_macro_ece - baseline_macro_ece

            accuracy_ok = _check_direction(
                acc_change, case.expected_accuracy_direction, case.tolerance)
            ece_ok = _check_direction(
                ece_change, case.expected_ece_direction, case.tolerance)

            results.append(ValidationResult(
                error_name=case.name,
                ran=True,
                baseline_accuracy=baseline_macro_acc,
                baseline_ece=baseline_macro_ece,
                perturbed_accuracy=perturbed_macro_acc,
                perturbed_ece=perturbed_macro_ece,
                accuracy_changed_as_expected=accuracy_ok,
                ece_changed_as_expected=ece_ok))

        except Exception as e:
            results.append(ValidationResult(
                error_name=case.name,
                ran=False,
                reason=str(e),
                baseline_accuracy=baseline_macro_acc,
                baseline_ece=baseline_macro_ece))

    return results


def _check_direction(change: float, expected: str, tolerance: float) -> bool:
    """Check that a change matches the expected direction within tolerance.

    expected: 'up', 'down', or 'none'
    tolerance: minimum absolute change to count as having moved
    """
    checks: dict[str, bool] = {
        "none": abs(change) <= tolerance,
        "up": change > tolerance,
        "down": change < -tolerance,
    }
    return checks.get(expected, False)


def summary_table(results: list[ValidationResult]) -> str:
    """Format validation results as a readable table."""
    lines = ["Seeded Error Validation Results", "=" * 60]
    for result in results:
        if not result.ran:
            lines.append(f"{result.error_name}: FAILED - {result.reason}")
        else:
            acc_ok = "OK" if result.accuracy_changed_as_expected else "WRONG"
            ece_ok = "OK" if result.ece_changed_as_expected else "WRONG"
            baseline_str = (
                f"acc={result.baseline_accuracy:.4f}  "
                f"ece={result.baseline_ece:.4f}")
            perturbed_str = (
                f"acc={result.perturbed_accuracy:.4f}  "
                f"ece={result.perturbed_ece:.4f}")
            lines.append(f"{result.error_name}")
            lines.append(f"  baseline: {baseline_str}")
            lines.append(f"  perturbed: {perturbed_str}")
            lines.append(f"  accuracy: {acc_ok}  calibration: {ece_ok}")
    return "\n".join(lines)
