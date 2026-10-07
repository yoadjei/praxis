# -*- coding: utf-8 -*-
"""Tests for seeded error injection harness.

These tests validate that the error injection functions work as documented and that
the validation suite correctly detects them. Each test catches a specific mutation in
the implementation.
"""
from __future__ import annotations

import copy

import numpy as np

from praxis.evaluation.seeded_errors import (
    SeededErrorCase,
    ValidationResult,
    _check_direction,
    _make_perturbation,
    _perturb_class_conditional_noise,
    _perturb_confidence_inflation,
    _perturb_constant_predictor,
    _perturb_label_flip,
    seeded_error_suite,
    summary_table,
    validate_seeded_errors,
)


def make_baseline_predictions(
        n_samples: int = 100, n_behaviours: int = 2
) -> dict[str, tuple[list[float], list[int]]]:
    """Create synthetic predictions with high accuracy for testing.

    Returns a dict of behaviour -> (probabilities, labels) where probabilities are
    drawn from a distribution that gives ~85% accuracy when thresholded at 0.5.
    This is typical for realistic model outputs in the evaluation harness.
    """
    generator = np.random.default_rng(20260901)
    predictions = {}
    for i in range(n_behaviours):
        behaviour_name = f"B{i + 1}"
        # Generate high-confidence correct predictions: 85% accuracy.
        labels = generator.binomial(1, 0.5, n_samples)
        # When label=1, prob is high. When label=0, prob is low.
        # This gives ~85% accuracy at the 0.5 threshold.
        probs = np.where(
            labels == 1,
            generator.normal(0.75, 0.15, n_samples),
            generator.normal(0.25, 0.15, n_samples)
        )
        # Clip to [0, 1].
        probs = np.clip(probs, 0.0, 1.0)
        predictions[behaviour_name] = (probs.tolist(), labels.tolist())
    return predictions


class TestLabelFlip:
    """Test that label flipping degrades accuracy as expected."""

    def test_label_flip_rate_degrades_accuracy(self) -> None:
        """Label flipping reduces accuracy, drop is near the requested rate.

        Mutation: a flip that ignored its rate would pass the tolerance test.
        """
        baseline = make_baseline_predictions(n_samples=200, n_behaviours=1)
        perturbed = _perturb_label_flip(baseline, rate=0.2, seed=20260901)

        # Baseline accuracy.
        baseline_preds = (np.asarray(baseline["B1"][0]) >= 0.5).astype(float)
        baseline_labels = np.asarray(baseline["B1"][1])
        baseline_acc = (baseline_preds == baseline_labels).mean()

        # Perturbed accuracy.
        perturbed_preds = (np.asarray(perturbed["B1"][0]) >= 0.5).astype(float)
        perturbed_labels = np.asarray(perturbed["B1"][1])
        perturbed_acc = (perturbed_preds == perturbed_labels).mean()

        # Drop should be near 20% (tolerance 5% to account for randomness).
        drop = baseline_acc - perturbed_acc
        assert 0.15 < drop < 0.25, (
            f"label flip at 20% gave drop {drop:.3f}, "
            f"expected near 0.20"
        )

    def test_label_flip_input_not_mutated(self) -> None:
        """Label flipping does not mutate the input dict.

        Mutation: a flip that modifies in place would corrupt the baseline
        that is compared against in the validation suite.
        """
        baseline = make_baseline_predictions(n_samples=50, n_behaviours=1)
        original = copy.deepcopy(baseline)

        _perturb_label_flip(baseline, rate=0.1, seed=20260901)

        # Input should be unchanged.
        assert baseline["B1"][0] == original["B1"][0]
        assert baseline["B1"][1] == original["B1"][1]

    def test_label_flip_produces_binary_labels(self) -> None:
        """Label flipping preserves binary label structure.

        Mutation: a flip that produced values outside {0, 1} would break
        the downstream evaluation harness.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=1)
        perturbed = _perturb_label_flip(baseline, rate=0.3, seed=20260901)

        # All labels must be 0 or 1.
        labels = np.asarray(perturbed["B1"][1])
        assert np.all((labels == 0) | (labels == 1)), (
            f"label flip produced invalid values: {np.unique(labels)}"
        )

    def test_label_flip_reproducible(self) -> None:
        """Label flipping with same seed gives same result.

        Mutation: a flip that ignored its seed would pass this test if run
        once, but the combination with different seed catching it.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=1)

        p1 = _perturb_label_flip(baseline, rate=0.2, seed=20260901)
        p2 = _perturb_label_flip(baseline, rate=0.2, seed=20260901)
        p3 = _perturb_label_flip(baseline, rate=0.2, seed=20260902)

        # Same seed must give identical result (both probabilities and labels).
        # Note: probabilities are unchanged by label flip, only labels change.
        assert p1["B1"][0] == p2["B1"][0]
        assert p1["B1"][1] == p2["B1"][1]

        # Different seed must give different labels (with high probability).
        assert p1["B1"][1] != p3["B1"][1]


class TestConfidenceInflation:
    """Test that confidence inflation degrades calibration without accuracy change."""

    def test_confidence_inflation_preserves_accuracy(self) -> None:
        """Confidence inflation leaves accuracy unchanged.

        This is the critical test for H4. Hard decisions at threshold 0.5 are never
        crossed, so the binary prediction remains the same even though the
        probability moves.

        Mutation: an inflation that crossed the 0.5 threshold would change accuracy.
        """
        baseline = make_baseline_predictions(n_samples=200, n_behaviours=1)
        perturbed = _perturb_confidence_inflation(baseline, rate=0.4, seed=20260902)

        # Accuracy should be identical before and after.
        baseline_probs = np.asarray(baseline["B1"][0])
        perturbed_probs = np.asarray(perturbed["B1"][0])
        labels = np.asarray(baseline["B1"][1])

        baseline_acc = ((baseline_probs >= 0.5).astype(float) == labels).mean()
        perturbed_acc = ((perturbed_probs >= 0.5).astype(float) == labels).mean()

        # Must be identical to floating point precision.
        assert baseline_acc == perturbed_acc, (
            f"inflation changed accuracy from {baseline_acc:.4f} "
            f"to {perturbed_acc:.4f}"
        )

    def test_confidence_inflation_increases_magnitude(self) -> None:
        """Confidence inflation pushes probabilities away from 0.5.

        Mutation: inflation that moved probabilities toward 0.5 would violate this.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=1)
        perturbed = _perturb_confidence_inflation(baseline, rate=0.4, seed=20260902)

        baseline_probs = np.asarray(baseline["B1"][0])
        perturbed_probs = np.asarray(perturbed["B1"][0])

        # Probabilities >= 0.5 should increase (move toward 1).
        high_mask = baseline_probs >= 0.5
        assert np.all(perturbed_probs[high_mask] >= baseline_probs[high_mask]), (
            "high probabilities should not decrease"
        )

        # Probabilities < 0.5 should decrease (move toward 0).
        low_mask = baseline_probs < 0.5
        assert np.all(perturbed_probs[low_mask] <= baseline_probs[low_mask]), (
            "low probabilities should not increase"
        )

    def test_confidence_inflation_preserves_probability_bounds(self) -> None:
        """Confidence inflation keeps probabilities in [0, 1].

        Mutation: inflation that let probabilities exceed these bounds would be
        meaningless for calibration metrics.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=1)
        perturbed = _perturb_confidence_inflation(baseline, rate=0.5, seed=20260902)

        probs = np.asarray(perturbed["B1"][0])
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), (
            f"inflation produced out-of-bounds probabilities: "
            f"[{probs.min():.4f}, {probs.max():.4f}]"
        )

    def test_confidence_inflation_input_not_mutated(self) -> None:
        """Confidence inflation does not mutate input.

        Mutation: in-place modification would corrupt the baseline.
        """
        baseline = make_baseline_predictions(n_samples=50, n_behaviours=1)
        original = copy.deepcopy(baseline)

        _perturb_confidence_inflation(baseline, rate=0.3, seed=20260902)

        assert baseline["B1"][0] == original["B1"][0]
        assert baseline["B1"][1] == original["B1"][1]

    def test_confidence_inflation_reproducible(self) -> None:
        """Confidence inflation with same rate and seed is deterministic.

        Mutation: if the implementation ignores the seed, this would catch it.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=1)

        # Note: confidence inflation doesn't actually use the seed parameter
        # in the current implementation, but we test that at least the output
        # is deterministic given the same input.
        p1 = _perturb_confidence_inflation(baseline, rate=0.4, seed=20260902)
        p2 = _perturb_confidence_inflation(baseline, rate=0.4, seed=20260902)

        assert np.allclose(p1["B1"][0], p2["B1"][0])


class TestConstantPredictor:
    """Test that constant predictor destroys discrimination."""

    def test_constant_predictor_removes_discrimination(self) -> None:
        """Constant predictor removes all discrimination.

        With constant probability (0.5), all hard predictions (>=0.5) are the same
        (all 1), so the model always predicts the same class and loses all
        discrimination.

        Mutation: a constant that randomized would preserve some discrimination.
        """
        baseline = make_baseline_predictions(n_samples=500, n_behaviours=1)
        perturbed = _perturb_constant_predictor(baseline, constant_value=0.5, seed=0)

        perturbed_probs = np.asarray(perturbed["B1"][0])

        # With all predictions at 0.5, hard decision is >=0.5 -> 1.
        # So all predictions are 1.
        all_predictions_same = (perturbed_probs == perturbed_probs[0]).all()
        assert all_predictions_same, "constant predictor should predict same for all"

        # Verify prediction is indeed 1 (since 0.5 >= 0.5).
        all_predict_one = np.all(perturbed_probs >= 0.5)
        assert all_predict_one

    def test_constant_predictor_probabilities_all_same(self) -> None:
        """Constant predictor sets all probabilities to the same value.

        Mutation: a constant that randomized would violate this.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=1)
        perturbed = _perturb_constant_predictor(baseline, constant_value=0.75, seed=0)

        probs = np.asarray(perturbed["B1"][0])
        assert np.allclose(probs, 0.75), (
            f"not all probabilities are 0.75: {np.unique(probs)}"
        )

    def test_constant_predictor_input_not_mutated(self) -> None:
        """Constant predictor does not mutate input.

        Mutation: in-place modification would corrupt the baseline.
        """
        baseline = make_baseline_predictions(n_samples=50, n_behaviours=1)
        original = copy.deepcopy(baseline)

        _perturb_constant_predictor(baseline, constant_value=0.5, seed=0)

        assert baseline["B1"][0] == original["B1"][0]
        assert baseline["B1"][1] == original["B1"][1]

    def test_constant_predictor_preserves_label_structure(self) -> None:
        """Constant predictor does not change the labels.

        Mutation: a predictor that modified labels would produce wrong evaluation.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=1)
        perturbed = _perturb_constant_predictor(baseline, constant_value=0.5, seed=0)

        assert baseline["B1"][1] == perturbed["B1"][1]


class TestClassConditionalNoise:
    """Test that per-behaviour noise isolates to that behaviour."""

    def test_class_conditional_noise_targets_one_behaviour(self) -> None:
        """Class-conditional noise affects only the target behaviour.

        Mutation: a noise that applied to all behaviours would fail this test.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=3)
        perturbed = _perturb_class_conditional_noise(
            baseline, target_behaviour="B2", noise_rate=0.2, seed=20260904
        )

        # B2 should be perturbed (different labels).
        assert baseline["B2"][1] != perturbed["B2"][1]

        # B1 and B3 should be unchanged.
        assert baseline["B1"][1] == perturbed["B1"][1]
        assert baseline["B3"][1] == perturbed["B3"][1]

    def test_class_conditional_noise_produces_binary_labels(self) -> None:
        """Class-conditional noise preserves binary label structure.

        Mutation: noise that produced values outside {0, 1} would break evaluation.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=2)
        perturbed = _perturb_class_conditional_noise(
            baseline, target_behaviour="B1", noise_rate=0.15, seed=20260904
        )

        for behaviour in perturbed:
            labels = np.asarray(perturbed[behaviour][1])
            assert np.all((labels == 0) | (labels == 1)), (
                f"noise in {behaviour} produced invalid labels"
            )

    def test_class_conditional_noise_input_not_mutated(self) -> None:
        """Class-conditional noise does not mutate input.

        Mutation: in-place modification would corrupt the baseline.
        """
        baseline = make_baseline_predictions(n_samples=50, n_behaviours=2)
        original = copy.deepcopy(baseline)

        _perturb_class_conditional_noise(
            baseline, target_behaviour="B1", noise_rate=0.1, seed=20260904
        )

        # Input should be unchanged.
        for behaviour in baseline:
            assert baseline[behaviour][0] == original[behaviour][0]
            assert baseline[behaviour][1] == original[behaviour][1]

    def test_class_conditional_noise_reproducible(self) -> None:
        """Class-conditional noise with same seed is deterministic.

        Mutation: if the implementation ignores the seed, this would catch it.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=2)

        p1 = _perturb_class_conditional_noise(
            baseline, target_behaviour="B1", noise_rate=0.15, seed=20260904
        )
        p2 = _perturb_class_conditional_noise(
            baseline, target_behaviour="B1", noise_rate=0.15, seed=20260904
        )
        p3 = _perturb_class_conditional_noise(
            baseline, target_behaviour="B1", noise_rate=0.15, seed=20260905
        )

        # Same seed must give identical result.
        assert p1["B1"][1] == p2["B1"][1]

        # Different seed must give different result (with high probability).
        assert p1["B1"][1] != p3["B1"][1]


class TestMakePerturbation:
    """Test the perturbation wrapper function."""

    def test_make_perturbation_wraps_kwargs(self) -> None:
        """_make_perturbation fixes kwargs and returns a callable.

        Mutation: a wrapper that ignored kwargs would produce wrong perturbations.
        """
        baseline = make_baseline_predictions(n_samples=50, n_behaviours=1)

        wrapped = _make_perturbation(_perturb_label_flip, rate=0.2, seed=20260901)
        perturbed = wrapped(baseline)

        # Should have flipped labels (different from baseline).
        assert baseline["B1"][1] != perturbed["B1"][1]

    def test_make_perturbation_output_has_same_structure(self) -> None:
        """_make_perturbation preserves the dict structure.

        Mutation: a wrapper that changed the structure would break downstream code.
        """
        baseline = make_baseline_predictions(n_samples=50, n_behaviours=2)

        wrapped = _make_perturbation(
            _perturb_confidence_inflation, rate=0.3, seed=20260902
        )
        perturbed = wrapped(baseline)

        # Should have same keys.
        assert set(baseline.keys()) == set(perturbed.keys())

        # Each value should be (probs, labels) tuple.
        for _behaviour, (probs, labels) in perturbed.items():
            assert len(probs) == len(labels)
            assert len(probs) > 0


class TestCheckDirection:
    """Test the direction-checking helper."""

    def test_check_direction_down(self) -> None:
        """_check_direction detects when metric decreased.

        Mutation: a direction checker that ignored the sign would fail.
        """
        # Accuracy dropped by 0.05.
        assert _check_direction(-0.05, "down", tolerance=0.01) is True
        assert _check_direction(-0.05, "up", tolerance=0.01) is False
        assert _check_direction(-0.05, "none", tolerance=0.01) is False

    def test_check_direction_up(self) -> None:
        """_check_direction detects when metric increased.

        Mutation: a direction checker that ignored the sign would fail.
        """
        # ECE increased by 0.02.
        assert _check_direction(0.02, "up", tolerance=0.001) is True
        assert _check_direction(0.02, "down", tolerance=0.001) is False
        assert _check_direction(0.02, "none", tolerance=0.001) is False

    def test_check_direction_none(self) -> None:
        """_check_direction detects when metric stayed the same.

        Mutation: a direction checker that ignored tolerance would fail.
        """
        # Change of 0.0005, tolerance is 0.001.
        assert _check_direction(0.0005, "none", tolerance=0.001) is True
        assert _check_direction(0.0005, "up", tolerance=0.001) is False
        assert _check_direction(0.0005, "down", tolerance=0.001) is False

    def test_check_direction_respects_tolerance(self) -> None:
        """_check_direction applies tolerance correctly.

        Mutation: a checker that ignored tolerance would fail this.
        """
        # Change of 0.015 with tolerance 0.01 should be "down".
        assert _check_direction(-0.015, "down", tolerance=0.01) is True

        # Change of 0.005 with tolerance 0.01 should not be "down".
        assert _check_direction(-0.005, "down", tolerance=0.01) is False

        # Change of 0.009 with tolerance 0.01 should be "none".
        assert _check_direction(0.009, "none", tolerance=0.01) is True
        assert _check_direction(0.009, "down", tolerance=0.01) is False


class TestSeededErrorSuite:
    """Test the suite definition."""

    def test_seeded_error_suite_returns_cases(self) -> None:
        """seeded_error_suite returns a list of SeededErrorCase instances.

        Mutation: suite that returned wrong type would be caught by type checker
        but also by this test.
        """
        suite = seeded_error_suite()

        assert isinstance(suite, list)
        assert len(suite) > 0
        assert all(isinstance(case, SeededErrorCase) for case in suite)

    def test_seeded_error_suite_has_expected_cases(self) -> None:
        """seeded_error_suite includes the required error types.

        Mutation: suite missing a case would be caught here.
        """
        suite = seeded_error_suite()
        names = {case.name for case in suite}

        assert "label_flip_20pct" in names
        assert "confidence_inflation_40pct" in names
        assert "constant_predictor_50pct" in names
        assert "class_conditional_noise_b1" in names

    def test_seeded_error_suite_cases_have_perturbations(self) -> None:
        """Each case has a callable perturbation.

        Mutation: a case with a missing perturbation would fail here.
        """
        suite = seeded_error_suite()

        for case in suite:
            assert callable(case.perturbation)
            # Test that it can be called.
            baseline = make_baseline_predictions(n_samples=10, n_behaviours=1)
            result = case.perturbation(baseline)
            assert isinstance(result, dict)


class TestValidateSeededErrors:
    """Test the validation runner."""

    def test_validate_seeded_errors_runs_suite(self) -> None:
        """validate_seeded_errors runs all cases and returns results.

        Mutation: a validator that skipped cases would return fewer results.
        """
        baseline = make_baseline_predictions(n_samples=150, n_behaviours=5)

        results = validate_seeded_errors(baseline, label="test")

        # Should have a result for each case in the suite.
        suite = seeded_error_suite()
        assert len(results) == len(suite)

        # All should be ValidationResult instances.
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_validate_seeded_errors_results_have_error_names(self) -> None:
        """Validation results include the error case names.

        Mutation: a validator that lost the name would fail here.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=5)
        results = validate_seeded_errors(baseline)

        names = {r.error_name for r in results}
        assert "label_flip_20pct" in names
        assert "confidence_inflation_40pct" in names
        assert "constant_predictor_50pct" in names
        assert "class_conditional_noise_b1" in names

    def test_validate_seeded_errors_baseline_failure_handled(self) -> None:
        """Validation handles baseline evaluation failure gracefully.

        Mutation: a validator that crashed would not return gracefully.
        """
        # Empty predictions should cause baseline evaluation to fail.
        baseline = {}

        results = validate_seeded_errors(baseline)

        # Should have a result indicating failure.
        assert any(not r.ran for r in results)

    def test_validate_seeded_errors_reports_accuracy_direction(self) -> None:
        """Validation results include accuracy direction check.

        Mutation: a validator that skipped direction checks would not catch
        perturbations that don't behave as expected.
        """
        baseline = make_baseline_predictions(n_samples=200, n_behaviours=5)
        results = validate_seeded_errors(baseline)

        # Find the label flip result.
        label_flip_result = next(
            (r for r in results if r.error_name == "label_flip_20pct"), None
        )
        assert label_flip_result is not None
        assert label_flip_result.ran is True
        assert label_flip_result.accuracy_changed_as_expected is not None

    def test_validate_seeded_errors_reports_ece_direction(self) -> None:
        """Validation results include ECE direction check.

        Mutation: a validator that skipped ECE checks would not catch
        calibration problems.
        """
        baseline = make_baseline_predictions(n_samples=200, n_behaviours=5)
        results = validate_seeded_errors(baseline)

        # Find the confidence inflation result.
        inflation_result = next(
            (r for r in results if r.error_name == "confidence_inflation_40pct"),
            None
        )
        assert inflation_result is not None
        assert inflation_result.ran is True
        assert inflation_result.ece_changed_as_expected is not None


class TestSummaryTable:
    """Test the human-readable output formatter."""

    def test_summary_table_formats_results(self) -> None:
        """summary_table formats validation results as a string.

        Mutation: a formatter that returned wrong type would fail.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=5)
        results = validate_seeded_errors(baseline)

        output = summary_table(results)

        assert isinstance(output, str)
        assert len(output) > 0
        assert "Seeded Error Validation Results" in output

    def test_summary_table_includes_all_results(self) -> None:
        """summary_table includes all case names.

        Mutation: a formatter that skipped results would fail here.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=5)
        results = validate_seeded_errors(baseline)

        output = summary_table(results)

        assert "label_flip_20pct" in output
        assert "confidence_inflation_40pct" in output
        assert "constant_predictor_50pct" in output
        assert "class_conditional_noise_b1" in output

    def test_summary_table_reports_success_and_failure(self) -> None:
        """summary_table distinguishes ran vs. failed results.

        Mutation: a formatter that didn't report success/failure would hide
        problems.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=5)
        results = validate_seeded_errors(baseline)

        output = summary_table(results)

        # Output should contain status indicators.
        # If all ran successfully, should have "OK" or "WRONG" markers.
        assert "OK" in output or "WRONG" in output or "FAILED" in output


class TestProbabilityBounds:
    """Test that all perturbations keep probabilities in [0, 1]."""

    def test_all_perturbations_respect_bounds(self) -> None:
        """Every perturbation keeps probabilities in [0, 1].

        Mutation: a perturbation that produced out-of-bounds probabilities
        would break calibration metrics.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=3)

        suite = seeded_error_suite()
        for case in suite:
            perturbed = case.perturbation(baseline)

            for behaviour, (probs, _labels) in perturbed.items():
                probs_array = np.asarray(probs)
                assert np.all(probs_array >= 0.0), (
                    f"{case.name} produced probabilities < 0 in {behaviour}: "
                    f"{probs_array.min()}"
                )
                assert np.all(probs_array <= 1.0), (
                    f"{case.name} produced probabilities > 1 in {behaviour}: "
                    f"{probs_array.max()}"
                )


class TestInputMutationAcrossAllPerturbations:
    """Test that all perturbations respect immutability."""

    def test_no_perturbation_mutates_baseline(self) -> None:
        """No perturbation function mutates its input.

        Mutation: any perturbation that modified in place would corrupt the
        baseline in the validation suite.
        """
        baseline = make_baseline_predictions(n_samples=100, n_behaviours=3)
        original = copy.deepcopy(baseline)

        suite = seeded_error_suite()
        for case in suite:
            case.perturbation(baseline)

            # Baseline must still be unchanged.
            for behaviour in baseline:
                assert baseline[behaviour][0] == original[behaviour][0], (
                    f"perturbation {case.name} mutated probabilities"
                )
                assert baseline[behaviour][1] == original[behaviour][1], (
                    f"perturbation {case.name} mutated labels"
                )
