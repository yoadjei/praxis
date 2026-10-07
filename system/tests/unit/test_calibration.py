# -*- coding: utf-8 -*-
"""Phase 5 acceptance tests: calibration metrics, temperature scaling, ensembles.

The four BUILD-SPEC Phase 5 acceptance tests are `test_ece_is_zero_when_perfectly_calibrated`,
`test_ece_matches_the_analytic_value_when_overconfident`,
`test_temperature_scaling_reduces_ece`, and `test_ensemble_members_have_different_weights`.

Every expected number here is derived in the test body rather than taken from a previous run,
because a calibration metric that silently drifts would move the thesis's headline figure.
"""
from __future__ import annotations

import random
from statistics import fmean

import numpy as np
import pytest

from praxis.confidence.ensemble import (
    EnsembleError,
    assert_members_differ,
    check_members_share_device,
    combine,
    mc_dropout_combine,
    mutual_information,
    pairwise_weight_distance,
)
from praxis.confidence.metrics import (
    DEFAULT_BINS,
    MAX_BINARY_ENTROPY,
    MIN_SAMPLES_PER_BIN,
    binary_entropy,
    brier_score,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    multiclass_calibration_report,
    negative_log_likelihood,
    reliability_bins,
    reliability_diagram,
)
from praxis.confidence.temperature import (
    Temperature,
    TemperatureScalingFailed,
    apply_temperature,
    fit_per_behaviour,
    fit_temperature,
    probabilities_to_logits,
)


def _group(probability: float, n: int, positives: int) -> tuple[list[float], list[int]]:
    """`n` predictions all at one probability, of which `positives` are true.

    Every point shares a probability, so the whole group lands in one bin and that bin's gap
    is exactly |probability - positives/n|. That is what makes ECE hand-computable below.
    """
    return [probability] * n, [1] * positives + [0] * (n - positives)


def _bin_of(probability: float, n_bins: int) -> int:
    """Which equal-width bin a probability falls in, restated from the definition.

    Bins are right-closed - bin `b` covers `(b/n, (b+1)/n]` - with the first also closed at
    zero so 0.0 has a home. Written out longhand here rather than imported, so that a test
    comparing a hand-derived ECE against the reported one is comparing two independent
    readings of the same definition and not the implementation against itself.
    """
    for b in range(n_bins):
        if probability <= (b + 1) / n_bins:
            return b
    return n_bins - 1


# ---------------------------------------------------------------------------
# Acceptance tests 1 and 2: ECE against known values
# ---------------------------------------------------------------------------

def test_ece_is_zero_when_perfectly_calibrated() -> None:
    """Five probabilities, each delivered at exactly its claimed rate. ECE is exactly zero."""
    probs: list[float] = []
    labels: list[int] = []
    for probability in (0.1, 0.3, 0.5, 0.7, 0.9):
        p, y = _group(probability, 100, round(probability * 100))
        probs += p
        labels += y

    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-12)
    assert maximum_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-12)

    report = calibration_report(probs, labels)
    assert report.ece == pytest.approx(0.0, abs=1e-12)
    assert all(b.gap == pytest.approx(0.0, abs=1e-12) for b in report.bins)


def test_ece_matches_the_analytic_value_when_overconfident() -> None:
    """Two groups, each entirely inside one bin, so ECE is a weighted mean of two gaps.

        group A  100 predictions at p=0.9, 60 true  ->  |0.9 - 0.60| = 0.30
        group B  100 predictions at p=0.2, 40 true  ->  |0.2 - 0.40| = 0.20
        ECE = (100*0.30 + 100*0.20) / 200          =  0.25
        MCE = max(0.30, 0.20)                      =  0.30
    """
    probs_a, labels_a = _group(0.9, 100, 60)
    probs_b, labels_b = _group(0.2, 100, 40)
    probs, labels = probs_a + probs_b, labels_a + labels_b

    assert expected_calibration_error(probs, labels) == pytest.approx(0.25, abs=1e-12)
    assert maximum_calibration_error(probs, labels) == pytest.approx(0.30, abs=1e-12)

    report = calibration_report(probs, labels)
    assert len(report.bins) == 2, "each group must land wholly in one bin for this to be exact"


def test_overconfidence_is_measured_against_the_right_reference() -> None:
    """The reference differs by mode, and using the wrong one inverts the answer.

    On the fixture below the model over-forecasts presence but is *more* often right than its
    mean probability suggests, so the two references disagree:

        mean predicted probability = (100*0.9 + 100*0.2) / 200 = 0.550
        base rate                  = (60 + 40) / 200           = 0.500  -> over-forecasting
        accuracy                   = (60 + 60) / 200           = 0.600

    Comparing 0.550 against accuracy would report the model as *under*-confident, which is
    the opposite of the truth and would land in H4's headline figure.
    """
    probs_a, labels_a = _group(0.9, 100, 60)
    probs_b, labels_b = _group(0.2, 100, 40)
    report = calibration_report(probs_a + probs_b, labels_a + labels_b,
                                mode="positive_class")

    assert report.mean_confidence == pytest.approx(0.55, abs=1e-12)
    assert report.base_rate == pytest.approx(0.50, abs=1e-12)
    assert report.accuracy == pytest.approx(0.60, abs=1e-12)

    assert report.mean_outcome == pytest.approx(report.base_rate), (
        "in positive_class mode the claim is about presence, so presence is the reference")
    assert report.overconfident is True
    assert report.signed_gap == pytest.approx(0.05, abs=1e-12)

    guo = calibration_report(probs_a + probs_b, labels_a + labels_b, mode="confidence")
    assert guo.mean_outcome == pytest.approx(guo.accuracy), (
        "in confidence mode the claim is about being right, so accuracy is the reference")


def test_brier_and_nll_match_hand_computed_values() -> None:
    """Brier is a mean square; NLL is a mean log. Both checked on four points."""
    probs = [0.9, 0.1, 0.8, 0.3]
    labels = [1, 0, 1, 0]

    expected_brier = ((0.9 - 1) ** 2 + (0.1 - 0) ** 2 + (0.8 - 1) ** 2 + (0.3 - 0) ** 2) / 4
    assert brier_score(probs, labels) == pytest.approx(expected_brier, abs=1e-12)

    expected_nll = -(np.log(0.9) + np.log(0.9) + np.log(0.8) + np.log(0.7)) / 4
    assert negative_log_likelihood(probs, labels) == pytest.approx(expected_nll, abs=1e-12)


def test_empty_bins_are_dropped_not_reported_as_zero() -> None:
    """A bin nobody predicted into is absent, because plotting it at zero invents a failure."""
    probs, labels = _group(0.5, 20, 10)
    bins = reliability_bins(probs, labels, n_bins=15)
    assert len(bins) == 1
    assert bins[0].count == 20


def test_the_two_binning_conventions_differ_and_both_are_available() -> None:
    """D21: positive-class ECE and Guo-style confidence ECE answer different questions.

    On a model that is systematically over-confident about presence, the positive-class view
    shows the gap directly. The confidence view folds the negatives in by their probability of
    absence, so the two do not coincide, and naming one while reporting the other would be a
    quiet error.
    """
    probs, labels = _group(0.9, 100, 60)
    positive_class = expected_calibration_error(probs, labels, mode="positive_class")
    confidence = expected_calibration_error(probs, labels, mode="confidence")

    assert positive_class == pytest.approx(0.30, abs=1e-12)
    assert confidence == pytest.approx(0.30, abs=1e-12), "all predicted positive here"

    mixed_probs = [0.9] * 50 + [0.1] * 50
    mixed_labels = [1] * 30 + [0] * 20 + [1] * 5 + [0] * 45
    assert (expected_calibration_error(mixed_probs, mixed_labels, mode="positive_class")
            != expected_calibration_error(mixed_probs, mixed_labels, mode="confidence"))


def test_metrics_refuse_malformed_input() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_calibration_error([1.4, 0.2], [1, 0])
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        expected_calibration_error([0.5, 0.2], [2, 0])
    with pytest.raises(ValueError, match="against"):
        expected_calibration_error([0.5, 0.2], [1])
    with pytest.raises(ValueError, match="no predictions"):
        expected_calibration_error([], [])


# ---------------------------------------------------------------------------
# Acceptance test 3: temperature scaling must reduce ECE
# ---------------------------------------------------------------------------

def _overconfident(n: int = 4000, sharpening: float = 3.0,
                   seed: int = 20260911) -> tuple[np.ndarray, np.ndarray]:
    """A model whose logits are `sharpening` times too large.

    True logits z give the real outcome probability. The model reports z * sharpening, so it
    is right about the ranking and wrong about the confidence, which is the failure mode
    temperature scaling exists to correct. The recoverable temperature is the sharpening.
    """
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(0.0, 1.5, size=n)
    labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-true_logits)))
    return true_logits * sharpening, labels


def test_temperature_scaling_reduces_ece() -> None:
    """The acceptance test, and the recovered temperature is checked against the truth."""
    logits, labels = _overconfident(sharpening=3.0)
    fitted = fit_temperature(logits, labels, behaviour="B1")

    assert fitted.ece_after < fitted.ece_before, fitted.summary()
    assert fitted.nll_after <= fitted.nll_before + 1e-9
    assert fitted.softened, "an over-confident model needs T > 1"
    assert fitted.value == pytest.approx(3.0, rel=0.15), (
        f"the sharpening factor should be recoverable; got T={fitted.value:.3f}")
    assert fitted.ece_improvement > 0.05, "a threefold sharpening is a large miscalibration"


def test_temperature_scaling_leaves_accuracy_untouched() -> None:
    """Dividing by a positive scalar cannot reorder logits or move one across zero."""
    logits, labels = _overconfident()
    fitted = fit_temperature(logits, labels)

    before = (apply_temperature(logits, 1.0) >= 0.5).astype(int)
    after = (apply_temperature(logits, fitted.value) >= 0.5).astype(int)
    assert np.array_equal(before, after)


def test_temperature_scaling_fails_loudly_when_it_does_not_calibrate() -> None:
    """BUILD-SPEC Phase 5: if ECE does not fall, the implementation is wrong; say so.

    Simulated by handing the checker a result that claims a worse ECE than it started with,
    which is what a broken fit would produce.
    """
    from praxis.confidence.temperature import _check_it_helped

    broken = Temperature(behaviour="B1", value=1.0, n_validation=100,
                         nll_before=0.5, nll_after=0.5,
                         ece_before=0.20, ece_after=0.28)
    with pytest.raises(TemperatureScalingFailed, match="implementation is wrong"):
        _check_it_helped(broken)

    diverged = Temperature(behaviour="B1", value=2.0, n_validation=100,
                           nll_before=0.5, nll_after=0.9,
                           ece_before=0.20, ece_after=0.10)
    with pytest.raises(TemperatureScalingFailed, match="did not converge"):
        _check_it_helped(diverged)


def test_an_already_calibrated_head_is_not_treated_as_a_failure() -> None:
    """ECE is not the objective, so it may move by noise on a head that needed no correction.

    NLL improvement is guaranteed because T=1 is inside the search interval; ECE improvement
    is not. Raising here would make a well-calibrated model look like a bug.
    """
    from praxis.confidence.temperature import _check_it_helped

    fine = Temperature(behaviour="B2", value=1.001, n_validation=500,
                       nll_before=0.40, nll_after=0.40,
                       ece_before=0.004, ece_after=0.006)
    _check_it_helped(fine)


def test_temperature_refuses_a_single_class_validation_split() -> None:
    with pytest.raises(TemperatureScalingFailed, match="only class"):
        fit_temperature([0.4, 1.2, -0.3], [1, 1, 1])
    with pytest.raises(TemperatureScalingFailed, match="no validation data"):
        fit_temperature([], [])


def test_fit_per_behaviour_gives_one_temperature_per_head() -> None:
    """Five heads with different base rates need five scalars, not one shared."""
    validation = {}
    for index, behaviour in enumerate(("B1", "B2", "B3", "B4", "B5")):
        logits, labels = _overconfident(n=2000, sharpening=1.5 + 0.5 * index,
                                        seed=100 + index)
        validation[behaviour] = (logits, labels)

    fitted = fit_per_behaviour(validation)
    assert sorted(fitted) == ["B1", "B2", "B3", "B4", "B5"]
    assert all(t.ece_after < t.ece_before for t in fitted.values())

    recovered = [fitted[b].value for b in ("B1", "B2", "B3", "B4", "B5")]
    assert recovered == sorted(recovered), "a sharper head should need a larger temperature"


def test_probabilities_round_trip_through_logits() -> None:
    probs = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    assert apply_temperature(probabilities_to_logits(probs), 1.0) == pytest.approx(probs)


# ---------------------------------------------------------------------------
# Acceptance test 4: ensemble members must genuinely differ
# ---------------------------------------------------------------------------

def _member(seed: int, backbone_seed: int = 0) -> dict[str, np.ndarray]:
    """A synthetic member: a shared frozen backbone plus its own trainable stack."""
    shared = np.random.default_rng(backbone_seed)
    own = np.random.default_rng(seed)
    return {
        "backbone.layer4.weight": shared.normal(size=(64, 32)),
        "backbone.layer4.bias": shared.normal(size=64),
        "tcn.0.weight": own.normal(size=(32, 16)),
        "heads.B1.weight": own.normal(size=(1, 32)),
    }


def test_ensemble_members_have_different_weights() -> None:
    """The acceptance test: verified by pairwise distance, not assumed from the seed."""
    members = [_member(seed) for seed in (1, 2, 3, 4, 5)]
    report = assert_members_differ(members)

    assert report.n_members == 5
    assert len(report.distances) == 10, "every pair is compared"
    assert report.identical_pairs == ()
    assert report.minimum > 0.5, "independent initialisations are far apart"


def test_identical_members_are_refused() -> None:
    """A seed that failed to vary produces zero epistemic uncertainty and no OOD signal."""
    members = [_member(7), _member(7), _member(9)]
    with pytest.raises(EnsembleError, match="identical"):
        assert_members_differ(members)


def test_the_shared_backbone_is_excluded_from_the_diversity_check() -> None:
    """D3's deviation: members share a frozen front end, so comparing it would mask failure.

    With the backbone included, two members differing only in their trainable stack look far
    more similar than they are, because the shared parameters dominate the vector. The check
    therefore compares trainable parameters only, and this test pins the difference.
    """
    members = [_member(1), _member(2)]

    trainable_only = pairwise_weight_distance(members)
    everything = pairwise_weight_distance(members, shared_prefixes=())

    assert trainable_only.excluded_keys == ("backbone.layer4.bias", "backbone.layer4.weight")
    assert trainable_only.n_parameters_compared == 32 * 16 + 32
    assert everything.n_parameters_compared == 32 * 16 + 32 + 64 * 32 + 64
    assert trainable_only.minimum > everything.minimum, (
        "including the shared backbone dilutes the measured difference")


def test_a_fully_shared_ensemble_cannot_pass_vacuously() -> None:
    """If every parameter is excluded there is nothing left to check, so refuse."""
    members = [{"backbone.w": np.ones(4)}, {"backbone.w": np.ones(4)}]
    with pytest.raises(EnsembleError, match="excluded as shared"):
        pairwise_weight_distance(members)


def test_members_must_share_a_device_model() -> None:
    """D14: kernel variance across GPU models would enter the epistemic signal undetected."""
    check_members_share_device(["nvidia-t4"] * 5)
    check_members_share_device([None, None])
    with pytest.raises(EnsembleError, match="different device models"):
        check_members_share_device(["nvidia-t4", "nvidia-t4", "nvidia-l4"])


# ---------------------------------------------------------------------------
# Uncertainty decomposition
# ---------------------------------------------------------------------------

def test_agreeing_members_carry_no_epistemic_uncertainty() -> None:
    """Total uncertainty may be high while epistemic is zero: the members all
    agree it is hard."""
    members = [[0.5, 0.9, 0.1]] * 4
    prediction = combine(members)

    assert prediction.epistemic_uncertainty == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert prediction.total_uncertainty[0] == pytest.approx(MAX_BINARY_ENTROPY, abs=1e-12)
    assert prediction.aleatoric_uncertainty == pytest.approx(prediction.total_uncertainty)


def test_disagreeing_members_carry_epistemic_uncertainty() -> None:
    """Two members certain of opposite answers: maximal disagreement, zero aleatoric.

        mean = 0.5            -> total     = H(0.5) = ln 2
        each member is certain -> aleatoric = 0
        epistemic = ln 2, the most a binary variable can carry.
    """
    prediction = combine([[1.0 - 1e-15], [1e-15]])

    assert prediction.total_uncertainty[0] == pytest.approx(MAX_BINARY_ENTROPY, abs=1e-9)
    assert prediction.aleatoric_uncertainty[0] == pytest.approx(0.0, abs=1e-9)
    assert prediction.epistemic_uncertainty[0] == pytest.approx(MAX_BINARY_ENTROPY, abs=1e-9)
    assert prediction.disagreement[0] == pytest.approx(1.0, abs=1e-9)


def test_mutual_information_is_never_negative() -> None:
    """Guaranteed by Jensen, since entropy is concave. Checked over random ensembles."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        members = rng.uniform(0.0, 1.0, size=(rng.integers(2, 8), 12))
        assert np.all(mutual_information(members) >= 0.0)


def test_total_decomposes_into_aleatoric_plus_epistemic() -> None:
    rng = np.random.default_rng(3)
    members = rng.uniform(0.02, 0.98, size=(5, 50))
    prediction = combine(members)
    assert prediction.total_uncertainty == pytest.approx(
        prediction.aleatoric_uncertainty + prediction.epistemic_uncertainty, abs=1e-12)


def test_an_ensemble_of_one_is_refused() -> None:
    with pytest.raises(EnsembleError, match="no disagreement"):
        combine([[0.5, 0.6]])


def test_mc_dropout_uses_the_same_decomposition() -> None:
    passes = [[0.8, 0.3], [0.6, 0.4], [0.7, 0.2], [0.9, 0.35]]
    assert mc_dropout_combine(passes).epistemic_uncertainty == pytest.approx(
        combine(passes).epistemic_uncertainty)


def test_binary_entropy_is_maximal_at_one_half() -> None:
    assert binary_entropy(0.5) == pytest.approx(MAX_BINARY_ENTROPY, abs=1e-12)
    assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-10)
    assert binary_entropy(1.0) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# The contract, and the figure
# ---------------------------------------------------------------------------

def test_ensemble_produces_a_populated_confidence_state() -> None:
    """Phase 5's definition of done: every detection carries a populated ConfidenceState."""
    prediction = combine([[0.9, 0.2], [0.7, 0.3], [0.85, 0.25]])
    state = prediction.to_confidence_state(
        index=0, raw_probability=0.88, ood_score=0.12, ood_flag=False,
        domain="microteaching")

    assert state.method == "ensemble"
    assert state.epistemic is not None and state.epistemic > 0
    assert state.calibrated_prob == pytest.approx(prediction.mean_probability[0])
    assert state.in_validated_domain == "microteaching"

    flagged = prediction.to_confidence_state(
        index=1, raw_probability=0.2, ood_score=0.97, ood_flag=True, domain="classroom")
    assert flagged.in_validated_domain == "unknown", "D16: a flagged detection has no domain"


def test_reliability_diagram_is_written(tmp_path) -> None:
    """The thesis figures are PNG on disk, so the rendering path is exercised in CI."""
    probs_a, labels_a = _group(0.9, 100, 60)
    probs_b, labels_b = _group(0.2, 100, 40)
    report = calibration_report(probs_a + probs_b, labels_a + labels_b)

    written = reliability_diagram(report, tmp_path / "figures" / "b1_ensemble.png",
                                  title="B1, ensemble, S0")
    assert written.exists() and written.stat().st_size > 1000
    assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Multiclass calibration. R4 for the codebook's categorical fields.
# ---------------------------------------------------------------------------

class TestMulticlassCalibration:
    """Five codebook fields are categorical, and R4 admits no exception for them.

    The strongest check available is agreement with the binary implementation at K = 2: a
    second binning rule would diverge at the edges first, which is exactly where an abstention
    threshold gets chosen.
    """

    def _two_class(self, n: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        probs = rng.uniform(size=n)
        labels = (rng.uniform(size=n) < probs).astype(int)
        return probs, labels, np.stack([1.0 - probs, probs], axis=1)

    def test_at_two_classes_it_agrees_with_the_binary_path(self) -> None:
        probs, labels, as_matrix = self._two_class()
        multi = multiclass_calibration_report(as_matrix, labels)
        binary = calibration_report(probs, labels, mode="confidence")

        assert multi.accuracy == pytest.approx(binary.accuracy)
        assert multi.ece == pytest.approx(binary.ece)
        assert multi.mce == pytest.approx(binary.mce)
        assert multi.mean_confidence == pytest.approx(binary.mean_confidence)

    def test_the_multiclass_brier_is_twice_the_binary_one_at_two_classes(self) -> None:
        """Documented rather than incidental: a reader comparing a binary field's Brier with a
        categorical field's needs to know they are not on the same scale."""
        probs, labels, as_matrix = self._two_class()
        assert (multiclass_calibration_report(as_matrix, labels).brier
                == pytest.approx(2.0 * calibration_report(probs, labels).brier))

    def test_a_calibrated_predictor_has_low_ece(self) -> None:
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(4000, 4))
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        labels = np.array([rng.choice(4, p=row) for row in probs])

        assert multiclass_calibration_report(probs, labels).ece < 0.05

    def test_sharpening_the_same_predictor_makes_it_overconfident(self) -> None:
        """The property H4 is about: confidence moving away from what was delivered."""
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(4000, 4))
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        labels = np.array([rng.choice(4, p=row) for row in probs])

        sharp = probs ** 4
        sharp /= sharp.sum(axis=1, keepdims=True)
        sharpened = multiclass_calibration_report(sharp, labels)

        assert sharpened.ece > multiclass_calibration_report(probs, labels).ece
        assert sharpened.overconfident

    def test_the_base_rate_is_the_most_common_class(self) -> None:
        """What accuracy has to beat before it means anything."""
        probs = np.full((10, 3), 1.0 / 3.0)
        labels = np.array([0] * 7 + [1, 1, 2])
        assert multiclass_calibration_report(probs, labels).base_rate == pytest.approx(0.7)

    @pytest.mark.parametrize("bad, message", [
        (("rows", np.array([[0.5, 0.2, 0.2]]), np.array([0])), "sum to 1"),
        (("range", np.array([[0.5, 0.5]]), np.array([5])), "must index"),
        (("shape", np.array([0.5, 0.5]), np.array([0])), "expected"),
        (("length", np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0])), "against"),
    ])
    def test_malformed_input_is_refused(self, bad, message) -> None:
        _, probs, labels = bad
        with pytest.raises(ValueError, match=message):
            multiclass_calibration_report(probs, labels)


class TestThinSamples:
    """D86. Eleven sessions is a small corpus, and ECE over 15 bins degrades quietly: it is
    population-weighted, so it keeps returning a number that looks like the others while most
    bins hold a handful of clips or none. MCE is worse - it is the single worst bin outright.
    The numbers stay computable; what is added is the report saying how far to trust them.
    """

    def sample(self, n: int, seed: int = 0):
        rng = random.Random(seed)
        probabilities = [rng.random() for _ in range(n)]
        return probabilities, [rng.random() < p for p in probabilities]

    def test_a_thin_sample_says_so(self) -> None:
        report = calibration_report(*self.sample(60))
        assert report.reliable is False
        assert report.caveat() is not None
        assert "60 samples" in report.caveat()

    def test_an_adequate_sample_carries_no_caveat(self) -> None:
        report = calibration_report(*self.sample(600))
        assert report.reliable is True
        assert report.caveat() is None

    def test_the_threshold_is_stated_rather_than_implied(self) -> None:
        """Ten per bin at the configured bin count. Asserted against the constant so that
        changing the bin count moves the requirement with it."""
        bins = 15
        just_under = calibration_report(*self.sample(MIN_SAMPLES_PER_BIN * bins - 1),
                                        n_bins=bins)
        exactly = calibration_report(*self.sample(MIN_SAMPLES_PER_BIN * bins), n_bins=bins)

        assert just_under.reliable is False
        assert exactly.reliable is True

    def test_the_numbers_are_still_produced(self) -> None:
        """Abstaining by returning None would propagate into every caller and into R4, which
        requires calibration reported wherever accuracy is. The figure is reported with its
        caveat, not withheld.

        The ECE is re-derived here from the definition rather than read back off the report,
        because the claim being tested is that marking a sample thin changes what the report
        *says* and not what it *computes*. Comparing the figure to itself would assert nothing.
        """
        probabilities, labels = self.sample(40)
        report = calibration_report(probabilities, labels)

        populated: dict[int, list[tuple[float, float]]] = {}
        for probability, label in zip(probabilities, labels, strict=True):
            populated.setdefault(_bin_of(probability, DEFAULT_BINS), []).append(
                (probability, float(label)))
        by_hand = sum(
            len(pairs) * abs(fmean([outcome for _, outcome in pairs])
                             - fmean([claimed for claimed, _ in pairs]))
            for pairs in populated.values()) / len(probabilities)

        assert report.reliable is False, "forty samples is thin and the report must say so"
        assert report.ece == pytest.approx(by_hand), (
            "the thin-sample caveat must not alter the arithmetic it caveats")
        assert 0.0 <= report.ece <= 1.0
        assert 0.0 <= report.accuracy <= 1.0

    def test_the_summary_line_carries_the_warning(self) -> None:
        """Whoever reads a summary without reading the object must still see it."""
        assert "[thin sample]" in calibration_report(*self.sample(40)).summary()
        assert "[thin sample]" not in calibration_report(*self.sample(600)).summary()

    def test_empty_bins_are_counted(self) -> None:
        """A report whose bins are mostly empty is the shape the warning is about."""
        probabilities = [0.5] * 30
        report = calibration_report(probabilities, [True] * 15 + [False] * 15)
        assert report.populated_bins == 1
        assert report.reliable is False
