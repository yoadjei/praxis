# -*- coding: utf-8 -*-
"""Phase 6 acceptance tests: domain shift, attribution, OOD detection, abstention.

The four BUILD-SPEC Phase 6 acceptance tests are `test_harness_refuses_leaked_s2_sessions`,
`test_attribution_reports_intervals_not_point_estimates`,
`test_auroc_matches_the_analytic_value`, and `test_threshold_sweep_produces_a_table`.

The AUROC expectation is analytic rather than remembered. For two unit-variance Gaussians
separated by `d`, the probability that a draw from the upper one exceeds a draw from the lower
is `Phi(d / sqrt(2))`, so a fixture with known separation has a known AUROC.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.stats import norm

from praxis.confidence.ood import (
    OodError,
    auroc,
    disagreement_score,
    evaluate_ood,
    fit_mahalanobis,
    flag_rate_by_domain,
    fpr_at_tpr,
    msp_score,
)
from praxis.contracts.manifest import SplitManifest
from praxis.evaluation.abstention import SweepRow, choose_threshold, threshold_sweep
from praxis.evaluation.attribution import attribute_degradation, attribution_feasibility
from praxis.evaluation.harness import (
    BehaviourEvaluation,
    ConfusionStructure,
    evaluate,
    evaluate_behaviour,
)
from praxis.evaluation.shift import ShiftError, ShiftLevel, build_shift_table
from praxis.ids import new_ulid

COVARIATES = ["camera_distance_m", "room_area_m2", "pupil_count",
              "ambient_noise_dba", "teacher_movement_range_m"]

# ULIDs because the columns these land in are CHAR(26) and a shorter value is stored
# blank-padded (D93). Crockford base32 excludes I, L, O and U, so neither of these can spell
# out the word it stands for.
SCHOOL = "01M2H0000000000000000SCH00"
COLLEGE = "01M2H0000000000000000CGE00"


def _manifest(**overrides) -> SplitManifest:
    base = dict(
        manifest_id=new_ulid(), created_at=datetime.now(UTC),
        config_sha256="a" * 64,
        train_teachers=["T1", "T2"], val_teachers=["T3"], test_teachers=["T4"],
        shift_eval_sessions=["S2a", "S2b"])
    base.update(overrides)
    return SplitManifest(**base)


# ---------------------------------------------------------------------------
# Acceptance test 1: the harness refuses leaked evaluation data
# ---------------------------------------------------------------------------

def test_harness_refuses_leaked_s2_sessions() -> None:
    """Four ways the same contamination arrives, each caught with its own reason."""
    from praxis.evaluation.shift import assert_no_leakage

    clean = _manifest()
    levels = {ShiftLevel.S2: ["S2a", "S2b"]}
    teachers = {"S2a": "T5", "S2b": "T6"}
    assert_no_leakage(clean, levels, teachers, training_sessions=["Tr1", "Tr2"])

    # 1. the session itself was trained on
    with pytest.raises(ShiftError, match="appear in a training partition"):
        assert_no_leakage(clean, levels, teachers, training_sessions=["S2a"])

    # 2. the session's teacher trained, which a session-level check alone would miss
    with pytest.raises(ShiftError, match="also appear in a training or validation"):
        assert_no_leakage(clean, levels, {"S2a": "T1", "S2b": "T6"})

    # 3. the manifest never declared the session held out
    with pytest.raises(ShiftError, match="not listed in the manifest"):
        assert_no_leakage(_manifest(shift_eval_sessions=["S2a"]), levels, teachers)

    # 4. no teacher recorded at all, so disjointness is uncheckable
    with pytest.raises(ShiftError, match="no teacher recorded"):
        assert_no_leakage(clean, levels, {"S2a": "T5"})


def test_harness_refuses_a_leaked_held_out_site() -> None:
    """S1 measures site shift, which a site the model trained on would erase.

    Was `test_harness_refuses_a_leaked_held_out_college` and asserted "Institutional shift
    cannot be measured". S1 is a held-out basic school now, not a held-out college, so the old
    wording described the wrong rung. The rule it protected is unchanged and still enforced: a
    teacher who contributes a held-out session may not appear in train or val. D98.
    """
    from praxis.evaluation.shift import assert_no_leakage

    with pytest.raises(ShiftError,
                       match=r"contribute S1 sessions and also appear in a training"):
        assert_no_leakage(_manifest(shift_eval_sessions=["S1a", "S2a", "S2b"]),
                          {ShiftLevel.S1: ["S1a"], ShiftLevel.S2: ["S2a", "S2b"]},
                          {"S1a": "T2", "S2a": "T5", "S2b": "T6"})


class TestTheLadderForAPracticumCorpus:
    """S1 is a held-out school and S2 a held-out college, because every session is practicum.

    The old ladder put the shift in the *domain*: S2 was classroom footage and the rule was that
    classroom footage is never trained on. When the whole corpus is classroom that rule either
    refuses every manifest or gets relaxed, and a leakage check relaxed to let the project run
    is the most dangerous object in this codebase. So the shift now lives in the site, and the
    checks are keyed on the site.

    Re-keying revealed that three of the four checks only ever looked at S2. An S1 session could
    sit in a training partition, go undeclared in `shift_eval_sessions`, or have no teacher
    recorded, and the harness would run. Under the old ladder S1 was microteaching like the
    training set and the hole was survivable; under the new one S1 is a held-out school and the
    hole is the leak. Every check now applies at every held-out level. D98.
    """

    def test_an_s1_session_in_a_training_partition_is_refused(self) -> None:
        """The hole. Only S2 was checked, so this passed before D98."""
        from praxis.evaluation.shift import assert_no_leakage

        with pytest.raises(ShiftError, match="appear in a training partition"):
            assert_no_leakage(_manifest(shift_eval_sessions=["S1a", "S2a", "S2b"]),
                              {ShiftLevel.S1: ["S1a"], ShiftLevel.S2: ["S2a", "S2b"]},
                              {"S1a": "T7", "S2a": "T5", "S2b": "T6"},
                              training_sessions=["S1a"])

    def test_an_s1_session_must_be_declared_held_out(self) -> None:
        """Nothing was protecting a session the manifest never named."""
        from praxis.evaluation.shift import assert_no_leakage

        with pytest.raises(ShiftError, match="not listed in the manifest"):
            assert_no_leakage(_manifest(shift_eval_sessions=["S2a", "S2b"]),
                              {ShiftLevel.S1: ["S1a"], ShiftLevel.S2: ["S2a", "S2b"]},
                              {"S1a": "T7", "S2a": "T5", "S2b": "T6"})

    def test_an_s1_session_with_no_teacher_recorded_is_refused(self) -> None:
        """teacher_id is the split key; disjointness cannot be checked without it."""
        from praxis.evaluation.shift import assert_no_leakage

        with pytest.raises(ShiftError, match="no teacher recorded"):
            assert_no_leakage(_manifest(shift_eval_sessions=["S1a", "S2a", "S2b"]),
                              {ShiftLevel.S1: ["S1a"], ShiftLevel.S2: ["S2a", "S2b"]},
                              {"S2a": "T5", "S2b": "T6"})

    def test_a_manifest_that_names_a_held_out_site_cannot_stand_down(self) -> None:
        """D83 let the teacher check stand down when the corpus left nobody to train on. That
        argument belongs to the domain-keyed rule: it was about there being no
        microteaching-only teachers. Once the shift is a held-out school or college, a teacher
        of a held-out session appearing in training is the leak itself, and nothing relaxes
        it."""
        from praxis.evaluation.shift import assert_no_leakage

        declared = _manifest(shift_axis="site", heldout_school=SCHOOL,
                             classroom_teachers_trained=True, shift_eval_sessions=["S1a"])
        with pytest.raises(ShiftError, match="also appear in a training or validation"):
            assert_no_leakage(declared, {ShiftLevel.S1: ["S1a"]}, {"S1a": "T1"})

        held_out_college = _manifest(shift_axis="site", heldout_college=COLLEGE,
                                     classroom_teachers_trained=True,
                                     shift_eval_sessions=["S2a"])
        with pytest.raises(ShiftError, match="also appear in a training or validation"):
            assert_no_leakage(held_out_college, {ShiftLevel.S2: ["S2a"]}, {"S2a": "T2"})

    def test_a_school_on_the_domain_axis_is_refused_rather_than_stored_unmeasured(self) -> None:
        """A held-out school has no rung on the domain ladder, so storing one would say a site
        was held out and measured when nothing did either."""
        with pytest.raises(ValueError, match="no rung on that ladder"):
            _manifest(heldout_school=SCHOOL)

    def test_the_site_axis_must_name_a_site(self) -> None:
        """Both levels would be empty, and an empty level reads as no shift found."""
        with pytest.raises(ValueError, match="S1 and S2 are both empty"):
            _manifest(shift_axis="site")

    def test_the_old_stand_down_still_holds_for_a_manifest_that_names_no_site(self) -> None:
        """D83 is not withdrawn. A manifest from the microteaching design declares no held-out
        school and no held-out college, and reads back behaving exactly as it did."""
        from praxis.evaluation.shift import assert_no_leakage

        old = _manifest(classroom_teachers_trained=True, shift_eval_sessions=["S2a"])
        assert old.heldout_school is None and old.heldout_college is None
        assert_no_leakage(old, {ShiftLevel.S2: ["S2a"]}, {"S2a": "T1"})

    def test_the_levels_name_the_site_they_hold_out(self) -> None:
        """A description that still says "microteaching" would be read as the design."""
        assert "practicum" in ShiftLevel.S0.description
        assert "school" in ShiftLevel.S1.description
        assert "college" in ShiftLevel.S2.description
        assert ShiftLevel.S1.isolates != ShiftLevel.S2.isolates


def test_every_leakage_problem_is_reported_in_one_pass() -> None:
    """Fixing one contamination and rediscovering the next is how a corpus gets
    rebuilt twice."""
    from praxis.evaluation.shift import assert_no_leakage

    with pytest.raises(ShiftError) as raised:
        assert_no_leakage(_manifest(shift_eval_sessions=[]),
                          {ShiftLevel.S2: ["S2a", "S2b"]},
                          {"S2a": "T1", "S2b": "T2"}, training_sessions=["S2a"])
    message = str(raised.value)
    assert "training partition" in message
    # The claim, not the rationale behind it. This check is D73's extra protection rather than
    # R2 itself - R2 is teacher-disjointness across partitions and `SplitManifest` enforces it
    # unconditionally - and asserting the old "R2 forbids" wording pinned a conflation that
    # made it look as though the rule could never be relaxed. D83.
    assert "contribute S2 sessions and also appear in a training" in message
    assert "shift_eval_sessions" in message


# ---------------------------------------------------------------------------
# Acceptance test 3: OOD AUROC against an analytic value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("separation", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_auroc_matches_the_analytic_value(separation: float) -> None:
    """Two unit Gaussians separated by d give AUROC = Phi(d / sqrt(2)), exactly."""
    rng = np.random.default_rng(20260913)
    n = 100_000
    in_distribution = rng.normal(0.0, 1.0, n)
    out_of_distribution = rng.normal(separation, 1.0, n)

    scores = np.concatenate([in_distribution, out_of_distribution])
    flags = np.concatenate([np.zeros(n, bool), np.ones(n, bool)])

    expected = float(norm.cdf(separation / np.sqrt(2.0)))
    assert auroc(scores, flags) == pytest.approx(expected, abs=5e-3)


def test_auroc_handles_ties_by_average_ranks() -> None:
    """A saturated detector scores many clips identically; naive sweeps flatter it.

    All scores equal means the detector separates nothing, so AUROC must be exactly 0.5.
    """
    assert auroc([1.0] * 20, [True] * 10 + [False] * 10) == pytest.approx(0.5)
    assert auroc([0.0, 0.0, 1.0, 1.0], [False, True, False, True]) == pytest.approx(0.5)
    assert auroc([0.0, 0.0, 1.0, 1.0], [False, False, True, True]) == pytest.approx(1.0)
    assert auroc([1.0, 1.0, 0.0, 0.0], [False, False, True, True]) == pytest.approx(0.0)


def test_auroc_refuses_one_sided_data() -> None:
    with pytest.raises(OodError, match="both kinds"):
        auroc([0.1, 0.2, 0.3], [False, False, False])


def test_every_ood_score_points_the_same_way() -> None:
    """Higher means more out-of-distribution, for all three. The sign
    convention is load-bearing.

    Inverting one would turn its AUROC from 0.9 into 0.1, which does not look
    like a bug. It looks like a method that failed, and it would be reported
    as one.
    """
    confident = msp_score([0.99, 0.01])
    uncertain = msp_score([0.5, 0.5])
    assert np.all(uncertain > confident), "MSP must be inverted: uncertain is more OOD"
    assert msp_score([0.5])[0] == pytest.approx(1.0)
    assert msp_score([1.0])[0] == pytest.approx(0.0)

    agreeing = disagreement_score([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    disagreeing = disagreement_score([[0.99, 0.5], [0.01, 0.5], [0.5, 0.5]])
    assert np.all(disagreeing[:1] > agreeing[:1])
    assert np.all((disagreement_score([[0.99], [0.01]]) >= 0.0)
                  & (disagreement_score([[0.99], [0.01]]) <= 1.0))


def test_mahalanobis_scores_unfamiliar_features_higher() -> None:
    """The feature-space method sees a room unlike anything in training, however confident."""
    rng = np.random.default_rng(5)
    features = np.concatenate([rng.normal(0, 1, (400, 8)),
                               rng.normal(4, 1, (400, 8))])
    labels = np.array([0] * 400 + [1] * 400)

    detector = fit_mahalanobis(features, labels, shrinkage=0.10)
    assert detector.n_features == 8 and detector.classes == (0, 1)

    near = detector.score(rng.normal(0, 1, (200, 8)))
    far = detector.score(rng.normal(12, 1, (200, 8)))
    assert far.mean() > near.mean() * 5

    evaluation = evaluate_ood(np.concatenate([near, far]),
                              np.array([False] * 200 + [True] * 200), "mahalanobis")
    assert evaluation.auroc > 0.99


def test_mahalanobis_refuses_what_it_cannot_estimate() -> None:
    with pytest.raises(OodError, match="single example"):
        fit_mahalanobis(np.zeros((2, 4)), [0, 1])
    with pytest.raises(OodError, match=r"shrinkage must lie"):
        fit_mahalanobis(np.zeros((6, 2)), [0, 0, 0, 1, 1, 1], shrinkage=2.0)
    detector = fit_mahalanobis(np.random.default_rng(1).normal(size=(40, 5)),
                               [0] * 20 + [1] * 20)
    with pytest.raises(OodError, match="dimensional but the detector"):
        detector.score(np.zeros((3, 9)))


def test_fpr_at_95_tpr_is_the_operating_cost() -> None:
    """Perfect separation costs nothing; no separation costs almost everything."""
    separated = np.concatenate([np.zeros(1000), np.ones(1000)])
    flags = np.array([False] * 1000 + [True] * 1000)
    fpr, _ = fpr_at_tpr(separated, flags, 0.95)
    assert fpr == pytest.approx(0.0, abs=1e-9)

    rng = np.random.default_rng(2)
    overlapping = rng.normal(0, 1, 2000)
    fpr, _ = fpr_at_tpr(overlapping, flags, 0.95)
    assert fpr > 0.80


def test_flag_rate_is_reported_per_domain_which_is_h5() -> None:
    """H5 asks whether classroom footage is flagged more than held-out microteaching."""
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
    domains = ["microteaching"] * 3 + ["classroom"] * 3
    rates = flag_rate_by_domain(scores, domains, threshold=0.5)

    assert rates == {"classroom": 1.0, "microteaching": 0.0}
    assert rates["classroom"] > rates["microteaching"]


# ---------------------------------------------------------------------------
# Acceptance test 2: attribution reports intervals
# ---------------------------------------------------------------------------

def _sessions(n_teachers: int = 12, per_teacher: int = 4,
              seed: int = 3) -> list[dict[str, object]]:
    """Synthetic sessions where the drop genuinely depends on camera distance
    and nothing else."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for teacher in range(n_teachers):
        offset = rng.normal(0, 0.02)          # a per-teacher random intercept
        for _ in range(per_teacher):
            distance = rng.uniform(1.5, 8.0)
            rows.append({
                "teacher_id": f"T{teacher}",
                "camera_distance_m": distance,
                "room_area_m2": rng.uniform(20, 80),
                "pupil_count": int(rng.integers(15, 60)),
                "ambient_noise_dba": rng.uniform(40, 75),
                "teacher_movement_range_m": rng.uniform(0.5, 6.0),
                # the drop rises with camera distance and with nothing else
                "drop": 0.02 * distance + offset + rng.normal(0, 0.01),
            })
    return rows


def test_attribution_reports_intervals_not_point_estimates() -> None:
    """Every coefficient carries an interval, and the real driver is identified."""
    result = attribute_degradation(_sessions(), COVARIATES)

    assert result.n_sessions == 48 and result.n_teachers == 12
    assert result.coefficients, "a fitted model must report its coefficients"
    for coefficient in result.coefficients:
        assert coefficient.ci_lower <= coefficient.estimate <= coefficient.ci_upper
        assert coefficient.ci_lower < coefficient.ci_upper, "an interval, not a point"

    assert "camera_distance_m" in result.attributable_to
    assert result.named("camera_distance_m").estimate > 0
    assert "pupil_count" not in result.attributable_to


def test_attribution_drops_sessions_missing_covariates_rather_than_imputing() -> None:
    """The covariates are "all measured, none assumed"; imputing invents the measurement."""
    rows = _sessions()
    rows[0]["camera_distance_m"] = None
    rows[1]["ambient_noise_dba"] = None

    result = attribute_degradation(rows, COVARIATES)
    assert result.n_dropped_missing_covariates == 2
    assert result.n_sessions == len(rows) - 2


def test_attribution_refuses_when_too_little_survives() -> None:
    rows = _sessions(n_teachers=2, per_teacher=1)
    with pytest.raises(ShiftError, match="cannot support"):
        attribute_degradation(rows, COVARIATES)
    with pytest.raises(ShiftError, match="nothing to attribute"):
        attribute_degradation(_sessions(), [])


def test_attribution_falls_back_audibly_when_a_mixed_model_will_not_fit() -> None:
    """One teacher gives no between-teacher variance, so OLS is used and the loss is stated."""
    rows = _sessions(n_teachers=1, per_teacher=30)
    result = attribute_degradation(rows, COVARIATES)

    assert result.model == "ols_fallback"
    assert any("understates the standard errors" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# The headline table
# ---------------------------------------------------------------------------

def _result(label: str, accuracy: float, ece: float, n: int = 2000, seed: int = 1):
    """An evaluation with a known accuracy and a known ECE.

    Every clip is predicted at one of two confidences, `c` or `1 - c`, and is right with
    probability `accuracy`. The bin at `c` then holds a fraction `accuracy` of positives, so
    the calibration gap is exactly `|accuracy - c|` in both bins and ECE is that number. Setting
    `c = accuracy + ece` therefore asks for the ECE directly rather than hoping for it.
    """
    rng = np.random.default_rng(seed)
    confidence = min(accuracy + ece, 0.99)
    truth = rng.binomial(1, 0.5, n).astype(float)
    correct = rng.random(n) < accuracy
    high = (truth == 1) == correct
    probs = np.where(high, confidence, 1.0 - confidence)
    return evaluate({"B1": (probs, truth)}, label=label, bootstrap=0)


def test_shift_table_reports_accuracy_and_ece_at_every_level() -> None:
    """R4 by the shape of the table: neither number appears at a level without the other."""
    table = build_shift_table({
        "temperature": {ShiftLevel.S0: _result("S0", 0.85, 0.05, seed=1),
                        ShiftLevel.S1: _result("S1", 0.80, 0.12, seed=2),
                        ShiftLevel.S2: _result("S2", 0.76, 0.28, seed=3)},
        "ensemble": {ShiftLevel.S0: _result("S0", 0.84, 0.04, seed=4),
                     ShiftLevel.S1: _result("S1", 0.81, 0.07, seed=5),
                     ShiftLevel.S2: _result("S2", 0.77, 0.11, seed=6)}})

    rows = table.as_table()
    assert len(rows) == 6
    assert all(row["accuracy"] is not None and row["ece"] is not None for row in rows)
    assert all(row["isolates"] for row in rows)

    s0 = [r for r in table.rows if r.level is ShiftLevel.S0]
    assert all(r.accuracy_drop is None for r in s0), "the baseline has nothing to drop from"


def test_degradation_ratio_expresses_h4() -> None:
    """H4: ECE degrades proportionally faster than accuracy. The ratio is the claim."""
    table = build_shift_table({
        "temperature": {ShiftLevel.S0: _result("S0", 0.85, 0.04, seed=1),
                        ShiftLevel.S2: _result("S2", 0.80, 0.18, seed=3)}})

    s0, s2 = table.for_method("temperature")
    assert s0.accuracy == pytest.approx(0.85, abs=0.03)
    assert s0.ece == pytest.approx(0.04, abs=0.03)
    assert s2.ece == pytest.approx(0.18, abs=0.03)

    ratio = table.degradation_ratio("temperature")
    assert ratio is not None and ratio > 1.0, (
        "accuracy fell 6 per cent and ECE rose several-fold; that gap is the H4 claim")

    degradation = table.degradation("temperature")
    assert degradation.ratio == ratio
    assert degradation.accuracy_fall == pytest.approx(s0.accuracy - s2.accuracy)
    assert degradation.ece_rise == pytest.approx(s2.ece - s0.ece)
    assert degradation.stable is True
    assert "absolute" in str(degradation), (
        "the absolute rise must travel with the ratio, which overstates the effect alone")


def test_a_ratio_off_a_near_zero_baseline_is_marked_unstable() -> None:
    """A well-calibrated S0 makes the denominator tiny and the ratio enormous. Say so."""
    table = build_shift_table({
        "ensemble": {ShiftLevel.S0: _result("S0", 0.90, 0.002, n=8000, seed=11),
                     ShiftLevel.S2: _result("S2", 0.88, 0.150, n=8000, seed=12)}})

    degradation = table.degradation("ensemble")
    assert degradation.stable is False
    assert "unstable" in str(degradation)
    assert degradation.ratio > 50, "which is the number that must not be quoted bare"


def test_no_ratio_is_reported_when_accuracy_did_not_fall() -> None:
    """The strongest version of H4 divides by zero, so it is reported in words instead."""
    table = build_shift_table({
        "ensemble": {ShiftLevel.S0: _result("S0", 0.80, 0.02, seed=21),
                     ShiftLevel.S2: _result("S2", 0.80, 0.20, seed=21)}})

    degradation = table.degradation("ensemble")
    assert degradation.ratio is None
    assert degradation.ece_rise > 0.1
    assert "accuracy did not fall" in str(degradation)
    assert table.degradation_ratio("ensemble") is None
    assert table.degradation("temperature") is None


def test_shift_table_refuses_a_method_without_a_baseline() -> None:
    with pytest.raises(ShiftError, match="no S0 baseline"):
        build_shift_table({"ensemble": {ShiftLevel.S2: _result("S2", 0.7, 0.2)}})


# ---------------------------------------------------------------------------
# Acceptance test 4: the sweep is a table, not a number
# ---------------------------------------------------------------------------

def _abstention_fixture(n: int = 2000, seed: int = 9):
    """Familiar clips predicted well; unfamiliar ones predicted badly and scored as OOD."""
    rng = np.random.default_rng(seed)
    familiar = int(n * 0.75)
    truth = rng.binomial(1, 0.5, n).astype(float)

    probs = np.empty(n)
    probs[:familiar] = np.where(truth[:familiar] == 1,
                                rng.uniform(0.75, 0.99, familiar),
                                rng.uniform(0.01, 0.25, familiar))
    probs[familiar:] = rng.uniform(0.3, 0.7, n - familiar)

    scores = np.concatenate([rng.uniform(0.0, 0.35, familiar),
                             rng.uniform(0.45, 1.0, n - familiar)])
    return scores, probs, truth


def test_threshold_sweep_produces_a_table() -> None:
    """A table of thresholds with their cost, not a single number handed down."""
    scores, probs, truth = _abstention_fixture()
    sweep = threshold_sweep(scores, probs, truth, steps=101)

    assert len(sweep) == 101
    rates = [row.suppression_rate for row in sweep]
    assert rates == sorted(rates, reverse=True), (
        "a higher threshold suppresses less, so the rate must fall monotonically")
    assert rates[0] == pytest.approx(1.0), "a threshold of 0 suppresses everything"
    assert rates[-1] < 0.05, "a threshold of 1 suppresses almost nothing"

    scored = [row for row in sweep if row.f1_on_presented is not None]
    assert len(scored) > 50
    assert all(row.accuracy_on_presented is not None for row in scored)
    assert all(row.ece_on_presented is not None for row in scored)


def test_threshold_is_chosen_by_a_stated_criterion() -> None:
    """Not by eye. The criterion is named, applied, and reported with its justification."""
    scores, probs, truth = _abstention_fixture()
    choice = choose_threshold(threshold_sweep(scores, probs, truth), max_suppression=0.30)

    assert choice.threshold is not None
    assert choice.row.suppression_rate <= 0.30
    assert "max_f1_at_suppression_rate_below_0.30" in choice.criterion
    assert "highest" in choice.justification and "suppressing" in choice.justification
    assert len(choice.as_table()) == 101, "the whole sweep travels with the choice"

    eligible = [r for r in choice.sweep
                if r.suppression_rate <= 0.30 and r.f1_on_presented is not None]
    assert choice.row.f1_on_presented == max(r.f1_on_presented for r in eligible)


def test_the_criterion_picks_the_best_eligible_row_not_the_first() -> None:
    """Hand-built rather than sampled, because a fixture can make the two coincide by luck.

    Added after a mutation test: replacing the selection with `eligible[0]` left the suite
    green, since on the random fixture the first eligible threshold happened to be the best.
    """
    sweep = [SweepRow(0.0, 1.00, 0, None, None, None),
             SweepRow(0.2, 0.45, 550, 0.90, 0.05, 0.90),      # best F1, but over the ceiling
             SweepRow(0.4, 0.28, 720, 0.82, 0.06, 0.82),      # first eligible, and not the best
             SweepRow(0.6, 0.20, 800, 0.86, 0.06, 0.86),      # the answer
             SweepRow(0.8, 0.10, 900, 0.80, 0.07, 0.80),
             SweepRow(1.0, 0.00, 1000, 0.78, 0.08, 0.78)]

    choice = choose_threshold(sweep)
    assert choice.threshold == 0.6
    assert choice.row.suppression_rate <= 0.30, "the ceiling excludes the 0.90 row entirely"
    assert choice.frugal.threshold == 0.6, "nothing cheaper comes within 0.01 of it"
    assert choice.suppression_bought_little is False


def test_a_flat_f1_curve_is_named_rather_than_silently_obeyed() -> None:
    """Found by running it: "maximise F1 under a ceiling" treats suppression as free.

    When F1 barely moves, the rule spends a fifth of the corpus for a fraction of a point,
    because nothing in it says suppression costs anything. The criterion still decides — a rule
    changed after seeing the curve is no rule — but the cheap alternative is reported beside it.
    """
    # built deterministically, because the effect being tested is smaller than the sampling
    # noise of a random fixture: the error rate rises from 0.25 to 0.29 across the score range,
    # so suppressing the whole ceiling buys six-tenths of an F1 point and no more
    n = 4000
    scores = np.linspace(0.0, 1.0, n)
    truth = (np.arange(n) % 2).astype(float)
    wrong = np.modf(scores * 997.0)[0] < 0.25 + 0.04 * scores
    probs = np.where(np.where(wrong, 1.0 - truth, truth) == 1, 0.9, 0.1)

    choice = choose_threshold(threshold_sweep(scores, probs, truth))

    assert choice.suppression_bought_little is True
    assert choice.frugal.suppression_rate < choice.row.suppression_rate
    assert "F1 is nearly flat here" in choice.justification
    assert choice.threshold == choice.row.threshold, (
        "the stated criterion still chooses; the note is information, not an override")


def test_a_genuinely_useful_threshold_is_not_flagged_as_flat() -> None:
    """The converse, so the warning means something when it appears."""
    scores, probs, truth = _abstention_fixture()
    choice = choose_threshold(threshold_sweep(scores, probs, truth))

    assert choice.suppression_bought_little is False
    assert "nearly flat" not in choice.justification


def test_an_impossible_ceiling_is_reported_rather_than_fudged() -> None:
    """If nothing meets the criterion, say so; do not return the least bad option quietly."""
    scores = np.ones(500)                     # a detector that flags every clip it is shown
    probs = np.random.default_rng(1).uniform(0, 1, 500)
    truth = np.random.default_rng(2).binomial(1, 0.5, 500).astype(float)

    choice = choose_threshold(threshold_sweep(scores, probs, truth), max_suppression=0.01)
    assert choice.threshold is None
    assert "no threshold" in choice.justification
    assert choice.sweep, "the sweep is still reported so the failure is inspectable"


# ---------------------------------------------------------------------------
# R4 at the return type
# ---------------------------------------------------------------------------

def test_an_evaluation_cannot_be_constructed_without_calibration() -> None:
    """The actual structural claim: the field is required, so no evaluation can omit ECE.

    Added after a mutation test. Making `calibration` optional with a `None` default left the
    suite green, because `evaluate_behaviour` still passed one. R4 rests on the field being
    required, and nothing was asserting that.
    """
    with pytest.raises(TypeError, match="calibration"):
        BehaviourEvaluation(behaviour="B1", n=10, accuracy=0.9, macro_f1=0.9,
                            confusion=ConfusionStructure(5, 1, 3, 1), recalls=())


def test_accuracy_cannot_be_obtained_without_calibration() -> None:
    """R4 enforced structurally: there is no accessor that returns one without the other."""
    rng = np.random.default_rng(7)
    truth = rng.binomial(1, 0.5, 400).astype(float)
    probs = np.clip(truth * 0.7 + rng.normal(0, 0.2, 400), 0.01, 0.99)

    evaluation = evaluate_behaviour("B1", probs, truth, bootstrap=50)
    assert evaluation.calibration is not None
    assert evaluation.ece == evaluation.calibration.ece

    result = evaluate({"B1": (probs, truth)}, bootstrap=0)
    assert result.macro_accuracy is not None and result.macro_ece is not None
    assert all("ece" in row for row in result.as_table())

    for recall in evaluation.recalls:
        assert recall.lower is not None and recall.upper is not None
        assert recall.lower <= recall.point <= recall.upper


def test_human_band_is_reported_beside_accuracy_and_not_compared_to_it() -> None:
    """D6 withdrew the absolute accuracy target, so the human band is the context a model number
    is read in. D97 then settled what that context is allowed to be: both numbers, named, and no
    verdict. The band is a chance-corrected coefficient and accuracy is a raw proportion, so a
    boolean saying the model "reached" the band would be comparing quantities on different
    scales, and reporting that as a result is how a thesis claims what it cannot defend."""
    rng = np.random.default_rng(8)
    truth = rng.binomial(1, 0.5, 300).astype(float)
    probs = np.clip(truth * 0.8 + rng.normal(0, 0.25, 300), 0.01, 0.99)

    stated = evaluate_behaviour("B1", probs, truth, human_band=(0.70, 0.85),
                                human_band_statistic="krippendorff_alpha", bootstrap=0)
    reported = stated.against_humans
    assert reported["accuracy"] == stated.accuracy
    assert reported["human_band"] == [0.70, 0.85]
    assert reported["statistic"] == "krippendorff_alpha"
    assert reported["commensurable"] is False

    # No key anywhere in the output answers "did it reach the band".
    assert not any(isinstance(v, bool) and v is True for v in reported.values())

    # A band with no named coefficient is refused rather than reported uninterpretably.
    with pytest.raises(ValueError, match="which coefficient produced it"):
        evaluate_behaviour("B1", probs, truth, human_band=(0.70, 0.85), bootstrap=0)

    # No band measured is an abstention, not a failure. D48.
    assert evaluate_behaviour("B1", probs, truth, bootstrap=0).against_humans is None


def test_the_table_carries_the_band_endpoints_and_no_verdict() -> None:
    """A verdict column would be read as the result whatever the prose beside it said."""
    rng = np.random.default_rng(9)
    truth = rng.binomial(1, 0.5, 200).astype(float)
    probs = np.clip(truth * 0.8 + rng.normal(0, 0.25, 200), 0.01, 0.99)

    result = evaluate({"B1": (probs, truth)}, human_bands={"B1": (0.70, 0.85)},
                      human_band_statistic="quadratic_kappa", bootstrap=0)
    row = result.as_table()[0]
    assert (row["human_band_low"], row["human_band_high"]) == (0.7, 0.85)
    assert row["human_band_statistic"] == "quadratic_kappa"
    assert "within_human_band" not in row


class TestAskingBeforeAttributing:
    """D86. With around eleven sessions and lesson plans whose covariate fields are
    inconsistently filled, attribution being unavailable is the expected path rather than the
    exceptional one. `attribute_degradation` still raises - it is asked for a regression and
    will not produce a meaningless one - but a Phase 6 report needs to say "attribution
    abstained, and here is why" without wrapping the call in the one `except` that also
    swallows real failures. D48: abstention is its own verdict.
    """

    def test_a_workable_corpus_reports_no_obstacle(self) -> None:
        assert attribution_feasibility(_sessions(), COVARIATES) is None

    def test_too_few_complete_sessions_is_named_rather_than_raised(self) -> None:
        reason = attribution_feasibility(_sessions(n_teachers=2, per_teacher=1), COVARIATES)
        assert reason is not None
        assert "sessions carry every covariate" in reason

    def test_the_reason_counts_what_was_dropped(self) -> None:
        """An operator reading it needs to know whether to chase missing measurements or
        accept that the corpus is too small."""
        rows = _sessions(n_teachers=3, per_teacher=1)
        rows[0]["camera_distance_m"] = None
        reason = attribution_feasibility(rows, COVARIATES)
        assert "dropped for missing measurements" in reason

    def test_no_covariates_is_its_own_reason(self) -> None:
        assert "nothing to attribute" in attribution_feasibility(_sessions(), [])

    def test_it_agrees_with_what_the_regression_actually_does(self) -> None:
        """The two must never disagree: a caller told attribution is feasible and then handed
        a ShiftError has been told the opposite of the truth."""
        for rows, covariates in ((_sessions(), COVARIATES),
                                 (_sessions(n_teachers=2, per_teacher=1), COVARIATES),
                                 (_sessions(), [])):
            feasible = attribution_feasibility(rows, covariates) is None
            try:
                attribute_degradation(rows, covariates)
                raised = False
            except ShiftError:
                raised = True
            assert feasible is not raised
