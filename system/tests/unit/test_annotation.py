# -*- coding: utf-8 -*-
"""Phase 2 acceptance tests: the agreement statistics, and the codebook's authority.

The four BUILD-SPEC Phase 2 acceptance tests are `test_alpha_matches_analytic_value`,
`test_not_fully_crossed_warning_fires`, `test_codebook_refuses_undefined_label`, and
`test_every_annotation_records_its_codebook_version`. The rest guard the arithmetic around them.

**Where the expected numbers come from.** Two sources, both independent of this code. The
alpha in `test_alpha_matches_analytic_value` is derived by hand in the test body, so a reader
can check it without running anything. The golden values in `test_alpha_matches_reference_*`
were cross-checked on 2026-09-13 against the `krippendorff` package across all four scale
types and agreed to machine precision; they are pinned here rather than recomputed so the
suite carries no dependency on that package. ICC is checked against Shrout and Fleiss (1979).
"""
from __future__ import annotations

import pytest

from praxis.annotation.codebook import (
    ACTIVE_CODEBOOK,
    CODEBOOK_V1,
    CodebookError,
    ScaleType,
)
from praxis.annotation.irr import Annotation, Rater, report
from praxis.annotation.statistics import (
    bootstrap_interval,
    icc_2k,
    krippendorff_alpha,
    mean_pairwise_kappa,
    quadratic_weighted_kappa,
)

# Cross-checked against the `krippendorff` package, 2026-09-13. See the module docstring.
CANONICAL = [
    [None, 1, None], [None, None, None], [None, 2, 2], [None, 1, 1], [None, 3, 3],
    [3, 3, 4], [4, 4, 4], [1, 3, None], [2, None, 2], [1, None, 1],
    [1, None, 1], [3, None, 3], [3, None, 3], [None, None, None], [3, None, 4],
]
CANONICAL_ALPHA = {
    ScaleType.NOMINAL: 0.6913580247,
    ScaleType.ORDINAL: 0.8067214199,
    ScaleType.INTERVAL: 0.8108448928,
    ScaleType.RATIO: 0.8089436708,
}

SHROUT_FLEISS = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                 [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]


# ---------------------------------------------------------------------------
# Acceptance test 1: alpha against a known value
# ---------------------------------------------------------------------------

def test_alpha_matches_analytic_value() -> None:
    """Alpha on a fixture whose value is derived by hand, to four decimal places.

    Two raters, four units, binary. Each unit has m=2 values, so each contributes 1/(m-1)=1
    to each of its two ordered pairs:

        units (0,0), (0,0), (1,1), (1,0)
        coincidences  o[0,0]=4  o[1,1]=2  o[0,1]=1  o[1,0]=1
        marginals     n_0 = 5   n_1 = 3   n = 8
        D_o  = off-diagonal sum                    = 2
        D_e  = sum of n_c n_k over c != k          = 5*3 + 3*5 = 30
        alpha = 1 - (n-1) D_o / D_e = 1 - 7*2/30   = 0.5333...
    """
    units = [[0, 0], [0, 0], [1, 1], [1, 0]]
    assert krippendorff_alpha(units, ScaleType.NOMINAL) == pytest.approx(1 - 14 / 30, abs=1e-10)
    assert round(krippendorff_alpha(units, ScaleType.NOMINAL), 4) == 0.5333


def test_alpha_is_one_on_perfect_agreement() -> None:
    units = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1], [3, 3, 3]]
    for scale in ScaleType:
        assert krippendorff_alpha(units, scale) == pytest.approx(1.0, abs=1e-12), scale


@pytest.mark.parametrize("scale", list(ScaleType))
def test_alpha_matches_reference_implementation(scale: ScaleType) -> None:
    """All four scale types against values cross-checked with the `krippendorff` package."""
    assert krippendorff_alpha(CANONICAL, scale) == pytest.approx(
        CANONICAL_ALPHA[scale], abs=1e-9)


def test_alpha_is_none_where_undefined() -> None:
    """Undefined is reported as undefined, never as 1.0.

    Every rater using one category throughout gives zero expected disagreement. Returning
    1.0 would claim perfect agreement on an instrument that never made a distinction.
    """
    assert krippendorff_alpha([[1, 1], [1, 1], [1, 1]]) is None
    assert krippendorff_alpha([[1, None], [2, None]]) is None
    assert krippendorff_alpha([]) is None


# ---------------------------------------------------------------------------
# Acceptance test 2: the not-fully-crossed warning
# ---------------------------------------------------------------------------

def test_not_fully_crossed_warning_fires() -> None:
    """ICC underestimates when the design is unbalanced, so it must say so.

    The fixture is deliberately unbalanced: the third rater skipped two clips.
    """
    unbalanced = [[0.25, 0.25, 0.5], [0.5, 0.5, None], [0.75, 0.5, 0.75],
                  [1.0, 0.75, None], [0.0, 0.25, 0.0]]
    result = icc_2k(unbalanced)

    assert result.fully_crossed is False
    assert result.units_dropped == 2
    assert result.warning is not None
    assert "not fully crossed" in result.warning
    assert "underestimates" in result.warning

    balanced = [[0.25, 0.25, 0.5], [0.75, 0.5, 0.75], [0.0, 0.25, 0.0], [1.0, 0.75, 1.0]]
    assert icc_2k(balanced).fully_crossed is True
    assert icc_2k(balanced).warning is None


def test_report_flags_an_uncrossed_design() -> None:
    annotations = [
        Annotation("c1", "r1", "B1", "v1.0-draft", {"b1_present": True}),
        Annotation("c1", "r2", "B1", "v1.0-draft", {"b1_present": True}),
        Annotation("c2", "r1", "B1", "v1.0-draft", {"b1_present": False}),
        # r2 never saw c2, so the design is not fully crossed.
    ]
    result = report(annotations, bootstrap=0)
    assert result.design_fully_crossed is False
    assert any("not fully crossed" in w for w in result.warnings)


def test_icc_matches_shrout_and_fleiss() -> None:
    """ICC(2,k) against the worked example in Shrout and Fleiss (1979), Table 1."""
    result = icc_2k(SHROUT_FLEISS)
    assert result.fully_crossed is True
    assert result.n_units == 6 and result.n_raters == 4
    assert result.icc == pytest.approx(0.6200505476, abs=1e-9)


# ---------------------------------------------------------------------------
# Acceptance test 3: the codebook is authoritative
# ---------------------------------------------------------------------------

def test_codebook_refuses_undefined_label() -> None:
    """A label the loaded version does not define is refused, naming the offending key."""
    with pytest.raises(CodebookError, match="b1_enthusiasm"):
        ACTIVE_CODEBOOK.validate_labels("B1", {"b1_enthusiasm": 3})

    with pytest.raises(CodebookError, match="b1_amplitude"):
        ACTIVE_CODEBOOK.validate_labels("B1", {"b1_amplitude": 9})

    with pytest.raises(CodebookError, match="b2_dominant"):
        ACTIVE_CODEBOOK.validate_labels("B2", {"b2_dominant": "ceiling"})

    with pytest.raises(CodebookError, match="b1_count"):
        ACTIVE_CODEBOOK.validate_labels("B1", {"b1_count": 21})

    ACTIVE_CODEBOOK.validate_labels("B1", {"b1_present": True, "b1_count": 3,
                                           "b1_amplitude": 2})


def test_codebook_excludes_inferred_constructs() -> None:
    """§0 forbids emotion, engagement, quality and intent. There is nowhere to record them.

    D4 removed facial expressiveness and D7 removed the openness scale. This asserts the
    absence stays an absence, because the tool can only emit what the codebook defines.
    """
    forbidden = ("emotion", "engagement", "enthusiasm", "confidence_level", "warmth",
                 "openness", "quality", "effective", "intent", "attention")
    for spec in ACTIVE_CODEBOOK.behaviours:
        for name in spec.field_names:
            for word in forbidden:
                assert word not in name.lower(), f"{spec.behaviour}.{name} codes {word!r}"


def test_every_behaviour_declares_a_scale_per_field() -> None:
    """The fourth transplant change: a statistic is chosen per field, so every field has one."""
    seen = set()
    for spec in ACTIVE_CODEBOOK.behaviours:
        assert spec.fields, f"{spec.behaviour} declares no fields"
        for field in spec.fields:
            assert isinstance(field.scale, ScaleType)
            seen.add(field.scale)
    # All four scale types really are in use; if they were not, one statistic would do.
    assert seen == set(ScaleType), f"only {sorted(s.value for s in seen)} in use"


# ---------------------------------------------------------------------------
# Acceptance test 4: version discipline
# ---------------------------------------------------------------------------

def test_every_annotation_records_its_codebook_version() -> None:
    """A row cannot exist without one, and versions are never pooled."""
    with pytest.raises(TypeError):
        Annotation("c1", "r1", "B1", labels={"b1_present": True})  # type: ignore[call-arg]

    mixed = [
        Annotation("c1", "r1", "B1", "v1.0-draft", {"b1_present": True}),
        Annotation("c1", "r2", "B1", "v2.0", {"b1_present": False}),
    ]
    with pytest.raises(ValueError, match="not comparable across versions"):
        report(mixed, bootstrap=0)


def test_codebook_binds_to_the_document_hash(repo_root) -> None:
    """A revision to the prose changes the hash even when the field set is identical."""
    bound = CODEBOOK_V1.with_document_hash(repo_root / "docs" / "CODEBOOK.md")
    assert bound.document_sha256 is not None
    assert len(bound.document_sha256) == 64
    assert bound.version == CODEBOOK_V1.version


# ---------------------------------------------------------------------------
# The other estimators
# ---------------------------------------------------------------------------

def test_quadratic_weighted_kappa_matches_sklearn() -> None:
    left, right = [1, 2, 3, 2, 1, 3, 2, 2, 1, 3], [1, 2, 3, 3, 1, 2, 2, 1, 1, 3]
    assert quadratic_weighted_kappa(left, right, (1, 2, 3)) == pytest.approx(
        0.7692307692, abs=1e-9)

    with pytest.raises(ValueError, match="not among the declared levels"):
        quadratic_weighted_kappa([1, 5], [1, 2], (1, 2, 3))


def test_declared_levels_fix_the_spacing_not_merely_the_domain() -> None:
    """Why the codebook's levels are used rather than the observed values.

    A band nobody used but which sits *between* two used bands pushes them apart, and that
    changes the weights. So taking the domain from the data would make agreement depend on
    which bands happened to appear, which is exactly the dependence declaring it removes.

    A band nobody used at the *end* of the scale has zero marginal in both the observed and
    the expected matrix, so it contributes to neither and the value is unchanged. Both facts
    are pinned here because the first is the reason for the design and the second is the
    thing a reader is likely to assume wrongly.
    """
    left = [1, 1, 2, 2, 4, 4, 1, 2, 4, 1]
    right = [1, 2, 2, 4, 4, 1, 1, 2, 4, 2]

    contiguous = quadratic_weighted_kappa(left, right, (1, 2, 4))
    with_interior_gap = quadratic_weighted_kappa(left, right, (1, 2, 3, 4))
    assert contiguous == pytest.approx(0.4615384615, abs=1e-9)
    assert with_interior_gap == pytest.approx(0.4966442953, abs=1e-9)
    assert contiguous != with_interior_gap

    trailing = quadratic_weighted_kappa(left, right, (1, 2, 4, 9))
    assert trailing == pytest.approx(contiguous, abs=1e-12)


def test_mean_pairwise_kappa_tolerates_a_varying_roster() -> None:
    units = [[1, 2, None], [2, 2, 3], [3, 3, 3], [1, 1, 2], [2, None, 2]]
    value = mean_pairwise_kappa(units, (1, 2, 3))
    assert value is not None and -1.0 <= value <= 1.0


def test_bootstrap_interval_brackets_its_point_and_is_deterministic() -> None:
    units = [[1, 1], [1, 2], [2, 2], [2, 2], [1, 1], [2, 1], [1, 1], [2, 2]]

    def estimator(u):
        return krippendorff_alpha(u, ScaleType.NOMINAL)

    first = bootstrap_interval(units, estimator, draws=200, seed=7)
    second = bootstrap_interval(units, estimator, draws=200, seed=7)

    assert first is not None
    assert first.lower <= first.point <= first.upper
    assert (first.lower, first.upper) == (second.lower, second.upper), "R7: seeded, so stable"
    assert bootstrap_interval(units, estimator, draws=200, seed=8) != first


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _synthetic_annotations(n_clips: int = 12) -> list[Annotation]:
    """Two raters who mostly agree, over every behaviour and field."""
    rows: list[Annotation] = []
    for index in range(n_clips):
        disagree = index % 4 == 0
        for rater, shift in (("r1", 0), ("r2", 1 if disagree else 0)):
            rows.append(Annotation(
                f"c{index}", rater, "B1", "v1.0-draft",
                {"b1_present": True, "b1_count": 3 + shift,
                 "b1_amplitude": 1 + (index + shift) % 3, "b1_nonscorable": False},
                session_college_id="COL-A"))
            rows.append(Annotation(
                f"c{index}", rater, "B2", "v1.0-draft",
                {"b2_facing_proportion": [0.0, 0.25, 0.5, 0.75, 1.0][(index + shift) % 5],
                 "b2_head_torso_divergence": 1 + (index + shift) % 3,
                 "b2_dominant": ["class", "board", "materials", "away"][(index + shift) % 4],
                 "b2_nonscorable": False},
                session_college_id="COL-A"))
    return rows


def test_report_produces_the_per_behaviour_table_with_intervals() -> None:
    """Phase 2's definition of done: the table H1 is tested against."""
    result = report(_synthetic_annotations(),
                    raters=[Rater("r1", "tutor", "COL-A"), Rater("r2", "lecturer", "COL-B")],
                    bootstrap=100)

    assert result.codebook_version == "v1.0-draft"
    assert result.n_raters == 2 and result.n_clips == 12
    assert len(result.behaviours) == 5

    b1 = result.behaviour("B1")
    assert b1.name == "Gesture production"
    assert {f.field for f in b1.fields} == {
        "b1_present", "b1_count", "b1_amplitude", "b1_nonscorable"}

    amplitude = next(f for f in b1.fields if f.field == "b1_amplitude")
    assert amplitude.scale is ScaleType.ORDINAL
    assert amplitude.primary_statistic == "weighted_kappa_quadratic"
    assert amplitude.alpha is not None
    assert amplitude.alpha.lower <= amplitude.alpha.point <= amplitude.alpha.upper

    facing = next(f for f in result.behaviour("B2").fields
                  if f.field == "b2_facing_proportion")
    assert facing.scale is ScaleType.INTERVAL
    assert facing.primary_statistic == "icc_2k"
    assert facing.icc is not None

    rows = result.as_table()
    assert len(rows) == sum(len(b.fields) for b in result.behaviours)
    assert all("alpha_ci_low" in row and "meets_alpha_gate" in row for row in rows)


def test_report_excludes_guesses_and_counts_them() -> None:
    """A ground truth built partly from guesses is not a ground truth (§1)."""
    rows = _synthetic_annotations(6)
    rows.append(Annotation("c99", "r1", "B1", "v1.0-draft",
                           {"b1_present": True}, rater_confidence="guess"))
    rows.append(Annotation("c99", "r2", "B1", "v1.0-draft",
                           {"b1_present": False}, rater_confidence="guess"))

    excluded = report(rows, bootstrap=0)
    assert excluded.excluded_guesses == 2
    assert excluded.n_clips == 6, "the guessed clip contributed nothing and is gone"

    included = report(rows, bootstrap=0, exclude_guesses=False)
    assert included.excluded_guesses == 0
    assert included.n_clips == 7


def test_restriction_of_range_is_reported_even_when_unflattering() -> None:
    """A field where every label is the same value produces meaningless agreement."""
    rows = [Annotation(f"c{i}", rater, "B1", "v1.0-draft", {"b1_present": True})
            for i in range(10) for rater in ("r1", "r2")]
    result = report(rows, bootstrap=0)

    check = result.artefacts.restriction_of_range["B1.b1_present"]
    assert check.proportion == 1.0 and check.n == 20
    assert any("nothing to disagree about" in w for w in result.artefacts.warnings)

    present = next(f for f in result.behaviour("B1").fields if f.field == "b1_present")
    assert present.alpha is None, "alpha is undefined when nobody ever disagreed"
    assert any("undefined" in note for note in present.notes)


def test_restriction_of_range_uses_a_measure_that_fits_the_scale() -> None:
    """A boolean has no interior, so "extreme categories" would always report 1.0.

    The measure is therefore chosen per scale: end-bands for a ranked scale with an
    interior, modal share otherwise. Without this every boolean field in the codebook
    reports full restriction and the check tells you nothing.
    """
    balanced = [Annotation(f"c{i}", rater, "B1", "v1.0-draft",
                           {"b1_present": i % 2 == 0, "b1_amplitude": 2})
                for i in range(20) for rater in ("r1", "r2")]
    checks = report(balanced, bootstrap=0).artefacts.restriction_of_range

    boolean = checks["B1.b1_present"]
    assert boolean.proportion == pytest.approx(0.5), "a balanced boolean is not restricted"
    assert boolean.measure == "share held by the most-used category"

    # b1_amplitude is ordinal with three bands, so the interior band 2 is not an extreme.
    ordinal = checks["B1.b1_amplitude"]
    assert ordinal.measure == "share in the top and bottom bands"
    assert ordinal.proportion == pytest.approx(0.0), "every label sits in the middle band"


def test_role_effect_is_not_computable_from_a_single_rater() -> None:
    """One rater in a role gives no within-role agreement, and that is said
    rather than faked."""
    rows = [Annotation(f"c{i}", rater, "B1", "v1.0-draft",
                       {"b1_present": i % 3 == 0, "b1_amplitude": 1 + i % 3})
            for i in range(9) for rater in ("r1", "r2", "r3")]
    result = report(rows, bootstrap=0,
                    raters=[Rater("r1", "tutor"), Rater("r2", "tutor"),
                            Rater("r3", "lecturer")])

    assert result.artefacts.rater_role_effects["lecturer"] is None
    assert result.artefacts.rater_role_effects["tutor"] is not None
    assert any("only one rater holds the role" in w for w in result.artefacts.warnings)


def test_observed_order_can_be_compared_with_the_registered_prediction() -> None:
    """§7 predicts B1 highest and B4 lowest before annotation begins."""
    result = report(_synthetic_annotations(), bootstrap=0)
    order = result.observed_order()
    assert set(order) <= {"B1", "B2", "B3", "B4", "B5"}
    assert len(order) == len(set(order))
