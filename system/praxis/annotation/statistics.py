# -*- coding: utf-8 -*-
"""Agreement estimators: Krippendorff's alpha, ICC(2,k), quadratic weighted kappa, bootstrap.

These compute. They do not decide what to compute them on; `irr.py` does that, driven by the
scale type each field declares in `codebook.py`.

**Why alpha carries most of the weight.** BUILD-SPEC Phase 2 chooses it "for categorical
layers, which handles incomplete data and any number of raters, unlike ICC". Both properties
are load-bearing here rather than nice to have: a rater marks a behaviour non-scorable
independently of the others, so most fields have gaps, and the rater roster varies between
calibration rounds.

**Where a statistic is undefined, these return None rather than a number.** Alpha is undefined
when every rater used a single category throughout, because expected disagreement is then zero
and the quotient has no value. Reporting 1.0 there would claim perfect agreement on an
instrument that never made a distinction, which is the opposite of what the data show.

Every estimator is deterministic given its seed, per R7.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from praxis.annotation.codebook import ScaleType

# One unit of analysis: one value per rater, None where that rater did not code it.
Unit = Sequence[Hashable | None]


@dataclass(frozen=True)
class Interval:
    """A bootstrap percentile interval, and what it was computed over."""

    point: float
    lower: float
    upper: float
    level: float
    draws: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass(frozen=True)
class IccResult:
    """ICC(2,k) with the design warning BUILD-SPEC Phase 2 requires alongside it."""

    icc: float | None
    n_units: int
    n_raters: int
    fully_crossed: bool
    units_dropped: int
    warning: str | None


# ---------------------------------------------------------------------------
# Krippendorff's alpha
# ---------------------------------------------------------------------------

def _coincidence(units: Sequence[Unit]) -> tuple[list, np.ndarray, np.ndarray, float]:
    """The coincidence matrix, its value set, marginals, and total.

    Each unit coded by m raters contributes 1/(m-1) to every ordered pair of its values.
    That weighting is what lets a unit coded by five raters count the same as one coded by
    two, which is how the statistic tolerates the missingness it was chosen for.
    """
    observed = [[value for value in unit if value is not None] for unit in units]
    usable = [values for values in observed if len(values) >= 2]

    catalogue = sorted({value for values in usable for value in values},
                       key=lambda v: (isinstance(v, str), v))
    index = {value: position for position, value in enumerate(catalogue)}
    size = len(catalogue)
    matrix = np.zeros((size, size), dtype=float)

    for values in usable:
        weight = 1.0 / (len(values) - 1)
        for i, left in enumerate(values):
            for j, right in enumerate(values):
                if i != j:
                    matrix[index[left], index[right]] += weight

    marginals = matrix.sum(axis=1)
    return catalogue, matrix, marginals, float(marginals.sum())


def _difference_matrix(catalogue: list, marginals: np.ndarray,
                       scale: ScaleType) -> np.ndarray:
    """Krippendorff's squared difference function for the given scale.

    The ordinal case is the one that cannot be replaced by something simpler: its distance
    between two ranks depends on how much of the data lies between them, so two adjacent but
    rarely-used ranks are closer together than two adjacent but heavily-used ones. That is
    what makes it an ordinal measure rather than an interval one over rank numbers.
    """
    size = len(catalogue)
    delta = np.zeros((size, size), dtype=float)

    if scale is ScaleType.NOMINAL:
        delta[:, :] = 1.0
        np.fill_diagonal(delta, 0.0)
        return delta

    if scale is ScaleType.ORDINAL:
        for c in range(size):
            for k in range(size):
                low, high = (c, k) if c <= k else (k, c)
                between = marginals[low:high + 1].sum()
                delta[c, k] = (between - (marginals[c] + marginals[k]) / 2.0) ** 2
        return delta

    values = np.asarray([float(value) for value in catalogue], dtype=float)

    if scale is ScaleType.INTERVAL:
        return (values[:, None] - values[None, :]) ** 2

    # Ratio: a true zero, so the same absolute gap matters more near the bottom of the range.
    totals = values[:, None] + values[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(totals == 0.0, 0.0,
                         ((values[:, None] - values[None, :]) / totals) ** 2)
    return delta


def krippendorff_alpha(units: Sequence[Unit],
                       scale: ScaleType = ScaleType.NOMINAL) -> float | None:
    """Krippendorff's alpha for any of the four scale types.

    `units` holds one sequence per unit of analysis, each carrying one value per rater with
    None where that rater did not code it. Units with fewer than two values contribute
    nothing and are skipped.

    Returns None where alpha is undefined: fewer than two coincident values anywhere, or
    zero expected disagreement because the instrument never made a distinction.
    """
    catalogue, matrix, marginals, total = _coincidence(units)
    if total < 2 or len(catalogue) < 2:
        return None

    delta = _difference_matrix(catalogue, marginals, scale)

    observed_disagreement = float((matrix * delta).sum())
    expected_disagreement = float((np.outer(marginals, marginals) * delta).sum())
    if expected_disagreement == 0.0:
        return None

    return 1.0 - (total - 1.0) * observed_disagreement / expected_disagreement


# ---------------------------------------------------------------------------
# Quadratic weighted kappa
# ---------------------------------------------------------------------------

def quadratic_weighted_kappa(left: Sequence, right: Sequence,
                             levels: Sequence) -> float | None:
    """Cohen's kappa with quadratic disagreement weights, for two raters on one ordinal field.

    Quadratic weights are the convention for ordinal scales: a two-band disagreement counts
    four times a one-band disagreement.

    `levels` is the ordered domain from the codebook rather than the observed values, so that
    a band nobody used still occupies its place on the scale. Taking the domain from the data
    would make the weights depend on which bands happened to appear.

    Returns None where expected disagreement is zero, which is both raters using one band
    throughout.
    """
    left = list(left)
    right = list(right)
    if not left or len(left) != len(right):
        return None

    index = {value: position for position, value in enumerate(levels)}
    size = len(levels)
    if size < 2:
        return None

    try:
        rows = np.array([index[value] for value in left], dtype=int)
        columns = np.array([index[value] for value in right], dtype=int)
    except KeyError as exc:
        raise ValueError(f"value {exc.args[0]!r} is not among the declared levels") from exc

    observed = np.zeros((size, size), dtype=float)
    np.add.at(observed, (rows, columns), 1.0)
    observed /= observed.sum()

    expected = np.outer(np.bincount(rows, minlength=size) / rows.size,
                        np.bincount(columns, minlength=size) / columns.size)

    positions = np.arange(size)
    weights = (positions[:, None] - positions[None, :]) ** 2 / float((size - 1) ** 2)

    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return None
    return 1.0 - float((weights * observed).sum()) / denominator


def mean_pairwise_kappa(units: Sequence[Unit], levels: Sequence) -> float | None:
    """Weighted kappa averaged over every rater pair that shares at least one unit.

    Kappa is defined for two raters. The roster here varies between rounds and raters skip
    units they judge non-scorable, so a single pairwise figure would not exist. Averaging
    over pairs is the usual resolution and is what the predecessor project did.
    """
    if not units:
        return None
    n_raters = max(len(unit) for unit in units)
    values: list[float] = []

    for first in range(n_raters):
        for second in range(first + 1, n_raters):
            paired = [(unit[first], unit[second]) for unit in units
                      if first < len(unit) and second < len(unit)
                      and unit[first] is not None and unit[second] is not None]
            if not paired:
                continue
            kappa = quadratic_weighted_kappa([a for a, _ in paired],
                                             [b for _, b in paired], levels)
            if kappa is not None:
                values.append(kappa)

    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# ICC(2,k)
# ---------------------------------------------------------------------------

def icc_2k(units: Sequence[Unit]) -> IccResult:
    """ICC(2,k): two-way random effects, absolute agreement, average measures.

    Average measures rather than single, because what matters is the reliability of the
    panel's consensus, which is what model performance is compared against, not the
    reliability of one rater drawn at random.

    **ICC needs a complete matrix, and that is the whole reason for the warning.** Units not
    coded by every rater are dropped. When the design is not fully crossed the retained
    sample is a biased slice of the data and ICC underestimates reliability, so
    `fully_crossed` and `warning` travel with the number and `irr.py` prints them beside it.
    BUILD-SPEC Phase 2 requires exactly this and `configs/default.yaml` sets
    `warn_when_not_fully_crossed: true`.
    """
    if not units:
        return IccResult(None, 0, 0, True, 0, "no units supplied")

    n_raters = max(len(unit) for unit in units)
    complete = [unit for unit in units
                if len(unit) == n_raters and all(value is not None for value in unit)]
    dropped = len(units) - len(complete)
    fully_crossed = dropped == 0

    warning = None if fully_crossed else (
        f"design is not fully crossed: {dropped} of {len(units)} units were not coded by "
        f"all {n_raters} raters and were dropped. ICC underestimates reliability in this "
        f"case, so read it as a lower bound and prefer Krippendorff's alpha, which "
        f"tolerates the missingness.")

    if len(complete) < 2 or n_raters < 2:
        return IccResult(None, len(complete), n_raters, fully_crossed, dropped,
                         warning or "fewer than two complete units or two raters")

    matrix = np.asarray([[float(value) for value in unit] for unit in complete], dtype=float)
    n, k = matrix.shape

    grand = matrix.mean()
    ss_rows = k * ((matrix.mean(axis=1) - grand) ** 2).sum()
    ss_columns = n * ((matrix.mean(axis=0) - grand) ** 2).sum()
    ss_error = ((matrix - grand) ** 2).sum() - ss_rows - ss_columns

    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_rows + (ms_columns - ms_error) / n
    if denominator == 0.0:
        return IccResult(None, n, k, fully_crossed, dropped,
                         warning or "no variance between units")

    return IccResult(float((ms_rows - ms_error) / denominator), n, k,
                     fully_crossed, dropped, warning)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_interval(units: Sequence[Unit],
                       estimator,
                       draws: int = 1000,
                       level: float = 0.95,
                       seed: int = 20260911) -> Interval | None:
    """Percentile interval for an agreement statistic, resampling units with replacement.

    **Units, not individual ratings.** A unit is the sampling unit, and resampling ratings
    within one would treat two raters' views of the same clip as two independent
    observations, which would narrow the interval by pretending to more evidence than exists.

    Draws where the statistic is undefined are discarded rather than counted as zero, and the
    number that survived is reported so a thin interval is visible as thin.
    """
    point = estimator(units)
    if point is None or len(units) < 2 or draws < 1:
        return None

    generator = np.random.default_rng(seed)
    positions = np.arange(len(units))
    values: list[float] = []

    for _ in range(draws):
        sample = [units[i] for i in generator.choice(positions, size=len(units), replace=True)]
        value = estimator(sample)
        if value is not None:
            values.append(value)

    if len(values) < 2:
        return None

    tail = (1.0 - level) / 2.0 * 100.0
    return Interval(point=float(point),
                    lower=float(np.percentile(values, tail)),
                    upper=float(np.percentile(values, 100.0 - tail)),
                    level=level, draws=len(values))
