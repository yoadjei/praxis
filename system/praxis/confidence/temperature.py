# -*- coding: utf-8 -*-
"""Temperature scaling (Guo et al., 2017): the baseline the thesis predicts will fail.

One scalar `T` per behaviour head, fitted by minimising negative log-likelihood on a held-out
i.i.d. validation split, never on test. Probabilities become `sigmoid(z / T)`. `T > 1` softens
an over-confident model; `T < 1` sharpens an under-confident one. Accuracy is untouched,
because dividing by a positive scalar cannot reorder logits or move any of them across zero.

**This is the comparison, not the product.** Guo et al. established that it works on i.i.d.
data; Ovadia et al. (2019) then found that post-hoc calibration of this kind falls short under
dataset shift, which is the condition this system lives in permanently. The thesis expects it
to hold at S0 and degrade by S2. Implementing it well is what makes that comparison fair.

**One scalar, fitted on a split the model never saw.** Fitting on training data would learn
the training distribution's over-confidence rather than the model's, and fitting on test would
be the leak the whole design exists to avoid.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from praxis.confidence.metrics import (
    DEFAULT_BINS,
    EPSILON,
    BinningMode,
    expected_calibration_error,
    negative_log_likelihood,
)

# T is searched on a log scale within these bounds. Outside them the sigmoid saturates and the
# objective goes flat, so a wider range buys nothing and risks a meaningless optimum.
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 100.0

# ECE may rise slightly on an already well-calibrated head without anything being wrong; see
# `_check_it_helped`. A rise beyond this, on a head that was miscalibrated to begin with, is
# treated as a bug rather than a finding.
ECE_REGRESSION_TOLERANCE = 1e-3
ALREADY_CALIBRATED = 0.01


class TemperatureScalingFailed(RuntimeError):
    """Raised when fitting produced a temperature that does not calibrate.

    BUILD-SPEC Phase 5: "Temperature scaling reduces ECE on the i.i.d. validation set. If it
    does not, fail loudly: the implementation is wrong." Silence here would let a broken
    baseline flatter the ensemble it is compared against, which would corrupt the headline
    finding rather than merely the baseline.
    """


@dataclass(frozen=True)
class Temperature:
    """A fitted temperature, with the evidence that it helped."""

    behaviour: str
    value: float
    n_validation: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float

    @property
    def softened(self) -> bool:
        """T > 1 means the model was over-confident, which is the usual direction."""
        return self.value > 1.0

    @property
    def ece_improvement(self) -> float:
        return self.ece_before - self.ece_after

    def summary(self) -> str:
        return (f"{self.behaviour}: T={self.value:.4f}  "
                f"ECE {self.ece_before:.4f} -> {self.ece_after:.4f}  "
                f"NLL {self.nll_before:.4f} -> {self.nll_after:.4f}  n={self.n_validation}")


def probabilities_to_logits(probabilities: Sequence[float]) -> np.ndarray:
    """The inverse sigmoid, for when a model hands back probabilities rather than logits.

    Clipped first: a probability of exactly 0 or 1 has infinite logit, and temperature
    scaling of an infinity is still an infinity, so the point would be immovable and would
    dominate the fit.
    """
    probs = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(probs / (1.0 - probs))


def apply_temperature(logits: Sequence[float], temperature: float) -> np.ndarray:
    """Scaled probabilities. Accuracy is unchanged by construction."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / temperature))


def _nll_at(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    return negative_log_likelihood(apply_temperature(logits, temperature), labels)


def fit_temperature(logits: Sequence[float], labels: Sequence[int | bool],
                    behaviour: str = "unnamed",
                    n_bins: int = DEFAULT_BINS,
                    mode: BinningMode = "positive_class",
                    check: bool = True) -> Temperature:
    """Fit one scalar by minimising NLL, and verify it calibrated.

    The search is over log T with a bounded scalar minimiser, which is deterministic and so
    satisfies R7 without a seed. Guo et al. used LBFGS on the parameter directly; on a single
    scalar the two agree to more places than any reported figure carries, and a deterministic
    minimiser is worth more here than fidelity to their optimiser.
    """
    raw = np.asarray(logits, dtype=float).ravel()
    truth = np.asarray(labels, dtype=float).ravel()
    if raw.size != truth.size:
        raise ValueError(f"{raw.size} logits against {truth.size} labels")
    if raw.size == 0:
        raise TemperatureScalingFailed(f"{behaviour}: no validation data to fit on")
    if len(np.unique(truth)) < 2:
        raise TemperatureScalingFailed(
            f"{behaviour}: the validation split contains only class {truth[0]:.0f}. A "
            f"temperature fitted on one class is fitted to the base rate, not to the model.")

    result = minimize_scalar(
        lambda log_t: _nll_at(float(np.exp(log_t)), raw, truth),
        bounds=(np.log(MIN_TEMPERATURE), np.log(MAX_TEMPERATURE)),
        method="bounded", options={"xatol": 1e-8})
    value = float(np.exp(result.x))

    before = apply_temperature(raw, 1.0)
    after = apply_temperature(raw, value)

    fitted = Temperature(
        behaviour=behaviour, value=value, n_validation=int(raw.size),
        nll_before=negative_log_likelihood(before, truth),
        nll_after=negative_log_likelihood(after, truth),
        ece_before=expected_calibration_error(before, truth, n_bins, mode),
        ece_after=expected_calibration_error(after, truth, n_bins, mode))

    if check:
        _check_it_helped(fitted)
    return fitted


def _check_it_helped(fitted: Temperature) -> None:
    """Fail loudly when the fit did not calibrate, and distinguish the two reasons it might not.

    **NLL must not worsen, and that is a guarantee rather than an expectation.** T = 1 lies
    inside the search interval and reproduces the original probabilities, so the minimiser can
    always return at least the starting objective. If NLL rose, the optimiser did not converge
    and nothing downstream is trustworthy.

    **ECE usually improves but is not guaranteed to**, because it is not the quantity being
    minimised. On a head that was already well calibrated the fit lands near T = 1 and ECE
    moves by noise in either direction, which is not a fault. So the loud failure is reserved
    for a head that *was* miscalibrated and came out no better: that combination is what a
    broken implementation produces, and BUILD-SPEC Phase 5 asks for it to be caught here
    rather than discovered at Phase 6 with the headline finding already contaminated.
    """
    if fitted.nll_after > fitted.nll_before + 1e-9:
        raise TemperatureScalingFailed(
            f"{fitted.summary()}\nNLL rose, which cannot happen from a converged fit: T=1 is "
            f"inside the search interval and reproduces the input exactly. The minimiser did "
            f"not converge.")

    if fitted.ece_before <= ALREADY_CALIBRATED:
        return

    if fitted.ece_after > fitted.ece_before + ECE_REGRESSION_TOLERANCE:
        raise TemperatureScalingFailed(
            f"{fitted.summary()}\nECE was {fitted.ece_before:.4f} before fitting and is "
            f"{fitted.ece_after:.4f} after. Temperature scaling is expected to reduce ECE on "
            f"an i.i.d. validation split; that it did not means the implementation is wrong, "
            f"not that the finding is interesting. Check that the split is genuinely i.i.d. "
            f"with the fitting data, and that logits rather than probabilities were passed.")


def fit_per_behaviour(validation: Mapping[str, tuple[Sequence[float], Sequence[int | bool]]],
                      n_bins: int = DEFAULT_BINS,
                      mode: BinningMode = "positive_class",
                      check: bool = True) -> dict[str, Temperature]:
    """One temperature per behaviour head, as BUILD-SPEC Phase 5 specifies.

    Per head rather than one shared scalar, because the five behaviours have different base
    rates and different difficulty: B1 is expected to be the easiest and B4 the hardest, so a
    single T would over-soften one to fix the other and calibrate neither.
    """
    return {behaviour: fit_temperature(logits, labels, behaviour, n_bins, mode, check)
            for behaviour, (logits, labels) in sorted(validation.items())}
