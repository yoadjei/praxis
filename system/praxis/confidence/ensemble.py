# -*- coding: utf-8 -*-
"""Deep ensembles (Ovadia et al., 2019): the production method, and the uncertainty it yields.

Ovadia et al. compared calibration methods under dataset shift and found that methods
marginalising over models hold up best while post-hoc rescaling falls short. This system lives
permanently in the shifted condition, so the ensemble is the production method and temperature
scaling is the baseline it is measured against.

**The decomposition is the point, not the averaging.** An ensemble gives two quantities a
single model cannot separate. Predictive entropy of the mean is *total* uncertainty. The mean
of the members' entropies is the part every member agrees is irreducible, which is aleatoric.
What is left is the members disagreeing with one another, which is *epistemic* uncertainty, and
it is the mutual information between the prediction and the choice of member. Epistemic
uncertainty is what rises when the model meets something it was not trained on, so it is what
the routing gate escalates on and what Phase 6 uses as its OOD score.

**The shared backbone is a deviation and it changes how diversity must be measured.** D3 makes
the M members independent TCN and head stacks over one frozen ResNet-50, because Phase 4 caches
its features and that makes the ensemble nearly free. Ovadia et al.'s members differ in every
parameter. Ours share a perceptual front end, so comparing whole state dicts would find them
99 per cent identical and conclude, wrongly, that training failed. Diversity is therefore
measured over the *trainable* parameters only, and the shared frozen prefix is excluded by
name. That exclusion is the thing most likely to be got wrong quietly, so it is explicit,
reported, and tested.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from praxis.confidence.metrics import MAX_BINARY_ENTROPY, binary_entropy
from praxis.contracts.confidence import ConfidenceState, ValidatedDomain

# Parameters shared by construction. Excluded from the diversity check, because they are
# identical by design and including them would mask real failures to diversify.
SHARED_PREFIXES: tuple[str, ...] = ("backbone.",)

# Below this relative L2 distance, two members are treated as the same model. Reached when a
# seed failed to vary or a training loop reused one initialisation.
IDENTICAL_TOLERANCE = 1e-6


class EnsembleError(RuntimeError):
    """Raised when an ensemble cannot deliver what an ensemble is for."""


@dataclass(frozen=True)
class EnsemblePrediction:
    """One clip's prediction, with total, aleatoric and epistemic uncertainty separated."""

    mean_probability: np.ndarray
    member_probabilities: np.ndarray          # (M, N)
    total_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    epistemic_uncertainty: np.ndarray         # mutual information, in nats

    @property
    def n_members(self) -> int:
        return int(self.member_probabilities.shape[0])

    @property
    def disagreement(self) -> np.ndarray:
        """Epistemic uncertainty as a fraction of the maximum a binary variable can carry.

        Reported alongside the raw mutual information because nats are not a scale anyone has
        intuitions about, and the routing gate's threshold is easier to defend on [0, 1].
        """
        return self.epistemic_uncertainty / MAX_BINARY_ENTROPY

    def to_confidence_state(self, index: int, raw_probability: float,
                            ood_score: float, ood_flag: bool,
                            domain: ValidatedDomain) -> ConfidenceState:
        """Build the contract R3 requires for one clip.

        `ood_score` and `ood_flag` are supplied by the caller rather than computed here: they
        come from Phase 6, and manufacturing a placeholder would put a number into a field a
        reviewer reads as measured.
        """
        return ConfidenceState(
            raw_prob=float(raw_probability),
            calibrated_prob=float(self.mean_probability[index]),
            method="ensemble",
            epistemic=float(self.epistemic_uncertainty[index]),
            ood_score=float(ood_score),
            ood_flag=bool(ood_flag),
            in_validated_domain="unknown" if ood_flag else domain)


def _member_matrix(member_probabilities: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(member_probabilities, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2:
        raise EnsembleError("member probabilities must be shaped (members, predictions)")
    if matrix.shape[0] < 2:
        raise EnsembleError(
            f"an ensemble of {matrix.shape[0]} has no disagreement to measure. Epistemic "
            f"uncertainty is the spread across members, so a single model cannot produce it.")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise EnsembleError("probabilities must lie in [0, 1]")
    return matrix


def combine(member_probabilities: Sequence[Sequence[float]]) -> EnsemblePrediction:
    """Average the members and decompose the uncertainty.

    Averaging probabilities rather than logits is what Ovadia et al. do and is the right
    choice here: the mean probability is the predictive distribution of a uniform mixture over
    members, which is the quantity the decomposition below is defined against. Averaging
    logits would give a different, sharper prediction whose entropy no longer decomposes.
    """
    members = _member_matrix(member_probabilities)
    mean_probability = members.mean(axis=0)

    total = binary_entropy(mean_probability)
    aleatoric = binary_entropy(members).mean(axis=0)
    # Non-negative by Jensen, since entropy is concave. Clipped only to absorb float error.
    epistemic = np.clip(total - aleatoric, 0.0, None)

    return EnsemblePrediction(
        mean_probability=mean_probability,
        member_probabilities=members,
        total_uncertainty=total,
        aleatoric_uncertainty=aleatoric,
        epistemic_uncertainty=epistemic)


def mutual_information(member_probabilities: Sequence[Sequence[float]]) -> np.ndarray:
    """Epistemic uncertainty on its own, for callers that need only the OOD score."""
    return combine(member_probabilities).epistemic_uncertainty


def mc_dropout_combine(passes: Sequence[Sequence[float]]) -> EnsemblePrediction:
    """MC dropout, the cheap middle option, decomposed the same way.

    The arithmetic is identical to an ensemble's: N stochastic forward passes stand in for M
    trained members. The interpretation is weaker, because the passes sample one model's
    dropout masks rather than independent fits, so their disagreement understates how much a
    differently-initialised model would have differed. It is reported as a third method and
    never as a substitute for the ensemble.
    """
    return combine(passes)


# ---------------------------------------------------------------------------
# Member diversity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiversityReport:
    """Evidence that the members are genuinely different models."""

    n_members: int
    n_parameters_compared: int
    excluded_keys: tuple[str, ...]
    distances: dict[tuple[int, int], float]
    identical_pairs: tuple[tuple[int, int], ...]

    @property
    def minimum(self) -> float:
        return min(self.distances.values()) if self.distances else 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.distances.values()))) if self.distances else 0.0

    def summary(self) -> str:
        return (f"{self.n_members} members, {self.n_parameters_compared:,} trainable "
                f"parameters compared, {len(self.excluded_keys)} shared keys excluded; "
                f"pairwise distance min {self.minimum:.4f}, mean {self.mean:.4f}")


def _flatten(state: Mapping[str, np.ndarray],
             shared_prefixes: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    kept, excluded = [], []
    for name in sorted(state):
        if any(name.startswith(prefix) for prefix in shared_prefixes):
            excluded.append(name)
            continue
        kept.append(np.asarray(state[name], dtype=float).ravel())
    return (np.concatenate(kept) if kept else np.zeros(0)), excluded


def pairwise_weight_distance(members: Sequence[Mapping[str, np.ndarray]],
                             shared_prefixes: Sequence[str] = SHARED_PREFIXES
                             ) -> DiversityReport:
    """Relative L2 distance between every pair of members, over trainable parameters only.

    Relative rather than absolute: ``||a - b|| / sqrt(||a|| ||b||)`` is scale-free, so the
    number means the same for a head with small weights as for one with large weights, and a
    threshold can be set once rather than per layer.
    """
    if len(members) < 2:
        raise EnsembleError(f"a diversity check needs at least two members, got {len(members)}")

    flattened, excluded = [], []
    for state in members:
        vector, dropped = _flatten(state, shared_prefixes)
        flattened.append(vector)
        excluded = dropped

    sizes = {vector.size for vector in flattened}
    if len(sizes) != 1:
        raise EnsembleError(
            f"members have different trainable parameter counts: {sorted(sizes)}"
        )
    if flattened[0].size == 0:
        raise EnsembleError(
            f"every parameter was excluded as shared. With prefixes {list(shared_prefixes)} "
            f"there is nothing left to compare, so the check would pass vacuously.")

    distances: dict[tuple[int, int], float] = {}
    identical: list[tuple[int, int]] = []
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            left, right = flattened[i], flattened[j]
            scale = float(np.sqrt(np.linalg.norm(left) * np.linalg.norm(right)))
            distance = float(np.linalg.norm(left - right) / scale) if scale > 0 else 0.0
            distances[(i, j)] = distance
            if distance < IDENTICAL_TOLERANCE:
                identical.append((i, j))

    return DiversityReport(
        n_members=len(members), n_parameters_compared=int(flattened[0].size),
        excluded_keys=tuple(excluded), distances=distances,
        identical_pairs=tuple(identical))


def assert_members_differ(members: Sequence[Mapping[str, np.ndarray]],
                          shared_prefixes: Sequence[str] = SHARED_PREFIXES) -> DiversityReport:
    """Refuse an ensemble whose members are the same model twice.

    BUILD-SPEC Phase 5 requires this to be verified rather than assumed. Identical members
    produce zero epistemic uncertainty everywhere, so the routing gate would never escalate
    and the Phase 6 OOD detector would have no signal — and both would fail silently, looking
    like a confident model rather than a broken one.
    """
    report = pairwise_weight_distance(members, shared_prefixes)
    if report.identical_pairs:
        raise EnsembleError(
            f"ensemble members {list(report.identical_pairs)} are identical to within "
            f"{IDENTICAL_TOLERANCE}. Members must differ by initialisation seed and data "
            f"order; check that the seed actually varied per member. {report.summary()}")
    return report


def check_members_share_device(device_models: Sequence[str | None]) -> None:
    """D14: every member trains on the same GPU model, or the ensemble is not one comparison.

    R7 forbids mixing devices within one experimental comparison and a deep ensemble is one.
    Members split across a T4 and an L4 would carry kernel variance into their disagreement,
    and disagreement is exactly the quantity Phase 6 reads as epistemic uncertainty. Nothing
    in any metric would reveal it.
    """
    distinct = {model for model in device_models if model is not None}
    if len(distinct) > 1:
        raise EnsembleError(
            f"ensemble members trained on different device models: {sorted(distinct)}. R7 "
            f"forbids mixing devices within one experimental comparison, and their "
            f"disagreement is read as epistemic uncertainty, so kernel differences would "
            f"enter the OOD signal undetected. Retrain the ensemble on one device model.")
