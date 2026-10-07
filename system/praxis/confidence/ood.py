# -*- coding: utf-8 -*-
"""Out-of-distribution detection: making the system notice when it has left campus.

Three methods, all required, because the comparison is the point. **Maximum softmax
probability** (Hendrycks and Gimpel, 2017) is the baseline every later method is measured
against. **Ensemble disagreement**, the mutual information from `ensemble.py`, is the primary
method, because it is the quantity that rises when members that agreed on training data stop
agreeing. **Mahalanobis distance** on penultimate backbone features (Lee et al., 2018) is the
feature-space method, and it sees a kind of shift the other two cannot: a room that looks
unlike anything in training, even when the model happens to be confident about it.

**Every score in this module points the same way: higher means more out-of-distribution.**
That is a convention, not a property of the underlying quantities — MSP is natively an
*in*-distribution score and is inverted here. Getting this backwards inverts every AUROC in
Phase 6, and it would not look wrong, it would look like a negative result. So the direction is
stated on every function and asserted in the tests.

`ood_score` and `ood_flag` in `ConfidenceState` are filled from here. The threshold that turns
a score into a flag is not chosen here: it comes from the sweep in `evaluation/shift.py`,
against a stated criterion, and is recorded in `docs/DECISIONS.md`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from praxis.confidence.ensemble import mutual_information
from praxis.confidence.metrics import MAX_BINARY_ENTROPY


class OodError(RuntimeError):
    """Raised when a detector is asked for a score it cannot honestly produce."""


# ---------------------------------------------------------------------------
# Scores. Higher is more out-of-distribution, without exception.
# ---------------------------------------------------------------------------

def msp_score(probabilities: Sequence[float]) -> np.ndarray:
    """Maximum softmax probability, inverted so that higher means more OOD.

    Hendrycks and Gimpel observe that a correctly classified in-distribution example tends to
    carry a higher maximum class probability than an out-of-distribution one. For a binary
    head that maximum is ``max(p, 1-p)``, which lies in [0.5, 1]. The score returned is
    ``1 - |2p - 1|``, which maps a confident prediction at either extreme to 0 and maximum
    uncertainty at p = 0.5 to 1.

    **This is the baseline, and it is weak on purpose.** It cannot distinguish a genuinely
    ambiguous in-distribution clip from an unfamiliar one, because both look uncertain. That
    limitation is what the other two methods are being measured against.
    """
    probs = np.asarray(probabilities, dtype=float)
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise OodError("probabilities must lie in [0, 1]")
    return 1.0 - np.abs(2.0 * probs - 1.0)


def disagreement_score(member_probabilities: Sequence[Sequence[float]]) -> np.ndarray:
    """Ensemble disagreement, normalised to [0, 1]. Higher means more OOD.

    The mutual information between the prediction and the choice of member, divided by the
    most a binary variable can carry. Unlike MSP this separates "hard" from "unfamiliar": on a
    genuinely ambiguous clip the members agree that it is ambiguous and disagreement stays
    low, while on an unfamiliar one they diverge.
    """
    return mutual_information(member_probabilities) / MAX_BINARY_ENTROPY


@dataclass(frozen=True)
class MahalanobisDetector:
    """Class-conditional Mahalanobis distance on penultimate features (Lee et al., 2018).

    Fitted on in-distribution features only. The score is the distance to the nearest class
    centroid under a covariance shared across classes, so a feature vector far from every
    class is scored high however confident the head above it happened to be.

    The tied covariance is Lee et al.'s choice and matters at this scale: a 2048-dimensional
    per-class covariance estimated from a few thousand clips would be singular, and its
    inverse would be noise. Shrinkage toward a scaled identity is applied on top, controlled
    by `ood.mahalanobis.shrinkage` in the config.
    """

    means: np.ndarray            # (n_classes, n_features)
    precision: np.ndarray        # (n_features, n_features)
    classes: tuple
    shrinkage: float
    n_fitted: int

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    def score(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Squared distance to the nearest class centroid. Higher means more OOD.

        Left as a squared distance rather than mapped to [0, 1]: it is unbounded by nature,
        any squashing would be an arbitrary choice, and only the ordering matters for AUROC
        and for a threshold chosen by sweep.
        """
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.n_features:
            raise OodError(
                f"features are {matrix.shape[1]}-dimensional but the detector was fitted on "
                f"{self.n_features}")

        distances = np.empty((matrix.shape[0], len(self.classes)), dtype=float)
        for index, mean in enumerate(self.means):
            centred = matrix - mean
            distances[:, index] = np.einsum("ij,jk,ik->i", centred, self.precision, centred)
        return distances.min(axis=1)


def fit_mahalanobis(features: Sequence[Sequence[float]], labels: Sequence,
                    shrinkage: float = 0.10) -> MahalanobisDetector:
    """Fit on in-distribution features only.

    Fitting on anything else would teach the detector that the unfamiliar is familiar, which
    is the one mistake that makes an OOD detector worse than none: it would report confidence
    precisely where the system is outside what it was validated on.
    """
    matrix = np.asarray(features, dtype=float)
    truth = np.asarray(labels)
    if matrix.ndim != 2:
        raise OodError("features must be shaped (samples, dimensions)")
    if matrix.shape[0] != truth.shape[0]:
        raise OodError(f"{matrix.shape[0]} feature rows against {truth.shape[0]} labels")
    if not 0.0 <= shrinkage <= 1.0:
        raise OodError(f"shrinkage must lie in [0, 1], got {shrinkage}")

    classes = tuple(sorted(set(truth.tolist())))
    if not classes:
        raise OodError("no labelled features to fit on")

    n_features = matrix.shape[1]
    means = np.empty((len(classes), n_features), dtype=float)
    scatter = np.zeros((n_features, n_features), dtype=float)
    degrees_of_freedom = 0

    for index, label in enumerate(classes):
        selected = matrix[truth == label]
        if selected.shape[0] < 1:
            raise OodError(f"class {label!r} has no examples")
        means[index] = selected.mean(axis=0)
        centred = selected - means[index]
        scatter += centred.T @ centred
        degrees_of_freedom += selected.shape[0] - 1

    if degrees_of_freedom < 1:
        raise OodError(
            "every class has a single example, so within-class covariance is undefined. A "
            "Mahalanobis detector fitted here would describe the sample, not the class.")

    covariance = scatter / degrees_of_freedom
    # Ledoit-Wolf style shrinkage toward a scaled identity. Without it a covariance estimated
    # from fewer clips than feature dimensions is singular and its inverse is noise.
    target = np.trace(covariance) / n_features * np.eye(n_features)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * target

    try:
        precision = np.linalg.inv(shrunk)
    except np.linalg.LinAlgError as exc:
        raise OodError(
            f"the shrunk covariance is singular at shrinkage={shrinkage}; raise it, or fit on "
            f"more clips than there are feature dimensions ({matrix.shape[0]} < {n_features})"
        ) from exc

    return MahalanobisDetector(means=means, precision=precision, classes=classes,
                               shrinkage=shrinkage, n_fitted=int(matrix.shape[0]))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OodEvaluation:
    """How well a score separates out-of-distribution from in-distribution."""

    method: str
    auroc: float
    fpr_at_95_tpr: float
    n_in_distribution: int
    n_out_of_distribution: int
    threshold_at_95_tpr: float

    def summary(self) -> str:
        return (f"{self.method}: AUROC={self.auroc:.4f}  FPR@95TPR={self.fpr_at_95_tpr:.4f}  "
                f"(n_id={self.n_in_distribution}, n_ood={self.n_out_of_distribution})")


def auroc(scores: Sequence[float], is_out_of_distribution: Sequence[bool]) -> float:
    """Area under the ROC, by the rank formulation, with ties handled by average ranks.

    Written out rather than imported so that the tie handling is visible: a detector that
    assigns an identical score to many clips, which a saturated one does, would otherwise get
    a silently optimistic number from a naive threshold sweep. The Mann-Whitney form is exact.

    Equals the probability that a randomly chosen OOD example scores above a randomly chosen
    in-distribution one, with ties counted as half.
    """
    values = np.asarray(scores, dtype=float)
    flags = np.asarray(is_out_of_distribution, dtype=bool)
    if values.size != flags.size:
        raise OodError(f"{values.size} scores against {flags.size} flags")

    n_ood = int(flags.sum())
    n_id = int(values.size - n_ood)
    if n_ood == 0 or n_id == 0:
        raise OodError(
            "AUROC needs examples of both kinds; got "
            f"{n_id} in-distribution and {n_ood} out-of-distribution")

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)

    # Average ranks within each group of tied scores.
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    return float((ranks[flags].sum() - n_ood * (n_ood + 1) / 2.0) / (n_ood * n_id))


def fpr_at_tpr(scores: Sequence[float], is_out_of_distribution: Sequence[bool],
               target_tpr: float = 0.95) -> tuple[float, float]:
    """False positive rate at the threshold giving at least `target_tpr`, and that threshold.

    The operational number. AUROC summarises every threshold at once; this one says what
    catching 95 per cent of the unfamiliar actually costs in in-distribution clips wrongly
    flagged, which is what an abstention policy has to pay.
    """
    values = np.asarray(scores, dtype=float)
    flags = np.asarray(is_out_of_distribution, dtype=bool)
    ood_scores = values[flags]
    id_scores = values[~flags]
    if ood_scores.size == 0 or id_scores.size == 0:
        raise OodError("both kinds are needed to trade one against the other")

    # The largest threshold that still catches the target fraction of OOD examples.
    threshold = float(np.quantile(ood_scores, 1.0 - target_tpr, method="lower"))
    return float((id_scores >= threshold).mean()), threshold


def evaluate_ood(scores: Sequence[float], is_out_of_distribution: Sequence[bool],
                 method: str, target_tpr: float = 0.95) -> OodEvaluation:
    """AUROC and FPR at the operating point, reported together."""
    flags = np.asarray(is_out_of_distribution, dtype=bool)
    fpr, threshold = fpr_at_tpr(scores, flags, target_tpr)
    return OodEvaluation(
        method=method,
        auroc=auroc(scores, flags),
        fpr_at_95_tpr=fpr,
        n_in_distribution=int((~flags).sum()),
        n_out_of_distribution=int(flags.sum()),
        threshold_at_95_tpr=threshold)


def flag_rate_by_domain(scores: Sequence[float], domains: Sequence[str],
                        threshold: float) -> dict[str, float]:
    """Proportion flagged in each domain, which is H5.

    H5 asks whether the system flags authentic classroom footage more often than held-out
    microteaching. Reported per domain rather than pooled, because a pooled rate would hide
    exactly the contrast being tested.
    """
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(domains)
    if values.size != labels.size:
        raise OodError(f"{values.size} scores against {labels.size} domain labels")
    return {str(domain): float((values[labels == domain] >= threshold).mean())
            for domain in sorted(set(labels.tolist()))}
