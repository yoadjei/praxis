# -*- coding: utf-8 -*-
"""The Adebayo et al. (2018) sanity checks, run on this model and this data.

Grad-CAM passes these on Inception v3 over ImageNet. That is why BUILD-SPEC chooses it, and it
is not evidence about a frozen ResNet-50 with a TCN over Ghanaian classroom video, so the checks
are run again here.

**The logic of the checks is inverted from the usual intuition, and getting it backwards makes
a broken method look good.** An explanation is supposed to describe a particular model's
computation. Randomise that model's parameters and the computation changes, so a faithful
explanation must change too. A method whose maps stay the same is not describing the
computation - it is describing the input, like the edge detector Adebayo et al. show producing
maps that resemble several popular saliency methods. So **high similarity after randomisation
is a failure**, which is why `fail_threshold_rank_corr` is an upper bound.

**Two checks, and they fail differently.**

The *parameter randomisation* test randomises layers from the top down, cascading, and measures
similarity to the original explanation at each stage. Cascading rather than one layer at a time
because a method can be insensitive to any single layer while still depending on the network as
a whole.

The *data randomisation* test compares explanations from a model trained on true labels with
one trained on permuted labels. A model fitted to noise has learned a different function; an
explanation that looks the same for both is not describing either.

**Deletion and insertion AUC** are not Adebayo checks and are reported separately. They measure
whether the regions a map ranks highest actually carry the prediction, which is a different
question from whether the map describes the computation, and a method can do well on one and
badly on the other.

**The decision rule is fixed in advance**, in `configs/default.yaml`, and applied by
`FidelityReport.passed`: fail either sanity check and the method is withdrawn from the
interface and the intrinsic explanation is used instead. The failure is reported, not buried.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from praxis.explain.base import ClipInput, ExplainError, Explanation, normalise


class FidelityError(RuntimeError):
    """A fidelity check could not be run, with the reason named."""


@dataclass(frozen=True)
class Similarity:
    """How alike two explanations are, by both measures BUILD-SPEC names.

    Both are reported because they disagree usefully. Rank correlation is sensitive to the
    ordering of attributions and blind to their spatial arrangement; SSIM is the reverse. A
    method that survives one and not the other has not passed.
    """

    rank_correlation: float
    ssim: float

    @property
    def worst(self) -> float:
        """The higher of the two, which is what a threshold on similarity must test.

        Taking the mean would let a method hide a near-identical structural match behind an
        uncorrelated ranking.
        """
        return max(self.rank_correlation, self.ssim)


def compare_maps(first: np.ndarray, second: np.ndarray) -> Similarity:
    """Rank correlation and structural similarity between two attribution maps.

    Constant maps are the case that matters here, because a deliberately broken explainer
    produces them and Phase 7's acceptance test requires that it be caught. Spearman's rho is
    undefined when either input has zero variance - scipy returns NaN - and a NaN propagated
    into the verdict would read as "no result" rather than as the perfect similarity it
    actually represents. Two constant maps are identical, so the correlation is 1.0.
    """
    from scipy.stats import spearmanr
    from skimage.metrics import structural_similarity

    a = np.asarray(first, dtype=float).ravel()
    b = np.asarray(second, dtype=float).ravel()
    if a.size != b.size:
        raise FidelityError(f"maps of {a.size} and {b.size} values cannot be compared")
    if a.size < 2:
        raise FidelityError("a map of fewer than two values has no structure to compare")

    a_flat = np.ptp(a) == 0
    b_flat = np.ptp(b) == 0
    if a_flat and b_flat:
        rho = 1.0
    elif a_flat or b_flat:
        # One constant and one not: no monotone relationship exists, which is zero rather than
        # undefined.
        rho = 0.0
    else:
        rho = float(spearmanr(a, b).statistic)
        if not np.isfinite(rho):
            rho = 0.0

    # SSIM over the normalised maps on a common data range, so that a method producing large
    # raw values is not scored differently from one producing small ones.
    first_n = normalise(np.asarray(first, dtype=float))
    second_n = normalise(np.asarray(second, dtype=float))
    if first_n.ndim >= 2 and min(first_n.shape[-2:]) >= 7:
        score = structural_similarity(first_n, second_n, data_range=1.0,
                                      channel_axis=0 if first_n.ndim == 3 else None)
    else:
        # Too small for SSIM's default 7x7 window. A 1-D attribution over frames is the normal
        # case for the intrinsic method, and an odd window no wider than the signal is the
        # honest reduction rather than refusing to compare at all.
        window = max(3, min(7, (first_n.size // 2) * 2 - 1))
        score = structural_similarity(first_n.ravel(), second_n.ravel(),
                                      data_range=1.0, win_size=min(window, first_n.size))
    return Similarity(rank_correlation=float(rho), ssim=float(score))


@dataclass(frozen=True)
class RandomisationStage:
    """One step of the cascading randomisation, and how similar the map stayed."""

    layer: str
    similarity: Similarity


@dataclass(frozen=True)
class CheckResult:
    """One sanity check's verdict, with the evidence that produced it."""

    check: str
    passed: bool
    worst_similarity: float
    threshold: float
    stages: tuple[RandomisationStage, ...] = ()
    detail: str = ""

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (f"{self.check}: {verdict}  worst similarity {self.worst_similarity:.3f} "
                f"against a threshold of {self.threshold:.2f}")


@dataclass(frozen=True)
class FidelityReport:
    """Everything measured about one method, and whether it may be shown to a reviewer."""

    method: str
    parameter_randomisation: CheckResult
    data_randomisation: CheckResult | None
    deletion_auc: float | None = None
    insertion_auc: float | None = None

    @property
    def sanity_checks(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in (self.parameter_randomisation, self.data_randomisation)
                     if c is not None)

    @property
    def passed(self) -> bool:
        """BUILD-SPEC's decision rule, fixed in advance.

        Both checks must pass. A check that could not be run is not a pass: `data_randomisation`
        is None when no permuted-label model was supplied, and `checks_run` reports that
        separately so a partial evaluation cannot be quoted as a clean one. D48.
        """
        return all(check.passed for check in self.sanity_checks)

    @property
    def complete(self) -> bool:
        """Whether both sanity checks actually ran."""
        return self.data_randomisation is not None

    def summary(self) -> list[str]:
        lines = [check.summary() for check in self.sanity_checks]
        if not self.complete:
            lines.append(
                "data_randomisation: NOT RUN - no permuted-label model supplied, so this "
                "report is not a clean pass and must not be quoted as one")
        if self.deletion_auc is not None:
            lines.append(f"deletion AUC {self.deletion_auc:.4f}  "
                         f"insertion AUC {self.insertion_auc:.4f}")
        return lines


def parameter_randomisation_test(
    explainer,
    clip: ClipInput,
    behaviour: str,
    field: str,
    *,
    randomise: Callable[[str], None],
    restore: Callable[[], None],
    layers: Sequence[str],
    threshold: float,
) -> CheckResult:
    """Adebayo's cascading parameter randomisation.

    Args:
        explainer: the method under test.
        clip, behaviour, field: what to explain.
        randomise: called with a layer name, reinitialises that layer's parameters in place.
            Supplied by the caller because which module owns the layers differs by method -
            Grad-CAM randomises the backbone, the intrinsic method the Stage B trunk.
        restore: puts the original parameters back. Always called, including on failure,
            because a randomised model left behind would silently poison every later check.
        layers: top-down order. Cascading, so each stage randomises one more.
        threshold: `explain.fidelity.fail_threshold_rank_corr`. Similarity **above** this after
            randomisation is a failure.

    Returns:
        A CheckResult whose `worst_similarity` is the highest similarity observed across the
        cascade, because a method that recovers at the last stage has still failed.
    """
    if not layers:
        raise FidelityError(
            "no layers to randomise. A check with nothing to randomise would pass "
            "unconditionally, which is the quiet way an invariant is deleted.")

    original = explainer.explain(clip, behaviour, field).comparable_map()
    stages: list[RandomisationStage] = []
    try:
        for layer in layers:
            randomise(layer)
            perturbed = explainer.explain(clip, behaviour, field).comparable_map()
            stages.append(RandomisationStage(layer, compare_maps(original, perturbed)))
    finally:
        restore()

    worst = max(stage.similarity.worst for stage in stages)
    return CheckResult(
        check="parameter_randomisation",
        passed=worst <= threshold,
        worst_similarity=worst,
        threshold=threshold,
        stages=tuple(stages),
        detail=("the explanation changed as the model did"
                if worst <= threshold else
                "the explanation survived randomisation of the model it claims to describe, "
                "so it is describing the input rather than the computation"))


def data_randomisation_test(
    explainer_on_trained,
    explainer_on_permuted,
    clip: ClipInput,
    behaviour: str,
    field: str,
    *,
    threshold: float,
) -> CheckResult:
    """Adebayo's data randomisation: true labels against permuted labels.

    A model fitted to permuted labels has learned a different function. An explanation that
    looks the same for both is not describing either of them.

    The two explainers are passed already bound to their models rather than being constructed
    here, because retraining is the caller's business - it costs a full Stage B run - and a
    function that silently retrained would hide minutes of work inside a check.
    """
    trained = explainer_on_trained.explain(clip, behaviour, field).comparable_map()
    permuted = explainer_on_permuted.explain(clip, behaviour, field).comparable_map()
    similarity = compare_maps(trained, permuted)

    return CheckResult(
        check="data_randomisation",
        passed=similarity.worst <= threshold,
        worst_similarity=similarity.worst,
        threshold=threshold,
        detail=("the explanation differs between a model fitted to the labels and one fitted "
                "to noise" if similarity.worst <= threshold else
                "the explanation is the same whether the model learned the task or not"))


def deletion_insertion_auc(
    explanation: Explanation,
    predict: Callable[[np.ndarray], float],
    baseline: np.ndarray,
    subject: np.ndarray,
    *,
    steps: int = 20,
) -> tuple[float, float]:
    """How much the prediction depends on the regions the map ranks highest.

    Deletion progressively replaces the highest-attributed elements with the baseline and
    watches the score fall; insertion starts from the baseline and adds them back. A faithful,
    informative map gives a low deletion AUC and a high insertion one.

    Reported separately from the sanity checks and never folded into the verdict. They answer a
    different question - whether the highlighted regions carry the prediction, rather than
    whether the map describes the computation - and a method can do well on one and badly on
    the other. Folding them together would let a persuasive-but-unfaithful method average its
    way to a pass.

    Args:
        explanation: the map whose ranking is followed.
        predict: the score for a modified input, as a single float.
        baseline: what a deleted element is replaced by, same shape as `subject`.
        subject: the original input the map describes.
        steps: how many fractions to evaluate, from `explain.fidelity.deletion_steps`.
    """
    if steps < 2:
        raise FidelityError("an AUC over fewer than two points is not a curve")
    if baseline.shape != subject.shape:
        raise FidelityError(
            f"baseline {baseline.shape} and subject {subject.shape} must have the same shape")

    attribution = explanation.comparable_map()
    flat_subject = subject.reshape(-1)
    if attribution.size != flat_subject.size:
        raise FidelityError(
            f"the map has {attribution.size} values and the input {flat_subject.size}; they "
            f"must describe the same elements for a ranking to mean anything")

    order = np.argsort(attribution)[::-1]
    fractions = np.linspace(0.0, 1.0, steps)
    deletion, insertion = [], []

    for fraction in fractions:
        count = round(fraction * order.size)
        chosen = order[:count]

        deleted = flat_subject.copy()
        deleted[chosen] = baseline.reshape(-1)[chosen]
        deletion.append(predict(deleted.reshape(subject.shape)))

        inserted = baseline.reshape(-1).copy()
        inserted[chosen] = flat_subject[chosen]
        insertion.append(predict(inserted.reshape(subject.shape)))

    return (float(np.trapezoid(deletion, fractions)),
            float(np.trapezoid(insertion, fractions)))


def run_fidelity(
    explainer,
    clip: ClipInput,
    behaviour: str,
    field: str,
    *,
    randomise: Callable[[str], None],
    restore: Callable[[], None],
    layers: Sequence[str],
    threshold: float,
    permuted_explainer=None,
) -> FidelityReport:
    """Both sanity checks, with a report that states which of them ran.

    `permuted_explainer` is optional because the data randomisation test needs a second, fully
    retrained model. Its absence is recorded as "not run" rather than as a pass, so a partial
    evaluation cannot be quoted as a clean one.
    """
    parameter = parameter_randomisation_test(
        explainer, clip, behaviour, field, randomise=randomise, restore=restore,
        layers=layers, threshold=threshold)

    data = None
    if permuted_explainer is not None:
        data = data_randomisation_test(explainer, permuted_explainer, clip, behaviour, field,
                                       threshold=threshold)

    return FidelityReport(method=getattr(explainer, "name", "unknown"),
                          parameter_randomisation=parameter, data_randomisation=data)


def reinitialise(module) -> None:
    """Reset a module's parameters in place, for the randomisation test.

    `reset_parameters` where the module defines it, and a normal draw otherwise. The point is
    that the layer no longer computes what it was trained to; matching any particular
    initialisation scheme is not required.
    """
    import torch

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
        return
    with torch.no_grad():
        for parameter in module.parameters(recurse=False):
            if parameter.dim() > 1:
                torch.nn.init.normal_(parameter, std=0.05)
            else:
                torch.nn.init.zeros_(parameter)
    if not any(True for _ in module.parameters(recurse=False)):
        raise ExplainError(
            f"{type(module).__name__} has no parameters of its own to randomise, so "
            f"randomising it would change nothing and the check would pass for free")
