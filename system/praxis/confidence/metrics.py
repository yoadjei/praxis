# -*- coding: utf-8 -*-
"""Calibration metrics: ECE, MCE, Brier, NLL, and the reliability diagram.

Rule R4 says calibration is reported wherever accuracy is reported, so these are not optional
diagnostics. They are half of every result the thesis states.

**These are binary metrics, per behaviour, and that is a deliberate reading.** The behaviour
model has five independent presence heads with sigmoid outputs, not one softmax over five
classes, so a clip may show several behaviours at once and usually does. Guo et al. frame ECE
over the maximum softmax probability of a multiclass classifier; the analogue for a set of
independent binary heads is a per-head ECE over the probability of presence. Both conventions
are implemented and the difference between them is stated in D21, because they answer
different questions and reporting one while naming the other would be a quiet error.

The default, `positive_class`, asks: when the model says 0.7, does the behaviour occur 70 per
cent of the time? That is the question `ConfidenceState.calibrated_prob` claims to answer, so
it is the one measured by default.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# Guo et al.'s standard setting, and configs/default.yaml's calibration.metrics.ece_bins.
DEFAULT_BINS = 15

# Below this many samples per bin, ECE and MCE are computable and not dependable: most
# bins hold a handful of clips or none, and MCE in particular is the worst bin, so one
# bin of two samples decides it. The numbers are still returned - abstaining by returning
# None would propagate into every caller - and `reliable` is how the report says it.
# This corpus is small enough that the distinction is not academic. D86.
MIN_SAMPLES_PER_BIN = 10

# Probabilities are clipped before any logarithm. A single confident mistake would otherwise
# send NLL to infinity and destroy the mean, which reports the clipping rather than the model.
EPSILON = 1e-12

BinningMode = Literal["positive_class", "confidence"]


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_rate: float

    @property
    def gap(self) -> float:
        """Signed gap. Positive means over-confident: claimed more than it delivered."""
        return self.mean_confidence - self.empirical_rate


@dataclass(frozen=True)
class CalibrationReport:
    """Every calibration number for one behaviour, computed together.

    They are returned as one object because R4 is enforced by the return type: there is no
    way to obtain accuracy from this module without also obtaining ECE.
    """

    n: int
    mode: BinningMode
    ece: float
    mce: float
    brier: float
    nll: float
    accuracy: float
    base_rate: float
    mean_confidence: float
    mean_outcome: float
    bins: tuple[ReliabilityBin, ...]
    # The bin count asked for, not the number that ended up populated. `_bins_from`
    # drops empty bins, so `len(bins)` understates the binning and `reliable` would
    # read a 30-sample report spread over one populated bin as well supported.
    n_bins: int = DEFAULT_BINS

    @property
    def overconfident(self) -> bool:
        """Whether the model claims more than it delivers on average.

        **The reference differs by mode and getting it wrong inverts the answer.** In
        `positive_class` mode the claim is "this behaviour is present with probability p", so
        what it must be measured against is how often the behaviour actually occurred, the
        base rate. In `confidence` mode the claim is "my prediction is right with probability
        p", so the reference is accuracy. `mean_outcome` is whichever of the two the current
        mode makes correct, which is why it is stored rather than recomputed at the call site.

        The thesis predicts this gap widens under shift faster than accuracy falls, which is
        H4, so a mode-dependent sign error here would land directly in the headline figure.
        """
        return self.mean_confidence > self.mean_outcome

    @property
    def signed_gap(self) -> float:
        """Mean claimed minus mean delivered. Positive is over-confident."""
        return self.mean_confidence - self.mean_outcome

    @property
    def populated_bins(self) -> int:
        """How many bins hold anything. `_bins_from` omits empty ones, so this is the
        length of the tuple - named rather than inlined because the difference between
        it and `n_bins` is the whole point of `reliable`."""
        return len(self.bins)

    @property
    def reliable(self) -> bool:
        """Whether this many samples can support a binned calibration estimate.

        Not a property of the model: a property of the sample. ECE averages over bins
        weighted by how full they are, so it degrades gracefully as n falls and keeps
        returning a number that looks like the others. MCE takes the worst bin outright
        and a bin holding two clips can set it. R4 requires calibration to be reported
        wherever accuracy is; it does not require the reader to guess how far to trust
        it, which is what this answers. D86."""
        return self.n >= MIN_SAMPLES_PER_BIN * self.n_bins

    def caveat(self) -> str | None:
        """The sentence a table or figure must carry when the estimate is thin."""
        if self.reliable:
            return None
        return (f"calibration estimated from {self.n} samples across "
                f"{self.n_bins} bins ({self.populated_bins} populated); "
                f"{MIN_SAMPLES_PER_BIN * self.n_bins} are needed for a dependable "
                f"ECE and MCE at this bin count")

    def summary(self) -> str:
        thin = "" if self.reliable else "  [thin sample]"
        return (f"n={self.n}  ECE={self.ece:.4f}  MCE={self.mce:.4f}  "
                f"Brier={self.brier:.4f}  NLL={self.nll:.4f}  "
                f"acc={self.accuracy:.4f}{thin}")

    def caption(self) -> str:
        """The same numbers over two lines, so a figure title does not run off the canvas."""
        return (f"n={self.n}   ECE={self.ece:.4f}   MCE={self.mce:.4f}\n"
                f"Brier={self.brier:.4f}   NLL={self.nll:.4f}   acc={self.accuracy:.3f}")


def _as_arrays(probabilities: Sequence[float],
               labels: Sequence[int | bool]) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float).ravel()
    truth = np.asarray(labels, dtype=float).ravel()

    if probs.size != truth.size:
        raise ValueError(f"{probs.size} probabilities against {truth.size} labels")
    if probs.size == 0:
        raise ValueError("no predictions supplied")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0, 1]; pass logits through a sigmoid")
    if not np.all(np.isin(truth, (0.0, 1.0))):
        raise ValueError("labels must be 0 or 1; these are per-behaviour presence metrics")
    return probs, truth


def _confidence_and_correctness(probs: np.ndarray, truth: np.ndarray,
                                mode: BinningMode) -> tuple[np.ndarray, np.ndarray]:
    """The two series every bin is built from.

    In `positive_class` mode the confidence is the probability of presence and the outcome is
    presence, so a bin answers "when it said 0.7, how often did it happen".

    In `confidence` mode the confidence is the probability of whichever class was predicted
    and the outcome is whether that prediction was right, which is Guo et al.'s formulation
    and is what makes a number comparable with the published multiclass literature.
    """
    if mode == "positive_class":
        return probs, truth
    predicted = (probs >= 0.5).astype(float)
    return np.maximum(probs, 1.0 - probs), (predicted == truth).astype(float)


def reliability_bins(probabilities: Sequence[float], labels: Sequence[int | bool],
                     n_bins: int = DEFAULT_BINS,
                     mode: BinningMode = "positive_class") -> list[ReliabilityBin]:
    """Equal-width bins over [0, 1]. Empty bins are dropped, not reported as zero.

    An empty bin contributes nothing to ECE and plotting it at zero would draw a bar where no
    prediction was ever made, which reads as a calibration failure rather than as no evidence.
    """
    probs, truth = _as_arrays(probabilities, labels)
    confidence, correct = _confidence_and_correctness(probs, truth, mode)
    return _bins_from(confidence, correct, n_bins)


def _bins_from(confidence: np.ndarray, correct: np.ndarray,
               n_bins: int) -> list[ReliabilityBin]:
    """The binning itself, over confidence and correctness series already computed.

    Split out so the multiclass path below bins identically rather than similarly. Two
    implementations of one binning rule would differ at the edges first, which is exactly where
    an abstention threshold is chosen.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins, with the first bin also closed on the left so 0.0 has a home.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    out: list[ReliabilityBin] = []
    for b in range(n_bins):
        selected = index == b
        count = int(selected.sum())
        if count == 0:
            continue
        out.append(ReliabilityBin(
            lower=float(edges[b]), upper=float(edges[b + 1]), count=count,
            mean_confidence=float(confidence[selected].mean()),
            empirical_rate=float(correct[selected].mean())))
    return out


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int | bool],
                               n_bins: int = DEFAULT_BINS,
                               mode: BinningMode = "positive_class") -> float:
    """ECE: the average gap between claimed and delivered, weighted by bin population."""
    bins = reliability_bins(probabilities, labels, n_bins, mode)
    total = sum(b.count for b in bins)
    if not total:
        return 0.0
    return sum(b.count * abs(b.gap) for b in bins) / total


def maximum_calibration_error(probabilities: Sequence[float], labels: Sequence[int | bool],
                              n_bins: int = DEFAULT_BINS,
                              mode: BinningMode = "positive_class") -> float:
    """MCE: the worst bin, regardless of how few predictions landed in it.

    Reported beside ECE because a small bin with a large gap is invisible in ECE and is
    exactly where an abstention policy has to act.
    """
    bins = reliability_bins(probabilities, labels, n_bins, mode)
    return max((abs(b.gap) for b in bins), default=0.0)


def brier_score(probabilities: Sequence[float], labels: Sequence[int | bool]) -> float:
    """Mean squared error of the probability. A proper scoring rule; lower is better."""
    probs, truth = _as_arrays(probabilities, labels)
    return float(np.mean((probs - truth) ** 2))


def negative_log_likelihood(probabilities: Sequence[float],
                            labels: Sequence[int | bool]) -> float:
    """Binary cross-entropy in nats. What temperature scaling minimises."""
    probs, truth = _as_arrays(probabilities, labels)
    clipped = np.clip(probs, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)))


def calibration_report(probabilities: Sequence[float], labels: Sequence[int | bool],
                       n_bins: int = DEFAULT_BINS,
                       mode: BinningMode = "positive_class") -> CalibrationReport:
    """Every calibration number at once. R4 is enforced by this being the only way out."""
    probs, truth = _as_arrays(probabilities, labels)
    confidence, correct = _confidence_and_correctness(probs, truth, mode)
    bins = reliability_bins(probs, truth, n_bins, mode)

    return CalibrationReport(
        n=int(probs.size),
        mode=mode,
        ece=expected_calibration_error(probs, truth, n_bins, mode),
        mce=maximum_calibration_error(probs, truth, n_bins, mode),
        brier=brier_score(probs, truth),
        nll=negative_log_likelihood(probs, truth),
        accuracy=float(((probs >= 0.5).astype(float) == truth).mean()),
        base_rate=float(truth.mean()),
        mean_confidence=float(confidence.mean()),
        mean_outcome=float(correct.mean()),
        bins=tuple(bins),
        n_bins=n_bins,
    )


def multiclass_calibration_report(probabilities: Sequence[Sequence[float]],
                                  labels: Sequence[int],
                                  n_bins: int = DEFAULT_BINS) -> CalibrationReport:
    """Guo et al.'s calibration, for a field whose codebook levels are more than two.

    R4 requires calibration wherever accuracy is, and four of the codebook's fields are
    categorical - `b2_dominant`, `b3_zone`, `b4_posture`, `b4_arms`, `b5_type`. Reporting their
    accuracy in the frozen comparison without an ECE beside it would put the invariant's
    exception in the one table the thesis quotes from.

    **The same binning as the binary path, not a similar one.** Confidence is the maximum class
    probability and the outcome is whether the argmax was right, which is `confidence` mode's
    formulation extended to K classes, and the bins come from the same `_bins_from`.

    **Brier and NLL are the multiclass forms.** Brier is the mean squared error of the whole
    probability vector against the one-hot truth, which reduces to twice the binary Brier at
    K = 2 - stated because a reader comparing a binary field's Brier with a categorical one's
    needs to know they are not on the same scale. NLL is the negative log probability of the
    true class.

    **`base_rate` is the most common true class**, which is what accuracy has to beat before it
    means anything. For a binary field the same attribute holds the positive rate, so the two
    are not interchangeable across modes; `overconfident` reads `mean_outcome`, which is
    accuracy here, and is unaffected.
    """
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int).ravel()

    if probs.ndim != 2:
        raise ValueError(f"expected (n, classes) probabilities, got shape {probs.shape}")
    if probs.shape[0] != truth.size:
        raise ValueError(f"{probs.shape[0]} probability rows against {truth.size} labels")
    if probs.size == 0:
        raise ValueError("no predictions supplied")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0, 1]; pass logits through a softmax")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each row must sum to 1; pass logits through a softmax")
    if truth.min() < 0 or truth.max() >= probs.shape[1]:
        raise ValueError(
            f"labels must index the {probs.shape[1]} classes, got range "
            f"{truth.min()} to {truth.max()}")

    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (predicted == truth).astype(float)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(truth.size), truth] = 1.0
    bins = _bins_from(confidence, correct, n_bins)
    total = sum(b.count for b in bins)
    counts = np.bincount(truth, minlength=probs.shape[1])

    return CalibrationReport(
        n=int(truth.size),
        mode="confidence",
        ece=(sum(b.count * abs(b.gap) for b in bins) / total) if total else 0.0,
        mce=max((abs(b.gap) for b in bins), default=0.0),
        brier=float(((probs - one_hot) ** 2).sum(axis=1).mean()),
        nll=float(-np.mean(np.log(
            np.clip(probs[np.arange(truth.size), truth], EPSILON, 1.0)))),
        accuracy=float(correct.mean()),
        base_rate=float(counts.max() / truth.size),
        mean_confidence=float(confidence.mean()),
        mean_outcome=float(correct.mean()),
        bins=tuple(bins),
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# Uncertainty decomposition
# ---------------------------------------------------------------------------

def binary_entropy(probabilities: Sequence[float] | float) -> np.ndarray:
    """Shannon entropy of a Bernoulli, in nats. Maximal at p = 0.5."""
    probs = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


MAX_BINARY_ENTROPY = math.log(2.0)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram(report: CalibrationReport, path: str | Path,
                        title: str = "Reliability", n_bins: int = DEFAULT_BINS) -> Path:
    """Render a reliability diagram to PNG, for the thesis figures.

    Bar height is the delivered rate and the diagonal is perfect calibration, so the visible
    gap between them is ECE's summand. Bin population is drawn underneath because a bar over
    four predictions and a bar over four hundred look identical otherwise, and the difference
    is what decides whether a gap is a finding or a sampling artefact.

    matplotlib is imported here rather than at module scope: the inference path needs the
    metrics and never needs the plot, and importing pyplot costs seconds on this CPU.
    """
    import matplotlib
    matplotlib.use("Agg")           # no display on a server or in a container
    import matplotlib.pyplot as plt

    width = 1.0 / n_bins
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.0, 6.0), height_ratios=[3, 1], sharex=True)

    top.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888888",
             label="perfect calibration", zorder=1)
    if report.bins:
        centres = [(b.lower + b.upper) / 2 for b in report.bins]
        top.bar(centres, [b.empirical_rate for b in report.bins], width=width * 0.9,
                color="#3B6EA5", edgecolor="white", linewidth=0.5, label="delivered", zorder=2)
        top.bar(centres, [b.gap for b in report.bins],
                bottom=[b.empirical_rate for b in report.bins], width=width * 0.9,
                color="#C44E52", alpha=0.45, edgecolor="none", label="gap", zorder=3)
        bottom.bar(centres, [b.count for b in report.bins], width=width * 0.9,
                   color="#999999", edgecolor="white", linewidth=0.5)

    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.set_ylabel("observed frequency")
    # Three lines at this size fit the canvas. On one line the metrics run off the right edge
    # and whichever comes last is silently truncated, which on a thesis figure is a defect
    # nobody notices until the number they wanted is the missing one.
    top.set_title(f"{title}\n{report.caption()}", fontsize=9, linespacing=1.5)
    top.legend(loc="upper left", fontsize=8, frameon=False)

    bottom.set_xlabel("predicted probability")
    bottom.set_ylabel("count")
    bottom.set_xlim(0, 1)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
