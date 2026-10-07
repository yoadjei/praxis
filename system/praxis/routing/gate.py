# -*- coding: utf-8 -*-
"""The routing gate: which detections a reviewer is shown, and which are withheld.

Three outcomes, and the order they are tested in is the policy:

    1. the model declined to answer          -> suppress, model_abstained
    2. the input is outside the validated domain -> suppress, out_of_distribution
    3. calibrated confidence is below the floor  -> suppress, low_confidence
    4. the ensemble members disagree             -> escalate
    5. otherwise                                 -> present

**The order is the decision, not an implementation detail.** BUILD-SPEC lists the three
outcomes without saying what happens when a detection satisfies more than one, and detections
routinely will: a clip can be both low-confidence and high-disagreement. Tested in the order
above, suppression always beats escalation. The reason is what escalation does — it puts the
model's indication in front of a *second* reviewer. If the system is outside the domain it was
validated on, showing an unreliable indication to two people is worse than showing it to one,
because the second opinion launders it. Escalation is therefore reserved for detections that
are in-domain and confident, where the disagreement between members is genuinely about the
clip rather than about the model being lost.

**Disagreement cannot escalate what the model could not measure.** `epistemic` is `None` for
temperature scaling and for MC dropout, and the one thing that must not happen is treating
`None` as zero: the gate would then never escalate, and a report of "no ambiguous cases" would
be a statement about the calibration method rather than about the corpus. A detection whose
method cannot produce disagreement is recorded as not eligible for escalation, and the count
is reported.

**The gate reads `ood_flag`; it does not re-derive it.** Phase 6 chooses the threshold that
turns `ood_score` into that boolean, by a stated criterion, and records it. Thresholding the
score again here would put a second threshold in the system, and the two would drift.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from praxis.contracts.confidence import ConfidenceState
from praxis.contracts.detection import Detection
from praxis.vocabulary import GateOutcome, SuppressionReason

# `epistemic` is mutual information in nats, as `confidence/ensemble.py` produces it. The
# maximum a binary variable can carry is ln 2, so a threshold of 0.15 nats is 21.6 per cent of
# the maximum. Stated because a threshold in nats and a threshold on the normalised [0, 1]
# disagreement score are different numbers and nothing in the config says which it is.
EPISTEMIC_UNITS = "nats"
MAX_BINARY_ENTROPY_NATS = 0.6931471805599453


class GateError(RuntimeError):
    """Raised when the gate is asked to route something it cannot route honestly."""


@dataclass(frozen=True)
class GatePolicy:
    """The thresholds, carried with every decision so a result can never be read without them.

    Both are marked PROVISIONAL in `configs/default.yaml`. `min_calibrated_prob` in particular
    has no sweep behind it yet: Phase 6 chose the OOD threshold by stated criterion, not this
    one. `provisional` is therefore part of the policy rather than a comment, and it travels
    into the decision so that a figure computed under an unswept threshold says so.
    """

    min_calibrated_prob: float = 0.70
    min_epistemic: float = 0.15
    require_no_ood_flag: bool = True
    provisional: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_calibrated_prob <= 1.0:
            raise GateError(
                f"min_calibrated_prob={self.min_calibrated_prob} is not a probability")
        if self.min_epistemic < 0.0:
            raise GateError(f"min_epistemic={self.min_epistemic} is negative; "
                            f"mutual information cannot be")
        if self.min_epistemic > MAX_BINARY_ENTROPY_NATS:
            raise GateError(
                f"min_epistemic={self.min_epistemic} {EPISTEMIC_UNITS} exceeds ln 2 = "
                f"{MAX_BINARY_ENTROPY_NATS:.4f}, the most a binary prediction can carry, so "
                f"nothing could ever escalate. Was this threshold meant to be on the "
                f"normalised [0, 1] disagreement score instead?")
        if not self.require_no_ood_flag:
            raise GateError(
                "require_no_ood_flag=False would present detections the system has flagged as "
                "outside the domain it was validated on. R3 does not permit it.")

    @classmethod
    def from_config(cls, config) -> GatePolicy:
        """Read the thresholds from the loaded configuration, never from a literal."""
        routing = config.routing
        return cls(min_calibrated_prob=routing.present_when.min_calibrated_prob,
                   min_epistemic=routing.escalate_when.min_epistemic,
                   require_no_ood_flag=not routing.present_when.ood_flag)

    def describe(self) -> str:
        marker = " (PROVISIONAL)" if self.provisional else ""
        return (f"present when calibrated_prob >= {self.min_calibrated_prob:.2f} and not "
                f"ood_flag; escalate when epistemic >= {self.min_epistemic:.3f} "
                f"{EPISTEMIC_UNITS}{marker}")


@dataclass(frozen=True)
class GateDecision:
    """One routing outcome, with the reason and the thresholds that produced it.

    The decision is auditable on its own: `explain()` reconstructs why, months later, without
    the configuration that was loaded at the time.
    """

    outcome: GateOutcome
    suppression_reason: SuppressionReason | None
    rule: str
    policy: GatePolicy
    epistemic_measurable: bool

    @property
    def shows_indication(self) -> bool:
        """Whether the reviewer sees the model's answer at all."""
        return self.outcome != "suppress"

    def explain(self) -> str:
        detail = f" ({self.suppression_reason})" if self.suppression_reason else ""
        return f"{self.outcome}{detail}: {self.rule}. Policy: {self.policy.describe()}"


def decide(confidence: ConfidenceState, policy: GatePolicy,
           abstained: bool = False) -> GateDecision:
    """Route one detection. Exhaustive and mutually exclusive by construction.

    `abstained` is the model's own non-scorable judgement. It is passed separately rather than
    read off the prediction because the gate must be usable before a `Detection` exists —
    inference calls it to *compute* `gate_outcome`, which is then stored on the row.
    """
    measurable = confidence.epistemic is not None

    if abstained:
        return GateDecision(
            "suppress", "model_abstained",
            "the model marked this clip non-scorable for this behaviour, which is the model "
            "declining to answer rather than answering uncertainly",
            policy, measurable)

    if policy.require_no_ood_flag and confidence.ood_flag:
        return GateDecision(
            "suppress", "out_of_distribution",
            f"ood_flag is set (score {confidence.ood_score:.4f}), so this recording is unlike "
            f"those the model was checked on",
            policy, measurable)

    if confidence.calibrated_prob < policy.min_calibrated_prob:
        return GateDecision(
            "suppress", "low_confidence",
            f"calibrated_prob {confidence.calibrated_prob:.4f} is below the floor of "
            f"{policy.min_calibrated_prob:.2f}",
            policy, measurable)

    if measurable and confidence.epistemic >= policy.min_epistemic:
        return GateDecision(
            "escalate", None,
            f"members disagree: epistemic {confidence.epistemic:.4f} {EPISTEMIC_UNITS} reaches "
            f"the threshold of {policy.min_epistemic:.3f}, and the detection is in-domain and "
            f"above the confidence floor, so a second reviewer is asked rather than nothing "
            f"being shown",
            policy, measurable)

    if not measurable:
        return GateDecision(
            "present", None,
            f"calibrated_prob {confidence.calibrated_prob:.4f} clears the floor and the domain "
            f"is validated. Disagreement was not assessed: method "
            f"{confidence.method!r} produces no epistemic term, so this detection could not "
            f"have escalated on any threshold",
            policy, measurable)

    return GateDecision(
        "present", None,
        f"calibrated_prob {confidence.calibrated_prob:.4f} clears the floor, the domain is "
        f"validated, and epistemic {confidence.epistemic:.4f} {EPISTEMIC_UNITS} is below "
        f"{policy.min_epistemic:.3f}",
        policy, measurable)


def route(detection: Detection, policy: GatePolicy) -> Detection:
    """Apply the gate to an existing detection, returning the routed copy.

    A new object rather than a mutation: `Detection` is frozen, and a routing decision that
    could be applied in place is a routing decision that could be applied twice.
    """
    decision = decide(detection.confidence, policy, abstained=detection.is_nonscorable)
    return detection.model_copy(update={
        "gate_outcome": decision.outcome,
        "suppression_reason": decision.suppression_reason})


@dataclass(frozen=True)
class RoutingSummary:
    """What the gate did to a batch. Reported per session, because the reviewer's load is."""

    counts: dict[str, int] = field(default_factory=dict)
    suppression_reasons: dict[str, int] = field(default_factory=dict)
    n_epistemic_not_measurable: int = 0
    policy: GatePolicy | None = None

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def suppression_rate(self) -> float:
        return self.counts.get("suppress", 0) / self.total if self.total else 0.0

    def caption(self) -> str:
        parts = [f"{name} {self.counts.get(name, 0)}"
                 for name in ("present", "suppress", "escalate")]
        line = f"{self.total} detections: " + ", ".join(parts)
        if self.n_epistemic_not_measurable:
            line += (f"; {self.n_epistemic_not_measurable} could not be assessed for "
                     f"disagreement and so could never escalate")
        return line


def summarise(decisions: Iterable[GateDecision]) -> RoutingSummary:
    """Count the outcomes, and count separately what could not be assessed.

    The second count is the one that matters and the one a plain tally would hide. If most
    detections came from a method with no epistemic term, a low escalation rate says nothing
    about how often the model was uncertain.
    """
    decisions = list(decisions)
    outcomes: Counter[str] = Counter(d.outcome for d in decisions)
    reasons: Counter[str] = Counter(
        d.suppression_reason for d in decisions if d.suppression_reason is not None)
    return RoutingSummary(
        counts=dict(outcomes),
        suppression_reasons=dict(reasons),
        n_epistemic_not_measurable=sum(1 for d in decisions if not d.epistemic_measurable),
        policy=decisions[0].policy if decisions else None)


def route_all(detections: Sequence[Detection],
              policy: GatePolicy) -> tuple[list[Detection], RoutingSummary]:
    """Route a session's detections and report what the gate did."""
    decisions = [decide(d.confidence, policy, abstained=d.is_nonscorable) for d in detections]
    routed = [d.model_copy(update={"gate_outcome": decision.outcome,
                                   "suppression_reason": decision.suppression_reason})
              for d, decision in zip(detections, decisions, strict=True)]
    return routed, summarise(decisions)
