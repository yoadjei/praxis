# -*- coding: utf-8 -*-
"""Phase 8: the routing gate, and the cognitive forcing function in front of it.

The gate's acceptance condition is not that each outcome is reachable. It is that **exactly
one** outcome is reachable for any input, because BUILD-SPEC lists three conditions that
overlap and says nothing about precedence. `test_every_input_has_exactly_one_outcome` sweeps
the combinations; the tests around it pin which one wins and why.
"""
from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from praxis.annotation.codebook import CodebookError
from praxis.contracts.confidence import ConfidenceState
from praxis.contracts.detection import Detection
from praxis.ids import new_ulid
from praxis.routing.gate import (
    MAX_BINARY_ENTROPY_NATS,
    GateError,
    GatePolicy,
    decide,
    route,
    route_all,
    summarise,
)
from praxis.routing.reveal import (
    RevealError,
    RevealPolicy,
    ReviewSession,
    elapsed,
    evidence_only,
)
from tests.conftest import example_prediction

POLICY = GatePolicy()
START = datetime(2026, 9, 13, 9, 0, 0, tzinfo=UTC)


def confidence(prob: float = 0.91, epistemic: float | None = 0.04, ood: bool = False,
               method: str | None = None) -> ConfidenceState:
    """D16 forbids an ensemble without an epistemic term, so the method follows from it here."""
    if method is None:
        method = "ensemble" if epistemic is not None else "temperature"
    return ConfidenceState(
        raw_prob=prob, calibrated_prob=prob, method=method, epistemic=epistemic,
        ood_score=0.94 if ood else 0.12, ood_flag=ood,
        in_validated_domain="unknown" if ood else "microteaching")


def detection_for(behaviour: str = "B1", nonscorable: bool = False, **kwargs) -> Detection:
    return Detection(
        detection_id=new_ulid(), session_id="S1", behaviour=behaviour,
        t_start_s=0.0, t_end_s=8.0,
        predicted=example_prediction(behaviour, nonscorable=nonscorable),
        confidence=kwargs.pop("confidence", confidence()),
        evidence_ref="/api/v1/evidence/x", model_version="b-resnet50-tcn-v3-ens5",
        gate_outcome=kwargs.pop("gate_outcome", "present"), **kwargs)


# ---------------------------------------------------------------------------
# Exhaustive and mutually exclusive
# ---------------------------------------------------------------------------

def test_every_input_has_exactly_one_outcome() -> None:
    """The gate is a total function. No input falls through and none matches twice."""
    outcomes = set()
    for prob, epistemic, ood, abstained in itertools.product(
            (0.0, 0.69, 0.70, 0.71, 1.0), (None, 0.0, 0.149, 0.15, 0.60),
            (False, True), (False, True)):
        decision = decide(confidence(prob, epistemic, ood), POLICY, abstained=abstained)
        assert decision.outcome in ("present", "suppress", "escalate")
        assert (decision.suppression_reason is not None) == (decision.outcome == "suppress"), (
            "a suppression without a reason, or a reason without a suppression")
        outcomes.add(decision.outcome)

    assert outcomes == {"present", "suppress", "escalate"}, (
        "the sweep must actually reach all three, or it proves only that one is reachable")


def test_suppression_beats_escalation() -> None:
    """The precedence decision. Escalating an out-of-domain clip would launder it.

    Escalation shows the indication to a *second* reviewer. Doing that when the system is
    outside the domain it was validated on is worse than showing it to one, so suppression
    wins every tie.
    """
    both = confidence(prob=0.95, epistemic=0.60, ood=True)
    decision = decide(both, POLICY)
    assert decision.outcome == "suppress"
    assert decision.suppression_reason == "out_of_distribution"

    low_and_disagreeing = confidence(prob=0.20, epistemic=0.60, ood=False)
    assert decide(low_and_disagreeing, POLICY).suppression_reason == "low_confidence"


def test_abstention_beats_everything() -> None:
    """A model that marked the clip non-scorable has declined, however confident it looks."""
    decision = decide(confidence(prob=0.99, epistemic=0.0), POLICY, abstained=True)
    assert decision.outcome == "suppress"
    assert decision.suppression_reason == "model_abstained"
    assert "declining to answer" in decision.rule

    detection = detection_for(nonscorable=True)
    assert detection.is_nonscorable
    assert route(detection, POLICY).suppression_reason == "model_abstained"


def test_thresholds_are_inclusive_because_they_are_named_min() -> None:
    """`min_calibrated_prob` is a minimum, so meeting it exactly passes. Stated, not assumed."""
    assert decide(confidence(prob=0.70, epistemic=0.0), POLICY).outcome == "present"
    assert decide(confidence(prob=0.6999, epistemic=0.0), POLICY).outcome == "suppress"

    assert decide(confidence(prob=0.9, epistemic=0.15), POLICY).outcome == "escalate"
    assert decide(confidence(prob=0.9, epistemic=0.1499), POLICY).outcome == "present"


# ---------------------------------------------------------------------------
# The trap: epistemic is None for two of the three calibration methods
# ---------------------------------------------------------------------------

def test_a_method_without_disagreement_never_escalates_and_says_so() -> None:
    """Treating a missing epistemic term as zero would report "no ambiguous cases" falsely.

    Temperature scaling produces no epistemic value at all. Silently reading `None` as 0.0
    makes every such detection present, and a low escalation rate would then be a fact about
    the calibration method rather than about the corpus.
    """
    for method in ("temperature", "mc_dropout"):
        decision = decide(confidence(prob=0.95, epistemic=None, method=method), POLICY)
        assert decision.outcome == "present"
        assert decision.epistemic_measurable is False
        assert "could not have escalated on any threshold" in decision.rule

    summary = summarise([decide(confidence(0.95, None, method="temperature"), POLICY),
                         decide(confidence(0.95, 0.02), POLICY)])
    assert summary.n_epistemic_not_measurable == 1
    assert "could not be assessed for disagreement" in summary.caption()


def test_a_method_without_disagreement_can_still_be_suppressed() -> None:
    """Only the escalation rule needs the epistemic term; the other three do not."""
    decision = decide(confidence(prob=0.3, epistemic=None, method="temperature"), POLICY)
    assert decision.outcome == "suppress" and decision.suppression_reason == "low_confidence"


def test_a_zero_threshold_still_does_not_escalate_an_unmeasurable_detection() -> None:
    """Where `None` read as zero stops being harmless. Found by a mutation test.

    At the default threshold of 0.15, `(None or 0.0) >= 0.15` is false and the bug hides. At a
    threshold of 0.0 — which the policy permits and which "escalate on any disagreement at all"
    would mean — it becomes true, and every temperature-scaled detection escalates on
    disagreement that was never measured.
    """
    permissive = GatePolicy(min_epistemic=0.0)

    assert decide(confidence(0.95, epistemic=0.0), permissive).outcome == "escalate", (
        "a measured zero does reach a zero threshold")
    assert decide(confidence(0.95, epistemic=None), permissive).outcome == "present", (
        "an unmeasured one must not, however low the threshold goes")


# ---------------------------------------------------------------------------
# The policy refuses what it cannot mean
# ---------------------------------------------------------------------------

def test_the_policy_refuses_a_threshold_on_the_wrong_scale() -> None:
    """`epistemic` is mutual information in nats, capped at ln 2. 0.8 is not a nats threshold.

    The normalised [0, 1] disagreement score from `ood.py` and the raw nats from
    `ensemble.py` are different numbers, and a threshold copied from one to the other would
    silently make escalation impossible.
    """
    with pytest.raises(GateError, match="normalised"):
        GatePolicy(min_epistemic=0.80)

    GatePolicy(min_epistemic=MAX_BINARY_ENTROPY_NATS)  # the ceiling itself is legal

    with pytest.raises(GateError, match="cannot be"):
        GatePolicy(min_epistemic=-0.01)
    with pytest.raises(GateError, match="not a probability"):
        GatePolicy(min_calibrated_prob=1.5)


def test_the_policy_refuses_to_present_flagged_detections() -> None:
    with pytest.raises(GateError, match="R3 does not permit it"):
        GatePolicy(require_no_ood_flag=False)


def test_the_policy_is_read_from_the_config_and_travels_with_the_decision(config) -> None:
    """A decision that cannot say which thresholds produced it is not auditable."""
    policy = GatePolicy.from_config(config)
    assert policy.min_calibrated_prob == config.routing.present_when.min_calibrated_prob
    assert policy.min_epistemic == config.routing.escalate_when.min_epistemic

    decision = decide(confidence(), policy)
    assert decision.policy is policy
    assert "PROVISIONAL" in decision.explain(), (
        "a figure computed under an unswept threshold must say so")
    assert f"{policy.min_calibrated_prob:.2f}" in decision.explain()


def test_routing_a_batch_reports_what_it_did() -> None:
    detections = [
        detection_for(confidence=confidence(0.95, 0.02)),
        detection_for(confidence=confidence(0.95, 0.40)),
        detection_for(confidence=confidence(0.30, 0.02)),
        detection_for(confidence=confidence(0.95, 0.02, ood=True)),
        detection_for(nonscorable=True),
    ]
    routed, summary = route_all(detections, POLICY)

    assert [d.gate_outcome for d in routed] == [
        "present", "escalate", "suppress", "suppress", "suppress"]
    assert summary.counts == {"present": 1, "escalate": 1, "suppress": 3}
    assert summary.suppression_reasons == {
        "low_confidence": 1, "out_of_distribution": 1, "model_abstained": 1}
    assert summary.suppression_rate == pytest.approx(0.6)
    assert all(d.suppression_reason is None for d in routed if d.gate_outcome != "suppress")


# ---------------------------------------------------------------------------
# Evidence before suggestion
# ---------------------------------------------------------------------------

def test_the_forcing_arm_withholds_the_indication_from_the_payload() -> None:
    """Not greyed out in a template. Absent from the dict, asserted on the JSON."""
    detection = detection_for()
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="dossier")

    raw = json.dumps(session.open(detection, START))
    assert '"predicted"' not in raw
    assert '"calibrated_prob"' not in raw
    assert '"confidence"' not in raw
    for name in detection.predicted:
        assert f'"{name}"' not in raw, f"{name} reached the reviewer before they asked"
    assert '"evidence_ref"' in raw, "the reviewer must still be able to find the clip"


def test_the_comparison_arm_sees_the_indication_immediately() -> None:
    """`no_dossier` is the always-on arm the reliance study compares against."""
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="no_dossier")
    payload = session.open(detection_for(), START)

    assert "predicted" in payload
    assert session.items[payload["detection_id"]].revealed is True, (
        "the comparison arm still records a reveal, so both arms answer the same question")


def test_the_forcing_function_is_never_applied_to_the_comparison_arm() -> None:
    """Even with the config switch on, which is what makes it a comparison."""
    policy = RevealPolicy(enabled=True, require_reveal_action=True)
    assert policy.forces("dossier") is True
    assert policy.forces("no_dossier") is False
    assert RevealPolicy(enabled=False).forces("dossier") is False


def test_revealing_a_suppressed_detection_is_refused() -> None:
    """"Revealed and there was nothing" and "never revealed" are different data points."""
    suppressed = detection_for(
        confidence=confidence(0.2), gate_outcome="suppress",
        suppression_reason="low_confidence")
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="dossier")
    payload = session.open(suppressed, START)
    assert payload["indication_available"] is False

    with pytest.raises(RevealError, match="no indication to reveal"):
        session.reveal(suppressed.detection_id, elapsed(START, 5))


def test_the_first_reveal_is_the_one_measured() -> None:
    """Re-reading is normal and must not overwrite the timestamp H6 is computed from."""
    detection = detection_for()
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="dossier")
    session.open(detection, START)

    session.reveal(detection.detection_id, elapsed(START, 12))
    session.reveal(detection.detection_id, elapsed(START, 40))

    item = session.items[detection.detection_id]
    assert item.seconds_before_reveal() == pytest.approx(12.0)
    assert session.reveal_order == [detection.detection_id], "re-reading is not a second reveal"


def test_reveal_order_and_instrumentation_are_recorded_for_h6_and_h7() -> None:
    first, second, third = detection_for(), detection_for(), detection_for()
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="dossier")
    for detection in (first, second, third):
        session.open(detection, START)

    session.reveal(third.detection_id, elapsed(START, 5))
    session.reveal(first.detection_id, elapsed(START, 30))
    session.replay_evidence(first.detection_id)
    assert session.replay_evidence(first.detection_id) == 2
    session.decide(first.detection_id, elapsed(START, 45))
    session.decide(second.detection_id, elapsed(START, 50))

    assert session.reveal_order == [third.detection_id, first.detection_id]
    assert session.items[third.detection_id].reveal_position == 0
    assert session.items[first.detection_id].reveal_position == 1

    instrumentation = session.instrumentation(first.detection_id)
    assert instrumentation == {"seconds_on_item": pytest.approx(45.0),
                               "evidence_replays": 2,
                               "revealed_indication": True,
                               "arm": "dossier"}

    assert session.unrevealed == [second.detection_id], (
        "deciding from evidence alone is the forcing function working, not an error")
    assert session.instrumentation(second.detection_id)["revealed_indication"] is False
    assert session.median_seconds_before_reveal() == pytest.approx(17.5)


def test_an_item_cannot_be_reopened_or_worked_on_unopened() -> None:
    detection = detection_for()
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="dossier")
    session.open(detection, START)

    with pytest.raises(RevealError, match="already open"):
        session.open(detection, elapsed(START, 10))
    with pytest.raises(RevealError, match="never opened"):
        session.reveal(new_ulid(), START)


def test_evidence_only_is_built_not_filtered() -> None:
    """A build-then-strip path leaks the first time a field is added and the strip forgotten.

    Asserted structurally: the evidence payload's keys are a fixed set that does not grow when
    the full payload does.
    """
    payload = evidence_only(detection_for())
    assert set(payload) == {"detection_id", "session_id", "behaviour", "t_start_s", "t_end_s",
                            "evidence_ref", "indication_available"}


# ---------------------------------------------------------------------------
# Found by adversarial review: a validator nobody calls is not a guarantee
# ---------------------------------------------------------------------------

def test_the_review_session_validates_an_edit_against_its_own_detection() -> None:
    """`Adjudication.validate_against` existed and nothing called it.

    The review session is the only place the adjudication and the detection it corrects are
    both in scope, so building the adjudication there means the check cannot be skipped by a
    call site that simply forgot it.
    """
    detection = detection_for()
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="dossier")
    session.open(detection, START)
    session.reveal(detection.detection_id, elapsed(START, 9))

    with pytest.raises(CodebookError, match="has no field"):
        session.adjudicate(detection.detection_id, "edit", elapsed(START, 30),
                           edited_value={"b1_vigour": 2},
                           rationale="the gesture was larger than that")

    with pytest.raises(CodebookError, match="does not accept"):
        session.adjudicate(detection.detection_id, "edit", elapsed(START, 30),
                           edited_value={"b1_count": 500},
                           rationale="the gesture was larger than that")


def test_an_adjudication_carries_the_instrumentation_that_was_observed() -> None:
    """Filled in from the session, not passed in: a caller could supply anything."""
    detection = detection_for()
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="dossier")
    session.open(detection, START)
    session.replay_evidence(detection.detection_id)
    session.reveal(detection.detection_id, elapsed(START, 20))

    decision = session.adjudicate(detection.detection_id, "edit", elapsed(START, 50),
                                  edited_value={"b1_count": 4},
                                  rationale="four gestures, not the two indicated")

    assert decision.seconds_on_item == pytest.approx(50.0)
    assert decision.evidence_replays == 1
    assert decision.revealed_indication is True
    assert decision.arm == "dossier"
    assert decision.detection_id == detection.detection_id
    assert session.items[detection.detection_id].decided_at == elapsed(START, 50)


def test_an_adjudication_without_a_reveal_records_that_too() -> None:
    """Deciding from evidence alone is the forcing function working, and must be recorded."""
    detection = detection_for()
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="dossier")
    session.open(detection, START)

    decision = session.adjudicate(detection.detection_id, "confirm", elapsed(START, 14))
    assert decision.revealed_indication is False
    assert decision.action == "confirm" and decision.edited_value is None


def test_an_edit_without_a_rationale_is_refused_before_it_reaches_the_trail() -> None:
    """API.md returns 422 for this; the contract refuses it before an endpoint could."""
    detection = detection_for()
    session = ReviewSession(reviewer_id="R1", session_id="S1", arm="no_dossier")
    session.open(detection, START)

    with pytest.raises(ValidationError, match="requires a rationale"):
        session.adjudicate(detection.detection_id, "edit", elapsed(START, 20),
                           edited_value={"b1_count": 4}, rationale="short")
