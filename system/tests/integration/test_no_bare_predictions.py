# -*- coding: utf-8 -*-
"""The acceptance test API.md names by path, asserted on the raw JSON.

docs/API.md, opening note: "A detection whose `gate_outcome` is `suppress` must never have its
predicted value appear in a response body. Not greyed out, not nulled at render time, not
present-but-hidden. **Absent from the payload.** Every serialiser is written against this rule
and `tests/integration/test_no_bare_predictions.py` asserts it against the raw JSON, not
against Python objects."

**On the JSON, and on every behaviour.** Asserting `"predicted" not in payload` against a dict
would pass while a field leaked through some other key, and testing B1 alone would pass while
B3's six fields leaked. So the assertion is made on the serialised text, for all five
behaviours, for every suppression reason, and against every field name the codebook defines —
because a value can escape without its enclosing key.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from praxis.annotation.codebook import ACTIVE_CODEBOOK
from praxis.contracts.confidence import ConfidenceState
from praxis.contracts.detection import Detection
from praxis.ids import new_ulid
from praxis.routing.gate import GatePolicy, route
from praxis.routing.reveal import ReviewSession, evidence_only
from praxis.vocabulary import BEHAVIOUR_IDS
from tests.conftest import example_prediction

SUPPRESSING = {
    "low_confidence": dict(prob=0.20, epistemic=0.02, ood=False, abstained=False),
    "out_of_distribution": dict(prob=0.95, epistemic=0.02, ood=True, abstained=False),
    "model_abstained": dict(prob=0.99, epistemic=0.01, ood=False, abstained=True),
}


def state(prob: float, epistemic: float, ood: bool) -> ConfidenceState:
    return ConfidenceState(
        raw_prob=prob, calibrated_prob=prob, method="ensemble", epistemic=epistemic,
        ood_score=0.94 if ood else 0.08, ood_flag=ood,
        in_validated_domain="unknown" if ood else "classroom")


def build(behaviour: str, prob: float, epistemic: float, ood: bool,
          abstained: bool) -> Detection:
    detection = Detection(
        detection_id=new_ulid(), session_id=new_ulid(), behaviour=behaviour,
        t_start_s=128.0, t_end_s=136.0,
        predicted=example_prediction(behaviour, nonscorable=abstained),
        confidence=state(prob, epistemic, ood),
        evidence_ref=f"/api/v1/evidence/{behaviour}",
        model_version="b-resnet50-tcn-v3-ens5", gate_outcome="present")
    return route(detection, GatePolicy())


def assert_nothing_leaked(raw: str, detection: Detection) -> None:
    """No enclosing key, and no field name or value from inside it."""
    assert '"predicted"' not in raw
    assert '"calibrated_prob"' not in raw
    assert '"epistemic"' not in raw
    assert '"model_version"' not in raw
    assert '"explanation_ref"' not in raw
    for name, value in detection.predicted.items():
        assert f'"{name}"' not in raw, f"{name} leaked without its enclosing key"
        if isinstance(value, str):
            assert f'"{value}"' not in raw, f"the value {value!r} of {name} leaked"


@pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
@pytest.mark.parametrize("reason", sorted(SUPPRESSING))
def test_a_suppressed_detection_serialises_without_its_prediction(
        behaviour: str, reason: str) -> None:
    """Fifteen cases: every behaviour against every way a detection can be suppressed."""
    detection = build(behaviour, **SUPPRESSING[reason])
    assert detection.gate_outcome == "suppress"
    assert detection.suppression_reason == reason

    raw = json.dumps(detection.to_payload())
    assert_nothing_leaked(raw, detection)

    payload = json.loads(raw)
    assert payload["suppression_reason"] == reason
    assert payload["evidence_ref"].endswith(behaviour), (
        "the reviewer must still be able to find and judge the clip themselves")
    assert payload["confidence"]["plain_language"], (
        "a suppressed detection still explains itself in words a teacher educator can read")


@pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
def test_a_presented_detection_carries_its_whole_prediction(behaviour: str) -> None:
    """The control. Without it the test above would pass on a
    serialiser that emitted nothing."""
    detection = build(behaviour, prob=0.95, epistemic=0.02, ood=False, abstained=False)
    assert detection.gate_outcome == "present"

    payload = json.loads(json.dumps(detection.to_payload()))
    assert payload["predicted"] == detection.predicted
    assert set(payload["predicted"]) == ACTIVE_CODEBOOK.behaviour(behaviour).field_names
    assert payload["codebook_version"] == detection.codebook_version
    assert "requires_second_reviewer" not in payload


@pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
def test_an_escalated_detection_is_a_present_payload_plus_a_flag(behaviour: str) -> None:
    """API.md §6: "same shape as `present`, plus `requires_second_reviewer`: true"."""
    detection = build(behaviour, prob=0.95, epistemic=0.40, ood=False, abstained=False)
    assert detection.gate_outcome == "escalate"

    payload = json.loads(json.dumps(detection.to_payload()))
    assert payload["requires_second_reviewer"] is True
    assert payload["predicted"] == detection.predicted

    presented = json.loads(json.dumps(
        build(behaviour, 0.95, 0.02, False, False).to_payload()))
    assert set(payload) - set(presented) == {"requires_second_reviewer"}


@pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
def test_the_pre_reveal_payload_withholds_it_too(behaviour: str) -> None:
    """The second place a prediction could leak: before the reviewer has asked to see it.

    Suppression and the forcing function are different mechanisms with the same requirement,
    and a system that got one right and the other wrong would look correct in every test that
    only checked the gate.
    """
    detection = build(behaviour, prob=0.95, epistemic=0.02, ood=False, abstained=False)
    session = ReviewSession(reviewer_id=new_ulid(), session_id="S1", arm="dossier")

    raw = json.dumps(session.open(
        detection, datetime(2026, 9, 13, 9, 0, tzinfo=UTC)))
    assert_nothing_leaked(raw, detection)
    assert json.loads(raw)["indication_available"] is True, (
        "the reviewer is told an indication exists; they are not told what it says")

    assert json.dumps(evidence_only(detection)) == raw


# The shape docs/API.md §6 documents, key for key. An undocumented key is a contract breach in
# a system whose serialiser is the thing standing between a suppressed value and a client.
PRESENT_KEYS = {"detection_id", "session_id", "behaviour", "gate_outcome", "t_start_s",
                "t_end_s", "evidence_ref", "predicted", "codebook_version", "confidence",
                "model_version"}
SUPPRESS_KEYS = {"detection_id", "session_id", "behaviour", "gate_outcome", "t_start_s",
                 "t_end_s", "evidence_ref", "suppression_reason", "confidence"}


@pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
def test_the_payload_carries_exactly_the_documented_keys(behaviour: str) -> None:
    """Drift in either direction is a finding. Found by a mutation test.

    A leak test that only checks the *suppressed* payload cannot see a second key carrying the
    prediction in the presented one, and a serialiser that grows an undocumented field is one
    refactor away from growing it on the suppressed branch too.
    """
    presented = build(behaviour, prob=0.95, epistemic=0.02, ood=False, abstained=False)
    assert set(presented.to_payload()) == PRESENT_KEYS

    escalated = build(behaviour, prob=0.95, epistemic=0.40, ood=False, abstained=False)
    assert set(escalated.to_payload()) == PRESENT_KEYS | {"requires_second_reviewer"}

    suppressed = build(behaviour, **SUPPRESSING["low_confidence"])
    assert set(suppressed.to_payload()) == SUPPRESS_KEYS

    with_explanation = presented.model_copy(
        update={"explanation_ref": "/api/v1/explanations/x"}
    )
    assert set(with_explanation.to_payload()) == PRESENT_KEYS | {"explanation_ref"}


def test_the_suppressed_payload_is_a_strict_subset_of_the_presented_one() -> None:
    """Suppression removes keys; it never adds one that could carry a value in disguise."""
    suppressed = build("B3", **SUPPRESSING["low_confidence"]).to_payload()
    presented = build("B3", prob=0.95, epistemic=0.02, ood=False, abstained=False).to_payload()

    added = set(suppressed) - set(presented)
    assert added == {"suppression_reason"}, f"suppression introduced {added}"
    assert set(suppressed["confidence"]) < set(presented["confidence"])
