# -*- coding: utf-8 -*-
"""The contracts Phase 8 changed: a prediction is a label, and a correction is one too.

`Detection.predicted` stopped being a float and became the behaviour's codebook fields, checked
through `Codebook.validate_labels` — the same function a rater's label goes through. That is
what makes a model output and a human label comparable, so these tests are about the check being
real rather than decorative: a mutation removing the call must fail here.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from praxis.annotation.codebook import ACTIVE_CODEBOOK, CodebookError, ScaleType, get_codebook
from praxis.contracts.adjudication import Adjudication
from praxis.contracts.confidence import ConfidenceState
from praxis.contracts.detection import Detection
from praxis.ids import new_ulid
from praxis.vocabulary import BEHAVIOUR_IDS
from tests.conftest import example_prediction

NOW = datetime(2026, 9, 13, 9, 0, tzinfo=UTC)


def build(behaviour: str = "B1", **overrides) -> Detection:
    base = dict(
        detection_id=new_ulid(), session_id=new_ulid(), behaviour=behaviour,
        t_start_s=0.0, t_end_s=8.0, predicted=example_prediction(behaviour),
        confidence=ConfidenceState(
            raw_prob=0.9, calibrated_prob=0.9, method="ensemble", epistemic=0.03,
            ood_score=0.1, ood_flag=False, in_validated_domain="classroom"),
        evidence_ref="/e", model_version="v1", gate_outcome="present")
    base.update(overrides)
    return Detection(**base)


# ---------------------------------------------------------------------------
# A prediction is checked against the codebook, not merely shaped like a dict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("behaviour", BEHAVIOUR_IDS)
def test_a_complete_in_domain_prediction_is_accepted(behaviour: str) -> None:
    detection = build(behaviour)
    assert set(detection.predicted) == ACTIVE_CODEBOOK.behaviour(behaviour).field_names


def test_a_prediction_from_the_wrong_behaviour_is_refused() -> None:
    """A B1 detection carrying B3's fields would show a reviewer a zone where a
    count belongs."""
    with pytest.raises(ValidationError, match="has no field"):
        build("B1", predicted=example_prediction("B3"))


def test_a_field_the_codebook_does_not_define_is_refused() -> None:
    labels = example_prediction("B1") | {"b1_enthusiasm": 4}
    with pytest.raises(ValidationError, match="b1_enthusiasm"):
        build("B1", predicted=labels)


def test_a_value_outside_its_declared_domain_is_refused() -> None:
    for name, bad in (("b1_count", 99), ("b1_amplitude", 9), ("b1_present", "yes")):
        with pytest.raises(ValidationError, match="does not accept"):
            build("B1", predicted=example_prediction("B1") | {name: bad})


def test_a_partial_prediction_is_refused() -> None:
    """Shown to a reviewer, a prediction missing a field looks like a complete one."""
    labels = example_prediction("B3")
    del labels["b3_proximity"]
    with pytest.raises(ValidationError, match="omits"):
        build("B3", predicted=labels)


def test_an_unregistered_codebook_version_is_refused() -> None:
    """A prediction written in a vocabulary nothing has cannot be checked against anything."""
    with pytest.raises(ValidationError, match="not registered"):
        build("B1", codebook_version="v9.9-imaginary")

    assert get_codebook(ACTIVE_CODEBOOK.version) is ACTIVE_CODEBOOK
    with pytest.raises(CodebookError, match="known versions are"):
        get_codebook("v2.0")


def test_the_version_travels_with_the_prediction() -> None:
    """A later revision must not silently reinterpret an earlier detection."""
    assert build().codebook_version == ACTIVE_CODEBOOK.version
    assert build().to_payload()["codebook_version"] == ACTIVE_CODEBOOK.version


# ---------------------------------------------------------------------------
# bool is a subclass of int, and `in` does not care
# ---------------------------------------------------------------------------

def test_a_count_cannot_masquerade_as_a_presence_flag() -> None:
    """`0 in (False, True)` is true in Python. Without a type check, 0 passes as False."""
    presence = ACTIVE_CODEBOOK.behaviour("B1").field("b1_present")
    assert presence.accepts(False) and presence.accepts(True)
    assert not presence.accepts(0), "0 is a count, not an absence"
    assert not presence.accepts(1)

    with pytest.raises(ValidationError, match="does not accept"):
        build("B1", predicted=example_prediction("B1") | {"b1_present": 1})


def test_a_presence_flag_cannot_masquerade_as_an_ordinal_band() -> None:
    """`True in (1, 2, 3)` is also true, so the confusion runs both ways."""
    amplitude = ACTIVE_CODEBOOK.behaviour("B1").field("b1_amplitude")
    assert amplitude.accepts(1) and not amplitude.accepts(True)

    with pytest.raises(ValidationError, match="does not accept"):
        build("B1", predicted=example_prediction("B1") | {"b1_amplitude": True})


def test_an_integer_valued_float_still_satisfies_an_ordinal_level() -> None:
    """The type check is about bool, not about int against float. 2.0 is the band 2."""
    duration = ACTIVE_CODEBOOK.behaviour("B5").field("b5_duration")
    assert duration.accepts(2) and duration.accepts(2.0)


def test_a_boolean_is_still_refused_by_a_numeric_field() -> None:
    count = ACTIVE_CODEBOOK.behaviour("B1").field("b1_count")
    assert count.scale is ScaleType.RATIO
    assert count.accepts(3) and not count.accepts(True) and not count.accepts("3")


# ---------------------------------------------------------------------------
# A correction is a label too
# ---------------------------------------------------------------------------

def adjudication(detection: Detection, **overrides) -> Adjudication:
    base = dict(
        adjudication_id=new_ulid(), detection_id=detection.detection_id,
        reviewer_id=new_ulid(), action="edit", edited_value={"b1_count": 5},
        rationale="the count was three, not one", seconds_on_item=31.0,
        evidence_replays=1, revealed_indication=True, arm="dossier", created_at=NOW)
    base.update(overrides)
    return Adjudication(**base)


def test_an_edit_carries_codebook_fields_not_a_number() -> None:
    """SCHEMA.md stores `edited_value JSONB`; a reviewer corrects a field, not "the value"."""
    detection = build()
    edit = adjudication(detection)
    assert edit.edited_value == {"b1_count": 5}
    edit.validate_against(detection)

    with pytest.raises(ValidationError):
        adjudication(detection, edited_value=0.83)


def test_a_partial_edit_is_allowed_where_a_partial_prediction_is_not() -> None:
    """A reviewer corrects what was wrong; the model must answer everything it was asked."""
    detection = build()
    adjudication(detection, edited_value={"b1_amplitude": 3}).validate_against(detection)


def test_an_edit_naming_a_field_the_codebook_lacks_is_refused() -> None:
    detection = build()
    with pytest.raises(CodebookError, match="has no field"):
        adjudication(detection, edited_value={"b1_vigour": 2}).validate_against(detection)

    with pytest.raises(CodebookError, match="does not accept"):
        adjudication(detection, edited_value={"b1_count": 500}).validate_against(detection)


def test_an_edit_is_checked_against_its_own_detection() -> None:
    """Validating against the wrong detection would check a B1 edit against B3's fields."""
    detection, other = build("B1"), build("B3")
    with pytest.raises(ValueError, match="is for detection"):
        adjudication(detection).validate_against(other)


def test_a_confirmation_needs_no_value_and_no_rationale() -> None:
    detection = build()
    confirmed = adjudication(detection, action="confirm", edited_value=None, rationale=None)
    confirmed.validate_against(detection)      # a no-op for anything but an edit

    with pytest.raises(ValidationError, match="requires a rationale"):
        adjudication(detection, action="reject", edited_value=None, rationale="too short")
    with pytest.raises(ValidationError, match="requires edited_value"):
        adjudication(detection, action="edit", edited_value=None)
    with pytest.raises(ValidationError, match="requires edited_value"):
        adjudication(detection, action="edit", edited_value={})


def test_an_adjudication_cannot_be_edited_in_place() -> None:
    """R5 in Python, mirroring the database trigger. A correction is a new row."""
    edit = adjudication(build())
    with pytest.raises(ValidationError):
        edit.action = "confirm"


def test_a_detection_built_after_a_new_codebook_is_activated_uses_it(monkeypatch) -> None:
    """The Delphi panel's revisions become v2.0, and detections made after must say so.

    `Field(default=ACTIVE_CODEBOOK.version)` reads the version once, when the class is defined.
    A detection built after a newer codebook was activated would then still claim the old one —
    a wrong provenance record rather than a crash, which is the kind that survives review.
    `default_factory` reads it at construction instead.
    """
    from praxis.annotation import codebook as codebook_module

    next_version = ACTIVE_CODEBOOK.model_copy(update={"version": "v2.0-test"})
    monkeypatch.setitem(codebook_module.CODEBOOKS, "v2.0-test", next_version)
    monkeypatch.setattr(codebook_module, "ACTIVE_CODEBOOK", next_version)
    monkeypatch.setattr("praxis.contracts.detection.ACTIVE_CODEBOOK", next_version)

    assert build().codebook_version == "v2.0-test"
    assert build(codebook_version=ACTIVE_CODEBOOK.version).codebook_version == "v1.0-draft", (
        "an explicit version still wins; only the default follows the active codebook")
