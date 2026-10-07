# -*- coding: utf-8 -*-
"""Phase 8: the append-only trail, its hash chain, and the replay that proves it complete.

Two acceptance conditions are met here. "Replaying the audit log reconstructs the final state
exactly" is tested as an *equality* against a state maintained live, not as an inspection of a
few remembered fields — which is what makes it able to catch a payload too thin to replay.
`test_audit_trail_immutable` stays in `tests/test_invariants.py`, where it greps the migrations;
it tightens to a live UPDATE and DELETE once PostgreSQL is installed.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from praxis.audit.chain import (
    GENESIS_PREV_HASH,
    AuditLog,
    AuditRecord,
    canonical_payload,
    canonical_timestamp,
    verify,
)
from praxis.audit.events import EVENTS, AuditError, spec_for
from praxis.audit.replay import SystemState, replay
from praxis.ids import new_ulid
from scripts.verify_audit_chain import load

START = datetime(2026, 9, 13, 9, 0, 0, tzinfo=UTC)


def at(seconds: int) -> datetime:
    return START + timedelta(seconds=seconds)


def adjudication_payload(detection_id: str, reviewer_id: str, action: str = "confirm",
                         **overrides) -> dict:
    payload = {
        "adjudication_id": new_ulid(), "detection_id": detection_id,
        "reviewer_id": reviewer_id, "action": action, "seconds_on_item": 31.5,
        "evidence_replays": 2, "revealed_indication": True, "arm": "dossier"}
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# The taxonomy is closed, and payloads are checked at append time
# ---------------------------------------------------------------------------

def test_an_undeclared_event_is_refused() -> None:
    """An event nobody declared is an event nothing knows how to replay."""
    log = AuditLog()
    with pytest.raises(AuditError, match="not an audited event type"):
        log.append("detection.peeked", "D1", {}, at(0))


def test_a_payload_too_thin_to_replay_is_refused_when_written() -> None:
    """The hole is found at the call site, not when the trail is finally asked a question."""
    log = AuditLog()
    with pytest.raises(AuditError, match="omits"):
        log.append("adjudication.created", "A1",
                   {"adjudication_id": "A1", "detection_id": "D1", "reviewer_id": "R1",
                    "action": "confirm"}, at(0))


def test_the_reliance_study_fields_are_required_by_the_taxonomy() -> None:
    """§4.4: "do not remove them as unused". A trail without them has lost H6 and H7's data."""
    required = set(spec_for("adjudication.created").required_fields)
    assert {"seconds_on_item", "evidence_replays", "revealed_indication"} <= required


def test_config_changed_exists_although_schema_md_omits_it() -> None:
    """BUILD-SPEC requires every config change logged; SCHEMA.md's table has no event for it."""
    assert "config.changed" in EVENTS
    assert set(EVENTS["config.changed"].required_fields) == {
        "config_key", "old_value", "new_value", "config_sha256"}


# ---------------------------------------------------------------------------
# Canonical encoding: the four things SCHEMA.md leaves open
# ---------------------------------------------------------------------------

def test_a_naive_timestamp_is_refused() -> None:
    """It would hash as whatever the writing machine's clock meant, and verify elsewhere."""
    log = AuditLog()
    with pytest.raises(AuditError, match="no timezone"):
        log.append("consent.withdrawn", "T1", {"teacher_id": "T1", "scope": "all"},
                   datetime(2026, 9, 13, 9, 0, 0))


def test_the_same_instant_in_two_zones_hashes_identically() -> None:
    """Normalised to UTC, so where a row was written cannot change its hash."""
    elsewhere = START.astimezone(timezone(timedelta(hours=5, minutes=30)))
    assert canonical_timestamp(elsewhere) == canonical_timestamp(START)
    assert canonical_timestamp(START).endswith("Z")
    assert canonical_timestamp(START) == "2026-09-13T09:00:00.000000Z"


def test_payload_key_order_does_not_change_the_hash() -> None:
    """Two equal payloads must have one representation, or verification is a coin toss."""
    assert canonical_payload({"b": 2, "a": 1}) == canonical_payload({"a": 1, "b": 2})
    assert canonical_payload({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_the_genesis_row_links_to_the_empty_string_not_to_the_text_null() -> None:
    """"NULL" is a plausible value of some other field; the empty string is unambiguous."""
    log = AuditLog()
    assert log.head == GENESIS_PREV_HASH == ""
    record = log.append("codebook.activated", "v1.0-draft",
                        {"version": "v1.0-draft", "document_sha256": "a" * 64}, at(0))
    assert record.prev_hash == ""
    assert record.as_row()["prev_hash"] is None, "stored as SQL NULL, hashed as empty"


def test_fields_are_framed_so_a_character_cannot_move_across_a_boundary() -> None:
    """The reason for length-prefixing. Plain concatenation cannot tell "ab"+"c" from "a"+"bc".

    The two fields must be **adjacent in the digest order** or the test proves nothing: an
    intervening field separates them and the concatenations differ anyway. `entity_type` and
    `entity_id` are adjacent, and `actor_user_id` and `actor_role` are the other such pair.
    Caught by a mutation test — with non-adjacent fields this passed with the framing removed.
    """
    for first, second in (("entity_type", "entity_id"),
                          ("actor_user_id", "actor_role")):
        base = dict(occurred_at=START, event_type="e", entity_type="t", entity_id="E",
                    payload={}, actor_user_id="u", actor_role="r")
        left = AuditRecord(**{**base, first: "ab", second: "c"})
        right = AuditRecord(**{**base, first: "a", second: "bc"})
        assert left.digest() != right.digest(), (
            f"{first} and {second} are concatenated without framing, so a character can be "
            f"moved from one into the other and every hash stays valid")


def test_entity_type_is_covered_although_the_formula_omits_it() -> None:
    """SCHEMA.md's formula leaves it out, which would let an edit to it pass unnoticed."""
    base = AuditRecord(occurred_at=START, event_type="e", entity_type="detection",
                       entity_id="D1", payload={})
    assert base.digest() != replace(base, entity_type="adjudication").digest()


def test_every_field_is_covered_by_the_hash() -> None:
    """A field outside the hash is a field an editor can change without detection."""
    base = AuditRecord(occurred_at=START, event_type="e", entity_type="t", entity_id="E1",
                       payload={"k": 1}, actor_user_id="U1", actor_role="supervisor")
    mutations = {
        "occurred_at": at(1), "event_type": "f", "entity_type": "u", "entity_id": "E2",
        "payload": {"k": 2}, "actor_user_id": "U2", "actor_role": "admin", "prev_hash": "x",
    }
    for name, value in mutations.items():
        assert replace(base, **{name: value}).digest() != base.digest(), (
            f"{name} is outside the hash and could be edited undetectably")


# ---------------------------------------------------------------------------
# Tamper evidence
# ---------------------------------------------------------------------------

def build_log() -> AuditLog:
    log = AuditLog()
    log.append("codebook.activated", "v1.0-draft",
               {"version": "v1.0-draft", "document_sha256": "c" * 64}, at(0),
               actor_user_id="U_admin", actor_role="admin")
    log.append("model.deployed", "b-resnet50-tcn-v3-ens5",
               {"model_version": "b-resnet50-tcn-v3-ens5", "config_sha256": "f" * 64}, at(10),
               actor_user_id="U_admin", actor_role="admin")
    log.append("session.ingested", "S1",
               {"session_id": "S1", "teacher_id": "T1", "domain": "classroom"}, at(20))
    log.append("detection.served", "D1",
               {"detection_id": "D1", "session_id": "S1", "behaviour": "B1",
                "gate_outcome": "present", "reviewer_id": "R1"}, at(30))
    log.append("adjudication.created", "A1",
               adjudication_payload("D1", "R1", adjudication_id="A1"), at(40),
               actor_user_id="R1", actor_role="supervisor")
    return log


def test_an_untampered_chain_verifies() -> None:
    log = build_log()
    assert len(log) == 5
    assert log.verify() is None
    assert log.head == log[-1].row_hash


def test_an_edit_in_place_is_detected_and_located() -> None:
    """The attack the REVOKE and the trigger cannot stop: a superuser changing a row."""
    records = list(build_log())
    tampered = replace(records[2], payload={"session_id": "S1", "teacher_id": "T_OTHER",
                                            "domain": "classroom"})
    records[2] = tampered

    found = verify(records)
    assert found is not None
    assert found.index == 2 and found.kind == "content"
    assert "edited in place" in found.describe()
    assert "session.ingested" in found.describe()


def test_a_deleted_row_is_detected_as_a_broken_link() -> None:
    records = list(build_log())
    del records[2]

    found = verify(records)
    assert found is not None
    assert found.index == 2 and found.kind == "link"
    assert "deleted, inserted or reordered" in found.describe()


def test_a_reordering_is_detected() -> None:
    records = list(build_log())
    records[1], records[2] = records[2], records[1]

    found = verify(records)
    assert found is not None and found.index == 1


def test_only_the_first_break_is_reported() -> None:
    """Everything after a break fails by construction; listing them all buries the finding."""
    records = list(build_log())
    records[1] = replace(records[1], entity_id="TAMPERED")
    records[3] = replace(records[3], entity_id="ALSO_TAMPERED")

    found = verify(records)
    assert found is not None and found.index == 1


def test_an_empty_chain_is_intact_and_proves_nothing() -> None:
    assert verify([]) is None


# ---------------------------------------------------------------------------
# The acceptance test: replay reconstructs the final state exactly
# ---------------------------------------------------------------------------

def test_replaying_the_log_reconstructs_the_final_state_exactly() -> None:
    """Equality against a state maintained live, not a spot check of a few fields.

    A payload too thin to reconstruct its own effect shows up here as a difference, which is
    the only way an under-specified event gets found before the corpus depends on it.
    """
    log = AuditLog()
    live = SystemState()

    def record(event: str, entity: str, payload: dict, when: datetime, apply) -> None:
        log.append(event, entity, payload, when)
        apply(live)

    record("codebook.activated", "v1.0-draft",
           {"version": "v1.0-draft", "document_sha256": "c" * 64}, at(0),
           lambda s: (setattr(s, "active_codebook", "v1.0-draft"),
                      setattr(s, "codebook_document_sha256", "c" * 64)))
    record("model.deployed", "m1", {"model_version": "m1", "config_sha256": "f" * 64}, at(5),
           lambda s: (setattr(s, "deployed_model", "m1"),
                      setattr(s, "model_config_sha256", "f" * 64)))
    record("session.ingested", "S1",
           {"session_id": "S1", "teacher_id": "T1", "domain": "classroom"}, at(10),
           lambda s: s.sessions.__setitem__("S1", {"teacher_id": "T1",
                                                   "domain": "classroom"}))
    record("session.ingested", "S2",
           {"session_id": "S2", "teacher_id": "T2", "domain": "microteaching"}, at(11),
           lambda s: s.sessions.__setitem__("S2", {"teacher_id": "T2",
                                                   "domain": "microteaching"}))
    record("session.rejected", "S2", {"session_id": "S2", "reason": "consent refused"}, at(12),
           lambda s: (s.rejected_sessions.__setitem__("S2", "consent refused"),
                      s.sessions.pop("S2", None)))
    record("teacher_track.confirmed", "S1",
           {"session_id": "S1", "track_id": "TR7", "confirmed_by": "U1"}, at(13),
           lambda s: s.confirmed_tracks.__setitem__(
               "S1", {"track_id": "TR7", "confirmed_by": "U1"}))
    record("annotation.created", "AN1",
           {"annotation_id": "AN1", "clip_id": "C1", "behaviour": "B1", "rater_id": "RA1",
            "codebook_version": "v1.0-draft"}, at(14),
           lambda s: s.annotations.__setitem__("AN1", {
               "clip_id": "C1", "behaviour": "B1", "rater_id": "RA1",
               "codebook_version": "v1.0-draft"}))
    record("detection.served", "D1",
           {"detection_id": "D1", "session_id": "S1", "behaviour": "B1",
            "gate_outcome": "escalate", "reviewer_id": "R1"}, at(20),
           lambda s: s.served_detections.__setitem__("D1", {
               "session_id": "S1", "behaviour": "B1", "gate_outcome": "escalate",
               "reviewer_id": "R1"}))
    record("export.generated", "X1",
           {"export_id": "X1", "recipient": "college-registry", "teacher_id": "T1"}, at(60),
           lambda s: s.exports.__setitem__("X1", {"recipient": "college-registry",
                                                  "teacher_id": "T1"}))
    record("consent.withdrawn", "T9", {"teacher_id": "T9", "scope": "all"}, at(70),
           lambda s: s.withdrawn_consent.__setitem__("T9", "all"))
    record("media.deleted", "M4", {"media_id": "M4", "reason": "retention, 5 years"}, at(80),
           lambda s: s.deleted_media.__setitem__("M4", "retention, 5 years"))
    record("config.changed", "routing.present_when.min_calibrated_prob",
           {"config_key": "routing.present_when.min_calibrated_prob", "old_value": 0.70,
            "new_value": 0.62, "config_sha256": "d" * 64}, at(90),
           lambda s: (s.config.__setitem__("routing.present_when.min_calibrated_prob", 0.62),
                      s.config_history.append({
                          "config_key": "routing.present_when.min_calibrated_prob",
                          "old_value": 0.70, "new_value": 0.62}),
                      setattr(s, "config_sha256", "d" * 64)))

    reconstructed = replay(log)
    assert reconstructed.differences(live) == []
    assert reconstructed == live
    assert reconstructed.n_events_applied == len(log)
    assert log.verify() is None


def test_a_correction_replays_as_a_new_row_with_the_earlier_one_kept() -> None:
    """Adjudications are append-only, so a revision is another row, not an edit.

    The trail must be able to answer what the reviewer first thought, which is the question
    the reliance study asks most often. Resolving corrections away at replay would lose it.
    """
    log = AuditLog()
    log.append("adjudication.created", "A1",
               adjudication_payload("D1", "R1", adjudication_id="A1", action="confirm"), at(0))
    log.append("adjudication.created", "A2",
               adjudication_payload("D1", "R1", adjudication_id="A2", action="reject",
                                    rationale="on review the hands are occluded throughout"),
               at(100))
    log.append("adjudication.created", "A3",
               adjudication_payload("D1", "R2", adjudication_id="A3", action="confirm"),
               at(120))

    state = replay(log)
    latest = state.adjudications[("D1", "R1")]
    assert latest.adjudication_id == "A2" and latest.action == "reject"
    assert latest.history == ["A1"], "the first decision must remain recoverable"
    assert latest.rationale.startswith("on review")

    assert len(state.latest_for("D1")) == 2, "two reviewers, one latest decision each"
    assert state.adjudications[("D1", "R2")].history == []


def test_an_edit_replays_with_the_corrected_fields_intact() -> None:
    """`edited_value` is a codebook-shaped object, not a float, so replay must
    carry the dict."""
    corrected = {"b1_count": 5, "b1_amplitude": 3}
    log = AuditLog()
    log.append("adjudication.created", "A1",
               adjudication_payload("D1", "R1", action="edit", edited_value=corrected,
                                    rationale="the count was three, not one"), at(0))

    assert replay(log).adjudications[("D1", "R1")].edited_value == corrected


def test_replay_refuses_an_event_it_has_no_rule_for() -> None:
    """Silently skipping would make the reconstruction quietly incomplete."""
    orphan = AuditRecord(occurred_at=START, event_type="detection.peeked",
                         entity_type="detection", entity_id="D1", payload={})
    with pytest.raises(AuditError, match="no rule for event type"):
        replay([orphan])


def test_every_declared_event_has_a_replay_rule() -> None:
    """The taxonomy and the replay cannot drift apart without this failing."""
    for event_type, spec in EVENTS.items():
        payload = {name: "x" for name in spec.required_fields}
        record = AuditRecord(occurred_at=START, event_type=event_type,
                             entity_type=spec.entity_type, entity_id="E1", payload=payload)
        replay([record])          # raises if a declared event has no rule


def test_differences_names_the_field_that_diverged() -> None:
    """A failing reconstruction must say which event lost information, not just "unequal"."""
    left, right = SystemState(), SystemState()
    left.sessions["S1"] = {"teacher_id": "T1", "domain": "classroom"}
    right.sessions["S1"] = {"teacher_id": "T2", "domain": "classroom"}
    right.deployed_model = "m1"

    differences = left.differences(right)
    assert any("sessions['S1']" in d for d in differences)
    assert any("deployed_model" in d for d in differences)


# ---------------------------------------------------------------------------
# The exported artefact the verification script reads
# ---------------------------------------------------------------------------

def test_an_exported_row_round_trips_through_json(tmp_path) -> None:
    """`scripts/verify_audit_chain.py` reads this shape; it must survive the trip."""
    log = build_log()
    path = tmp_path / "audit.json"
    path.write_text(json.dumps([record.as_row() for record in log]), encoding="utf-8")

    assert verify(load(path)) is None

    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[2]["entity_id"] = "S_TAMPERED"
    path.write_text(json.dumps(rows), encoding="utf-8")
    found = verify(load(path))
    assert found is not None and found.index == 2 and found.kind == "content"


# ---------------------------------------------------------------------------
# Found by adversarial review, after the first implementation passed its own tests
# ---------------------------------------------------------------------------

def test_a_payload_that_is_not_json_is_refused_rather_than_stringified() -> None:
    """`default=str` would coerce, and coercion breaks injectivity.

    `Decimal("1.5")` and the string `"1.5"` stringify identically, so two genuinely different
    rows would share a hash — in a chain whose entire purpose is that they cannot.
    """
    from decimal import Decimal

    log = AuditLog()
    for bad in (Decimal("1.5"), START, {1, 2}, float("nan")):
        with pytest.raises(AuditError, match="must be JSON"):
            log.append("config.changed", "k",
                       {"config_key": "k", "old_value": bad, "new_value": 1,
                        "config_sha256": "a" * 64}, at(0))

    assert canonical_payload({"a": Decimal("1.5").__str__()}) == '{"a":"1.5"}'


def test_a_sealed_record_does_not_alias_the_callers_dict() -> None:
    """`frozen=True` stops the field being reassigned, not the dict being mutated.

    A caller holding a reference could otherwise change the payload after the row was sealed,
    leaving row_hash describing content the record no longer holds — and `verify` would then
    report a break that no attacker caused.
    """
    payload = {"teacher_id": "T1", "scope": "all"}
    record = AuditRecord(occurred_at=START, event_type="consent.withdrawn",
                         entity_type="consent", entity_id="T1", payload=payload).sealed("")
    payload["teacher_id"] = "T_SOMEONE_ELSE"

    assert record.payload["teacher_id"] == "T1"
    assert record.digest() == record.row_hash

    log = AuditLog()
    shared = {"media_id": "M1", "reason": "retention"}
    appended = log.append("media.deleted", "M1", shared, at(0))
    shared["reason"] = "changed my mind"
    assert appended.payload["reason"] == "retention"
    assert log.verify() is None


def test_an_exported_row_does_not_alias_the_record() -> None:
    record = AuditRecord(occurred_at=START, event_type="media.deleted", entity_type="media",
                         entity_id="M1", payload={"media_id": "M1", "reason": "r"}).sealed("")
    row = record.as_row()
    row["payload"]["reason"] = "tampered"
    assert record.payload["reason"] == "r"


def test_a_rejection_and_an_ingest_of_one_session_never_both_survive() -> None:
    """Either order. A session cannot be both ingested and rejected, whichever arrived first."""
    ingest = {"session_id": "S1", "teacher_id": "T1", "domain": "classroom"}
    rejection = {"session_id": "S1", "reason": "consent refused"}

    rejected_first = AuditLog()
    rejected_first.append("session.rejected", "S1", rejection, at(0))
    rejected_first.append("session.ingested", "S1", ingest, at(10))
    state = replay(rejected_first)
    assert "S1" in state.sessions and "S1" not in state.rejected_sessions

    ingested_first = AuditLog()
    ingested_first.append("session.ingested", "S1", ingest, at(0))
    ingested_first.append("session.rejected", "S1", rejection, at(10))
    state = replay(ingested_first)
    assert "S1" in state.rejected_sessions and "S1" not in state.sessions


def test_replay_keeps_who_confirmed_the_teacher_track() -> None:
    """R1 turns on a human confirming which track is the teacher; the audit is asked who."""
    log = AuditLog()
    log.append("teacher_track.confirmed", "S1",
               {"session_id": "S1", "track_id": "TR7", "confirmed_by": "U_alice"}, at(0))

    assert replay(log).confirmed_tracks["S1"] == {"track_id": "TR7",
                                                  "confirmed_by": "U_alice"}


def test_replay_keeps_what_a_config_value_changed_from() -> None:
    """`old_value` is required of the payload, so something must read it back."""
    log = AuditLog()
    log.append("config.changed", "routing.present_when.min_calibrated_prob",
               {"config_key": "routing.present_when.min_calibrated_prob",
                "old_value": 0.70, "new_value": 0.62, "config_sha256": "d" * 64}, at(0))
    log.append("config.changed", "routing.present_when.min_calibrated_prob",
               {"config_key": "routing.present_when.min_calibrated_prob",
                "old_value": 0.62, "new_value": 0.68, "config_sha256": "e" * 64}, at(10))

    state = replay(log)
    assert state.config["routing.present_when.min_calibrated_prob"] == 0.68
    assert [c["old_value"] for c in state.config_history] == [0.70, 0.62]
    assert state.config_sha256 == "e" * 64


def test_every_required_payload_field_is_read_by_a_replay_rule() -> None:
    """The structural version of the two findings above.

    A required field nothing reads back is a field that will quietly stop being supplied. This
    compares each event's declared fields against what its handler stores, by giving every
    field a distinguishable value and asserting it appears somewhere in the reconstruction.
    """
    unread: list[str] = []
    for event_type, spec in EVENTS.items():
        payload = {name: f"<{name}>" for name in spec.required_fields}
        state = replay([AuditRecord(occurred_at=START, event_type=event_type,
                                    entity_type=spec.entity_type, entity_id="E1",
                                    payload=payload).sealed("")])
        rendered = repr(state)          # dict keys include tuples, so repr, not json
        for name in spec.required_fields:
            if name.endswith("_id") or name.endswith("_sha256") or name == "version":
                continue          # identifiers are keys, and keys are in the rendering anyway
            if f"<{name}>" not in rendered:
                unread.append(f"{event_type}.{name}")
    assert not unread, f"declared but never reconstructed: {unread}"
