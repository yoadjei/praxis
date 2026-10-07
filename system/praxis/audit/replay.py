# -*- coding: utf-8 -*-
"""Replaying the trail: reconstructing what the system held from the log of what happened.

BUILD-SPEC Phase 8 asks that "replaying the audit log reconstructs the final state exactly".
That is a stronger requirement than it looks, and it is the one that makes the trail worth
keeping. A log that records *that* something happened without recording *what* lets you say a
decision was made and not what it was, which in a thesis defence is the same as having nothing.

So the test is not that replay produces something plausible. It is that replay produces a state
identical to the one maintained live, and `SystemState` supports equality for exactly that
comparison. Any event whose payload is too thin to reconstruct its effect shows up as a
difference, which is how an under-specified payload gets found before the corpus depends on it.

**Corrections are replayed, not resolved away.** An adjudication revised three times leaves
three rows, and the reconstructed state holds the latest while `history` keeps all of them. The
API returns the latest by `created_at`; a replay that discarded the earlier ones would make the
trail unable to answer what a reviewer first thought, which is the question the reliance study
asks most often.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from praxis.audit.chain import AuditRecord
from praxis.audit.events import AuditError


@dataclass
class AdjudicationState:
    """The latest decision on one detection by one reviewer, and everything before it."""

    adjudication_id: str
    detection_id: str
    reviewer_id: str
    action: str
    edited_value: dict[str, Any] | None
    rationale: str | None
    seconds_on_item: float
    evidence_replays: int
    revealed_indication: bool
    arm: str | None
    history: list[str] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdjudicationState):
            return NotImplemented
        return (self.adjudication_id == other.adjudication_id
                and self.detection_id == other.detection_id
                and self.reviewer_id == other.reviewer_id
                and self.action == other.action
                and self.edited_value == other.edited_value
                and self.rationale == other.rationale
                and self.seconds_on_item == other.seconds_on_item
                and self.evidence_replays == other.evidence_replays
                and self.revealed_indication == other.revealed_indication
                and self.arm == other.arm
                and self.history == other.history)


@dataclass
class SystemState:
    """Everything the trail is expected to be able to rebuild.

    Compared by value, because the acceptance test is an equality against the state maintained
    live rather than an inspection of a few fields someone remembered to check.
    """

    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected_sessions: dict[str, str] = field(default_factory=dict)
    preprocessed_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    camera_setups: dict[str, dict[str, Any]] = field(default_factory=dict)
    confirmed_tracks: dict[str, dict[str, Any]] = field(default_factory=dict)
    recovered_rankings: dict[str, dict[str, Any]] = field(default_factory=dict)
    excluded_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_codebook: str | None = None
    codebook_document_sha256: str | None = None
    deployed_model: str | None = None
    model_config_sha256: str | None = None
    served_detections: dict[str, dict[str, Any]] = field(default_factory=dict)
    adjudications: dict[tuple[str, str], AdjudicationState] = field(default_factory=dict)
    exports: dict[str, dict[str, Any]] = field(default_factory=dict)
    withdrawn_consent: dict[str, str] = field(default_factory=dict)
    deleted_media: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    config_history: list[dict[str, Any]] = field(default_factory=list)
    config_sha256: str | None = None

    # How many events were applied. Excluded from equality: two logs that reach the same state
    # by different numbers of events are the same state, and the acceptance test is about the
    # state.
    n_events_applied: int = field(default=0, compare=False, repr=False)

    def latest_for(self, detection_id: str) -> list[AdjudicationState]:
        """Every reviewer's latest decision on one detection."""
        return [state for (did, _), state in sorted(self.adjudications.items())
                if did == detection_id]

    def differences(self, other: SystemState) -> list[str]:
        """Where two states diverge, named field by field.

        Returned rather than asserted so that a failing reconstruction test says *which* event
        lost information, not merely that the two objects were unequal.
        """
        out: list[str] = []
        for name in ("sessions", "rejected_sessions", "preprocessed_sessions", "camera_setups",
                     "confirmed_tracks", "recovered_rankings", "excluded_sessions",
                     "annotations",
                     "active_codebook", "codebook_document_sha256", "deployed_model",
                     "model_config_sha256", "served_detections", "adjudications", "exports",
                     "withdrawn_consent", "deleted_media", "config", "config_history",
                     "config_sha256"):
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine == theirs:
                continue
            if isinstance(mine, dict) and isinstance(theirs, dict):
                for key in sorted(set(mine) | set(theirs), key=str):
                    if mine.get(key) != theirs.get(key):
                        out.append(f"{name}[{key!r}]: {mine.get(key)!r} != {theirs.get(key)!r}")
            else:
                out.append(f"{name}: {mine!r} != {theirs!r}")
        return out


def _adjudication(payload: dict[str, Any], previous: AdjudicationState | None
                  ) -> AdjudicationState:
    history = [] if previous is None else [*previous.history, previous.adjudication_id]
    return AdjudicationState(
        adjudication_id=payload["adjudication_id"],
        detection_id=payload["detection_id"],
        reviewer_id=payload["reviewer_id"],
        action=payload["action"],
        edited_value=payload.get("edited_value"),
        rationale=payload.get("rationale"),
        seconds_on_item=payload["seconds_on_item"],
        evidence_replays=payload["evidence_replays"],
        revealed_indication=payload["revealed_indication"],
        arm=payload.get("arm"),
        history=history)


def replay(records: Iterable[AuditRecord]) -> SystemState:
    """Rebuild the final state from the log, in the order it was written.

    Order is the log's order, not a sort by timestamp: the chain fixes the sequence, and two
    events written in the same microsecond would otherwise be applied in whichever order the
    sort happened to produce.
    """
    state = SystemState()

    for record in records:
        payload = record.payload
        kind = record.event_type

        if kind == "session.ingested":
            state.sessions[payload["session_id"]] = {
                "teacher_id": payload["teacher_id"], "domain": payload["domain"]}
            # Symmetric with the rejection below. A session id that was rejected and later
            # ingested has had its rejection superseded, and leaving it in both dictionaries
            # would reconstruct a state the system can never actually have been in.
            state.rejected_sessions.pop(payload["session_id"], None)
        elif kind == "session.rejected":
            state.rejected_sessions[payload["session_id"]] = payload["reason"]
            state.sessions.pop(payload["session_id"], None)
        elif kind == "session.preprocessed":
            # `original_deleted` and `faces_blurred` are read rather than merely recorded: the
            # run destroyed the unblurred source, and a trail that could not answer whether the
            # deletion was verified would be unable to answer the one question this event
            # exists for. `config_sha256` and `pose_sha256` are what make the artefact
            # identifiable afterwards - R7.
            state.preprocessed_sessions[payload["session_id"]] = {
                "pose_sha256": payload["pose_sha256"],
                "frames_processed": payload["frames_processed"],
                "faces_blurred": payload["faces_blurred"],
                "original_deleted": payload["original_deleted"],
                "config_sha256": payload["config_sha256"]}
        elif kind == "camera_setup.defined":
            state.camera_setups[payload["setup_id"]] = {
                "college_id": payload["college_id"], "defined_by": payload["defined_by"]}
        elif kind == "teacher_track.confirmed":
            # `confirmed_by` is required of the payload, so it is read: R1 turns on a human
            # confirming which track is the teacher, and "who confirmed it" is the part an
            # audit is asked for.
            state.confirmed_tracks[payload["session_id"]] = {
                "track_id": payload["track_id"], "confirmed_by": payload["confirmed_by"]}
        elif kind == "session.excluded":
            # Kept apart from `rejected_sessions`, which the quality gate writes. A rejection
            # says the file did not pass; an exclusion says a researcher will not label a file
            # that did. Merging them would let a research decision read as a measurement.
            state.excluded_sessions[payload["session_id"]] = {
                "reason": payload["reason"], "excluded_by": payload["excluded_by"]}
        elif kind == "teacher_track.ranking_recovered":
            # Kept apart from `confirmed_tracks`: a recovered ranking is what a reviewer will be
            # shown, not what anybody decided. Folding the two together would let a replay
            # report a session as having a teacher because its candidates were rebuilt.
            state.recovered_rankings[payload["session_id"]] = {
                "source": payload["source"], "pose_sha256": payload["pose_sha256"],
                "candidate_count": payload["candidate_count"]}
        elif kind == "annotation.created":
            state.annotations[payload["annotation_id"]] = {
                "clip_id": payload["clip_id"], "behaviour": payload["behaviour"],
                "rater_id": payload["rater_id"],
                "codebook_version": payload["codebook_version"]}
        elif kind == "codebook.activated":
            state.active_codebook = payload["version"]
            state.codebook_document_sha256 = payload["document_sha256"]
        elif kind == "model.deployed":
            state.deployed_model = payload["model_version"]
            state.model_config_sha256 = payload["config_sha256"]
        elif kind == "detection.served":
            state.served_detections[payload["detection_id"]] = {
                "session_id": payload["session_id"], "behaviour": payload["behaviour"],
                "gate_outcome": payload["gate_outcome"],
                "reviewer_id": payload["reviewer_id"]}
        elif kind == "adjudication.created":
            key = (payload["detection_id"], payload["reviewer_id"])
            state.adjudications[key] = _adjudication(payload, state.adjudications.get(key))
        elif kind == "export.generated":
            state.exports[payload["export_id"]] = {
                "recipient": payload["recipient"], "teacher_id": payload["teacher_id"]}
        elif kind == "consent.withdrawn":
            state.withdrawn_consent[payload["teacher_id"]] = payload["scope"]
        elif kind == "media.deleted":
            state.deleted_media[payload["media_id"]] = payload["reason"]
        elif kind == "config.changed":
            state.config[payload["config_key"]] = payload["new_value"]
            # `old_value` is required of the payload so that the trail can answer what a
            # threshold was before someone moved it. Reconstructing only the current value
            # would leave that field declared and never read, which is how a required field
            # quietly stops being supplied.
            state.config_history.append({"config_key": payload["config_key"],
                                         "old_value": payload["old_value"],
                                         "new_value": payload["new_value"]})
            state.config_sha256 = payload["config_sha256"]
        else:
            # Unreachable through AuditLog.append, which refuses unknown types. Reached only by
            # a record built directly, and silently skipping it would make the reconstruction
            # quietly incomplete, which is the one failure this test exists to catch.
            raise AuditError(
                f"replay has no rule for event type {kind!r}. Every declared event must have "
                f"one, or the reconstructed state is missing whatever that event did.")

        state.n_events_applied += 1

    return state
