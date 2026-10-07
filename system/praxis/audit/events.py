# -*- coding: utf-8 -*-
"""The event taxonomy, and what each event's payload must carry to be replayable.

SCHEMA.md §9 lists eleven event types and says when each occurs. It does not say what goes in
`payload`, and BUILD-SPEC's acceptance test — "replaying the audit log reconstructs the final
state exactly" — cannot be met by a log whose payloads are unspecified. What a replay can
reconstruct is exactly what the payloads carried, so the required fields are declared here and
checked at append time rather than discovered to be missing when someone needs the trail.

**`config.changed` is added.** BUILD-SPEC requires "every config change" logged; SCHEMA.md's
table omits an event for it. Recorded as a deviation rather than silently reconciled.

**`adjudication.created` carries the whole adjudication, not a reference to it.** A payload of
`{"adjudication_id": ...}` would make the audit log depend on the adjudications table to say
what happened, and the trail is supposed to be the thing that proves the table. It also carries
`seconds_on_item`, `evidence_replays` and `revealed_indication`: §4.4 says these are how H6 and
H7 are answered, so a trail that cannot reproduce them has lost the study's data.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class AuditError(RuntimeError):
    """Raised when an event cannot be recorded in a form the trail could replay."""


@dataclass(frozen=True)
class EventSpec:
    """One event type: when it happens, and the payload fields a replay will need."""

    event_type: str
    entity_type: str
    when: str
    required_fields: tuple[str, ...]

    def check(self, entity_type: str, payload: Mapping[str, Any]) -> None:
        if entity_type != self.entity_type:
            raise AuditError(
                f"{self.event_type} describes a {self.entity_type}, not a {entity_type!r}")
        missing = [name for name in self.required_fields if name not in payload]
        if missing:
            raise AuditError(
                f"{self.event_type} payload omits {missing}. The trail must be able to "
                f"reconstruct the final state from the log alone, and these are the fields a "
                f"replay of this event reads.")


EVENTS: dict[str, EventSpec] = {
    spec.event_type: spec for spec in (
        EventSpec("session.ingested", "session", "media accepted",
                  ("session_id", "teacher_id", "domain")),
        EventSpec("session.rejected", "session", "quality gate or consent refusal",
                  ("session_id", "reason")),
        # Preprocessing destroys the unblurred original, so the trail has to record that it
        # happened, what came out, and whether the deletion was verified.
        EventSpec("session.preprocessed", "session",
                  "pose extracted, faces blurred, original deleted",
                  ("session_id", "pose_sha256", "frames_processed", "faces_blurred",
                   "original_deleted", "config_sha256")),
        EventSpec("teacher_track.confirmed", "session",
                  "a human confirms the teacher identity",
                  ("session_id", "track_id", "confirmed_by")),
        # A researcher decides a session will not be annotated. The ground is required because
        # an exclusion recorded without one is indistinguishable from unfinished work, which is
        # the state 0010 exists to end.
        EventSpec("session.excluded", "session",
                  "a researcher decides a session will not be annotated, and says why",
                  ("session_id", "reason", "excluded_by")),
        # 0009 gave a declined proposal a row and a candidate ranking; the rows written before
        # it took a server default saying the ranking was never recorded. Rebuilding one from
        # the stored pose artefact changes what a reviewer is shown, so it is an audited event
        # rather than a repair. `source` is required because a keypoint hull and a detection box
        # are different measurements of the same track, and a recovered ranking that did not say
        # which it came from would be indistinguishable from the original run's.
        EventSpec("teacher_track.ranking_recovered", "session",
                  "a candidate ranking is rebuilt from the stored pose artefact",
                  ("session_id", "source", "pose_sha256", "candidate_count")),
        EventSpec("camera_setup.defined", "session", "an operator marks the room's zones",
                  ("setup_id", "college_id", "defined_by")),
        EventSpec("annotation.created", "annotation", "any rater label",
                  ("annotation_id", "clip_id", "behaviour", "rater_id", "codebook_version")),
        EventSpec("codebook.activated", "codebook", "a version becomes active",
                  ("version", "document_sha256")),
        EventSpec("model.deployed", "model", "a model version goes live",
                  ("model_version", "config_sha256")),
        EventSpec("detection.served", "detection",
                  "a detection is shown to a reviewer, with the gate outcome",
                  ("detection_id", "session_id", "behaviour", "gate_outcome", "reviewer_id")),
        EventSpec("adjudication.created", "adjudication", "confirm, edit, or reject",
                  ("adjudication_id", "detection_id", "reviewer_id", "action",
                   "seconds_on_item", "evidence_replays", "revealed_indication")),
        EventSpec("export.generated", "export", "a report leaves the system, with recipient",
                  ("export_id", "recipient", "teacher_id")),
        EventSpec("consent.withdrawn", "consent", "withdrawal recorded",
                  ("teacher_id", "scope")),
        EventSpec("media.deleted", "media", "deletion executed, with reason",
                  ("media_id", "reason")),
        # Required by BUILD-SPEC Phase 8 work item 4, absent from SCHEMA.md's table.
        EventSpec("config.changed", "config", "a configuration value is changed",
                  ("config_key", "old_value", "new_value", "config_sha256")),
    )
}


def spec_for(event_type: str) -> EventSpec:
    try:
        return EVENTS[event_type]
    except KeyError:
        raise AuditError(
            f"{event_type!r} is not an audited event type. The taxonomy is closed: an event "
            f"nobody declared is an event nothing knows how to replay. Known types are "
            f"{sorted(EVENTS)}.") from None
