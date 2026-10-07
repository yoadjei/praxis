# -*- coding: utf-8 -*-
"""Evidence before suggestion: the cognitive forcing function, and the instrumentation for H6.

BUILD-SPEC Phase 8: "The interface presents evidence first and requires an interaction before
revealing any model indication." The point is not politeness about ordering. Anchoring is the
failure mode the reliance study exists to measure: a reviewer shown a confident indication
first tends to look for evidence that supports it, and the evidence panel then confirms rather
than informs.

**Two arms, and the participant is assigned to one, never the client.** In `dossier` the
indication is withheld until the reviewer asks for it. In `no_dossier` it is there from the
start. The comparison between them is H6. A client that could choose its own arm could choose
differently per item and the contrast would dissolve, so `ReviewSession` takes the arm at
construction and there is no setter.

**Withholding is structural here too, for the same reason it is in the gate.** The pre-reveal
payload is *built* without the indication rather than built with it and stripped: a build-then-
strip path leaks the first time someone adds a field and forgets the strip list. `evidence_only`
constructs its own dict and there is no code path from a `Detection` to a pre-reveal payload
that passes through the full one.

**A suppressed detection has nothing to reveal.** Revealing one is refused rather than
returning an empty indication, because "the reviewer revealed it and there was nothing there"
and "the reviewer never revealed it" are different data points for H6 and must not merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from praxis.contracts.adjudication import Action, Adjudication, Arm
from praxis.contracts.detection import Detection
from praxis.ids import new_ulid


class RevealError(RuntimeError):
    """Raised when the forcing function is asked to do something that would void the study."""


@dataclass(frozen=True)
class RevealPolicy:
    """Whether the forcing function is active, and for whom."""

    enabled: bool = True
    require_reveal_action: bool = True

    @classmethod
    def from_config(cls, config) -> RevealPolicy:
        section = config.routing.evidence_before_suggestion
        return cls(enabled=section.enabled,
                   require_reveal_action=section.require_reveal_action)

    def forces(self, arm: Arm) -> bool:
        """Whether this participant must act before seeing an indication.

        `no_dossier` is the always-on comparison arm the spec requires, so the forcing function
        never applies to it however the config is set.
        """
        return self.enabled and self.require_reveal_action and arm == "dossier"


def evidence_only(detection: Detection) -> dict[str, Any]:
    """What a reviewer sees before revealing: where to look, and nothing about what to think.

    Built from scratch rather than filtered from `to_payload()`. The gate outcome is included
    because navigation needs it — a suppressed item is still listed, marked, and reviewable —
    but no indication, no probability and no explanation reference appear.
    """
    return {
        "detection_id": detection.detection_id,
        "session_id": detection.session_id,
        "behaviour": detection.behaviour,
        "t_start_s": detection.t_start_s,
        "t_end_s": detection.t_end_s,
        "evidence_ref": detection.evidence_ref,
        "indication_available": detection.gate_outcome != "suppress",
    }


@dataclass
class ReviewItem:
    """One detection in front of one reviewer, and everything H6 and H7 need recorded.

    Mutable on purpose: this is the live state of an item being worked on, not a contract
    crossing a stage boundary. It becomes an immutable `Adjudication` when the reviewer decides.
    """

    detection: Detection
    opened_at: datetime
    revealed_at: datetime | None = None
    reveal_position: int | None = None
    evidence_replays: int = 0
    decided_at: datetime | None = None

    @property
    def revealed(self) -> bool:
        return self.revealed_at is not None

    @property
    def seconds_on_item(self) -> float:
        """Wall-clock from opening the item to deciding it, or to now if still open."""
        end = self.decided_at or self.opened_at
        return max((end - self.opened_at).total_seconds(), 0.0)

    def seconds_before_reveal(self) -> float | None:
        """How long the reviewer spent with evidence alone. The forcing function's whole point.

        A reviewer who reveals instantly has complied with the interaction requirement without
        doing what it exists to make them do, and that is measurable only here.
        """
        if self.revealed_at is None:
            return None
        return max((self.revealed_at - self.opened_at).total_seconds(), 0.0)

    def replay(self) -> None:
        self.evidence_replays += 1


@dataclass
class ReviewSession:
    """A reviewer working through one session's detections, in one study arm."""

    reviewer_id: str
    session_id: str
    arm: Arm
    policy: RevealPolicy = field(default_factory=RevealPolicy)
    items: dict[str, ReviewItem] = field(default_factory=dict)
    reveal_order: list[str] = field(default_factory=list)

    def open(self, detection: Detection, at: datetime) -> dict[str, Any]:
        """Put an item in front of the reviewer and return what they may see now.

        In the comparison arm the indication is present immediately, which is the point of
        having the arm: it is the condition the forcing function is being compared against.
        """
        if detection.detection_id in self.items:
            raise RevealError(
                f"detection {detection.detection_id} is already open in this review session; "
                f"reopening would restart its timer and lose the time already on it")

        item = ReviewItem(detection=detection, opened_at=at)
        self.items[detection.detection_id] = item

        if self.policy.forces(self.arm):
            return evidence_only(detection)

        # The comparison arm sees everything at once, and that is still recorded as a reveal so
        # that "was the indication visible when they decided" has one answer in both arms.
        item.revealed_at = at
        item.reveal_position = len(self.reveal_order)
        self.reveal_order.append(detection.detection_id)
        return detection.to_payload()

    def reveal(self, detection_id: str, at: datetime) -> dict[str, Any]:
        """The reviewer asks for the model's indication. Records that they did, and when."""
        item = self._item(detection_id)
        if item.detection.gate_outcome == "suppress":
            raise RevealError(
                f"detection {detection_id} was suppressed "
                f"({item.detection.suppression_reason}); there is no indication to reveal. "
                f"Returning an empty one would record a reveal that showed nothing, which is "
                f"not the same event as a reviewer choosing not to reveal.")
        if item.revealed:
            # Not an error: re-reading is normal. But the first reveal is the one H6 measures,
            # so the timestamp and position are not overwritten.
            return item.detection.to_payload()

        item.revealed_at = at
        item.reveal_position = len(self.reveal_order)
        self.reveal_order.append(detection_id)
        return item.detection.to_payload()

    def replay_evidence(self, detection_id: str) -> int:
        """The reviewer played the clip again. Counted; it is one of H7's covariates."""
        item = self._item(detection_id)
        item.replay()
        return item.evidence_replays

    def decide(self, detection_id: str, at: datetime) -> ReviewItem:
        """Close an item.

        Deciding without revealing is allowed and is a finding, not an error: a reviewer who
        judges from evidence alone is exactly what the forcing function is for, and
        `unrevealed` counts them.
        """
        item = self._item(detection_id)
        item.decided_at = at
        return item

    def adjudicate(self, detection_id: str, action: Action, at: datetime,
                   reviewer_id: str | None = None, edited_value: dict[str, Any] | None = None,
                   rationale: str | None = None,
                   was_seeded_error: bool = False) -> Adjudication:
        """Close an item and build the reviewer's decision, checked against its own detection.

        This is the only place both the adjudication and the detection it corrects are in
        scope, so it is where `Adjudication.validate_against` is called. A validator that has
        to be *remembered* at a call site is not a guarantee; making the constructor that
        assembles the adjudication also run it means an edit naming a field the codebook does
        not define cannot reach storage by a route that simply forgot.

        The instrumentation is filled in from what was observed rather than passed in, because
        a caller supplying its own `seconds_on_item` is a caller that could supply anything.
        """
        item = self.decide(detection_id, at)
        adjudication = Adjudication(
            adjudication_id=new_ulid(),
            detection_id=detection_id,
            reviewer_id=reviewer_id or self.reviewer_id,
            action=action,
            edited_value=edited_value,
            rationale=rationale,
            was_seeded_error=was_seeded_error,
            created_at=at,
            **self.instrumentation(detection_id))
        adjudication.validate_against(item.detection)
        return adjudication

    def instrumentation(self, detection_id: str) -> dict[str, Any]:
        """The fields an `Adjudication` must carry, assembled from what was observed.

        §4.4 of BUILD-SPEC: "do not remove them as unused". They are how H6 and H7 are
        answered, so they are produced here rather than left to a caller to remember.
        """
        item = self._item(detection_id)
        return {
            "seconds_on_item": item.seconds_on_item,
            "evidence_replays": item.evidence_replays,
            "revealed_indication": item.revealed,
            "arm": self.arm,
        }

    @property
    def unrevealed(self) -> list[str]:
        """Items decided without the reviewer ever asking for the indication."""
        return [key for key, item in self.items.items()
                if item.decided_at is not None and not item.revealed]

    def median_seconds_before_reveal(self) -> float | None:
        """Whether the forcing function bought any looking at all, in one number."""
        waits = sorted(w for w in (item.seconds_before_reveal() for item in self.items.values())
                       if w is not None)
        if not waits:
            return None
        middle = len(waits) // 2
        if len(waits) % 2:
            return waits[middle]
        return (waits[middle - 1] + waits[middle]) / 2.0

    def _item(self, detection_id: str) -> ReviewItem:
        try:
            return self.items[detection_id]
        except KeyError:
            raise RevealError(
                f"detection {detection_id} was never opened in this review session, so nothing "
                f"is known about how long the reviewer spent with it") from None


def elapsed(start: datetime, seconds: float) -> datetime:
    """Small helper so callers and tests express timings in seconds, not timedeltas."""
    return start + timedelta(seconds=seconds)
