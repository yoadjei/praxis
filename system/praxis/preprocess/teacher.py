# -*- coding: utf-8 -*-
"""Which track is the teacher. A proposal, never a decision.

BUILD-SPEC Phase 3 item 3: "The heuristic proposes; a human confirms at ingest review. Never
trust it silently. Store both the proposal and the confirmation." SCHEMA.md enforces the same
thing structurally - `teacher_tracks.confirmed_by` is nullable and `confirmed_at` must accompany
it - and `preprocess.teacher_id.require_human_confirmation` is documented `NEVER set false`.

So this module returns a `TeacherProposal` and there is no function here that returns "the
teacher". The distinction is not ceremony: R1 says only the teacher is classified, and every
downstream phase filters to the teacher track. A heuristic trusted silently would mean the
system classifies whoever it happened to pick, which is the one mistake R1 exists to prevent.

The score is the weighted sum the configuration defines, over three signals that are cheap and
independent: how long the track is present, how large it is, and how much time it spends in the
front zone. A teacher is usually the person who is there throughout, closest to the camera, and
in front of the class. Usually is exactly why a human confirms.

**The floor is scaled to the evidence that existed.** `min_score_to_propose` was set against
all three signals. A session whose camera setup has no marked zones cannot measure the third,
so thirty per cent of the weight is unreachable and the best attainable score falls with it -
on the default weights, from 1.00 to 0.70. Comparing that score against the unscaled floor
refuses a session for evidence nobody collected, and this module used to do exactly that while
printing a caveat saying the score "is not comparable with a session that has them". The caveat
was right and the comparison went ahead anyway. So the floor is multiplied by the fraction of
weight actually measured. That is a derivation from the configured weights, decided before the
scores it would change were computed, and it can only ever admit a proposal, never withdraw
one.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from praxis.preprocess.tracking import Track
from praxis.preprocess.zones import CameraSetup, normalise


@dataclass(frozen=True)
class TrackSignals:
    """The three measured quantities, kept so a reviewer can see why a track scored so."""

    track_id: int
    presence_fraction: float
    median_area_fraction: float
    front_zone_fraction: float

    def score(self, weight_presence: float, weight_area: float, weight_front: float) -> float:
        return (weight_presence * self.presence_fraction
                + weight_area * self.median_area_fraction
                + weight_front * self.front_zone_fraction)

    def as_json(self, weights: tuple[float, float, float]) -> dict[str, object]:
        """One candidate as the reviewer's screen receives it, score included so the ranking
        does not have to be recomputed by whoever reads the row."""
        return {"track_id": self.track_id,
                "presence_fraction": self.presence_fraction,
                "median_area_fraction": self.median_area_fraction,
                "front_zone_fraction": self.front_zone_fraction,
                "score": self.score(*weights)}


@dataclass(frozen=True)
class TeacherProposal:
    """What the heuristic thinks, and everything needed to disagree with it.

    `track_id` is None when no track cleared the floor. That is a legitimate outcome - an empty
    room, a camera pointed at the board - and it must reach the reviewer as "no proposal" rather
    than as the least-bad track. It is now also a row: until 0009 a declined proposal stored
    nothing, so the reviewer saw neither the score nor the ranking, and `confirm_teacher` would
    not accept a session it had no row for.

    `weights` and `zones_available` are carried rather than passed in again at the database
    boundary, because the stored candidate scores must be the ones this ranking was made with.
    Two copies of the weights is how the ranking and the scores beside it come to disagree.
    """

    track_id: int | None
    score: float | None
    signals: tuple[TrackSignals, ...]
    reason: str
    weights: tuple[float, float, float]
    zones_available: bool

    @property
    def is_confirmed(self) -> bool:
        """Always False. Confirmation is a database fact written by a person, and no object
        this module constructs can carry it."""
        return False

    def candidates(self, *, source: str, limit: int) -> dict[str, object]:
        """The ranking, capped, with where its numbers came from.

        `source` is the caller's to state and has no default. A detection box and a keypoint
        hull are different measurements of the same track, and a run re-emitted from a stored
        pose artifact after D18 deleted the original can only have the second. Defaulting it
        would relabel one as the other, which is how a metric gets silently redefined.
        """
        ranked = self.signals[:limit]
        return {"source": source,
                "zones_available": self.zones_available,
                "ranked": [signal.as_json(self.weights) for signal in ranked],
                "truncated": len(self.signals) - len(ranked)}

    def as_row(self, *, source: str, limit: int) -> dict[str, object]:
        return {"track_id": self.track_id, "proposed_by": "heuristic",
                "heuristic_score": self.score, "confirmed_by": None, "confirmed_at": None,
                "reason": self.reason,
                "candidates": self.candidates(source=source, limit=limit)}


def measure(tracks: list[Track], *, frame_count: int, frame_width: int, frame_height: int,
            setup: CameraSetup | None, track_positions: dict[int, list[tuple[float, float]]],
            track_areas: dict[int, list[float]]) -> list[TrackSignals]:
    """The three signals per track, each normalised to 0..1 so the weights mean what they say.

    `track_positions` and `track_areas` come from the per-frame detections rather than from the
    track's final box, because a track's last position says nothing about where it spent the
    session.
    """
    if frame_count <= 0:
        raise ValueError("cannot measure presence against a session with no frames")

    frame_area = float(frame_width * frame_height)
    signals = []
    for track in tracks:
        observed = track_positions.get(track.track_id, [])
        areas = track_areas.get(track.track_id, [])

        presence = len(observed) / frame_count
        median_area = (median(areas) / frame_area) if areas else 0.0

        if setup is None or not observed:
            # No marked zones means the front-zone signal is unmeasured, not zero-by-default.
            # It contributes nothing and `propose` says so, rather than quietly penalising every
            # track equally and pretending the score means the same thing.
            front = 0.0
        else:
            in_front = sum(
                1 for point in observed
                if setup.zone_front.contains(normalise(point, frame_width, frame_height)))
            front = in_front / len(observed)

        signals.append(TrackSignals(
            track_id=track.track_id,
            presence_fraction=min(presence, 1.0),
            median_area_fraction=min(median_area, 1.0),
            front_zone_fraction=front))
    return signals


def effective_floor(teacher_config, zones_available: bool) -> float:
    """`min_score_to_propose`, scaled to the share of the weight that could be measured.

    With no marked zones the front-zone term is unmeasured rather than zero, so the weight on it
    is not evidence against any track - it is evidence nobody has. Holding the full floor over a
    score that cannot reach it refuses the session for the operator's omission. Scaling by the
    measured share asks the same question of the two signals that exist.

    Returns the floor unchanged when every signal was measurable, so a session with zones is
    judged exactly as before.
    """
    measured = teacher_config.weight_presence_duration + teacher_config.weight_median_bbox_area
    total = measured + teacher_config.weight_front_zone_time
    if zones_available or total <= 0:
        return teacher_config.min_score_to_propose
    return teacher_config.min_score_to_propose * measured / total


def propose(signals: list[TrackSignals], teacher_config,
            zones_available: bool) -> TeacherProposal:
    """Rank the tracks and propose the best, if it clears the floor the evidence allows."""
    weights = (teacher_config.weight_presence_duration,
               teacher_config.weight_median_bbox_area,
               teacher_config.weight_front_zone_time)
    if not signals:
        return TeacherProposal(None, None, (), "no tracks were formed in this session",
                               weights, zones_available)

    ranked = sorted(signals, key=lambda s: s.score(*weights), reverse=True)
    best = ranked[0]
    best_score = best.score(*weights)
    floor = effective_floor(teacher_config, zones_available)

    caveat = ("" if zones_available else
              f" The camera setup has no marked zones, so the front-zone term contributed "
              f"nothing: this score is not comparable with a session that has them, and the "
              f"{teacher_config.min_score_to_propose} floor is scaled to {floor:.3f} for the "
              f"share of the weight that could be measured.")

    if best_score < floor:
        return TeacherProposal(
            None, best_score, tuple(ranked),
            f"the strongest track scored {best_score:.3f}, below the "
            f"{floor:.3f} floor, so nothing is proposed.{caveat}",
            weights, zones_available)

    runner_up = ranked[1].score(*weights) if len(ranked) > 1 else 0.0
    margin = best_score - runner_up
    return TeacherProposal(
        best.track_id, best_score, tuple(ranked),
        f"track {best.track_id} scored {best_score:.3f} "
        f"(present {best.presence_fraction:.0%}, median area {best.median_area_fraction:.1%} "
        f"of frame, front zone {best.front_zone_fraction:.0%}), "
        f"{margin:.3f} ahead of the next.{caveat}",
        weights, zones_available)


def accuracy_on_labelled(proposals: dict[str, TeacherProposal],
                         truth: dict[str, int]) -> dict[str, float | int]:
    """How often the heuristic proposed the right track on a fixture where it is known.

    BUILD-SPEC requires this be reported and that confirmation stay mandatory regardless of the
    number, so this returns a report and changes nothing. Sessions where the heuristic declined
    to propose are counted separately: declining is not a wrong answer, and folding it into the
    error rate would make a cautious heuristic look worse than a confidently wrong one.
    """
    scored = {s: p for s, p in proposals.items() if s in truth}
    proposed = {s: p for s, p in scored.items() if p.track_id is not None}
    correct = sum(1 for s, p in proposed.items() if p.track_id == truth[s])

    return {
        "sessions": len(scored),
        "proposed": len(proposed),
        "declined": len(scored) - len(proposed),
        "correct": correct,
        "accuracy_when_proposed": correct / len(proposed) if proposed else 0.0,
        "coverage": len(proposed) / len(scored) if scored else 0.0,
    }
