# -*- coding: utf-8 -*-
"""Learner evidence, as counts, with the identities discarded.

This is where R1 is enforced in the pipeline rather than in the schema. SCHEMA.md removes the
*place* to store a learner identity; this module removes the identity itself, at the point
between computing the aggregates and writing anything down.

BUILD-SPEC Phase 3 item 5: "After computing aggregate learner counts (hands raised per minute,
gross motion index), drop non-teacher track identifiers. R1 is enforced here."

The order matters and is the whole design. Counting hands raised requires knowing which
detections belong to the same person across frames, so tracks must exist first. What R1
forbids is those identifiers surviving the computation, so `summarise` takes tracked detections
and returns bins that contain no identifier of any kind.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from praxis.preprocess.pose import Detection
from praxis.preprocess.zones import CameraSetup, normalise

# COCO indices used for the raised-hand test.
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


@dataclass(frozen=True)
class LearnerBin:
    """One time bin of learner evidence. Deliberately carries no identifier.

    Matches `learner_aggregates` exactly: the primary key is (session_id, t_start_s) and there
    is no column this could populate that would identify anybody.

    `hands_raised` counts raise *events* in the bin, not frames in which a hand was up.
    `gross_motion` is a speed, in frame diagonals per second. Both are therefore independent of
    `preprocess.sample_fps`: a hand raised once is one event whether it was sampled at 8 fps or
    at 2, and lowering the sampling rate does not make a class look calmer. A per-frame tally
    would have changed value under a configuration edit that changed nothing about the room.

    `gross_motion` is None when no learner was present in two consecutive sampled frames, which
    is a different statement from zero. An empty room did not hold still.
    """

    t_start_s: float
    hands_raised: int
    gross_motion: float | None
    person_count: int

    def as_row(self, session_id: str) -> dict[str, object]:
        return {"session_id": session_id, "t_start_s": self.t_start_s,
                "hands_raised": self.hands_raised, "gross_motion": self.gross_motion,
                "person_count": self.person_count}


def hand_is_raised(detection: Detection) -> bool:
    """A wrist above the shoulder on the same side, both observed.

    Deliberately crude and deliberately not a classifier. R1 forbids modelling learners, and a
    raised-hand *detector* trained on pupils would be exactly that. This is a geometric test on
    keypoints that already exist, and it is reported as a count per time bin, never per person.

    Image coordinates put y increasing downward, so "above" is a smaller y.
    """
    keypoints = detection.keypoints
    for wrist, shoulder in ((LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)):
        observed = keypoints[wrist, 2] > 0 and keypoints[shoulder, 2] > 0
        if observed and keypoints[wrist, 1] < keypoints[shoulder, 1]:
            return True
    return False


def _gross_motion(previous: dict[int, tuple[float, float]],
                  current: dict[int, tuple[float, float]], diagonal: float,
                  seconds_between_frames: float) -> float | None:
    """Mean centroid speed of the people present in both frames, in frame diagonals per second.

    Divided by the frame diagonal so two rooms recorded at different distances are comparable,
    and by the interval so two sessions preprocessed at different sampling rates are.

    None when nobody was present in both frames: there was no motion to measure, which is not
    the same claim as no motion.
    """
    shared = set(previous) & set(current)
    if not shared or diagonal <= 0 or seconds_between_frames <= 0:
        return None
    moved = [float(np.hypot(current[k][0] - previous[k][0], current[k][1] - previous[k][1]))
             for k in shared]
    return float(np.mean(moved) / diagonal / seconds_between_frames)


def summarise(frames: Sequence[tuple[int, list[Detection], list[int | None]]], *,
              teacher_track_id: int | None, bin_seconds: float, sampled_fps: float,
              frame_width: int, frame_height: int,
              setup: CameraSetup | None) -> list[LearnerBin]:
    """Aggregate learner evidence into time bins, discarding every identifier on the way out.

    `frames` is (frame_index, detections, track_ids) with track_ids aligned to detections.
    `teacher_track_id` is excluded from every count; when it is None nothing is excluded, and
    the caller must know that the counts then include the teacher. The pipeline only calls this
    after a proposal exists, and records which case applied.

    `setup` restricts counting to `learner_region` when the operator has marked one, which is
    what stops a colleague standing at the back of the room being counted as a class.

    The two metrics are defined so that `preprocess.sample_fps` does not appear in their value:
    hands are counted as events and motion as a speed. See `LearnerBin`.
    """
    if bin_seconds <= 0 or sampled_fps <= 0:
        raise ValueError("bin_seconds and sampled_fps must both be positive")

    diagonal = float(np.hypot(frame_width, frame_height))
    frames_per_bin = max(1, round(bin_seconds * sampled_fps))
    seconds_between_frames = 1.0 / sampled_fps

    bins: dict[int, dict[str, float]] = {}
    previous_centroids: dict[int, tuple[float, float]] = {}
    # Last known hand state per track, carried across frames where the track was not detected so
    # that a momentary miss does not read as the hand being lowered and raised again.
    raised: dict[int, bool] = {}

    for frame_index, detections, track_ids in frames:
        bin_index = frame_index // frames_per_bin
        bucket = bins.setdefault(bin_index, {"raises": 0.0, "motion": 0.0, "moved_frames": 0.0,
                                             "people": 0.0, "frames": 0.0})

        centroids: dict[int, tuple[float, float]] = {}
        people = 0

        for detection, track_id in zip(detections, track_ids, strict=True):
            if track_id is not None and track_id == teacher_track_id:
                continue
            if setup is not None:
                point = normalise(detection.centroid, frame_width, frame_height)
                if not setup.learner_region.contains(point):
                    continue

            people += 1
            if track_id is None:
                # A raise is a transition, and a transition needs an identity to belong to. An
                # untracked detection still counts as a person present; it cannot contribute an
                # event, and inventing one per frame is how the per-frame tally went wrong.
                continue

            centroids[track_id] = detection.centroid
            up = hand_is_raised(detection)
            if up and not raised.get(track_id, False):
                bucket["raises"] += 1
            raised[track_id] = up

        bucket["people"] += people
        bucket["frames"] += 1
        moved = _gross_motion(previous_centroids, centroids, diagonal, seconds_between_frames)
        if moved is not None:
            bucket["motion"] += moved
            bucket["moved_frames"] += 1
        previous_centroids = centroids

    # Identifiers exist only inside this function. What leaves it is counts. R1.
    return [
        LearnerBin(
            t_start_s=round(index * frames_per_bin / sampled_fps, 6),
            hands_raised=int(bucket["raises"]),
            gross_motion=(round(bucket["motion"] / bucket["moved_frames"], 6)
                          if bucket["moved_frames"] else None),
            person_count=(round(bucket["people"] / bucket["frames"])
                          if bucket["frames"] else 0))
        for index, bucket in sorted(bins.items())
    ]


def hands_per_minute(bins: Sequence[LearnerBin], bin_seconds: float) -> float:
    """The rate BUILD-SPEC names, derived rather than stored, so the bins stay the only record.

    Raise events over elapsed minutes. Because `hands_raised` counts events, this is a count of
    hands going up per minute rather than a count of hand-frames scaled by the sampling rate.
    """
    if not bins or bin_seconds <= 0:
        return 0.0
    total_minutes = len(bins) * bin_seconds / 60.0
    return sum(b.hands_raised for b in bins) / total_minutes if total_minutes else 0.0
