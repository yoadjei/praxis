# -*- coding: utf-8 -*-
"""ByteTrack: detections become person tracks.

Implemented here rather than imported. `ultralytics.trackers.byte_tracker` resolves DNS and
pip-installs missing extras while being imported, so it cannot sit in the inference path under
R6 no matter what is set afterwards. D57.

This follows Zhang et al. (2022), "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box", and the distinguishing idea is the second association: low-confidence boxes are
not discarded, they are matched against tracks the first pass failed to cover. A teacher who
turns away from the camera drops in confidence rather than disappearing, and a tracker that
ignores weak boxes gives that teacher a new identity every time they turn around, which would
fragment exactly the track the teacher heuristic depends on.

Every threshold comes from `preprocess.tracking` in the configuration. None is hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from praxis.preprocess.pose import Detection, box_iou_matrix


class TrackState(Enum):
    TENTATIVE = "tentative"      # seen once, not yet confirmed
    TRACKED = "tracked"          # matched in the most recent frame
    LOST = "lost"                # unmatched, still within the buffer
    REMOVED = "removed"          # buffer exhausted


@dataclass
class Track:
    """One person's identity across frames.

    Motion is a constant-velocity prediction on the box centre rather than a full Kalman
    filter. At 8 fps a person moves far between frames and the velocity estimate is what keeps
    IoU overlapping at all; the covariance a Kalman filter adds would be tuned against nothing,
    since no labelled tracking fixture exists for this corpus.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    state: TrackState = TrackState.TENTATIVE
    first_frame: int = 0
    last_frame: int = 0
    hits: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)
    history: list[int] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def predicted(self) -> tuple[float, float, float, float]:
        vx, vy = self.velocity
        x1, y1, x2, y2 = self.bbox
        return x1 + vx, y1 + vy, x2 + vx, y2 + vy

    def update(self, detection: Detection, frame_index: int) -> None:
        old_cx, old_cy = self.centroid
        self.bbox = detection.bbox
        new_cx, new_cy = self.centroid
        self.velocity = (new_cx - old_cx, new_cy - old_cy)
        self.confidence = detection.confidence
        self.state = TrackState.TRACKED
        self.last_frame = frame_index
        self.hits += 1
        self.history.append(frame_index)


def iou_matrix(tracks: list[Track], detections: list[Detection]) -> np.ndarray:
    """Pairwise IoU between predicted track boxes and detection boxes."""
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    return box_iou_matrix(
        np.array([track.predicted() for track in tracks], dtype=np.float64),
        np.array([detection.bbox for detection in detections], dtype=np.float64))


def _associate(tracks: list[Track], detections: list[Detection],
               min_iou: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment on IoU, with pairs below `min_iou` rejected afterwards."""
    scores = iou_matrix(tracks, detections)
    if scores.size == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    track_rows, detection_cols = linear_sum_assignment(-scores)
    matched = [(int(t), int(d)) for t, d in zip(track_rows, detection_cols, strict=True)
               if scores[t, d] >= min_iou]

    used_tracks = {t for t, _ in matched}
    used_detections = {d for _, d in matched}
    return (matched,
            [i for i in range(len(tracks)) if i not in used_tracks],
            [j for j in range(len(detections)) if j not in used_detections])


@dataclass
class ByteTrackConfig:
    """Mirrors `preprocess.tracking`, so the tracker never reads the global config itself."""

    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer_frames: int
    match_thresh: float

    @property
    def min_iou(self) -> float:
        """`match_thresh` is a distance, not a similarity, and 0.80 therefore means IoU >= 0.20.

        That is its meaning in Zhang et al. and in every configuration that inherits ByteTrack's
        defaults, which this one does: the assignment cost is `1 - IoU` and a pair is kept when
        the cost is at most `match_thresh`. Read instead as a minimum IoU, the configured value
        would be four times stricter than intended, and at 8 fps a teacher walking across the
        room would fail to match their own predicted box and be given a new identity every few
        frames - the exact fragmentation the second association exists to prevent.
        """
        return 1.0 - self.match_thresh

    @classmethod
    def from_config(cls, tracking) -> ByteTrackConfig:
        return cls(track_high_thresh=tracking.track_high_thresh,
                   track_low_thresh=tracking.track_low_thresh,
                   new_track_thresh=tracking.new_track_thresh,
                   track_buffer_frames=tracking.track_buffer_frames,
                   match_thresh=tracking.match_thresh)


class ByteTracker:
    """Stateful across a session. One instance per video, never reused between sessions."""

    def __init__(self, config: ByteTrackConfig) -> None:
        self.config = config
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], frame_index: int) -> list[int | None]:
        """Advance one frame. Returns a track id per detection, aligned to the input order.

        None means the detection was not assigned an identity: it was too weak to start a track
        and matched nothing. It is still a real detection and still counts toward the learner
        aggregates, which is why the caller receives it rather than having it dropped here.
        """
        high = [i for i, d in enumerate(detections)
                if d.confidence >= self.config.track_high_thresh]
        low = [i for i, d in enumerate(detections)
               if self.config.track_low_thresh <= d.confidence < self.config.track_high_thresh]

        assigned: list[int | None] = [None] * len(detections)
        active = [t for t in self.tracks if t.state in (TrackState.TRACKED, TrackState.LOST)]

        # First association: confident detections against every live track.
        matched, unmatched_tracks, unmatched_high = _associate(
            active, [detections[i] for i in high], self.config.min_iou)
        for track_pos, detection_pos in matched:
            index = high[detection_pos]
            active[track_pos].update(detections[index], frame_index)
            assigned[index] = active[track_pos].track_id

        # Second association, the part that makes this ByteTrack rather than plain IoU
        # tracking: the weak boxes are offered to the tracks the first pass left unmatched.
        remaining = [active[i] for i in unmatched_tracks]
        low_matched, still_unmatched, _ = _associate(
            remaining, [detections[i] for i in low], self.config.min_iou)
        for track_pos, detection_pos in low_matched:
            index = low[detection_pos]
            remaining[track_pos].update(detections[index], frame_index)
            assigned[index] = remaining[track_pos].track_id

        for position in still_unmatched:
            track = remaining[position]
            if track.state is TrackState.TRACKED:
                track.state = TrackState.LOST

        # A new identity is only started by a detection confident enough to be worth one.
        for detection_pos in unmatched_high:
            index = high[detection_pos]
            if detections[index].confidence < self.config.new_track_thresh:
                continue
            track = Track(track_id=self._next_id, bbox=detections[index].bbox,
                          confidence=detections[index].confidence, state=TrackState.TRACKED,
                          first_frame=frame_index, last_frame=frame_index,
                          history=[frame_index])
            self._next_id += 1
            self.tracks.append(track)
            assigned[index] = track.track_id

        self._retire(frame_index)
        return assigned

    def _retire(self, frame_index: int) -> None:
        for track in self.tracks:
            if (track.state is TrackState.LOST
                    and frame_index - track.last_frame > self.config.track_buffer_frames):
                track.state = TrackState.REMOVED

    def finished(self) -> list[Track]:
        """Every track the session produced, including those that were lost and retired."""
        return list(self.tracks)
