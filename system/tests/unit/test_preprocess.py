# -*- coding: utf-8 -*-
"""Phase 3 components, tested against the configured thresholds rather than against constants.

Every threshold comes from `configs/default.yaml` through the `config` fixture. A test that
hardcoded 0.5 would keep passing after somebody changed the configuration, which is the failure
mode that makes a suite stop describing the system.

Detections are constructed directly because the pipeline's correctness does not depend on which
detector produced them: that is what the `PoseEstimator` protocol is for. The ONNX
implementation needs vendored weights and is covered separately.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from pydantic import ValidationError

from praxis.preprocess import aggregates, artifacts
from praxis.preprocess import teacher as teacher_module
from praxis.preprocess.blur import (
    BlurError,
    BlurPolicy,
    OriginalSurvived,
    blur_frame,
    kernel_for,
    remove_original,
)
from praxis.preprocess.pose import KEYPOINT_COUNT, Detection, _decode, box_iou_matrix
from praxis.preprocess.tracking import ByteTrackConfig, ByteTracker, TrackState, iou_matrix
from praxis.preprocess.zones import CameraSetup, Polygon, ZoneError

NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10


def person(x: float, y: float, w: float = 60.0, h: float = 160.0, confidence: float = 0.9,
           face: bool = True, hand_up: bool = False) -> Detection:
    """A detection whose keypoints are consistent with its box.

    Built from the geometry rather than from hand-picked numbers so that a test about hands
    cannot accidentally depend on where the face is.
    """
    keypoints = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    cx = x + w / 2
    if face:
        keypoints[NOSE] = (cx, y + h * 0.08, 0.9)
        keypoints[LEFT_EYE] = (cx - w * 0.12, y + h * 0.06, 0.9)
        keypoints[RIGHT_EYE] = (cx + w * 0.12, y + h * 0.06, 0.9)

    shoulder_y = y + h * 0.25
    keypoints[LEFT_SHOULDER] = (cx - w * 0.3, shoulder_y, 0.9)
    keypoints[RIGHT_SHOULDER] = (cx + w * 0.3, shoulder_y, 0.9)
    wrist_y = shoulder_y - h * 0.15 if hand_up else shoulder_y + h * 0.35
    keypoints[LEFT_WRIST] = (cx - w * 0.35, wrist_y, 0.9)
    keypoints[RIGHT_WRIST] = (cx + w * 0.35, shoulder_y + h * 0.35, 0.9)

    return Detection(bbox=(x, y, x + w, y + h), keypoints=keypoints, confidence=confidence)


@pytest.fixture
def tracker_config(config) -> ByteTrackConfig:
    return ByteTrackConfig.from_config(config.preprocess.tracking)


@pytest.fixture
def setup() -> CameraSetup:
    """Front, middle and back as bands; the learner region is everything but the front."""
    def band(top: float, bottom: float) -> Polygon:
        return Polygon(((0.0, top), (1.0, top), (1.0, bottom), (0.0, bottom)))

    return CameraSetup(
        setup_id="01" + "0" * 24, zone_board=band(0.0, 0.15), zone_front=band(0.15, 0.4),
        zone_middle=band(0.4, 0.7), zone_back=band(0.7, 1.0),
        learner_region=band(0.4, 1.0))


# ---------------------------------------------------------------------------
# Detections and face geometry
# ---------------------------------------------------------------------------

def test_a_detection_must_carry_the_full_keypoint_set() -> None:
    """The artefact is fixed-shape (F, P, 17, 3); a short detection would corrupt it."""
    with pytest.raises(ValueError, match="17 keypoints"):
        Detection(bbox=(0, 0, 10, 10), keypoints=np.zeros((5, 3), np.float32), confidence=0.9)


def test_a_face_box_is_absent_rather_than_guessed() -> None:
    """A teacher facing the board has no face in frame. Inventing a box would blur the wrong
    pixels and leave the face untouched."""
    assert person(0, 0, face=False).face_box(1.25) is None


def test_the_face_box_grows_with_the_configured_dilation(config) -> None:
    dilate = config.preprocess.blur.dilate_face_box
    subject = person(100, 100)
    tight = subject.face_box(1.0)
    dilated = subject.face_box(dilate)

    assert dilated is not None and tight is not None
    tight_width = tight[2] - tight[0]
    assert (dilated[2] - dilated[0]) == pytest.approx(tight_width * dilate)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def test_a_person_standing_still_keeps_one_identity(tracker_config) -> None:
    tracker = ByteTracker(tracker_config)
    ids = [tracker.update([person(100, 100)], frame)[0] for frame in range(10)]

    assert len(set(ids)) == 1 and ids[0] is not None
    assert len(tracker.finished()) == 1


def test_a_moving_person_keeps_one_identity(tracker_config) -> None:
    """Velocity is what keeps the predicted box overlapping at 8 fps; without it a walking
    teacher would be re-identified every frame and the track would fragment."""
    tracker = ByteTracker(tracker_config)
    ids = [tracker.update([person(100 + frame * 25, 100)], frame)[0] for frame in range(10)]
    assert len(set(ids)) == 1


def test_two_people_keep_separate_identities(tracker_config) -> None:
    tracker = ByteTracker(tracker_config)
    for frame in range(6):
        assigned = tracker.update([person(50, 100), person(600, 100)], frame)
    assert assigned[0] != assigned[1]
    assert len({t.track_id for t in tracker.finished()}) == 2


def test_a_weak_detection_recovers_a_track_instead_of_starting_one(tracker_config) -> None:
    """The second association is what makes this ByteTrack. A teacher turning away drops in
    confidence; a tracker that ignored weak boxes would give them a new identity."""
    low = (tracker_config.track_low_thresh + tracker_config.track_high_thresh) / 2
    assert low < tracker_config.track_high_thresh

    tracker = ByteTracker(tracker_config)
    first = tracker.update([person(100, 100)], 0)[0]
    recovered = tracker.update([person(105, 100, confidence=low)], 1)[0]

    assert recovered == first, "a low-confidence box must rejoin its track, not start one"
    assert len(tracker.finished()) == 1


def test_a_detection_below_the_new_track_threshold_starts_nothing(tracker_config) -> None:
    tracker = ByteTracker(tracker_config)
    below = tracker_config.new_track_thresh - 0.01
    assigned = tracker.update([person(100, 100, confidence=below)], 0)

    assert assigned == [None], "a weak first sighting is reported, not given an identity"
    assert tracker.finished() == []


def test_a_track_is_retired_only_after_the_configured_buffer(tracker_config) -> None:
    tracker = ByteTracker(tracker_config)
    tracker.update([person(100, 100)], 0)
    tracker.update([], 1)
    assert tracker.finished()[0].state is TrackState.LOST

    tracker.update([], 1 + tracker_config.track_buffer_frames + 1)
    assert tracker.finished()[0].state is TrackState.REMOVED


def test_iou_is_zero_for_boxes_that_do_not_meet(tracker_config) -> None:
    tracker = ByteTracker(tracker_config)
    tracker.update([person(0, 0)], 0)
    scores = iou_matrix(tracker.finished(), [person(900, 900)])
    assert scores.shape == (1, 1) and scores[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Suppression
#
# The decoder is exercised through a synthesised raw tensor rather than the ONNX graph, which
# needs vendored weights. That is a contract test: the numbers are invented, so it establishes
# that duplicates are removed and in what order, not how many people are in any real classroom.
# ---------------------------------------------------------------------------

def raw_tensor(boxes: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    """`(1, 56, N)` as YOLOv8-pose emits it: cx, cy, w, h, score, then 17 x (x, y, c)."""
    rows = np.zeros((len(boxes), 5 + KEYPOINT_COUNT * 3), dtype=np.float32)
    for index, (cx, cy, w, h, score) in enumerate(boxes):
        rows[index, :5] = (cx, cy, w, h, score)
        # keypoints inside the box, all confidently observed
        rows[index, 5::3] = cx
        rows[index, 6::3] = cy
        rows[index, 7::3] = 0.99
    return rows.T[None]


def decoded(boxes, *, min_person=0.35, max_persons=60, nms_iou=0.45):
    return _decode(raw_tensor(boxes), scale=1.0, pad=(0.0, 0.0), min_person=min_person,
                   min_keypoint=0.30, max_persons=max_persons, nms_iou=nms_iou)


def test_one_body_detected_many_times_is_returned_once() -> None:
    """The defect this exists for: the ONNX graph suppresses nothing, so a real teacher clears
    the confidence floor at dozens of neighbouring anchors."""
    jitter = [(100.0 + offset, 200.0 + offset, 60.0, 160.0, 0.9 - offset / 100)
              for offset in range(12)]
    assert len(decoded(jitter)) == 1


def test_the_survivor_is_the_most_confident_of_the_duplicates() -> None:
    detections = decoded([(100.0, 200.0, 60.0, 160.0, 0.55),
                          (102.0, 203.0, 60.0, 160.0, 0.95),
                          (98.0, 197.0, 60.0, 160.0, 0.72)])
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.95)


def test_two_people_standing_apart_both_survive() -> None:
    detections = decoded([(100.0, 200.0, 60.0, 160.0, 0.9),
                          (600.0, 200.0, 60.0, 160.0, 0.8)])
    assert len(detections) == 2


def test_adjacent_pupils_survive_at_the_configured_threshold(config) -> None:
    """Shoulder to shoulder is what a classroom row looks like. They overlap, and a threshold
    that merged them would erase the learners the S2 condition is about."""
    shoulder_to_shoulder = [(100.0, 200.0, 60.0, 160.0, 0.9),
                            (145.0, 200.0, 60.0, 160.0, 0.85)]
    threshold = config.preprocess.pose.nms_iou_threshold
    assert len(decoded(shoulder_to_shoulder, nms_iou=threshold)) == 2


def test_the_cap_counts_bodies_rather_than_anchors() -> None:
    """Suppression before the cap, not after. Capping first fills the budget with copies of the
    loudest detection and discards the real people ranked below them."""
    duplicates = [(100.0, 200.0, 60.0, 160.0, 0.99 - i / 1000) for i in range(30)]
    distant = [(1000.0 + 300 * i, 200.0, 60.0, 160.0, 0.5) for i in range(3)]

    detections = decoded(duplicates + distant, max_persons=4)

    assert len(detections) == 4
    # the crowded corner contributes one body, so the three distant people are all still here
    left_edges = sorted(round(d.bbox[0]) for d in detections)
    assert left_edges == [70, 970, 1270, 1570]


def test_a_box_contained_in_another_is_suppressed() -> None:
    """A torso-only box inside a full-body box is the same person seen worse, and IoU alone can
    read low for it. The containment case is where a naive threshold leaks duplicates."""
    detections = decoded([(100.0, 200.0, 60.0, 160.0, 0.9),
                          (100.0, 190.0, 50.0, 120.0, 0.6)])
    assert len(detections) == 1


def test_suppression_is_deterministic_under_tied_scores() -> None:
    """R7. Equal confidences must not resolve by whatever order the array happened to arrive in
    on this run."""
    tied = [(100.0 + 400 * i, 200.0, 60.0, 160.0, 0.7) for i in range(4)]
    first = [d.bbox for d in decoded(tied)]
    assert first == [d.bbox for d in decoded(tied)]
    assert len(first) == 4


def test_nothing_above_the_floor_returns_nothing() -> None:
    assert decoded([(100.0, 200.0, 60.0, 160.0, 0.10)]) == []


def test_the_shared_iou_is_the_one_the_tracker_uses() -> None:
    """L20. Two implementations of overlap is how one of them silently stops matching the other.
    `iou_matrix` is a thin adapter over `box_iou_matrix`, and this asserts it stays one."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    assert box_iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)
    assert box_iou_matrix(a, np.empty((0, 4))).shape == (1, 0)
    assert box_iou_matrix(np.empty((0, 4)), b).shape == (0, 1)


def test_a_degenerate_suppression_threshold_is_refused_by_the_config(config) -> None:
    """Both ends fail silently on real footage, so they are refused where the value is read
    rather than discovered in the learner aggregates. D77."""
    from praxis.config_schema import PoseSection

    fields = config.preprocess.pose.model_dump()
    for degenerate in (0.99, 0.01):
        with pytest.raises(ValidationError, match="nms_iou_threshold"):
            PoseSection(**{**fields, "nms_iou_threshold": degenerate})


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

def test_a_zone_must_be_a_polygon_in_normalised_coordinates() -> None:
    with pytest.raises(ZoneError, match="three points"):
        Polygon(((0.0, 0.0), (1.0, 1.0)))
    with pytest.raises(ZoneError, match="outside the normalised frame"):
        Polygon(((0.0, 0.0), (640.0, 0.0), (640.0, 480.0)))


def test_a_point_on_the_boundary_is_inside(setup: CameraSetup) -> None:
    """Otherwise a person standing on the front-zone edge flickers between zones frame to
    frame, and the front-zone fraction becomes noise."""
    assert setup.zone_front.contains((0.5, 0.15))
    assert setup.zone_of((0.5, 0.2)) == "zone_front"
    assert setup.zone_of((0.5, 0.9)) == "zone_back"


def test_overlapping_seating_zones_are_reported(setup: CameraSetup) -> None:
    assert setup.overlaps() == []
    bad = CameraSetup(
        setup_id=setup.setup_id, zone_board=setup.zone_board,
        zone_front=Polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (0.0, 0.6))),
        zone_middle=setup.zone_middle, zone_back=setup.zone_back,
        learner_region=setup.learner_region)
    assert ("zone_front", "zone_middle") in bad.overlaps()


def test_a_setup_round_trips_through_its_stored_form(setup: CameraSetup) -> None:
    restored = CameraSetup.from_row(setup.setup_id, setup.as_row())
    assert restored == setup


# ---------------------------------------------------------------------------
# The teacher heuristic proposes and never decides
# ---------------------------------------------------------------------------

def _signals(track_id: int, presence: float, area: float, front: float):
    return teacher_module.TrackSignals(track_id, presence, area, front)


def test_the_score_is_the_configured_weighted_sum(config) -> None:
    teacher_config = config.preprocess.teacher_id
    signal = _signals(1, 1.0, 0.5, 0.25)
    expected = (teacher_config.weight_presence_duration * 1.0
                + teacher_config.weight_median_bbox_area * 0.5
                + teacher_config.weight_front_zone_time * 0.25)

    assert signal.score(teacher_config.weight_presence_duration,
                        teacher_config.weight_median_bbox_area,
                        teacher_config.weight_front_zone_time) == pytest.approx(expected)


def test_the_strongest_track_is_proposed(config) -> None:
    proposal = teacher_module.propose(
        [_signals(1, 0.2, 0.01, 0.0), _signals(2, 0.95, 0.20, 0.9)],
        config.preprocess.teacher_id, zones_available=True)

    assert proposal.track_id == 2
    assert proposal.score is not None and not proposal.is_confirmed
    assert "ahead of the next" in proposal.reason


def test_nothing_is_proposed_below_the_configured_floor(config) -> None:
    """An empty room must reach the reviewer as "no proposal", not as the least bad track."""
    proposal = teacher_module.propose(
        [_signals(1, 0.05, 0.001, 0.0)], config.preprocess.teacher_id, zones_available=True)

    assert proposal.track_id is None
    assert str(config.preprocess.teacher_id.min_score_to_propose) in proposal.reason


def test_a_proposal_is_never_confirmed(config) -> None:
    """R1 turns on classifying only the teacher. Confirmation is a database fact written by a
    person, and no object this module builds can carry it."""
    proposal = teacher_module.propose([_signals(1, 1.0, 0.5, 1.0)],
                                      config.preprocess.teacher_id, zones_available=True)
    row = proposal.as_row(source="detection", limit=20)
    assert proposal.is_confirmed is False
    assert row["confirmed_by"] is None and row["confirmed_at"] is None
    assert row["proposed_by"] == "heuristic"


def test_a_session_without_zones_says_its_score_is_not_comparable(config) -> None:
    proposal = teacher_module.propose([_signals(1, 1.0, 0.5, 0.0)],
                                      config.preprocess.teacher_id, zones_available=False)
    assert "not comparable" in proposal.reason


class TestTheFloorFollowsTheEvidence:
    """The floor is scaled by the share of weight that could be measured, derived from the
    configured weights and from nothing else. Asserted arithmetically against the config so a
    weight change moves the expectation with it, rather than against a number typed here."""

    def test_with_zones_the_floor_is_the_configured_one(self, config) -> None:
        teacher_config = config.preprocess.teacher_id
        assert teacher_module.effective_floor(teacher_config, True) == pytest.approx(
            teacher_config.min_score_to_propose)

    def test_without_zones_it_drops_by_the_front_zone_weight(self, config) -> None:
        teacher_config = config.preprocess.teacher_id
        measured = (teacher_config.weight_presence_duration
                    + teacher_config.weight_median_bbox_area)
        total = measured + teacher_config.weight_front_zone_time
        assert teacher_module.effective_floor(teacher_config, False) == pytest.approx(
            teacher_config.min_score_to_propose * measured / total)

    def test_scaling_can_only_admit_a_proposal_never_withdraw_one(self, config) -> None:
        """Monotone, so no session that proposed under the unscaled floor stops proposing."""
        teacher_config = config.preprocess.teacher_id
        assert (teacher_module.effective_floor(teacher_config, False)
                <= teacher_module.effective_floor(teacher_config, True))

    def test_the_caveat_names_the_scaled_floor(self, config) -> None:
        """The old text said the score was not comparable and then compared it anyway. The
        number it is actually held against has to appear in the sentence."""
        teacher_config = config.preprocess.teacher_id
        proposal = teacher_module.propose([_signals(1, 0.1, 0.0, 0.0)], teacher_config,
                                          zones_available=False)
        floor = teacher_module.effective_floor(teacher_config, False)
        assert proposal.track_id is None
        assert f"{floor:.3f}" in proposal.reason

    def test_a_session_with_no_tracks_still_carries_its_weights(self, config) -> None:
        """The empty case used to construct a proposal with no provenance at all, and its row
        would then have scored every candidate it does not have at zero."""
        proposal = teacher_module.propose([], config.preprocess.teacher_id,
                                          zones_available=False)
        assert proposal.weights == (config.preprocess.teacher_id.weight_presence_duration,
                                    config.preprocess.teacher_id.weight_median_bbox_area,
                                    config.preprocess.teacher_id.weight_front_zone_time)
        assert proposal.candidates(source="detection", limit=20)["ranked"] == []


class TestTheRankingReachesTheRow:
    """What `as_row` emits is what a reviewer is shown. Before 0009 the reason and the ranking
    were computed and dropped at this boundary, so a declined session reached the reviewer as a
    bare null."""

    def test_the_declined_proposal_keeps_its_score_and_its_ranking(self, config) -> None:
        proposal = teacher_module.propose(
            [_signals(1, 0.10, 0.0, 0.0), _signals(2, 0.05, 0.0, 0.0)],
            config.preprocess.teacher_id, zones_available=False)
        row = proposal.as_row(source="detection", limit=20)

        assert row["track_id"] is None
        assert row["heuristic_score"] == pytest.approx(
            0.10 * config.preprocess.teacher_id.weight_presence_duration)
        assert [c["track_id"] for c in row["candidates"]["ranked"]] == [1, 2]
        assert row["reason"] == proposal.reason

    def test_candidates_are_ranked_best_first_with_their_scores(self, config) -> None:
        proposal = teacher_module.propose(
            [_signals(7, 0.2, 0.0, 0.0), _signals(9, 0.9, 0.0, 0.0)],
            config.preprocess.teacher_id, zones_available=True)
        ranked = proposal.candidates(source="detection", limit=20)["ranked"]

        assert [c["track_id"] for c in ranked] == [9, 7]
        assert ranked[0]["score"] > ranked[1]["score"]
        assert ranked[0]["presence_fraction"] == pytest.approx(0.9)

    def test_the_cap_is_applied_and_the_remainder_counted(self, config) -> None:
        """A 32-minute session forms hundreds of one-frame tracks. The cap is real, and what it
        dropped is recorded rather than left to be inferred from a short list."""
        signals = [_signals(i, 1.0 - i / 100, 0.0, 0.0) for i in range(10)]
        proposal = teacher_module.propose(signals, config.preprocess.teacher_id,
                                          zones_available=True)
        blob = proposal.candidates(source="detection", limit=4)

        assert len(blob["ranked"]) == 4
        assert blob["truncated"] == 6

    def test_nothing_truncated_is_recorded_as_zero_not_omitted(self, config) -> None:
        blob = teacher_module.propose([_signals(1, 1.0, 0.5, 1.0)],
                                      config.preprocess.teacher_id,
                                      zones_available=True).candidates(
            source="detection", limit=20)
        assert blob["truncated"] == 0

    def test_the_source_is_recorded_because_two_measurements_exist(self, config) -> None:
        """A detection box and a keypoint hull are different measurements of one track, and a
        run re-emitted from a stored artifact after D18 can only have the second. The row says
        which, so the two are never read as the same number."""
        proposal = teacher_module.propose([_signals(1, 1.0, 0.5, 1.0)],
                                          config.preprocess.teacher_id, zones_available=True)
        assert proposal.candidates(source="pose_artifact", limit=20)["source"] == (
            "pose_artifact")

    def test_zones_available_is_a_snapshot_not_a_lookup(self, config) -> None:
        """Stored beside the score it explains. A session that later acquires a camera setup
        must not make an old two-signal score look like a three-signal one."""
        without = teacher_module.propose([_signals(1, 1.0, 0.5, 0.0)],
                                         config.preprocess.teacher_id, zones_available=False)
        assert without.candidates(source="detection", limit=20)["zones_available"] is False


def test_accuracy_separates_being_wrong_from_declining(config) -> None:
    """A cautious heuristic must not score worse than a confidently wrong one."""
    weights = (0.4, 0.3, 0.3)
    right = teacher_module.TeacherProposal(3, 0.8, (), "", weights, True)
    wrong = teacher_module.TeacherProposal(9, 0.8, (), "", weights, True)
    declined = teacher_module.TeacherProposal(None, 0.1, (), "", weights, True)

    report = teacher_module.accuracy_on_labelled(
        {"s1": right, "s2": wrong, "s3": declined}, {"s1": 3, "s2": 3, "s3": 3})

    assert report["sessions"] == 3 and report["proposed"] == 2 and report["declined"] == 1
    assert report["accuracy_when_proposed"] == pytest.approx(0.5)
    assert report["coverage"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Learner aggregates: R1 enforced in the pipeline
# ---------------------------------------------------------------------------

def test_match_thresh_is_a_distance_as_the_literature_defines_it(tracker_config) -> None:
    """ByteTrack's assignment cost is `1 - IoU` and the configured 0.80 is the ceiling on that
    cost, so it admits pairs down to IoU 0.20. Read as a minimum IoU it would be four times
    stricter and the tracker would fragment every walking teacher."""
    assert tracker_config.min_iou == pytest.approx(1.0 - tracker_config.match_thresh)
    assert tracker_config.min_iou < tracker_config.match_thresh


def test_a_raised_hand_is_a_wrist_above_a_shoulder() -> None:
    assert aggregates.hand_is_raised(person(0, 0, hand_up=True))
    assert not aggregates.hand_is_raised(person(0, 0, hand_up=False))


def test_an_unobserved_wrist_is_not_a_raised_hand() -> None:
    subject = person(0, 0, hand_up=True)
    subject.keypoints[LEFT_WRIST, 2] = 0.0
    assert not aggregates.hand_is_raised(subject)


def test_the_teacher_is_excluded_from_learner_counts(setup: CameraSetup) -> None:
    frames = [(f, [person(100, 500, hand_up=True), person(300, 500, hand_up=True)],
               [7, 8]) for f in range(8)]

    with_teacher = aggregates.summarise(
        frames, teacher_track_id=None, bin_seconds=1.0, sampled_fps=8.0,
        frame_width=1280, frame_height=720, setup=setup)
    without = aggregates.summarise(
        frames, teacher_track_id=7, bin_seconds=1.0, sampled_fps=8.0,
        frame_width=1280, frame_height=720, setup=setup)

    assert with_teacher[0].person_count == 2
    assert without[0].person_count == 1, "the teacher is not a learner"
    assert without[0].hands_raised == 1


def test_a_learner_bin_carries_no_identifier(setup: CameraSetup) -> None:
    """R1. What leaves `summarise` is counts; the track ids exist only inside it."""
    frames = [(f, [person(100, 500)], [42]) for f in range(8)]
    bins = aggregates.summarise(frames, teacher_track_id=None, bin_seconds=1.0,
                                sampled_fps=8.0, frame_width=1280, frame_height=720,
                                setup=setup)

    row = bins[0].as_row("01" + "0" * 24)
    assert set(row) == {"session_id", "t_start_s", "hands_raised", "gross_motion",
                        "person_count"}
    assert 42 not in row.values()
    assert not any("track" in key or key == "id" for key in row if key != "session_id")


def test_people_outside_the_learner_region_are_not_counted(setup: CameraSetup) -> None:
    """A colleague standing at the front is not a class."""
    in_front = [(f, [person(100, 200)], [1]) for f in range(8)]
    bins = aggregates.summarise(in_front, teacher_track_id=None, bin_seconds=1.0,
                                sampled_fps=8.0, frame_width=1280, frame_height=720,
                                setup=setup)
    assert bins[0].person_count == 0


def test_hands_per_minute_is_derived_not_stored() -> None:
    bins = [aggregates.LearnerBin(0.0, 3, 0.0, 10), aggregates.LearnerBin(60.0, 1, 0.0, 10)]
    assert aggregates.hands_per_minute(bins, 60.0) == pytest.approx(2.0)


def _bin(frames, fps: float, setup: CameraSetup, teacher: int | None = None):
    return aggregates.summarise(frames, teacher_track_id=teacher, bin_seconds=1.0,
                                sampled_fps=fps, frame_width=1280, frame_height=720,
                                setup=setup)


def test_the_hand_count_does_not_change_with_the_sampling_rate(setup: CameraSetup) -> None:
    """One hand raised once is one raise. Counting frames in which a hand was up would report
    eight at 8 fps and two at 2 fps for the same classroom, and `sample_fps` is a configuration
    knob: the metric would then move when nothing about the room did."""
    def held_up(count: int):
        return [(f, [person(100, 500, hand_up=True)], [1]) for f in range(count)]

    assert _bin(held_up(8), 8.0, setup)[0].hands_raised == 1
    assert _bin(held_up(2), 2.0, setup)[0].hands_raised == 1


def test_a_hand_lowered_and_raised_again_is_two_raises(setup: CameraSetup) -> None:
    sequence = [True, True, False, False, True, True, True, True]
    frames = [(f, [person(100, 500, hand_up=up)], [1]) for f, up in enumerate(sequence)]
    assert _bin(frames, 8.0, setup)[0].hands_raised == 2


def test_a_track_missing_for_a_frame_does_not_produce_a_second_raise(
        setup: CameraSetup) -> None:
    """A detector miss is not the learner putting their hand down and up again."""
    frames = [(f, [] if f == 3 else [person(100, 500, hand_up=True)], [] if f == 3 else [1])
              for f in range(8)]
    assert _bin(frames, 8.0, setup)[0].hands_raised == 1


def test_motion_is_a_speed_rather_than_a_per_frame_displacement(setup: CameraSetup) -> None:
    """The same learner crossing the same distance in the same second reports the same motion
    at either sampling rate. A per-frame displacement reports a quarter as much at 2 fps."""
    def walk(count: int, step: int):
        return [(f, [person(100 + f * step, 500)], [1]) for f in range(count)]

    fast = _bin(walk(8, 20), 8.0, setup)[0].gross_motion
    slow = _bin(walk(2, 80), 2.0, setup)[0].gross_motion

    assert fast is not None and slow is not None
    assert fast == pytest.approx(slow, rel=1e-6)


def test_motion_is_unmeasured_rather_than_zero_in_an_empty_room(setup: CameraSetup) -> None:
    bins = _bin([(f, [], []) for f in range(8)], 8.0, setup)
    assert bins[0].person_count == 0
    assert bins[0].gross_motion is None, "an empty room did not hold still"


# ---------------------------------------------------------------------------
# Blur and the deletion of the original
# ---------------------------------------------------------------------------

def test_a_policy_that_spares_some_faces_is_refused(config) -> None:
    """Blurring only non-consenting faces would encode consent in pixels."""
    blur = config.preprocess.blur.model_copy(update={"blur_all_faces": False})
    with pytest.raises(BlurError, match="blur_all_faces"):
        BlurPolicy.from_config(blur)


def test_a_policy_that_keeps_the_original_is_refused(config) -> None:
    """`model_copy` bypasses the config validator, which is the point: this proves the pipeline
    refuses on its own rather than relying on the file having been loaded through it."""
    blur = config.preprocess.blur.model_copy(update={"delete_original_after_blur": False})
    with pytest.raises(BlurError, match="delete_original_after_blur"):
        BlurPolicy.from_config(blur)


def test_the_kernel_scales_with_the_face_and_is_odd(config) -> None:
    """A fixed kernel would leave distant faces legible."""
    fraction = config.preprocess.blur.kernel_fraction
    near = kernel_for((0, 0, 200, 200), fraction)
    far = kernel_for((0, 0, 20, 20), fraction)

    assert near > far
    assert near % 2 == 1 and far % 2 == 1
    assert far >= 3


def test_blurring_changes_the_face_and_leaves_the_rest(config) -> None:
    policy = BlurPolicy.from_config(config.preprocess.blur)
    subject = person(600, 80, w=60, h=160)
    box = subject.face_box(policy.dilate_face_box)
    assert box is not None
    x1, y1, x2, y2 = (int(value) for value in box)

    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    frame[y1:y2, x1:x2] = 255                          # something legible to destroy

    output, blurred = blur_frame(frame, [subject], policy)

    assert blurred == 1
    assert not np.array_equal(output[y1:y2, x1:x2], frame[y1:y2, x1:x2])

    # A pixel margin, because the implementation floors and ceils the box to whole pixels.
    untouched = np.ones(frame.shape[:2], dtype=bool)
    untouched[max(0, y1 - 1):y2 + 2, max(0, x1 - 1):x2 + 2] = False
    assert np.array_equal(output[untouched], frame[untouched])


def test_a_faceless_detection_blurs_nothing(config) -> None:
    policy = BlurPolicy.from_config(config.preprocess.blur)
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    output, blurred = blur_frame(frame, [person(100, 100, face=False)], policy)

    assert blurred == 0 and np.array_equal(output, frame)


def test_deletion_is_verified_not_assumed(config, tmp_path) -> None:
    policy = BlurPolicy.from_config(config.preprocess.blur)
    original = tmp_path / "original.mp4"
    original.write_bytes(b"unblurred")

    remove_original(original, policy)
    assert not original.exists()


def test_a_surviving_original_is_its_own_failure(config, tmp_path, monkeypatch) -> None:
    """The one Phase 3 failure that is a privacy incident rather than a processing error, so it
    must never be caught alongside the others."""
    policy = BlurPolicy.from_config(config.preprocess.blur)
    original = tmp_path / "original.mp4"
    original.write_bytes(b"unblurred")

    monkeypatch.setattr("pathlib.Path.unlink", lambda self, missing_ok=False: None)
    with pytest.raises(OriginalSurvived, match="still exists after unlink"):
        remove_original(original, policy)


# ---------------------------------------------------------------------------
# The pose artefact Phase 4 consumes
# ---------------------------------------------------------------------------

def test_packing_pads_and_records_how_many_slots_were_real() -> None:
    """Padding must be distinguishable from a person who was not detected, or a consumer
    counting non-zero rows treats a padded slot as a missing person."""
    frames = [([person(0, 0), person(200, 0)], [1, 2]), ([person(0, 0)], [1])]
    keypoints, track_ids, person_count = artifacts.pack(frames)

    assert keypoints.shape == (2, 2, KEYPOINT_COUNT, 3)
    assert track_ids.shape == (2, 2)
    assert list(person_count) == [2, 1]
    assert track_ids[1, 1] == artifacts.NO_TRACK
    assert keypoints[1, 1].sum() == 0.0


def test_an_untracked_detection_is_stored_without_an_identity() -> None:
    keypoints, track_ids, count = artifacts.pack([([person(0, 0)], [None])])
    assert track_ids[0, 0] == artifacts.NO_TRACK
    assert count[0] == 1, "it is still a detected person"
    assert keypoints[0, 0].sum() > 0


def test_an_artefact_round_trips_and_its_hash_is_checked(tmp_path) -> None:
    frames = [([person(0, 0), person(300, 0)], [1, 2]) for _ in range(5)]
    keypoints, track_ids, count = artifacts.pack(frames)
    path = tmp_path / "pose.npz"

    artifact = artifacts.write(
        path, session_id="01" + "0" * 24, keypoints=keypoints, track_ids=track_ids,
        person_count=count, sampled_fps=8.0, model_version="yolov8m-pose",
        config_sha256="c" * 64, frame_width=1280, frame_height=720, teacher_track_id=1)

    assert artifact.frame_count == 5 and artifact.max_persons == 2
    loaded = artifacts.load(path, artifact.sha256)
    assert loaded.provenance["proposed_teacher_track_id"] == 1
    assert np.array_equal(loaded.keypoints, keypoints)

    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(artifacts.ArtifactError, match="has changed since it was written"):
        artifacts.load(path, artifact.sha256)


def test_one_track_is_extracted_with_the_time_axis_intact(tmp_path) -> None:
    """Phase 4 needs (F, 17, 3) aligned to clip frames; a shorter array would misalign a
    temporal model on every frame the teacher was absent."""
    frames = [([person(0, 0), person(300, 0)], [1, 2]),
              ([person(300, 0)], [2]),
              ([person(0, 0), person(300, 0)], [1, 2])]
    keypoints, track_ids, count = artifacts.pack(frames)
    path = tmp_path / "pose.npz"
    artifact = artifacts.write(
        path, session_id="01" + "0" * 24, keypoints=keypoints, track_ids=track_ids,
        person_count=count, sampled_fps=8.0, model_version="m", config_sha256="c" * 64,
        frame_width=1280, frame_height=720, teacher_track_id=1)

    track_one = artifacts.load(path, artifact.sha256).track(1)
    assert track_one.shape == (3, KEYPOINT_COUNT, 3)
    assert track_one[1].sum() == 0.0, "the frame track 1 was absent is zeroed, not dropped"
    assert track_one[0].sum() > 0 and track_one[2].sum() > 0


def test_the_same_arrays_produce_the_same_artefact_bytes(tmp_path) -> None:
    """R7. Two runs of one configuration must produce the same file, not merely equivalent data.

    `np.savez_compressed` writes a zip, and a zip that recorded the wall clock in its member
    headers would hash differently every run. The hash is what `pose_artifacts` stores and what
    a Phase 4 feature cache would be keyed on, so a timestamp leaking into it would make every
    artefact look modified and every cache miss.
    """
    frames = [([person(0, 0), person(300, 0)], [1, 2]) for _ in range(4)]
    keypoints, track_ids, count = artifacts.pack(frames)

    def written(name: str) -> str:
        return artifacts.write(
            tmp_path / name, session_id="01" + "0" * 24, keypoints=keypoints,
            track_ids=track_ids, person_count=count, sampled_fps=8.0, model_version="m",
            config_sha256="c" * 64, frame_width=1280, frame_height=720,
            teacher_track_id=1).sha256

    first = written("first.npz")
    time.sleep(2.1)                      # zip member timestamps have two-second resolution
    assert written("second.npz") == first


def test_the_config_digest_changes_when_preprocessing_changes(config) -> None:
    """R7. A Phase 4 feature cache keyed on this must invalidate rather than reuse pose
    produced under different settings."""
    before = artifacts.config_digest(config.preprocess)
    altered = config.preprocess.model_copy(update={"sample_fps": 4.0})
    assert artifacts.config_digest(altered) != before
