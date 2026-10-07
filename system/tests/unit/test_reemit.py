# -*- coding: utf-8 -*-
"""Rebuilding a ranking from a pose artefact, and what it is not allowed to touch.

The dangerous outcome here is not a crash. It is a row that looks exactly like the original
run's and carries numbers measured a different way, because a stored metric would then have been
redefined with every column still consistent. So the assertions that matter most are the ones
about what `recover` leaves alone, and they are checked against the database in
`tests/integration/test_reemit.py`; this file pins the measurement itself.

Synthetic keypoints. They validate that presence, area and zone fractions are computed from the
artefact as described - nothing here speaks to pose quality or heuristic accuracy on real
footage.
"""
from __future__ import annotations

import numpy as np
import pytest

from praxis.config import load_config
from praxis.preprocess.artifacts import KEYPOINT_COUNT, NO_TRACK, LoadedPose
from praxis.preprocess.reemit import SOURCE, ReemitError, rebuild, signals_from_pose
from praxis.preprocess.zones import CameraSetup, Polygon

WIDTH, HEIGHT = 1000, 1000


@pytest.fixture(scope="module")
def teacher_config():
    return load_config().preprocess.teacher_id


def person(x: float, y: float, *, spread: float = 100.0) -> np.ndarray:
    points = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    points[:, 0] = np.linspace(x, x + spread, KEYPOINT_COUNT)
    points[:, 1] = np.linspace(y, y + spread, KEYPOINT_COUNT)
    points[:, 2] = 0.9
    return points


def artefact(track_ids: np.ndarray, keypoints: np.ndarray) -> LoadedPose:
    return LoadedPose(
        keypoints=keypoints, track_ids=track_ids,
        person_count=(track_ids != NO_TRACK).sum(axis=1).astype(np.int32),
        provenance={"frame_width": WIDTH, "frame_height": HEIGHT, "sampled_fps": 8.0})


def two_people(frames: int, *, first_present: int, second_present: int) -> LoadedPose:
    """Track 1 for `first_present` frames at the top-left, track 2 for `second_present` at the
    bottom-right, both out of `frames`."""
    track_ids = np.full((frames, 2), NO_TRACK, dtype=np.int32)
    track_ids[:first_present, 0] = 1
    track_ids[:second_present, 1] = 2
    keypoints = np.zeros((frames, 2, KEYPOINT_COUNT, 3), dtype=np.float32)
    keypoints[:first_present, 0] = person(100, 100, spread=300)
    keypoints[:second_present, 1] = person(700, 700, spread=50)
    return artefact(track_ids, keypoints)


def square(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon(points=((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


def setup_with_front_at_the_top() -> CameraSetup:
    return CameraSetup(
        setup_id="S1",
        zone_board=square(0.0, 0.0, 1.0, 0.05),
        zone_front=square(0.0, 0.05, 1.0, 0.5),
        zone_middle=square(0.0, 0.5, 1.0, 0.75),
        zone_back=square(0.0, 0.75, 1.0, 1.0),
        learner_region=square(0.0, 0.5, 1.0, 1.0))


class TestPresenceIsTheSameQuantityAsBefore:
    def test_it_is_the_share_of_sampled_frames_the_track_was_tracked_in(self) -> None:
        """The one signal unaffected by the change of source: the tracker's own assignment is
        in the artefact, so this number is the original run's number."""
        signals = signals_from_pose(two_people(100, first_present=80, second_present=25),
                                    setup=None)
        by_track = {s.track_id: s for s in signals}
        assert by_track[1].presence_fraction == pytest.approx(0.80)
        assert by_track[2].presence_fraction == pytest.approx(0.25)

    def test_a_frame_with_no_visible_keypoints_still_counts_as_presence(self) -> None:
        """The tracker saw the person there. Dropping the frame would understate how long they
        were in the room, which is the signal carrying the most weight."""
        frames = 10
        track_ids = np.full((frames, 1), 1, dtype=np.int32)
        keypoints = np.stack([person(400, 400)[None] for _ in range(frames)]).reshape(
            frames, 1, KEYPOINT_COUNT, 3)
        keypoints[:4] = 0.0
        signals = signals_from_pose(artefact(track_ids, keypoints), setup=None)
        assert signals[0].presence_fraction == pytest.approx(1.0)

    def test_untracked_detections_are_not_a_track(self) -> None:
        """`NO_TRACK` marks a detection the tracker would not commit to. Ranking it would offer
        a reviewer a candidate that has no identity across frames."""
        frames = 6
        track_ids = np.full((frames, 2), NO_TRACK, dtype=np.int32)
        track_ids[:, 0] = 1
        keypoints = np.zeros((frames, 2, KEYPOINT_COUNT, 3), dtype=np.float32)
        keypoints[:, 0] = person(300, 300)
        keypoints[:, 1] = person(800, 800)
        assert [s.track_id for s in signals_from_pose(artefact(track_ids, keypoints),
                                                      setup=None)] == [1]


class TestAreaIsTheHullAndSaysSo:
    def test_a_larger_person_scores_a_larger_area(self) -> None:
        signals = {s.track_id: s for s in
                   signals_from_pose(two_people(50, first_present=50, second_present=50),
                                     setup=None)}
        assert signals[1].median_area_fraction > signals[2].median_area_fraction

    def test_the_fraction_is_of_the_frame_recorded_in_the_provenance(self) -> None:
        """A 300-pixel-square hull in a 1000x1000 frame is 9 per cent of it. Measured against
        the file's own dimensions instead, every area in the corpus would shift."""
        signals = signals_from_pose(two_people(10, first_present=10, second_present=0),
                                    setup=None)
        assert signals[0].median_area_fraction == pytest.approx(0.09, abs=0.04)

    def test_a_track_never_visible_has_no_area_rather_than_a_guess(self) -> None:
        frames = 5
        track_ids = np.full((frames, 1), 7, dtype=np.int32)
        keypoints = np.zeros((frames, 1, KEYPOINT_COUNT, 3), dtype=np.float32)
        signals = signals_from_pose(artefact(track_ids, keypoints), setup=None)
        assert signals[0].median_area_fraction == 0.0
        assert signals[0].presence_fraction == pytest.approx(1.0)


class TestTheFrontZoneNeedsZones:
    def test_without_a_setup_the_fraction_is_zero_for_every_track(self) -> None:
        """Unmeasured, not zero-by-default. `propose` scales its floor for exactly this, and
        the caveat it writes is what tells a reader the score is not comparable."""
        signals = signals_from_pose(two_people(20, first_present=20, second_present=20),
                                    setup=None)
        assert all(s.front_zone_fraction == 0.0 for s in signals)

    def test_with_a_setup_the_person_at_the_front_scores_and_the_other_does_not(self) -> None:
        signals = {s.track_id: s for s in
                   signals_from_pose(two_people(20, first_present=20, second_present=20),
                                     setup=setup_with_front_at_the_top())}
        assert signals[1].front_zone_fraction == pytest.approx(1.0)
        assert signals[2].front_zone_fraction == pytest.approx(0.0)

    def test_it_is_a_share_of_the_frames_the_person_could_be_located_in(self) -> None:
        """Dividing by every tracked frame instead would report a teacher who stood at the
        front throughout as having been there for part of it, because the frames where the
        estimator saw nobody clearly would count against them."""
        frames = 10
        track_ids = np.full((frames, 1), 1, dtype=np.int32)
        keypoints = np.stack([person(100, 100, spread=200)[None] for _ in range(frames)]
                             ).reshape(frames, 1, KEYPOINT_COUNT, 3)
        keypoints[:6] = 0.0
        signals = signals_from_pose(artefact(track_ids, keypoints),
                                    setup=setup_with_front_at_the_top())
        assert signals[0].front_zone_fraction == pytest.approx(1.0)
        assert signals[0].presence_fraction == pytest.approx(1.0)


class TestRebuildGoesThroughTheRealProposeFunction:
    def test_the_ranking_is_ordered_and_the_source_is_named(self, teacher_config) -> None:
        proposal = rebuild(two_people(100, first_present=95, second_present=20),
                           teacher_config=teacher_config, setup=None)
        payload = proposal.candidates(source=SOURCE, limit=10)
        assert payload["source"] == SOURCE
        scores = [candidate["score"] for candidate in payload["ranked"]]
        assert scores == sorted(scores, reverse=True)

    def test_the_missing_zone_caveat_survives_the_rebuild(self, teacher_config) -> None:
        """The reason reaches a reviewer's screen. A rebuild that dropped the caveat would show
        a score against a scaled floor without saying either thing had happened."""
        proposal = rebuild(two_people(100, first_present=95, second_present=20),
                           teacher_config=teacher_config, setup=None)
        assert "no marked zones" in proposal.reason
        assert proposal.zones_available is False

    def test_zones_available_follows_the_setup(self, teacher_config) -> None:
        proposal = rebuild(two_people(100, first_present=95, second_present=20),
                           teacher_config=teacher_config,
                           setup=setup_with_front_at_the_top())
        assert proposal.zones_available is True
        assert proposal.candidates(source=SOURCE, limit=10)["zones_available"] is True

    def test_an_empty_session_declines_rather_than_proposing_nobody(self,
                                                                    teacher_config) -> None:
        frames = 4
        track_ids = np.full((frames, 1), NO_TRACK, dtype=np.int32)
        keypoints = np.zeros((frames, 1, KEYPOINT_COUNT, 3), dtype=np.float32)
        proposal = rebuild(artefact(track_ids, keypoints), teacher_config=teacher_config,
                           setup=None)
        assert proposal.track_id is None
        assert "no tracks" in proposal.reason

    def test_the_weights_travel_with_the_ranking(self, teacher_config) -> None:
        """Two copies of the weights is how the stored scores and the ranking beside them come
        to disagree, which is why `TeacherProposal` carries them."""
        proposal = rebuild(two_people(50, first_present=50, second_present=10),
                           teacher_config=teacher_config, setup=None)
        assert proposal.weights == (teacher_config.weight_presence_duration,
                                    teacher_config.weight_median_bbox_area,
                                    teacher_config.weight_front_zone_time)


class TestAnArtefactThatCannotBeMeasuredIsRefused:
    def test_no_frames(self) -> None:
        subject = LoadedPose(
            keypoints=np.zeros((0, 0, KEYPOINT_COUNT, 3), dtype=np.float32),
            track_ids=np.zeros((0, 0), dtype=np.int32),
            person_count=np.zeros(0, dtype=np.int32),
            provenance={"frame_width": WIDTH, "frame_height": HEIGHT, "sampled_fps": 8.0})
        with pytest.raises(ReemitError, match="covers no frames"):
            signals_from_pose(subject, setup=None)

    @pytest.mark.parametrize("provenance", [
        {"frame_width": 0, "frame_height": 1000, "sampled_fps": 8.0},
        {"frame_width": 1000, "frame_height": 0, "sampled_fps": 8.0},
        {},
    ])
    def test_a_frame_size_no_fraction_could_be_measured_against(self, provenance) -> None:
        subject = LoadedPose(
            keypoints=np.zeros((2, 1, KEYPOINT_COUNT, 3), dtype=np.float32),
            track_ids=np.ones((2, 1), dtype=np.int32),
            person_count=np.ones(2, dtype=np.int32), provenance=provenance)
        with pytest.raises(ReemitError, match="would mean anything"):
            signals_from_pose(subject, setup=None)
