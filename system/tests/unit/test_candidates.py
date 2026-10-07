# -*- coding: utf-8 -*-
"""Locating a candidate track in its own footage.

The screen this serves asks a reviewer which person is the teacher, and it can only be answered
if the crop beside "track 2" contains track 2. So the geometry is pinned here rather than
inspected by eye: a hull that drifts by a quarter of the frame still renders a plausible-looking
picture of the wrong person, and nothing downstream would notice.

Synthetic keypoints throughout. They validate the arithmetic that turns an artefact into a
rectangle, which is all this module does; nothing here says anything about pose quality on real
footage.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from praxis.preprocess.artifacts import KEYPOINT_COUNT, NO_TRACK, LoadedPose
from praxis.preprocess.candidates import (
    MIN_BOX_PIXELS,
    PAD_FRACTION,
    Thumbnail,
    hull,
    present_frames,
    thumbnails,
)

WIDTH, HEIGHT = 1920, 1080
FPS = 8.0


def person(x: float, y: float, *, spread: float = 100.0,
           confidence: float = 0.9) -> np.ndarray:
    """A (17, 3) block whose visible keypoints span a `spread`-sized square at (x, y)."""
    points = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    points[:, 0] = np.linspace(x, x + spread, KEYPOINT_COUNT)
    points[:, 1] = np.linspace(y, y + spread, KEYPOINT_COUNT)
    points[:, 2] = confidence
    return points


def pose(track_ids: np.ndarray, keypoints: np.ndarray, *, width: int = WIDTH,
         height: int = HEIGHT, fps: float = FPS) -> LoadedPose:
    return LoadedPose(
        keypoints=keypoints, track_ids=track_ids,
        person_count=(track_ids != NO_TRACK).sum(axis=1).astype(np.int32),
        provenance={"frame_width": width, "frame_height": height, "sampled_fps": fps})


def one_track(frames: int, *, track_id: int = 1, x: float = 400.0,
              y: float = 300.0) -> LoadedPose:
    track_ids = np.full((frames, 1), track_id, dtype=np.int32)
    keypoints = np.stack([person(x, y)[None, :, :] for _ in range(frames)])
    return pose(track_ids, keypoints.reshape(frames, 1, KEYPOINT_COUNT, 3))


class TestTheHullFindsThePerson:
    def test_it_spans_the_visible_keypoints_with_padding(self) -> None:
        box = hull(person(400, 300, spread=200), frame_width=WIDTH, frame_height=HEIGHT)
        assert box is not None
        x, y, width, height = box
        margin = 200 * PAD_FRACTION
        assert x == pytest.approx(400 - margin, abs=2)
        assert y == pytest.approx(300 - margin, abs=2)
        assert width == pytest.approx(200 + 2 * margin, abs=4)
        assert height == pytest.approx(200 + 2 * margin, abs=4)

    def test_a_keypoint_with_no_confidence_does_not_drag_the_box_to_the_origin(self) -> None:
        """The estimator zeroes a keypoint it is unsure of, so a zeroed one sits at (0, 0).
        Counting it would crop the top-left corner of the room instead of the person."""
        points = person(900, 600, spread=50)
        points[0] = 0.0
        box = hull(points, frame_width=WIDTH, frame_height=HEIGHT)
        assert box is not None
        assert box[0] > 800 and box[1] > 500

    def test_nothing_visible_is_none_and_not_a_corner(self) -> None:
        assert hull(np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32),
                    frame_width=WIDTH, frame_height=HEIGHT) is None

    def test_a_distant_person_is_widened_to_a_legible_minimum(self) -> None:
        box = hull(person(500, 500, spread=2), frame_width=WIDTH, frame_height=HEIGHT)
        assert box is not None
        assert box[2] >= MIN_BOX_PIXELS and box[3] >= MIN_BOX_PIXELS

    def test_the_box_never_leaves_the_frame(self) -> None:
        """A person at the edge pads past it, and a crop filter given a region outside the
        frame fails rather than clipping."""
        for x, y in ((0, 0), (WIDTH - 10, HEIGHT - 10), (0, HEIGHT - 5), (WIDTH - 5, 0)):
            box = hull(person(x, y, spread=300), frame_width=WIDTH, frame_height=HEIGHT)
            assert box is not None
            left, top, width, height = box
            assert left >= 0 and top >= 0
            assert left + width <= WIDTH, (x, y, box)
            assert top + height <= HEIGHT, (x, y, box)

    def test_dimensions_are_even(self) -> None:
        """Odd dimensions are refused by some encoders, and an ffmpeg exit code is a worse way
        to discover it than a rounding rule."""
        for spread in (3, 17, 51, 199):
            box = hull(person(300, 300, spread=spread), frame_width=WIDTH, frame_height=HEIGHT)
            assert box is not None
            assert box[2] % 2 == 0 and box[3] % 2 == 0

    def test_zero_padding_is_the_hull_itself(self) -> None:
        """What `reemit` measures area from: the person, not the person plus context."""
        box = hull(person(400, 300, spread=200), frame_width=WIDTH, frame_height=HEIGHT,
                   pad=0.0)
        assert box is not None
        assert box[2] == pytest.approx(200, abs=MIN_BOX_PIXELS + 2)


class TestPresenceIsReadFromTheTrackIds:
    def test_only_the_frames_the_track_appears_in(self) -> None:
        track_ids = np.array([[1, 2], [NO_TRACK, 2], [1, NO_TRACK]], dtype=np.int32)
        keypoints = np.zeros((3, 2, KEYPOINT_COUNT, 3), dtype=np.float32)
        subject = pose(track_ids, keypoints)
        assert present_frames(subject, 1).tolist() == [0, 2]
        assert present_frames(subject, 2).tolist() == [0, 1]

    def test_an_absent_track_is_empty_not_an_error(self) -> None:
        assert len(present_frames(one_track(4), 99)) == 0


class TestTheStripSpreadsAcrossThePresence:
    def test_it_returns_what_was_asked_for_when_there_is_enough(self) -> None:
        strip = thumbnails(one_track(40), 1, count=5)
        assert len(strip) == 5
        assert [t.index for t in strip] == [0, 1, 2, 3, 4]

    def test_the_stills_span_the_whole_presence(self) -> None:
        """Five consecutive frames of a thirty-minute lesson show one pose of one person. What
        identifies a teacher is where they are over the session."""
        strip = thumbnails(one_track(240), 1, count=5)
        assert [t.frame for t in strip] == [23, 71, 119, 167, 215]

    def test_the_strip_never_ends_on_the_last_frame_of_the_session(self) -> None:
        """That frame sits within one frame interval of the end of the video, so a seek to it
        decodes nothing and the entry is a dead still. It was one on every session."""
        for frames in (2, 8, 40, 241, 1215):
            for count in (1, 3, 5, 12):
                strip = thumbnails(one_track(frames), 1, count=count)
                assert strip, (frames, count)
                assert strip[-1].frame < frames - 1, (frames, count)

    def test_a_track_seen_only_in_that_frame_keeps_it(self) -> None:
        """There is no other moment to show, and an empty strip would say the track was never
        visible - which is a different and false claim."""
        track_ids = np.array([[NO_TRACK], [NO_TRACK], [1]], dtype=np.int32)
        keypoints = np.zeros((3, 1, KEYPOINT_COUNT, 3), dtype=np.float32)
        keypoints[2, 0] = person(400, 300)
        strip = thumbnails(pose(track_ids, keypoints), 1, count=3)
        assert [t.frame for t in strip] == [2]

    def test_no_two_positions_land_on_the_same_frame(self) -> None:
        """An endpoint spacing rounded two positions onto one frame for a short presence and
        served the same picture twice under two timestamps."""
        for frames in range(1, 30):
            for count in (2, 3, 5, 12):
                strip = thumbnails(one_track(frames), 1, count=count)
                seen = [t.frame for t in strip]
                assert len(set(seen)) == len(seen), (frames, count, seen)

    def test_a_short_track_yields_fewer_rather_than_repeats(self) -> None:
        """Frame 2 is the session's last and is dropped, so three frames offer two."""
        strip = thumbnails(one_track(3), 1, count=5)
        assert [t.frame for t in strip] == [0, 1]

    def test_the_timestamp_is_the_sampled_rate_not_the_source_rate(self) -> None:
        """Frame index over `sampled_fps`, which is what the pipeline stepped by. Using the
        source rate would put every still in the first eighth of the lesson."""
        strip = thumbnails(one_track(80), 1, count=2)
        assert [t.frame for t in strip] == [19, 59]
        assert strip[0].at_seconds == pytest.approx(19 / FPS)
        assert strip[-1].at_seconds == pytest.approx(59 / FPS)

    def test_a_frame_with_nothing_visible_is_skipped_not_emitted_blank(self) -> None:
        frames = 10
        track_ids = np.full((frames, 1), 1, dtype=np.int32)
        keypoints = np.stack([person(400, 300)[None] for _ in range(frames)]).reshape(
            frames, 1, KEYPOINT_COUNT, 3)
        keypoints[::2] = 0.0
        strip = thumbnails(pose(track_ids, keypoints), 1, count=frames)
        # Frames 1, 3, 5 and 7. The even ones have nothing visible and frame 9 is the session's
        # last, which is dropped before the positions are picked.
        assert [t.frame for t in strip] == [1, 3, 5, 7]

    def test_indices_are_consecutive_after_a_skip(self) -> None:
        """The index is the position in the strip, because it is what the thumbnail URL carries;
        a gap would make the second URL fetch the third picture."""
        frames = 8
        track_ids = np.full((frames, 1), 1, dtype=np.int32)
        keypoints = np.stack([person(400, 300)[None] for _ in range(frames)]).reshape(
            frames, 1, KEYPOINT_COUNT, 3)
        keypoints[0] = 0.0
        strip = thumbnails(pose(track_ids, keypoints), 1, count=frames)
        assert [t.index for t in strip] == list(range(len(strip)))

    def test_a_track_never_visible_is_an_empty_strip(self) -> None:
        frames = 4
        track_ids = np.full((frames, 1), 1, dtype=np.int32)
        keypoints = np.zeros((frames, 1, KEYPOINT_COUNT, 3), dtype=np.float32)
        assert thumbnails(pose(track_ids, keypoints), 1, count=3) == []

    def test_asking_for_none_returns_none(self) -> None:
        assert thumbnails(one_track(10), 1, count=0) == []

    def test_each_still_locates_the_right_person_in_a_crowded_frame(self) -> None:
        """Two tracks, far apart. A strip that read the wrong person slot would place both
        crops in the same corner, and the picture would look fine."""
        frames = 6
        track_ids = np.tile(np.array([1, 2], dtype=np.int32), (frames, 1))
        keypoints = np.zeros((frames, 2, KEYPOINT_COUNT, 3), dtype=np.float32)
        for frame in range(frames):
            keypoints[frame, 0] = person(100, 100, spread=80)
            keypoints[frame, 1] = person(1500, 800, spread=80)
        subject = pose(track_ids, keypoints)

        near = thumbnails(subject, 1, count=2)
        far = thumbnails(subject, 2, count=2)
        assert all(t.box[0] < 400 and t.box[1] < 400 for t in near)
        assert all(t.box[0] > 1200 and t.box[1] > 600 for t in far)


class TestBlankProvenanceIsRefused:
    @pytest.mark.parametrize("provenance", [
        {"frame_width": 0, "frame_height": 1080, "sampled_fps": 8.0},
        {"frame_width": 1920, "frame_height": 0, "sampled_fps": 8.0},
        {"frame_width": 1920, "frame_height": 1080, "sampled_fps": 0.0},
        {},
    ])
    def test_a_geometry_that_would_produce_nothing_says_so(self, provenance) -> None:
        """A zero width yields boxes of nothing and a zero rate puts every still at the start.
        Both would read as a video fault rather than a missing provenance field."""
        subject = LoadedPose(
            keypoints=np.zeros((2, 1, KEYPOINT_COUNT, 3), dtype=np.float32),
            track_ids=np.ones((2, 1), dtype=np.int32),
            person_count=np.ones(2, dtype=np.int32), provenance=provenance)
        with pytest.raises(ValueError, match="not a geometry"):
            thumbnails(subject, 1, count=1)


class TestTheWireShape:
    def test_a_thumbnail_serialises_to_what_the_screen_reads(self) -> None:
        payload = Thumbnail(track_id=3, index=1, frame=16, at_seconds=2.0,
                            box=(10, 20, 30, 40)).as_json()
        assert json.loads(json.dumps(payload)) == {
            "index": 1, "at_seconds": 2.0, "frame": 16,
            "box": {"x": 10, "y": 20, "width": 30, "height": 40}}
