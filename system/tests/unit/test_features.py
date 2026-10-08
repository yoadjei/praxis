# -*- coding: utf-8 -*-
"""Stage A: the feature cache.

The failures worth testing here are the quiet ones. A cache that is stale, or incomplete, or
built from frames the teacher was not in, loads without complaint and trains to a number. So the
assertions are about what gets *refused*, and about the arithmetic that decides which pixels a
clip is made of - because a frame index off by one is invisible in every downstream metric.

Synthetic throughout. A fake backbone returns the mean of each crop, which makes "did the right
pixels reach the model" a thing a test can assert rather than a thing a human eyeballs.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from praxis.annotation.clips import ClipRef
from praxis.behaviour.features import (
    MIN_LOCATED_FRACTION,
    CachedClip,
    CacheProvenance,
    FeatureCacheError,
    build_clip,
    clip_directory,
    embed,
    is_cached,
    located_boxes,
    pose_frames_at,
    read_clip,
    sample_times,
    write_clip,
)
from praxis.preprocess.artifacts import LoadedPose

SESSION = "01M3VY6DJA12BF7WNY2E1CPSA4"
FRAME_WIDTH, FRAME_HEIGHT, SAMPLED_FPS = 640, 360, 8.0
FEATURE_DIM = 4


def a_pose(frames: int = 128, *, track: int = 1, absent: tuple[int, ...] = ()) -> LoadedPose:
    """A one-person artefact where the track stands still in the middle of the room.

    `absent` names frames the tracker lost them in, which is the condition the carried-box path
    exists for.
    """
    keypoints = np.zeros((frames, 1, 17, 3), dtype=np.float32)
    # A plausible body: every joint inside a 120x200 box, all confident.
    keypoints[:, 0, :, 0] = np.linspace(260, 380, 17)
    keypoints[:, 0, :, 1] = np.linspace(80, 280, 17)
    keypoints[:, 0, :, 2] = 0.9

    track_ids = np.full((frames, 1), track, dtype=np.int32)
    for frame in absent:
        track_ids[frame, 0] = -1

    return LoadedPose(
        keypoints=keypoints, track_ids=track_ids,
        person_count=np.ones(frames, dtype=np.int32),
        provenance={"frame_width": FRAME_WIDTH, "frame_height": FRAME_HEIGHT,
                    "sampled_fps": SAMPLED_FPS, "sha256": "a" * 64})


class FakeBackbone:
    """Returns the mean intensity of each crop, so what reached it is readable from the output.

    `model_version` is real because the staleness check compares it, and a stub that returned a
    constant would make that test pass without the field being carried.
    """

    def __init__(self, version: str = "fake/v1") -> None:
        self.model_version = version
        self.batches: list[int] = []

    def embed(self, tensor):
        import torch

        self.batches.append(int(tensor.shape[0]))
        means = tensor.float().mean(dim=(1, 2, 3))
        return torch.stack([means] * FEATURE_DIM, dim=1)


class FakeFfmpeg:
    """Stands in for the decoder. Each frame is a flat image whose value encodes its timestamp,
    so a test can assert which timestamps were decoded and in what order."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, tuple]] = []

    def command(self, *args):
        return ("ffmpeg", *args)


def fake_rgb(monkeypatch, recorder: list) -> None:
    """Replace the decoder with one that paints the timestamp into the pixels."""
    def rgb_at(ffmpeg, path, *, at_seconds, width, height, box=None):
        recorder.append((round(float(at_seconds), 6), box))
        value = int(np.clip(at_seconds * 10, 0, 255))
        return np.full((height, width, 3), value, dtype=np.uint8)

    monkeypatch.setattr("praxis.behaviour.features.rgb_at", rgb_at)


def a_clip(index: int = 0, start: float = 0.0, length: float = 8.0) -> ClipRef:
    return ClipRef(session_id=SESSION, clip_index=index,
                   start_seconds=start, end_seconds=start + length)


class TestWhichFramesAClipIsMadeOf:
    def test_samples_are_midpoints_inside_the_window(self) -> None:
        """D95. An endpoint sits on the clip boundary, where a seek decodes the next clip's
        first frame or nothing at all."""
        times = sample_times(a_clip(start=16.0), frames=4)

        assert times[0] > 16.0, "the first sample sits on the opening boundary"
        assert times[-1] < 24.0, "the last sample sits on the closing boundary"
        np.testing.assert_allclose(times, [17.0, 19.0, 21.0, 23.0])

    def test_the_samples_are_evenly_spaced(self) -> None:
        times = sample_times(a_clip(), frames=64)
        gaps = np.diff(times)
        np.testing.assert_allclose(gaps, gaps[0])

    def test_a_window_with_no_span_is_refused(self) -> None:
        empty = ClipRef(session_id=SESSION, clip_index=0, start_seconds=8.0, end_seconds=8.0)
        with pytest.raises(FeatureCacheError, match="not a window"):
            sample_times(empty, frames=64)

    def test_timestamps_map_onto_the_artefacts_own_sample_rate(self) -> None:
        """The clip is 8 fps and the artefact need not be. A frame index off by one here is
        invisible in every downstream metric."""
        pose = a_pose(frames=128)
        frames = pose_frames_at(np.array([0.0, 1.0, 2.0]), pose)

        np.testing.assert_array_equal(frames, [0, 8, 16])

    def test_a_timestamp_past_the_artefact_clamps_rather_than_raising(self) -> None:
        """A clip ending within one sample of the end of the video is ordinary."""
        pose = a_pose(frames=16)
        assert int(pose_frames_at(np.array([999.0]), pose)[0]) == 15

    def test_an_artefact_with_no_sample_rate_is_refused(self) -> None:
        pose = a_pose()
        pose.provenance.pop("sampled_fps")
        with pytest.raises(FeatureCacheError, match="no sample rate"):
            pose_frames_at(np.array([0.0]), pose)


class TestAFrameTheTrackerLost:
    def test_a_gap_takes_the_box_from_the_nearest_located_frame(self) -> None:
        """A black crop would put a signal in the training data the room never produced."""
        pose = a_pose(frames=32, absent=(5,))
        boxes, found = located_boxes(pose, 1, np.arange(8), pad=0.2)

        assert found == 7, "the carried frame is not counted as located"
        assert boxes[5] == boxes[4] or boxes[5] == boxes[6]
        assert all(box is not None for box in boxes)

    def test_a_clip_the_teacher_never_appears_in_is_refused(self) -> None:
        """R1. A behaviour label describes a person, and there is nobody here for it to
        describe."""
        pose = a_pose(frames=32, absent=tuple(range(8)))
        with pytest.raises(FeatureCacheError, match="none of the"):
            located_boxes(pose, 1, np.arange(8), pad=0.2)

    def test_the_count_is_what_the_tracker_produced_not_what_was_filled(self) -> None:
        pose = a_pose(frames=32, absent=(1, 2, 3))
        _, found = located_boxes(pose, 1, np.arange(8), pad=0.2)
        assert found == 5


class TestBuildingAClip:
    def test_it_produces_one_tensor_per_variant_of_the_right_shape(self, monkeypatch) -> None:
        decoded: list = []
        fake_rgb(monkeypatch, decoded)
        backbone = FakeBackbone()

        cached = build_clip(
            a_clip(), pose=a_pose(), video="blurred.mp4", backbone=backbone,
            ffmpeg=FakeFfmpeg(), track_id=1, frames=16, crop_size=32, crop_pad=0.2,
            variants=("identity", "hflip"), batch_frames=8, dtype="float16",
            config_sha256="c" * 64)

        assert set(cached.features) == {"identity", "hflip"}
        for tensor in cached.features.values():
            assert tensor.shape == (16, FEATURE_DIM)
            assert tensor.dtype == np.float16
        assert cached.keypoints.shape == (16, 17, 3)

    def test_it_decodes_the_timestamps_the_sampler_chose(self, monkeypatch) -> None:
        """The whole chain in one assertion: if the frame arithmetic drifts, these move."""
        decoded: list = []
        fake_rgb(monkeypatch, decoded)

        build_clip(a_clip(start=8.0), pose=a_pose(), video="v.mp4",
                   backbone=FakeBackbone(), ffmpeg=FakeFfmpeg(), track_id=1, frames=4,
                   crop_size=32, crop_pad=0.2, variants=("identity",), batch_frames=4,
                   dtype="float16", config_sha256="c" * 64)

        assert [at for at, _ in decoded] == [9.0, 11.0, 13.0, 15.0]

    def test_the_flip_is_embedded_from_the_same_pixels_not_a_second_decode(
            self, monkeypatch) -> None:
        """Two decodes could disagree - a seek is not guaranteed to land on the same frame
        twice - and the pair would then describe different moments."""
        decoded: list = []
        fake_rgb(monkeypatch, decoded)

        build_clip(a_clip(), pose=a_pose(), video="v.mp4", backbone=FakeBackbone(),
                   ffmpeg=FakeFfmpeg(), track_id=1, frames=8, crop_size=32, crop_pad=0.2,
                   variants=("identity", "hflip"), batch_frames=8, dtype="float16",
                   config_sha256="c" * 64)

        assert len(decoded) == 8, "the clip was decoded once per variant"

    def test_it_embeds_in_the_configured_batches(self, monkeypatch) -> None:
        """`stage_a_batch_frames` exists because inference holds the batch resident."""
        fake_rgb(monkeypatch, [])
        backbone = FakeBackbone()

        build_clip(a_clip(), pose=a_pose(), video="v.mp4", backbone=backbone,
                   ffmpeg=FakeFfmpeg(), track_id=1, frames=16, crop_size=32, crop_pad=0.2,
                   variants=("identity",), batch_frames=6, dtype="float16",
                   config_sha256="c" * 64)

        assert backbone.batches == [6, 6, 4]

    def test_a_mostly_carried_clip_is_refused(self, monkeypatch) -> None:
        """Past the floor the crops are a fabrication rather than a short gap."""
        fake_rgb(monkeypatch, [])
        # The clip samples pose frames 2, 6, 10 ... 62, so losing 1-39 loses ten of the
        # sixteen and leaves six located: 37.5 per cent, below the floor.
        pose = a_pose(frames=64, absent=tuple(range(1, 40)))

        with pytest.raises(FeatureCacheError, match="below the"):
            build_clip(a_clip(), pose=pose, video="v.mp4", backbone=FakeBackbone(),
                       ffmpeg=FakeFfmpeg(), track_id=1, frames=16, crop_size=32,
                       crop_pad=0.2, variants=("identity",), batch_frames=8,
                       dtype="float16", config_sha256="c" * 64)

    def test_a_variant_stage_a_cannot_build_is_named_rather_than_skipped(
            self, monkeypatch) -> None:
        """A silently skipped variant trains with less augmentation than the config claims."""
        fake_rgb(monkeypatch, [])
        with pytest.raises(FeatureCacheError, match="no Stage A path"):
            build_clip(a_clip(), pose=a_pose(), video="v.mp4", backbone=FakeBackbone(),
                       ffmpeg=FakeFfmpeg(), track_id=1, frames=8, crop_size=32, crop_pad=0.2,
                       variants=("colour0",), batch_frames=8, dtype="float16",
                       config_sha256="c" * 64)

    def test_the_provenance_records_what_it_was_built_from(self, monkeypatch) -> None:
        fake_rgb(monkeypatch, [])
        cached = build_clip(
            a_clip(), pose=a_pose(frames=64, absent=(3,)), video="v.mp4",
            backbone=FakeBackbone("resnet50/IMAGENET1K_V1"), ffmpeg=FakeFfmpeg(),
            track_id=1, frames=16, crop_size=32, crop_pad=0.2, variants=("identity",),
            batch_frames=8, dtype="float16", config_sha256="c" * 64)

        assert cached.provenance.backbone_version == "resnet50/IMAGENET1K_V1"
        assert cached.provenance.pose_sha256 == "a" * 64
        assert cached.provenance.feature_dim == FEATURE_DIM
        assert cached.provenance.track_id == 1
        assert cached.provenance.frames_located <= 16


def a_cached(clip_id: str = f"{SESSION}:00000", **overrides) -> CachedClip:
    provenance = dict(
        backbone_version="fake/v1", pose_sha256="a" * 64, config_sha256="c" * 64,
        feature_dim=FEATURE_DIM, frames=8, crop_size=32, crop_pad=0.2, track_id=1,
        frames_located=8)
    provenance.update(overrides)
    return CachedClip(
        clip_id=clip_id,
        features={"identity": np.zeros((8, FEATURE_DIM), dtype=np.float16)},
        keypoints=np.zeros((8, 17, 3), dtype=np.float32),
        provenance=CacheProvenance(**provenance))


class TestReadingItBack:
    def test_a_written_clip_reads_back_identical(self, tmp_path) -> None:
        cached = a_cached()
        cached.features["identity"][:] = np.arange(8 * FEATURE_DIM).reshape(8, FEATURE_DIM)
        write_clip(tmp_path, cached)

        back = read_clip(tmp_path, cached.clip_id, variants=("identity",))
        np.testing.assert_array_equal(back.features["identity"], cached.features["identity"])
        np.testing.assert_array_equal(back.keypoints, cached.keypoints)
        assert back.provenance == cached.provenance

    def test_clips_of_one_session_sit_together(self, tmp_path) -> None:
        write_clip(tmp_path, a_cached(f"{SESSION}:00000"))
        write_clip(tmp_path, a_cached(f"{SESSION}:00001"))
        assert sorted(p.name for p in (tmp_path / SESSION).iterdir()) == ["00000", "00001"]

    def test_an_uncached_clip_says_how_to_build_it(self, tmp_path) -> None:
        with pytest.raises(FeatureCacheError, match=r"build_features\.py"):
            read_clip(tmp_path, f"{SESSION}:00000", variants=("identity",))

    def test_a_cache_from_a_different_backbone_is_refused(self, tmp_path) -> None:
        """The shapes agree, the loss falls, and the result describes a model that is gone."""
        write_clip(tmp_path, a_cached())
        expected = a_cached(backbone_version="resnet50/IMAGENET1K_V2").provenance

        with pytest.raises(FeatureCacheError, match="backbone_version"):
            read_clip(tmp_path, f"{SESSION}:00000", variants=("identity",), expect=expected)

    def test_a_cache_from_a_different_pose_artefact_is_refused(self, tmp_path) -> None:
        write_clip(tmp_path, a_cached())
        expected = a_cached(pose_sha256="b" * 64).provenance

        with pytest.raises(FeatureCacheError, match="pose_sha256"):
            read_clip(tmp_path, f"{SESSION}:00000", variants=("identity",), expect=expected)

    def test_every_disagreement_is_reported_in_one_read(self, tmp_path) -> None:
        """Rebuilding the cache three times to discover three reasons is how an afternoon
        goes."""
        write_clip(tmp_path, a_cached())
        expected = a_cached(backbone_version="other", crop_size=224, frames=64).provenance

        with pytest.raises(FeatureCacheError) as raised:
            read_clip(tmp_path, f"{SESSION}:00000", variants=("identity",), expect=expected)

        said = str(raised.value)
        assert "backbone_version" in said and "crop_size" in said and "frames" in said

    def test_how_many_frames_were_located_is_not_a_staleness_reason(self, tmp_path) -> None:
        """It measures the clip rather than the settings, so a caller has nothing to compare
        it against and a difference means nothing went wrong."""
        write_clip(tmp_path, a_cached(frames_located=5))
        expected = a_cached(frames_located=8).provenance

        read_clip(tmp_path, f"{SESSION}:00000", variants=("identity",), expect=expected)

    def test_a_missing_variant_is_refused_rather_than_trained_without(self, tmp_path) -> None:
        write_clip(tmp_path, a_cached())
        with pytest.raises(FeatureCacheError, match=r"no 'hflip' variant"):
            read_clip(tmp_path, f"{SESSION}:00000", variants=("identity", "hflip"))

    def test_a_half_written_clip_does_not_look_finished(self, tmp_path) -> None:
        """The provenance is written last for this reason."""
        cached = a_cached()
        write_clip(tmp_path, cached)
        (clip_directory(tmp_path, cached.clip_id) / "provenance.json").unlink()

        assert not is_cached(tmp_path, cached.clip_id)

    def test_is_cached_reports_a_complete_entry(self, tmp_path) -> None:
        assert not is_cached(tmp_path, f"{SESSION}:00000")
        write_clip(tmp_path, a_cached())
        assert is_cached(tmp_path, f"{SESSION}:00000")

    def test_the_provenance_on_disk_is_readable_without_this_module(self, tmp_path) -> None:
        """JSON rather than a pickle, because the question it answers - what built this - is
        one somebody asks when the code that wrote it will not import."""
        write_clip(tmp_path, a_cached())
        raw = json.loads(
            (clip_directory(tmp_path, f"{SESSION}:00000") / "provenance.json")
            .read_text(encoding="utf-8"))
        assert raw["backbone_version"] == "fake/v1"


class TestTheFloorIsStated:
    def test_it_is_a_fraction_between_a_half_and_one(self) -> None:
        """Documented rather than asserted loosely: below a half the clip is mostly carried,
        and at one a single lost frame would discard an otherwise good clip."""
        assert 0.5 <= MIN_LOCATED_FRACTION < 1.0


class TestEmbedding:
    def test_a_batch_of_zero_frames_is_refused(self) -> None:
        with pytest.raises(FeatureCacheError, match="cannot be embedded"):
            embed(FakeBackbone(), np.zeros((4, 8, 8, 3), dtype=np.uint8),
                  batch_frames=0, dtype="float16")

    def test_the_backbone_receives_channels_first(self) -> None:
        """ResNet wants (N, 3, H, W) and the decoder produces (N, H, W, 3). Transposing in the
        wrong direction gives a 3-pixel-wide image of 224 channels, which still runs."""
        seen = {}

        class Shapes(FakeBackbone):
            def embed(self, tensor):
                seen["shape"] = tuple(tensor.shape)
                return super().embed(tensor)

        embed(Shapes(), np.zeros((4, 32, 32, 3), dtype=np.uint8),
              batch_frames=4, dtype="float32")
        assert seen["shape"] == (4, 3, 32, 32)
