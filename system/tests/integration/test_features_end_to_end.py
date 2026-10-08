# -*- coding: utf-8 -*-
"""Stage A against a real video file and the real vendored backbone.

`tests/unit/test_features.py` covers the arithmetic and the refusals with a fake decoder and a
fake backbone, which is where those belong. What it cannot establish is that the pieces fit:
that ffmpeg accepts the filter chain this module builds, that a crop rectangle derived from a
pose artefact lands inside a real frame, and that ResNet-50 accepts the tensor shape and dtype
it is handed. Each of those has exactly one correct form and several that run.

The backbone is the vendored file, not a stub. D58 forbids downloading one at runtime, so a test
that silently fetched weights would be testing a rule this project exists to keep.
"""
from __future__ import annotations

import numpy as np
import pytest

from praxis.annotation.clips import ClipRef
from praxis.behaviour.backbone import BackboneError, build_backbone
from praxis.behaviour.features import build_clip, cache_digest, read_clip, write_clip
from praxis.config import load_config
from praxis.preprocess.artifacts import LoadedPose
from praxis.tools import resolve
from tests.integration.conftest import make_clip

SESSION = "01M3VY6DJA12BF7WNY2E1CPSA4"
CLIP_SECONDS = 2.0
WIDTH, HEIGHT = 320, 240
SAMPLED_FPS = 8.0


@pytest.fixture(scope="module")
def backbone():
    """The real ResNet-50, or an honest refusal naming what is missing.

    Returned rather than raised, and never skipped: the project forbids skip and xfail because
    a skipped test reads as a passing one in a summary. Each test fails with the reason named,
    so a missing vendored weight file is reported as the state of the estate that it is.
    """
    try:
        return build_backbone(load_config())
    except BackboneError as missing:
        return missing


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A real encoded file. `testsrc` gives moving content, so two frames differ."""
    return make_clip(tmp_path_factory.mktemp("stagea") / "blurred.mp4", CLIP_SECONDS + 1.0,
                     with_audio=False, size=f"{WIDTH}x{HEIGHT}")


def a_pose(frames: int = 32) -> LoadedPose:
    """One person, upright, well inside a 320x240 frame."""
    keypoints = np.zeros((frames, 1, 17, 3), dtype=np.float32)
    keypoints[:, 0, :, 0] = np.linspace(120, 200, 17)
    keypoints[:, 0, :, 1] = np.linspace(40, 180, 17)
    keypoints[:, 0, :, 2] = 0.9
    return LoadedPose(
        keypoints=keypoints, track_ids=np.ones((frames, 1), dtype=np.int32),
        person_count=np.ones(frames, dtype=np.int32),
        provenance={"frame_width": WIDTH, "frame_height": HEIGHT,
                    "sampled_fps": SAMPLED_FPS, "sha256": "e" * 64})


def built(video, backbone, config, **overrides):
    arguments = dict(
        pose=a_pose(), video=video, backbone=backbone, ffmpeg=resolve("ffmpeg",
                                                                     config.tools.ffmpeg),
        track_id=1, frames=8, crop_size=config.behaviour.clip.crop_size,
        crop_pad=(config.behaviour.clip.crop_padding - 1.0) / 2.0,
        variants=("identity",), batch_frames=4,
        dtype=config.behaviour.backbone.cache_dtype, config_sha256=cache_digest(config))
    arguments.update(overrides)
    return build_clip(ClipRef(SESSION, 0, 0.0, CLIP_SECONDS), **arguments)


class TestThePiecesFit:
    def test_a_real_clip_becomes_features_of_the_declared_width(self, video, backbone) -> None:
        """2048 is not a number this module chooses; it is what the configured backbone emits,
        and a mismatch means the weights are not the architecture the config names."""
        if isinstance(backbone, BackboneError):
            pytest.fail(f"the vendored backbone is missing: {backbone}")

        config = load_config()
        cached = built(video, backbone, config)

        assert cached.features["identity"].shape == (8, config.behaviour.backbone.output_dim)
        assert cached.features["identity"].dtype == np.float16
        assert cached.keypoints.shape == (8, 17, 3)

    def test_different_frames_produce_different_features(self, video, backbone) -> None:
        """The decode, the crop and the embed all in one assertion. Identical rows mean the
        seek returned the same frame every time, which is what a wrong timestamp looks like -
        and it would train a temporal model on a still image without any error."""
        if isinstance(backbone, BackboneError):
            pytest.fail(f"the vendored backbone is missing: {backbone}")

        features = built(video, backbone, load_config()).features["identity"]
        spread = features.astype("float32").std(axis=0).mean()
        assert spread > 0.0, "every frame embedded identically; the clip is one frame repeated"

    def test_the_flip_is_a_different_tensor_from_the_identity(self, video, backbone) -> None:
        """If the flip were a no-op the augmentation would be absent while the config, the
        policy and the cache all said it was present."""
        if isinstance(backbone, BackboneError):
            pytest.fail(f"the vendored backbone is missing: {backbone}")

        cached = built(video, backbone, load_config(), variants=("identity", "hflip"))
        assert not np.array_equal(cached.features["identity"], cached.features["hflip"])

    def test_it_round_trips_through_the_cache(self, video, backbone, tmp_path) -> None:
        if isinstance(backbone, BackboneError):
            pytest.fail(f"the vendored backbone is missing: {backbone}")

        cached = built(video, backbone, load_config())
        write_clip(tmp_path, cached)
        back = read_clip(tmp_path, cached.clip_id, variants=("identity",),
                         expect=cached.provenance)

        np.testing.assert_array_equal(back.features["identity"], cached.features["identity"])
        assert back.provenance.backbone_version == cached.provenance.backbone_version

    def test_the_cached_version_is_the_configured_one(self, video, backbone) -> None:
        """What `declared_version` promises without loading the weights has to be what the
        loaded weights call themselves, or the staleness check compares two different things."""
        if isinstance(backbone, BackboneError):
            pytest.fail(f"the vendored backbone is missing: {backbone}")

        from praxis.behaviour.backbone import declared_version

        cached = built(video, backbone, load_config())
        assert cached.provenance.backbone_version == declared_version(load_config())
