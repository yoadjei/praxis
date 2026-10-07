# -*- coding: utf-8 -*-
"""The Stage B network.

What is checked here is structure, not skill: that the clip length survives the dilations, that
attention is a distribution over time, that the temporal stack actually looks at time rather
than collapsing to a per-frame classifier, and that what comes out the other end is something
the codebook accepts. Whether the thing learns anything needs the corpus, and no test in this
repository claims it does.
"""
from __future__ import annotations

import pytest
import torch

from praxis.annotation import CODEBOOK_V1
from praxis.behaviour.heads import HeadKind
from praxis.behaviour.model import (
    BehaviourModel,
    ModelShape,
    build_model,
    shape_from_config,
)
from praxis.vocabulary import BEHAVIOUR_IDS

SMALL = ModelShape(
    feature_dim=32, keypoints=17, keypoint_hidden=16, keypoint_out=8,
    tcn_channels=24, tcn_kernel=3, tcn_dilations=(1, 2, 4), tcn_dropout=0.0,
    attention_dim=8,
)


@pytest.fixture
def model() -> BehaviourModel:
    torch.manual_seed(0)
    return BehaviourModel(SMALL, CODEBOOK_V1)


def batch(n: int = 2, frames: int = 64, shape: ModelShape = SMALL):
    torch.manual_seed(1)
    return (torch.randn(n, frames, shape.feature_dim),
            torch.randn(n, frames, shape.keypoints, 3))


class TestShapes:
    def test_every_behaviour_and_field_gets_an_output(self, model) -> None:
        output = model(*batch())
        assert set(output.raw) == set(BEHAVIOUR_IDS)
        for behaviour in BEHAVIOUR_IDS:
            expected = {h.field for h in model.head_specs[behaviour]}
            assert set(output.raw[behaviour]) == expected

    def test_binary_and_scalar_heads_are_one_wide_and_categorical_are_not(self, model) -> None:
        output = model(*batch(n=3))
        for behaviour, specs in model.head_specs.items():
            for head in specs:
                tensor = output.raw[behaviour][head.field]
                if head.kind is HeadKind.CATEGORICAL:
                    assert tensor.shape == (3, head.width), head.field
                else:
                    assert tensor.shape == (3,), head.field

    @pytest.mark.parametrize("frames", [16, 32, 64, 65])
    def test_the_clip_length_survives_the_dilations(self, model, frames: int) -> None:
        """Symmetric padding, so the output length matches the input. An off-by-one here would
        misalign the attention weights against the frames they are supposed to describe."""
        _, attention = model(*batch(frames=frames)).raw, model(*batch(frames=frames)).attention
        assert attention.shape == (2, frames)


class TestAttention:
    def test_the_weights_are_a_distribution_over_time(self, model) -> None:
        attention = model(*batch(n=4, frames=32)).attention
        assert torch.allclose(attention.sum(dim=1), torch.ones(4), atol=1e-5)
        assert (attention >= 0).all()

    def test_the_weights_come_back_with_the_prediction_not_on_request(self, model) -> None:
        """Phase 7's intrinsic fallback needs the weights that produced *this* prediction. A
        second forward pass under dropout would give different ones, and the explanation would
        then not describe the output it is displayed beside."""
        output = model(*batch())
        assert output.attention.shape[0] == output.raw["B1"]["b1_present"].shape[0]


class TestTemporalReceptiveField:
    def test_the_model_looks_across_time_not_at_single_frames(self, model) -> None:
        """Three layers at dilations 1, 2 and 4 with kernel 3 reach 15 frames. If the stack
        ever collapsed to a per-frame classifier - a wrong padding, a dropped block - the
        clip-level output would stop depending on where in the clip a change happened, and
        nothing about the shapes would look different."""
        features, keypoints = batch(n=1, frames=32)
        baseline = model(features, keypoints).raw["B1"]["b1_present"]

        early, late = features.clone(), features.clone()
        early[0, 2] += 5.0
        late[0, 20] += 5.0

        moved_early = model(early, keypoints).raw["B1"]["b1_present"]
        moved_late = model(late, keypoints).raw["B1"]["b1_present"]

        assert not torch.allclose(moved_early, baseline, atol=1e-6)
        assert not torch.allclose(moved_early, moved_late, atol=1e-6), (
            "the same perturbation at two points in the clip produces the same output, so "
            "the temporal stack is not using time")

    def test_the_keypoint_branch_reaches_the_output(self, model) -> None:
        """Otherwise the pose tensor is expensive decoration and the concatenation is wrong."""
        features, keypoints = batch()
        changed = keypoints.clone()
        changed[0, :, 9] += 3.0  # left wrist

        assert not torch.allclose(model(features, keypoints).raw["B1"]["b1_present"],
                                  model(features, changed).raw["B1"]["b1_present"],
                                  atol=1e-6)


class TestPrediction:
    def test_decoded_predictions_are_labels_the_codebook_accepts(self, model) -> None:
        """The contract `Detection` enforces, checked here rather than at serialisation."""
        decoded = model.predict(*batch(n=3))

        for behaviour in BEHAVIOUR_IDS:
            assert len(decoded[behaviour]) == 3
            for clip in decoded[behaviour]:
                CODEBOOK_V1.validate_labels(behaviour, clip)
                missing = CODEBOOK_V1.behaviour(behaviour).field_names - set(clip)
                assert not missing, f"{behaviour} prediction omits {missing}"

    def test_prediction_does_not_leave_the_model_in_train_mode(self, model) -> None:
        """`predict` switches to eval for dropout and batch norm. A caller that trains after
        predicting would otherwise get whichever mode was last set, which is a bug that shows
        up as unstable training rather than as an error."""
        model.train()
        model.predict(*batch())
        assert not model.training


class TestDeterminism:
    def test_the_same_seed_builds_the_same_network(self) -> None:
        """R7. Weight initialisation is part of what one config and one seed must reproduce."""
        torch.manual_seed(42)
        first = BehaviourModel(SMALL, CODEBOOK_V1)
        torch.manual_seed(42)
        second = BehaviourModel(SMALL, CODEBOOK_V1)

        for (name, a), (_, b) in zip(first.named_parameters(),
                                     second.named_parameters(), strict=True):
            assert torch.equal(a, b), name

    def test_a_forward_pass_in_eval_mode_repeats(self, model) -> None:
        model.eval()
        features, keypoints = batch()
        with torch.no_grad():
            assert torch.equal(model(features, keypoints).raw["B3"]["b3_transitions"],
                               model(features, keypoints).raw["B3"]["b3_transitions"])


class TestRefusals:
    def test_a_clip_and_its_pose_must_be_the_same_length(self, model) -> None:
        """Otherwise every frame after the mismatch is scored against another frame's pose."""
        features, _ = batch(frames=64)
        _, keypoints = batch(frames=32)
        with pytest.raises(ValueError, match="same length"):
            model(features, keypoints)

    def test_features_of_the_wrong_width_are_refused(self, model) -> None:
        with pytest.raises(ValueError, match="expected features"):
            model(torch.randn(2, 64, SMALL.feature_dim + 1), torch.randn(2, 64, 17, 3))


def test_the_shape_comes_from_the_config(config) -> None:
    """The config is read in one place, so the network and the cached features cannot
    disagree about the backbone width."""
    shape = shape_from_config(config)

    assert shape.feature_dim == config.behaviour.backbone.output_dim
    assert shape.tcn_dilations == tuple(config.behaviour.temporal.dilations)
    assert shape.keypoints == config.behaviour.keypoint_branch.input_keypoints
    assert shape.behaviours == tuple(config.behaviour.heads.presence_behaviours)


def test_the_configured_model_builds_and_runs(config) -> None:
    """At the real widths, to catch a config that describes a network that cannot be built."""
    torch.manual_seed(0)
    model = build_model(config, CODEBOOK_V1)
    frames = config.behaviour.clip.frames

    output = model(torch.randn(1, frames, config.behaviour.backbone.output_dim),
                   torch.randn(1, frames, config.behaviour.keypoint_branch.input_keypoints, 3))

    assert output.attention.shape == (1, frames)
    assert set(output.raw) == set(config.behaviour.heads.presence_behaviours)
