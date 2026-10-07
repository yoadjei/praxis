# -*- coding: utf-8 -*-
"""The three required baselines.

BUILD-SPEC calls none of them optional, so the test that matters most is the dull one: that
`behaviour.baselines` in the config names exactly the three that exist, and that an unknown
name raises rather than being skipped. A comparison table quietly missing a row is how
"compared against all three baselines" becomes true of a document and false of the work.

The rest check the properties a baseline has to have to be comparable at all - probabilities
rather than hard labels, fitted on training data only, one output per classification head at
the codebook's full width.
"""
from __future__ import annotations

import numpy as np
import pytest

from praxis.annotation import CODEBOOK_V1
from praxis.behaviour.baselines import (
    REQUIRED_BASELINES,
    BaselineError,
    FrameAveragedResnet,
    KeypointsGBDT,
    LabelledClips,
    MajorityClass,
    build_baselines,
    keypoint_features,
)
from praxis.behaviour.heads import HeadKind, behaviour_heads
from praxis.vocabulary import BEHAVIOUR_IDS

HEADS = {b: behaviour_heads(CODEBOOK_V1, b) for b in BEHAVIOUR_IDS}
FRAMES, DIM, JOINTS = 6, 5, 17


def classification_fields():
    for behaviour, specs in HEADS.items():
        for head in specs:
            if head.kind is not HeadKind.SCALAR and not head.is_nonscorable_flag:
                yield behaviour, head


def clips(n: int = 20, *, seed: int = 0, unscorable: tuple[str, ...] = (),
          single_class: bool = False) -> LabelledClips:
    rng = np.random.default_rng(seed)
    positive = np.arange(n) % 2 == 0
    if single_class:
        positive = np.ones(n, dtype=bool)

    features = np.where(positive[:, None, None], 1.0, -1.0) + 0.05 * rng.normal(
        size=(n, FRAMES, DIM))
    keypoints = np.concatenate([
        np.where(positive[:, None, None, None], 0.8, 0.2)
        + 0.05 * rng.normal(size=(n, FRAMES, JOINTS, 2)),
        rng.uniform(0.5, 1.0, size=(n, FRAMES, JOINTS, 1)),
    ], axis=-1)

    targets = {}
    for _, head in classification_fields():
        if head.kind is HeadKind.BINARY:
            targets[head.field] = positive.astype(int)
        else:
            targets[head.field] = np.where(positive, 0, min(1, head.width - 1))

    scorable = {b: np.full(n, b not in unscorable) for b in BEHAVIOUR_IDS}
    return LabelledClips(features=features, keypoints=keypoints,
                         targets=targets, scorable=scorable)


ALL = [MajorityClass, KeypointsGBDT, FrameAveragedResnet]


class TestRegistry:
    def test_the_config_names_exactly_the_baselines_that_exist(self, config) -> None:
        """BUILD-SPEC requires all three in the frozen report."""
        assert tuple(config.behaviour.baselines) == REQUIRED_BASELINES

    def test_an_unknown_baseline_raises_rather_than_being_skipped(self) -> None:
        with pytest.raises(BaselineError, match="no baseline called"):
            build_baselines(["majority_class", "a_better_one"])

    def test_building_the_required_set_gives_three_distinct_baselines(self) -> None:
        built = build_baselines()
        assert [b.name for b in built] == list(REQUIRED_BASELINES)


@pytest.mark.parametrize("factory", ALL)
class TestEveryBaseline:
    """Properties all three must have to be comparable with the deep model at all."""

    def test_it_answers_every_classification_head(self, factory) -> None:
        baseline = factory()
        baseline.fit(clips(), HEADS)
        out = baseline.predict_proba(clips(8, seed=1))

        expected = {head.field for _, head in classification_fields()}
        assert set(out) == expected

    def test_scalar_heads_and_the_abstention_flag_are_not_answered(self, factory) -> None:
        """Asking a majority-class predictor for a gesture count would be inventing a
        baseline nobody specified, and the flag measures abstention rather than recognition."""
        baseline = factory()
        baseline.fit(clips(), HEADS)
        out = baseline.predict_proba(clips(4, seed=2))

        assert "b1_count" not in out
        assert not any(field.endswith("_nonscorable") for field in out)

    def test_outputs_are_probabilities_at_the_codebook_width(self, factory) -> None:
        """R4 is enforced by the evaluation return type: a baseline emitting hard labels could
        not have an ECE computed and would drop out of the comparison silently."""
        baseline = factory()
        baseline.fit(clips(), HEADS)
        n = 7
        out = baseline.predict_proba(clips(n, seed=3))

        for _, head in classification_fields():
            values = out[head.field]
            if head.kind is HeadKind.BINARY:
                assert values.shape == (n,), head.field
                assert ((values >= 0.0) & (values <= 1.0)).all(), head.field
            else:
                assert values.shape == (n, head.width), head.field
                assert np.allclose(values.sum(axis=1), 1.0, atol=1e-6), head.field

    def test_predicting_before_fitting_is_refused(self, factory) -> None:
        with pytest.raises(BaselineError, match="was not fitted"):
            factory().predict_proba(clips(4))

    def test_a_head_with_one_training_class_falls_back_to_a_constant(self, factory) -> None:
        """A rare behaviour the training partition never shows is a real state, not an error.
        The constant the data supports is recorded and shows up in the reported ECE."""
        baseline = factory()
        baseline.fit(clips(single_class=True), HEADS)
        out = baseline.predict_proba(clips(5, seed=4))

        for _, head in classification_fields():
            assert np.isfinite(out[head.field]).all(), head.field

    def test_an_entirely_unscorable_behaviour_does_not_crash_the_fit(self, factory) -> None:
        baseline = factory()
        baseline.fit(clips(unscorable=("B3",)), HEADS)
        out = baseline.predict_proba(clips(4, seed=5))
        assert np.isfinite(out["b3_zone_changed"]).all()

    def test_the_same_seed_gives_the_same_predictions(self, factory) -> None:
        """R7 does not stop at the thing being measured. A baseline that moved between runs
        would make the comparison move with it."""
        first, second = factory(), factory()
        first.fit(clips(seed=9), HEADS)
        second.fit(clips(seed=9), HEADS)

        a = first.predict_proba(clips(6, seed=10))
        b = second.predict_proba(clips(6, seed=10))
        for field in a:
            assert np.array_equal(a[field], b[field]), field


class TestMajorityClass:
    def test_it_predicts_the_training_base_rate(self) -> None:
        baseline = MajorityClass()
        baseline.fit(clips(20), HEADS)
        out = baseline.predict_proba(clips(4, seed=6))

        # The fixture alternates, so the base rate is one half.
        assert out["b1_present"] == pytest.approx(np.full(4, 0.5))

    def test_it_ignores_the_input_entirely(self) -> None:
        """It is the floor. If it responded to the features it would not be one."""
        baseline = MajorityClass()
        baseline.fit(clips(20), HEADS)

        assert np.array_equal(baseline.predict_proba(clips(4, seed=7))["b1_present"],
                              baseline.predict_proba(clips(4, seed=8))["b1_present"])


class TestKeypointFeatures:
    def test_the_vector_is_built_from_position_spread_confidence_and_path(self) -> None:
        design = keypoint_features(np.zeros((3, FRAMES, JOINTS, 3)))
        assert design.shape == (3, JOINTS * 2 + JOINTS * 2 + JOINTS + JOINTS)

    def test_a_single_frame_clip_has_zero_path_length_not_a_missing_value(self) -> None:
        design = keypoint_features(np.ones((2, 1, JOINTS, 3)))
        assert np.isfinite(design).all()
        assert design[:, -JOINTS:].sum() == 0.0

    def test_movement_shows_up_in_the_path_length(self) -> None:
        """Path length is the motion signal B3 mobility and B1 gesture both depend on."""
        still = np.ones((1, FRAMES, JOINTS, 3))
        moving = still.copy()
        moving[0, :, :, 0] = np.arange(FRAMES)[:, None]

        assert (keypoint_features(moving)[:, -JOINTS:].sum()
                > keypoint_features(still)[:, -JOINTS:].sum())

    def test_a_wrong_shape_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="must be"):
            keypoint_features(np.zeros((2, FRAMES, JOINTS)))


class TestLearnedBaselinesActuallyLearn:
    """Otherwise the comparison is three constants wearing different names."""

    @pytest.mark.parametrize("factory", [KeypointsGBDT, FrameAveragedResnet])
    def test_a_separable_signal_is_learned(self, factory) -> None:
        baseline = factory()
        baseline.fit(clips(40, seed=12), HEADS)
        held_out = clips(20, seed=13)
        predicted = baseline.predict_proba(held_out)["b1_present"] >= 0.5

        accuracy = float((predicted == held_out.targets["b1_present"].astype(bool)).mean())
        assert accuracy > 0.8, (
            "the fixture is linearly separable by construction; a baseline that cannot fit it "
            "is not fitting anything, and the comparison in the frozen report would be "
            "three constants under three names")

    def test_the_two_learned_baselines_use_different_inputs(self) -> None:
        """One reads pose, the other reads backbone features. If they agreed exactly, one of
        them would be reading the wrong array."""
        pose, frames = KeypointsGBDT(), FrameAveragedResnet()
        pose.fit(clips(40, seed=14), HEADS)
        frames.fit(clips(40, seed=14), HEADS)

        held_out = clips(20, seed=15)
        assert not np.array_equal(pose.predict_proba(held_out)["b1_present"],
                                  frames.predict_proba(held_out)["b1_present"])


def test_mismatched_features_and_keypoints_are_refused() -> None:
    with pytest.raises(BaselineError, match="different numbers of clips"):
        LabelledClips(features=np.zeros((4, FRAMES, DIM)),
                      keypoints=np.zeros((3, FRAMES, JOINTS, 3)),
                      targets={}, scorable={})
