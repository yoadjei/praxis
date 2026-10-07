# -*- coding: utf-8 -*-
"""Clip sampling and the per-behaviour augmentation mask.

The rule this file exists to hold is one line of BUILD-SPEC: a horizontal flip is allowed for
B1 and B4 and forbidden for B2 and B3, because it inverts orientation and position. The config
key is in the `locked` list. What makes it hard to get right is that the model is multi-task, so
a flipped clip is not simply excluded - it is a valid example for two behaviours and a corrupt
one for the other three, in the same forward pass.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from praxis.annotation import CODEBOOK_V1
from praxis.behaviour.dataset import (
    HFLIP,
    IDENTITY,
    AugmentationPolicy,
    ClipDataset,
    ClipRecord,
    DatasetError,
    balance_by_session,
    clips_per_session,
    collate,
    policy_from_config,
)
from praxis.behaviour.heads import behaviour_heads
from praxis.vocabulary import BEHAVIOUR_IDS

FRAMES = 16
FEATURE_DIM = 8

HEADS = {behaviour: behaviour_heads(CODEBOOK_V1, behaviour)
         for behaviour in BEHAVIOUR_IDS}


def labels_for(behaviour: str, *, nonscorable: bool = False) -> dict:
    values = {}
    for head in HEADS[behaviour]:
        values[head.field] = head.levels[0] if head.levels else head.minimum
    values[f"{behaviour.lower()}_nonscorable"] = nonscorable
    return values


def a_record(clip_id: str = "S:00000", *, variants=(IDENTITY,),
             nonscorable: tuple[str, ...] = (), session_id: str = "S") -> ClipRecord:
    torch.manual_seed(0)
    return ClipRecord(
        clip_id=clip_id,
        session_id=session_id,
        teacher_id="T1",
        features={name: torch.randn(FRAMES, FEATURE_DIM) for name in variants},
        keypoints=torch.randn(FRAMES, 17, 3),
        labels={b: labels_for(b, nonscorable=b in nonscorable) for b in BEHAVIOUR_IDS},
        variants=tuple(variants),
    )


def a_policy(**overrides) -> AugmentationPolicy:
    parameters = dict(flip_enabled=True, flip_applies_to=("B1", "B4"),
                      flip_probability=0.5, temporal_jitter_frames=4)
    parameters.update(overrides)
    return AugmentationPolicy(**parameters)


class TestFlipMask:
    """The locked rule, enforced rather than documented."""

    def test_the_policy_refuses_to_flip_b2_or_b3(self) -> None:
        """A flip inverts orientation and position. Accepting this would corrupt the labels of
        the two behaviours the thesis measures orientation and mobility with."""
        for forbidden in ("B2", "B3"):
            with pytest.raises(DatasetError, match="orientation and position"):
                a_policy(flip_applies_to=("B1", forbidden))

    def test_a_typo_in_applies_to_is_refused_not_ignored(self) -> None:
        """Silently disabling the flip would be the failure: the config key is locked, and a
        locked key that quietly stops applying is worse than one that errors."""
        with pytest.raises(DatasetError, match=r"not \s*behaviours|are not"):
            a_policy(flip_applies_to=("B1", "B9"))

    def test_the_flipped_variant_supplies_only_the_permitted_behaviours(self) -> None:
        variants = {v.name: v for v in a_policy().variants}
        flipped = variants[HFLIP]

        assert flipped.supplies("B1")
        assert flipped.supplies("B4")
        for behaviour in ("B2", "B3", "B5"):
            assert not flipped.supplies(behaviour)

    def test_the_identity_variant_supplies_everything(self) -> None:
        identity = next(v for v in a_policy().variants if v.name == IDENTITY)
        assert all(identity.supplies(b) for b in BEHAVIOUR_IDS)

    def test_a_flipped_sample_masks_out_the_behaviours_it_would_corrupt(self) -> None:
        """The claim that matters. One forward pass answers all five behaviours, so a flipped
        clip must contribute loss for B1 and B4 and nothing at all for B2, B3 and B5."""
        dataset = ClipDataset([a_record(variants=(IDENTITY, HFLIP))], HEADS,
                              frames=FRAMES, policy=a_policy(flip_probability=1.0),
                              seed=0, train=True)
        sample = dataset[0]

        assert sample.variant == HFLIP
        assert sample.scorable["B1"] is True
        assert sample.scorable["B4"] is True
        assert sample.scorable["B2"] is False
        assert sample.scorable["B3"] is False
        assert sample.scorable["B5"] is False

    def test_disabling_the_flip_removes_the_variant_entirely(self) -> None:
        names = {v.name for v in a_policy(flip_enabled=False).variants}
        assert names == {IDENTITY}


class TestNonScorableMask:
    def test_a_nonscorable_behaviour_is_masked_and_the_others_are_not(self) -> None:
        """Non-scorability is recorded per behaviour, which is the whole reason a clip
        unscorable for gesture stays in the batch for spatial position."""
        dataset = ClipDataset([a_record(nonscorable=("B1",))], HEADS,
                              frames=FRAMES, policy=a_policy(flip_enabled=False),
                              seed=0, train=True)
        sample = dataset[0]

        assert sample.scorable["B1"] is False
        assert all(sample.scorable[b] for b in ("B2", "B3", "B4", "B5"))

    def test_both_reasons_to_mask_combine(self) -> None:
        """Unscorable for B4 and flipped: B4 is masked for one reason and B2 for the other."""
        dataset = ClipDataset([a_record(variants=(IDENTITY, HFLIP), nonscorable=("B4",))],
                              HEADS, frames=FRAMES,
                              policy=a_policy(flip_probability=1.0), seed=0, train=True)
        sample = dataset[0]

        assert sample.scorable["B1"] is True
        assert sample.scorable["B4"] is False
        assert sample.scorable["B2"] is False


class TestEvaluationMode:
    def test_evaluation_never_augments(self) -> None:
        """An augmented evaluation set measures the model on clips the corpus does not
        contain, so the accuracy and ECE reported at S0 would describe neither."""
        dataset = ClipDataset([a_record(variants=(IDENTITY, HFLIP))], HEADS,
                              frames=FRAMES, policy=a_policy(flip_probability=1.0),
                              seed=0, train=False)
        for _ in range(10):
            assert dataset[0].variant == IDENTITY

    def test_evaluation_does_not_jitter_the_window(self) -> None:
        record = a_record()
        dataset = ClipDataset([record], HEADS, frames=FRAMES,
                              policy=a_policy(), seed=0, train=False)
        assert torch.equal(dataset[0].features, record.features[IDENTITY])


class TestTemporalJitter:
    def test_the_window_stays_the_configured_length(self) -> None:
        dataset = ClipDataset([a_record()], HEADS, frames=FRAMES,
                              policy=a_policy(temporal_jitter_frames=4), seed=3, train=True)
        for _ in range(20):
            sample = dataset[0]
            assert sample.features.shape == (FRAMES, FEATURE_DIM)
            assert sample.keypoints.shape == (FRAMES, 17, 3)

    def test_features_and_keypoints_are_jittered_together(self) -> None:
        """A clip scored against another frame's pose is silent corruption: the shapes stay
        right and every number afterwards is wrong."""
        record = a_record()
        dataset = ClipDataset([record], HEADS, frames=FRAMES,
                              policy=a_policy(temporal_jitter_frames=3), seed=7, train=True)
        sample = dataset[0]

        rows = record.features[IDENTITY]
        for position in range(FRAMES):
            matches = (rows == sample.features[position]).all(dim=1).nonzero().flatten()
            assert len(matches) >= 1
            source = int(matches[0])
            assert torch.equal(sample.keypoints[position], record.keypoints[source]), (
                "frame and pose came from different positions in the clip")

    def test_the_window_clamps_rather_than_wraps(self) -> None:
        """Wrapping splices the end of a clip onto its beginning and presents the result as
        eight continuous seconds. For B3 that manufactures a zone crossing that never
        happened."""
        record = a_record()
        dataset = ClipDataset([record], HEADS, frames=FRAMES,
                              policy=a_policy(temporal_jitter_frames=6), seed=11, train=True)

        rows = record.features[IDENTITY]
        for _ in range(30):
            sample = dataset[0]
            indices = [int((rows == sample.features[p]).all(dim=1).nonzero().flatten()[0])
                       for p in range(FRAMES)]
            assert indices == sorted(indices), "a wrapped window is not monotonic in time"


class TestCollate:
    def test_a_batch_carries_targets_and_masks_for_every_behaviour(self) -> None:
        records = [a_record(f"S:{i:05d}", variants=(IDENTITY,)) for i in range(4)]
        dataset = ClipDataset(records, HEADS, frames=FRAMES,
                              policy=a_policy(flip_enabled=False), seed=0, train=True)
        batch = collate([dataset[i] for i in range(4)], HEADS)

        assert batch["features"].shape == (4, FRAMES, FEATURE_DIM)
        assert batch["keypoints"].shape == (4, FRAMES, 17, 3)
        for behaviour in BEHAVIOUR_IDS:
            assert batch["scorable"][behaviour].shape == (4,)
            for head in HEADS[behaviour]:
                assert batch["targets"][behaviour][head.field].shape == (4,)

    def test_categorical_targets_are_indices_and_the_rest_are_floats(self) -> None:
        """Cross-entropy wants class indices; the other two losses want values. Getting this
        wrong raises deep inside torch with a message about neither."""
        from praxis.behaviour.heads import HeadKind

        dataset = ClipDataset([a_record()], HEADS, frames=FRAMES,
                              policy=a_policy(flip_enabled=False), seed=0, train=True)
        batch = collate([dataset[0]], HEADS)

        for behaviour in BEHAVIOUR_IDS:
            for head in HEADS[behaviour]:
                tensor = batch["targets"][behaviour][head.field]
                expected = torch.long if head.kind is HeadKind.CATEGORICAL else torch.float32
                assert tensor.dtype == expected, head.field


class TestRefusals:
    def test_an_empty_dataset_is_refused(self) -> None:
        with pytest.raises(DatasetError, match="no clips"):
            ClipDataset([], HEADS, frames=FRAMES, policy=a_policy(), seed=0)

    def test_a_missing_cache_variant_names_the_cause(self) -> None:
        """Stage A writes one tensor per variant. A missing one means the cache was built under
        a different augmentation policy, which is a stale-cache bug, not a data bug."""
        dataset = ClipDataset([a_record(variants=(IDENTITY,))], HEADS, frames=FRAMES,
                              policy=a_policy(flip_probability=1.0), seed=0, train=True)
        record = dataset.records[0]
        object.__setattr__(record, "variants", (IDENTITY, HFLIP))

        with pytest.raises(DatasetError, match="no cached features for variant"):
            dataset[0]


def test_the_policy_comes_from_the_config(config) -> None:
    policy = policy_from_config(config)

    assert policy.flip_applies_to == tuple(
        config.behaviour.augmentation.horizontal_flip.applies_to)
    assert set(policy.flip_applies_to) & {"B2", "B3"} == set(), (
        "configs/default.yaml must not permit flipping the orientation behaviours; the key "
        "is in the locked list")
    assert policy.temporal_jitter_frames == config.behaviour.augmentation.temporal_jitter_frames


def test_an_unimplementable_augmentation_is_a_hard_error_not_a_no_op(config) -> None:
    """D75. A colour jitter acts on pixels and the cache holds activations. Honouring the flag
    quietly would mean training with no colour augmentation while the config says otherwise,
    which is configuration that lies - the failure mode the invariants exist to prevent."""
    assert config.behaviour.augmentation.colour_jitter.applied is False

    # A stand-in rather than a mutated config: PraxisConfig is frozen, because a run may not
    # edit its own configuration. What is under test is the guard, not pydantic.
    augmentation = config.behaviour.augmentation
    asked_for = SimpleNamespace(behaviour=SimpleNamespace(
        augmentation=SimpleNamespace(
            colour_jitter=SimpleNamespace(applied=True),
            horizontal_flip=augmentation.horizontal_flip,
            temporal_jitter_frames=augmentation.temporal_jitter_frames),
        heads=config.behaviour.heads))

    with pytest.raises(DatasetError, match="not built"):
        policy_from_config(asked_for)


class TestSessionBalance:
    """D84. Session length in the corpus runs 59 seconds to 32 minutes, so two sessions supply
    most of the clips. Every clip from one session shares a teacher, a room, a camera and a
    light level, and a model fitted on that learns the room."""

    def corpus(self, shape: dict[str, int]) -> list[ClipRecord]:
        return [a_record(clip_id=f"{s}:{i:04d}", session_id=s)
                for s, n in shape.items() for i in range(n)]

    # the shape measured on the real corpus
    REAL: ClassVar[dict[str, int]] = {
        "s01": 243, "s02": 182, "s03": 43, "s04": 26, "s05": 18, "s06": 15,
        "s07": 15, "s08": 14, "s09": 13, "s10": 8, "s11": 7}

    def test_the_dominant_session_stops_dominating(self) -> None:
        records = self.corpus(self.REAL)
        before = clips_per_session(records)
        assert max(before.values()) / sum(before.values()) > 0.40

        after = clips_per_session(balance_by_session(records, "median", seed=7))
        assert max(after.values()) / sum(after.values()) < 0.15

    def test_the_median_leaves_the_smaller_half_untouched(self) -> None:
        """Which is what makes it the right cap rather than an arbitrary one."""
        records = self.corpus(self.REAL)
        after = clips_per_session(balance_by_session(records, "median", seed=7))
        for session, count in self.REAL.items():
            if count <= 15:
                assert after[session] == count, f"{session} was trimmed and should not be"

    def test_an_integer_cap_is_honoured_exactly(self) -> None:
        after = clips_per_session(balance_by_session(self.corpus(self.REAL), 40, seed=7))
        assert max(after.values()) == 40

    def test_no_cap_keeps_every_clip(self) -> None:
        records = self.corpus(self.REAL)
        assert len(balance_by_session(records, None, seed=7)) == len(records)

    def test_selection_is_deterministic_for_a_seed(self) -> None:
        """R7. The same corpus and seed must train on the same clips."""
        records = self.corpus(self.REAL)
        first = [r.clip_id for r in balance_by_session(records, "median", seed=7)]
        assert first == [r.clip_id for r in balance_by_session(records, "median", seed=7)]
        assert first != [r.clip_id for r in balance_by_session(records, "median", seed=8)]

    def test_adding_a_session_does_not_reshuffle_the_others(self) -> None:
        """Seeded per session rather than once over the corpus. Otherwise ingesting one more
        session silently changes which clips every earlier session contributes, and two runs of
        the same experiment stop being comparable."""
        shape = dict(self.REAL)
        kept = {r.clip_id for r in balance_by_session(self.corpus(shape), 20, seed=7)
                if r.session_id == "s01"}

        shape["s12"] = 30
        after = {r.clip_id for r in balance_by_session(self.corpus(shape), 20, seed=7)
                 if r.session_id == "s01"}

        assert kept == after

    def test_a_cap_of_zero_is_refused_rather_than_emptying_the_corpus(self) -> None:
        with pytest.raises(DatasetError, match="contributing no clips"):
            balance_by_session(self.corpus({"a": 4}), 0, seed=7)

    def test_an_evaluation_dataset_is_never_capped(self) -> None:
        """The decisive one. Capping a held-out set would report accuracy and ECE on a
        subsample of those sessions while naming them in full."""
        records = self.corpus({"big": 60, "small": 4})

        evaluation = ClipDataset(records, HEADS, frames=FRAMES, policy=a_policy(),
                                 seed=0, train=False, max_clips_per_session="median")
        training = ClipDataset(records, HEADS, frames=FRAMES, policy=a_policy(),
                               seed=0, train=True, max_clips_per_session="median")

        assert len(evaluation) == len(records)
        assert len(training) < len(records)

    def test_an_uncapped_training_set_is_the_default(self) -> None:
        """The cap arrives from config, so a caller that does not ask for one gets every clip
        and the decision stays visible at the call site."""
        records = self.corpus({"big": 60, "small": 4})
        assert len(ClipDataset(records, HEADS, frames=FRAMES, policy=a_policy(),
                               seed=0, train=True)) == len(records)
