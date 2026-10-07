# -*- coding: utf-8 -*-
"""Stage B training.

The acceptance test BUILD-SPEC names is here: training twice with the same config and seed
produces identical validation metrics. Alongside it are the two properties that make a
training run trustworthy rather than merely finished - that early stopping returns the weights
it reported a score for, and that a teacher cannot appear in both partitions.

None of this claims the model learns anything useful. The fixtures are synthetic and validate
the training contract, not real-world performance.
"""
from __future__ import annotations

import math

import pytest
import torch

from praxis.annotation import CODEBOOK_V1
from praxis.behaviour.dataset import IDENTITY, AugmentationPolicy, ClipDataset, ClipRecord
from praxis.behaviour.heads import behaviour_heads
from praxis.behaviour.losses import LossWeights
from praxis.behaviour.model import BehaviourModel, ModelShape
from praxis.behaviour.train import (
    TrainingError,
    TrainingPlan,
    cosine_with_warmup,
    plan_from_config,
    train,
    validation_scores,
    weights_from_config,
)
from praxis.vocabulary import BEHAVIOUR_IDS

FRAMES = 8
FEATURE_DIM = 6
HEADS = {b: behaviour_heads(CODEBOOK_V1, b) for b in BEHAVIOUR_IDS}
WEIGHTS = LossWeights(focal_alpha=0.25, focal_gamma=2.0, intensity_weight=0.30)

SHAPE = ModelShape(
    feature_dim=FEATURE_DIM, keypoints=17, keypoint_hidden=8, keypoint_out=4,
    tcn_channels=8, tcn_kernel=3, tcn_dilations=(1, 2), tcn_dropout=0.0, attention_dim=4)

POLICY = AugmentationPolicy(flip_enabled=False, flip_applies_to=(),
                            flip_probability=0.0, temporal_jitter_frames=0)


def labels_for(behaviour: str, positive: bool) -> dict:
    values = {}
    for head in HEADS[behaviour]:
        if head.levels is not None:
            values[head.field] = head.levels[-1 if positive else 0]
        else:
            values[head.field] = head.maximum if positive else head.minimum
    values[f"{behaviour.lower()}_nonscorable"] = False
    return values


def dataset(n: int, teacher: str, *, seed: int = 0) -> ClipDataset:
    torch.manual_seed(seed)
    records = []
    for index in range(n):
        positive = index % 2 == 0
        # A signal the model can actually fit, so early stopping has something to stop on.
        base = torch.full((FRAMES, FEATURE_DIM), 1.0 if positive else -1.0)
        records.append(ClipRecord(
            clip_id=f"{teacher}:{index:05d}",
            session_id=teacher,
            teacher_id=teacher,
            features={IDENTITY: base + 0.01 * torch.randn(FRAMES, FEATURE_DIM)},
            keypoints=torch.zeros(FRAMES, 17, 3),
            labels={b: labels_for(b, positive) for b in BEHAVIOUR_IDS},
            variants=(IDENTITY,)))
    return ClipDataset(records, HEADS, frames=FRAMES, policy=POLICY, seed=seed, train=True)


def a_plan(**overrides) -> TrainingPlan:
    parameters = dict(epochs=3, batch_size=4, learning_rate=1e-3, weight_decay=1e-4,
                      warmup_epochs=1, grad_clip_norm=1.0, patience=10, min_delta=0.0)
    parameters.update(overrides)
    return TrainingPlan(**parameters)


def run(plan=None, *, seed: int = 5):
    torch.manual_seed(seed)
    model = BehaviourModel(SHAPE, CODEBOOK_V1)
    return train(model, dataset(8, "T1"), dataset(4, "T2"), HEADS,
                 plan=plan or a_plan(), weights=WEIGHTS, seed=seed), model


class TestSchedule:
    def test_warmup_rises_linearly_to_one(self) -> None:
        assert cosine_with_warmup(0, 10, 2) == pytest.approx(0.5)
        assert cosine_with_warmup(1, 10, 2) == pytest.approx(1.0)

    def test_cosine_decays_to_zero_at_the_last_epoch(self) -> None:
        assert cosine_with_warmup(10, 10, 2) == pytest.approx(0.0, abs=1e-9)

    def test_the_multiplier_never_leaves_the_unit_interval(self) -> None:
        """A negative multiplier would flip the gradient step and train the model backwards."""
        for epoch in range(0, 14):
            assert 0.0 <= cosine_with_warmup(epoch, 10, 3) <= 1.0

    def test_no_warmup_is_a_plain_cosine(self) -> None:
        assert cosine_with_warmup(0, 10, 0) == pytest.approx(1.0)

    def test_a_schedule_with_no_epochs_is_refused(self) -> None:
        with pytest.raises(TrainingError, match="at least one epoch"):
            cosine_with_warmup(0, 0, 0)


class TestDeterminism:
    """BUILD-SPEC's acceptance test for this phase."""

    def test_two_runs_with_the_same_seed_give_identical_validation_metrics(self) -> None:
        first, _ = run(seed=11)
        second, _ = run(seed=11)

        assert [e.val_macro_f1 for e in first.epochs] == [e.val_macro_f1 for e in second.epochs]
        assert [e.val_loss for e in first.epochs] == [e.val_loss for e in second.epochs]
        assert [e.train_loss for e in first.epochs] == [e.train_loss for e in second.epochs]

    def test_a_different_seed_gives_different_metrics(self) -> None:
        """Otherwise the previous test would pass on a training loop that ignored the data."""
        first, _ = run(seed=11)
        other, _ = run(seed=12)
        assert ([e.train_loss for e in first.epochs]
                != [e.train_loss for e in other.epochs])

    def test_the_trained_weights_are_identical_too(self) -> None:
        _, first = run(seed=3)
        _, second = run(seed=3)
        for (name, a), (_, b) in zip(first.state_dict().items(),
                                     second.state_dict().items(), strict=True):
            assert torch.equal(a, b), name


class TestEarlyStopping:
    def test_it_stops_when_the_monitor_stalls(self) -> None:
        result, _ = run(a_plan(epochs=40, patience=2, min_delta=10.0))
        assert result.stopped_early
        assert len(result.epochs) < 40

    def test_it_returns_the_weights_it_reported_a_score_for(self) -> None:
        """Stopping without restoring the best checkpoint reports a figure measured at the best
        epoch while returning the model from `patience` epochs later."""
        result, model = run(a_plan(epochs=30, patience=2, min_delta=10.0))

        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, result.best_state[name]), name

    def test_the_best_epoch_carries_the_best_score(self) -> None:
        result, _ = run(a_plan(epochs=6, patience=10))
        best = max(result.epochs, key=lambda e: e.val_macro_f1)
        assert result.best_score == pytest.approx(best.val_macro_f1)

    def test_a_full_run_is_not_marked_as_stopped_early(self) -> None:
        result, _ = run(a_plan(epochs=3, patience=10, min_delta=0.0))
        assert len(result.epochs) == 3
        assert not result.stopped_early


class TestLeakageGuard:
    def test_a_teacher_in_both_partitions_is_refused(self) -> None:
        """R2. A validation score measured on someone the model trained on is not a
        validation score, and nothing downstream would notice."""
        model = BehaviourModel(SHAPE, CODEBOOK_V1)
        with pytest.raises(TrainingError, match="R2 forbids"):
            train(model, dataset(8, "T1"), dataset(4, "T1"), HEADS,
                  plan=a_plan(), weights=WEIGHTS, seed=1)


class TestValidationScores:
    def test_masked_clips_are_dropped_before_scoring(self) -> None:
        """Scoring a model against a label the rater said could not be read is not a
        measurement. At S2 that would be a large and invisible share of the set."""
        torch.manual_seed(0)
        model = BehaviourModel(SHAPE, CODEBOOK_V1)
        batches = []
        for keep in (True, False):
            data = dataset(4, "T2")
            batch = _one_batch(data)
            batch["scorable"] = {b: torch.full((4,), keep) for b in BEHAVIOUR_IDS}
            batches.append(batch)

        _, macro, per_behaviour, _ = validation_scores(model, batches, HEADS, WEIGHTS)
        assert 0.0 <= macro <= 1.0
        assert set(per_behaviour) <= set(BEHAVIOUR_IDS)

    def test_a_wholly_masked_validation_set_scores_zero_rather_than_crashing(self) -> None:
        torch.manual_seed(0)
        model = BehaviourModel(SHAPE, CODEBOOK_V1)
        batch = _one_batch(dataset(4, "T2"))
        batch["scorable"] = {b: torch.zeros(4, dtype=torch.bool) for b in BEHAVIOUR_IDS}

        loss, macro, per_behaviour, _mae = validation_scores(model, [batch], HEADS, WEIGHTS)
        assert macro == 0.0
        assert per_behaviour == {}
        assert math.isfinite(loss)

    def test_the_nonscorable_flag_is_excluded_from_the_monitor(self) -> None:
        """Early stopping tracks behaviour recognition. A model that learned to abstain well
        and recognise nothing would otherwise look like it was still improving."""
        torch.manual_seed(0)
        model = BehaviourModel(SHAPE, CODEBOOK_V1)
        _, _, _, mae = validation_scores(model, [_one_batch(dataset(4, "T2"))],
                                         HEADS, WEIGHTS)
        assert not any(field.endswith("_nonscorable") for field in mae)

    def test_scalar_heads_are_reported_as_mae_not_folded_into_f1(self) -> None:
        torch.manual_seed(0)
        model = BehaviourModel(SHAPE, CODEBOOK_V1)
        _, _, _, mae = validation_scores(model, [_one_batch(dataset(4, "T2"))],
                                         HEADS, WEIGHTS)
        assert "b1_count" in mae and "b2_facing_proportion" in mae
        assert all(value >= 0.0 for value in mae.values())


def _one_batch(data: ClipDataset) -> dict:
    from praxis.behaviour.dataset import collate
    return collate([data[i] for i in range(len(data))], HEADS)


def test_mixed_precision_is_refused_until_it_is_measured(config) -> None:
    """BUILD-SPEC requires it measured before it is kept, and there is no CPU autocast path
    that would make it faster. Silently ignoring the flag would be configuration that lies."""
    model = BehaviourModel(SHAPE, CODEBOOK_V1)
    with pytest.raises(TrainingError, match="measured"):
        train(model, dataset(4, "T1"), dataset(4, "T2"), HEADS,
              plan=a_plan(mixed_precision=True), weights=WEIGHTS, seed=1)


def test_the_plan_and_the_weights_come_from_the_config(config) -> None:
    plan = plan_from_config(config)
    weights = weights_from_config(config)

    assert plan.epochs == config.behaviour.train.epochs
    assert plan.patience == config.behaviour.train.early_stopping.patience
    assert plan.monitor == config.behaviour.train.early_stopping.monitor == "val_macro_f1"
    assert weights.focal_gamma == config.behaviour.loss.focal_gamma
    assert weights.intensity_weight == config.behaviour.loss.intensity_weight
