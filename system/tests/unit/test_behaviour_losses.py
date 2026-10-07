# -*- coding: utf-8 -*-
"""The multi-task loss.

Three properties carry real risk and are tested directly rather than through the total: that
masking removes a clip's contribution entirely, that an all-masked batch yields zero with a
usable gradient rather than NaN, and that the non-scorable head is exempt from its own mask.
The third is the subtle one - getting it wrong leaves the model unable to abstain, and nothing
in a training curve would show it.
"""
from __future__ import annotations

import pytest
import torch

from praxis.annotation import CODEBOOK_V1
from praxis.behaviour.heads import behaviour_heads, encode_labels
from praxis.behaviour.losses import (
    LossWeights,
    behaviour_loss,
    categorical_loss,
    focal_loss,
    masked_mse,
    multitask_loss,
)
from praxis.vocabulary import BEHAVIOUR_IDS

WEIGHTS = LossWeights(focal_alpha=0.25, focal_gamma=2.0, intensity_weight=0.30)


def b1_batch(n: int = 4, *, scorable: list[bool] | None = None):
    heads = behaviour_heads(CODEBOOK_V1, "B1")
    outputs = {
        "b1_present": torch.zeros(n, requires_grad=True),
        "b1_count": torch.full((n,), 0.5, requires_grad=True),
        "b1_amplitude": torch.full((n,), 0.5, requires_grad=True),
        "b1_nonscorable": torch.zeros(n, requires_grad=True),
    }
    targets = {
        "b1_present": torch.ones(n),
        "b1_count": torch.full((n,), 0.25),
        "b1_amplitude": torch.zeros(n),
        "b1_nonscorable": torch.zeros(n),
    }
    mask = torch.tensor(scorable if scorable is not None else [True] * n)
    return heads, outputs, targets, mask


class TestFocalLoss:
    def test_a_confident_correct_prediction_costs_less_than_an_unsure_one(self) -> None:
        """The modulating factor is the whole point of focal loss."""
        mask = torch.ones(1)
        target = torch.ones(1)
        confident = focal_loss(torch.tensor([6.0]), target, mask, alpha=0.25, gamma=2.0)
        unsure = focal_loss(torch.tensor([0.0]), target, mask, alpha=0.25, gamma=2.0)
        assert confident < unsure

    def test_gamma_zero_reduces_to_weighted_cross_entropy(self) -> None:
        """A property with a known closed form, so the implementation is checked against
        arithmetic rather than against itself."""
        logits, target, mask = torch.tensor([0.7]), torch.ones(1), torch.ones(1)
        alpha = 0.25
        expected = alpha * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target)
        assert focal_loss(logits, target, mask, alpha=alpha, gamma=0.0) == pytest.approx(
            float(expected), abs=1e-6)

    def test_it_is_finite_at_a_saturated_logit(self) -> None:
        """Computed from logits, not probabilities. log(sigmoid(-40)) as a probability is -inf,
        and focal loss is used precisely where saturation happens."""
        for logit in (-40.0, 40.0):
            for target in (0.0, 1.0):
                value = focal_loss(torch.tensor([logit]), torch.tensor([target]),
                                   torch.ones(1), alpha=0.25, gamma=2.0)
                assert torch.isfinite(value), f"logit={logit} target={target}"


class TestMasking:
    def test_a_masked_clip_contributes_nothing(self) -> None:
        """Not "less". A clip the rater could not score must not move the loss at all."""
        predictions = torch.tensor([0.0, 99.0])
        targets = torch.tensor([0.0, 0.0])

        only_first = masked_mse(predictions, targets, torch.tensor([True, False]))
        alone = masked_mse(predictions[:1], targets[:1], torch.tensor([True]))
        assert only_first == pytest.approx(float(alone))

    def test_an_entirely_masked_batch_is_zero_and_still_differentiable(self) -> None:
        """A batch where every clip is unscorable for one behaviour is a real state. A NaN here
        would propagate through the shared trunk and destroy every behaviour, not just this
        one."""
        predictions = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = masked_mse(predictions, torch.zeros(2), torch.tensor([False, False]))

        assert float(loss.detach()) == 0.0
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.equal(predictions.grad, torch.zeros(2))

    def test_the_mean_is_over_the_masked_entries_not_the_batch(self) -> None:
        """Dividing by the batch size would make the loss shrink as more clips are masked,
        which reads as the model improving."""
        predictions = torch.tensor([2.0, 0.0, 0.0, 0.0])
        targets = torch.zeros(4)
        assert masked_mse(predictions, targets,
                          torch.tensor([True, False, False, False])) == pytest.approx(4.0)


class TestNonScorableExemption:
    """The head that must survive its own mask."""

    def test_the_flag_head_is_trained_on_clips_the_others_are_masked_out_of(self) -> None:
        heads, outputs, targets, _ = b1_batch(n=2)
        targets["b1_nonscorable"] = torch.ones(2)
        nothing_scorable = torch.zeros(2, dtype=torch.bool)

        _, parts = behaviour_loss(outputs, targets, heads, nothing_scorable, WEIGHTS)

        assert parts["b1_present"] == 0.0, "a masked field contributes nothing"
        assert parts["b1_count"] == 0.0
        assert parts["b1_nonscorable"] > 0.0, (
            "masking the flag on the clips it is raised for removes every positive example it "
            "has; the model could then never abstain and routing would never see "
            "model_abstained")

    def test_the_exemption_applies_to_exactly_one_head_per_behaviour(self) -> None:
        for behaviour in BEHAVIOUR_IDS:
            heads = behaviour_heads(CODEBOOK_V1, behaviour)
            exempt = [h.field for h in heads if h.is_nonscorable_flag]
            assert exempt == [f"{behaviour.lower()}_nonscorable"]


class TestBehaviourLoss:
    def test_every_field_appears_in_the_reported_parts(self) -> None:
        heads, outputs, targets, mask = b1_batch()
        _, parts = behaviour_loss(outputs, targets, heads, mask, WEIGHTS)
        assert set(parts) == {h.field for h in heads}

    def test_the_intensity_weight_scales_only_the_scalar_fields(self) -> None:
        heads, outputs, targets, mask = b1_batch()
        full = behaviour_loss(outputs, targets, heads, mask, WEIGHTS)[1]
        halved = behaviour_loss(outputs, targets, heads, mask,
                                LossWeights(0.25, 2.0, 0.15))[1]

        assert halved["b1_count"] == pytest.approx(full["b1_count"] / 2)
        assert halved["b1_present"] == pytest.approx(full["b1_present"])

    def test_the_total_is_differentiable(self) -> None:
        heads, outputs, targets, mask = b1_batch()
        total, _ = behaviour_loss(outputs, targets, heads, mask, WEIGHTS)
        total.backward()
        assert all(torch.isfinite(tensor.grad).all() for tensor in outputs.values())


class TestMultitaskLoss:
    def test_behaviours_are_summed_not_averaged(self) -> None:
        """An average lets a behaviour whose clips are mostly unscorable - and whose loss is
        therefore near zero by masking rather than by fit - pull the total down."""
        heads_b1, out_b1, tgt_b1, mask_b1 = b1_batch()
        total_one, _ = behaviour_loss(out_b1, tgt_b1, heads_b1, mask_b1, WEIGHTS)

        heads_b1b, out_b1b, tgt_b1b, mask_b1b = b1_batch()
        combined, parts = multitask_loss(
            {"B1": out_b1, "Bx": out_b1b},
            {"B1": tgt_b1, "Bx": tgt_b1b},
            {"B1": heads_b1, "Bx": heads_b1b},
            {"B1": mask_b1, "Bx": mask_b1b},
            WEIGHTS)

        assert float(combined.detach()) == pytest.approx(
            2 * float(total_one.detach()), rel=1e-5)
        assert set(parts) == {"B1", "Bx"}


def test_a_real_codebook_label_set_produces_a_finite_loss() -> None:
    """End to end on the actual codebook rather than a fixture, for every behaviour."""
    for behaviour in BEHAVIOUR_IDS:
        heads = behaviour_heads(CODEBOOK_V1, behaviour)
        labels = {h.field: (h.levels[0] if h.levels else h.minimum) for h in heads}
        encoded = encode_labels(heads, labels)

        outputs, targets = {}, {}
        for head in heads:
            width = head.width
            outputs[head.field] = (torch.zeros(3, width, requires_grad=True) if width > 1
                                   else torch.zeros(3, requires_grad=True))
            targets[head.field] = torch.full((3,), encoded[head.field])

        total, parts = behaviour_loss(outputs, targets, heads, torch.ones(3), WEIGHTS)
        assert torch.isfinite(total), behaviour
        assert all(value >= 0.0 for value in parts.values())


def test_categorical_loss_is_lowest_at_the_right_level() -> None:
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    right = categorical_loss(logits, torch.tensor([0]), torch.ones(1))
    wrong = categorical_loss(logits, torch.tensor([3]), torch.ones(1))
    assert right < wrong
