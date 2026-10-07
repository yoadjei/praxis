# -*- coding: utf-8 -*-
"""Multi-task loss: focal for binary fields, cross-entropy for categorical, masked MSE for
scalar ones.

BUILD-SPEC Phase 4 step 3, generalised the way `heads.py` generalises the output layout: focal
loss for presence, masked mean squared error for intensity, and non-scorable clips masked out
of both. Categorical fields are not mentioned there and are required by `Detection`, so they
are trained with cross-entropy and the choice is recorded in D74 rather than left implicit.

**The mask is per clip and per behaviour, never per clip alone.** A clip can be unscorable for
gesture and perfectly scorable for spatial position - the teacher's hands are out of frame while
their feet are not - which is why CODEBOOK.md records `bN_nonscorable` separately for each of
the five. A clip masked for B1 stays in the batch for B2 through B5.

**The non-scorable head is exempt from its own mask.** Masking the clips a behaviour is
unscorable on would remove every positive example that head has, so it could never learn to
raise the flag and `routing.gate` would never emit `model_abstained`. The flag is what the mask
is derived from; it cannot also be what the mask removes.

**Empty masks yield zero, not NaN.** A batch in which every clip is unscorable for one
behaviour is a real state, not an error, and `sum / count` with a count of zero would put NaN
into the gradient and silently destroy the whole model rather than that one behaviour. The
denominator is clamped, which keeps the term in the autograd graph while contributing nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from praxis.behaviour.heads import HeadKind, HeadSpec


@dataclass(frozen=True)
class LossWeights:
    """What the loss is made of, from `config.behaviour.loss`."""

    focal_alpha: float
    focal_gamma: float
    intensity_weight: float
    mask_nonscorable: bool = True


def _masked_mean(per_element: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the masked entries, and zero when there are none.

    `clamp(min=1)` rather than a branch on `mask.sum() == 0`. A branch returning a fresh zero
    tensor would drop out of the autograd graph, and `backward()` on a total made entirely of
    such zeros raises rather than doing nothing - which is the failure a batch of wholly
    unscorable clips would produce, at some unpredictable point in a long training run.
    """
    mask = mask.to(per_element.dtype)
    while mask.ndim < per_element.ndim:
        mask = mask.unsqueeze(-1)
    return (per_element * mask).sum() / mask.sum().clamp(min=1.0)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
               *, alpha: float, gamma: float) -> torch.Tensor:
    """Binary focal loss (Lin et al., 2017), computed from logits.

    From logits rather than probabilities, via `binary_cross_entropy_with_logits`, because the
    log of a sigmoid that has saturated to 0 or 1 is -inf. Focal loss is used precisely where
    the class balance is extreme, which is exactly where saturation happens, so the unstable
    formulation would fail on the data it was chosen for.

    `p_t` is recovered as `exp(-bce)`, which is the model's probability for the true class, and
    the modulating factor `(1 - p_t) ** gamma` shrinks the contribution of examples already
    classified confidently.
    """
    targets = targets.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return _masked_mean(alpha_t * (1.0 - p_t) ** gamma * bce, mask)


def categorical_loss(logits: torch.Tensor, targets: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over a codebook field's declared levels.

    Plain cross-entropy rather than a focal variant: BUILD-SPEC prescribes focal loss for
    presence, where one class is rare by construction, and the categorical fields - which zone,
    which posture, which arm configuration - are not imbalanced in that way. Using focal here
    would be extending a prescription past its stated reason. D74.
    """
    per_element = F.cross_entropy(logits, targets.long(), reduction="none")
    return _masked_mean(per_element, mask)


def masked_mse(predictions: torch.Tensor, targets: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the scorable clips only.

    Targets arrive already normalised to [0, 1] by `HeadSpec.encode`, so the fields carry equal
    weight regardless of their units.
    """
    return _masked_mean((predictions - targets.to(predictions.dtype)) ** 2, mask)


def behaviour_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    heads: tuple[HeadSpec, ...],
    scorable: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One behaviour's total loss, with the per-field parts reported alongside.

    Args:
        outputs: raw head outputs. Binary and scalar heads give (N,) or (N, 1); categorical
            heads give (N, width) of logits.
        targets: encoded targets, (N,). Categorical targets are class indices.
        heads: the behaviour's layout, from `behaviour_heads`.
        scorable: (N,) mask, true where this clip is scorable for this behaviour.
        weights: focal parameters and the intensity weight.

    Returns:
        The total, and a dict of the detached per-field values for logging. The parts are
        returned rather than logged here so that nothing in this module has to know how a run
        records itself.
    """
    parts: dict[str, float] = {}
    total = torch.zeros((), dtype=torch.float32,
                        device=next(iter(outputs.values())).device)

    for head in heads:
        prediction = outputs[head.field]
        target = targets[head.field]

        # The flag is exempt: it is what the mask is made of, and masking it would leave the
        # head with no positive examples at all.
        mask = scorable
        if head.is_nonscorable_flag or not weights.mask_nonscorable:
            mask = torch.ones_like(scorable)

        if head.kind is HeadKind.BINARY:
            term = focal_loss(prediction.squeeze(-1) if prediction.ndim > 1 else prediction,
                              target, mask,
                              alpha=weights.focal_alpha, gamma=weights.focal_gamma)
        elif head.kind is HeadKind.CATEGORICAL:
            term = categorical_loss(prediction, target, mask)
        else:
            term = weights.intensity_weight * masked_mse(
                prediction.squeeze(-1) if prediction.ndim > 1 else prediction, target, mask)

        parts[head.field] = float(term.detach())
        total = total + term

    return total, parts


def multitask_loss(
    outputs: dict[str, dict[str, torch.Tensor]],
    targets: dict[str, dict[str, torch.Tensor]],
    heads: dict[str, tuple[HeadSpec, ...]],
    scorable: dict[str, torch.Tensor],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    """Every behaviour summed, with each behaviour's parts kept separate.

    Summed rather than averaged across behaviours. An average would let a behaviour whose clips
    are mostly unscorable - and whose loss is therefore near zero by masking rather than by
    fit - pull the reported total down and look like progress.
    """
    device = next(iter(next(iter(outputs.values())).values())).device
    total = torch.zeros((), dtype=torch.float32, device=device)
    parts: dict[str, dict[str, float]] = {}

    for behaviour in sorted(heads):
        behaviour_total, behaviour_parts = behaviour_loss(
            outputs[behaviour], targets[behaviour], heads[behaviour],
            scorable[behaviour], weights)
        total = total + behaviour_total
        parts[behaviour] = behaviour_parts

    return total, parts
