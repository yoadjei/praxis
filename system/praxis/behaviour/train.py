# -*- coding: utf-8 -*-
"""Stage B training: cosine schedule, early stopping on validation macro-F1, deterministic.

BUILD-SPEC Phase 4 step 4, and its acceptance test: "training twice with the same config and
seed produces identical validation metrics". On this estate R7 means bit-exact on CPU, so the
seeding goes through `praxis.config.seed_everything` and the data order is drawn from a
generator seeded per epoch rather than from the ambient random state.

**Macro-F1 is not redefined here.** `ConfusionStructure.f1` in `evaluation/harness.py` owns the
formula, and this module builds confusion counts and calls it. Two implementations of one
statistic is lesson L20, and the one that drifts is always the one used for early stopping,
because nothing reports it next to the other.

**What the monitor covers, and what it deliberately does not.** The score averages macro-F1
over each behaviour's classification heads and then over behaviours, so a behaviour with five
fields does not outvote one with three. The scalar heads - counts, proportions, ordinal
intensity - have no F1 and are reported as masked mean absolute error alongside, without gating
on it, because BUILD-SPEC names macro-F1 as the early-stopping monitor and swapping in a
composite would be redefining the stopping rule.

**The non-scorable flag is excluded from the monitor.** It is a prediction the model makes and
it is trained, but early stopping should track behaviour recognition rather than abstention:
a model that learns to abstain well and recognise nothing would otherwise stop late and look
like it was still improving.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from praxis.behaviour.dataset import ClipDataset, collate
from praxis.behaviour.heads import HeadKind, HeadSpec
from praxis.behaviour.losses import LossWeights, multitask_loss
from praxis.behaviour.model import BehaviourModel
from praxis.behaviour.scoring import macro_f1_binary, macro_f1_multiclass
from praxis.config import seed_everything


class TrainingError(RuntimeError):
    """Training cannot proceed, with the reason named."""


@dataclass(frozen=True)
class TrainingPlan:
    """Everything `config.behaviour.train` decides."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    grad_clip_norm: float
    patience: int
    min_delta: float
    monitor: str = "val_macro_f1"
    mixed_precision: bool = False


def plan_from_config(config) -> TrainingPlan:
    train = config.behaviour.train
    return TrainingPlan(
        epochs=train.epochs,
        batch_size=train.batch_size,
        learning_rate=train.learning_rate,
        weight_decay=train.weight_decay,
        warmup_epochs=train.warmup_epochs,
        grad_clip_norm=train.grad_clip_norm,
        patience=train.early_stopping.patience,
        min_delta=train.early_stopping.min_delta,
        monitor=train.early_stopping.monitor,
        mixed_precision=train.mixed_precision,
    )


def weights_from_config(config) -> LossWeights:
    loss = config.behaviour.loss
    return LossWeights(
        focal_alpha=loss.focal_alpha,
        focal_gamma=loss.focal_gamma,
        intensity_weight=loss.intensity_weight,
        mask_nonscorable=loss.mask_nonscorable,
    )


@dataclass(frozen=True)
class EpochRecord:
    """One epoch, as the run manifest records it."""

    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    learning_rate: float
    per_behaviour_f1: dict[str, float] = field(default_factory=dict)
    scalar_mae: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingRun:
    """The history and the weights that produced the best validation score."""

    epochs: tuple[EpochRecord, ...]
    best_epoch: int
    best_score: float
    stopped_early: bool
    best_state: dict[str, torch.Tensor]

    @property
    def history(self) -> list[dict[str, float]]:
        return [{"epoch": e.epoch, "train_loss": e.train_loss, "val_loss": e.val_loss,
                 "val_macro_f1": e.val_macro_f1, "lr": e.learning_rate}
                for e in self.epochs]


def cosine_with_warmup(epoch: int, total: int, warmup: int) -> float:
    """The learning-rate multiplier for one epoch.

    Linear warmup then cosine decay to zero. Warmup is counted in epochs rather than steps
    because Stage B trains on cached features and an epoch is small - a step-based warmup at
    these sizes would be over before the first epoch ended and would amount to nothing.
    """
    if total <= 0:
        raise TrainingError("a schedule needs at least one epoch")
    if warmup > 0 and epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _macro_f1_for_head(head: HeadSpec, scores: torch.Tensor,
                       truth: torch.Tensor) -> float | None:
    """Macro-F1 for one classification head, through the shared definition in `scoring`.

    Binary heads average positive-class and negative-class F1, which is what
    `evaluate_behaviour` reports, so the monitor and the frozen report are the same number.
    Categorical heads average one-vs-rest F1 over the levels that occur.
    """
    if truth.numel() == 0:
        return None

    if head.kind is HeadKind.BINARY:
        predicted = (torch.sigmoid(scores) >= 0.5).to(torch.int64).numpy()
        return macro_f1_binary(predicted, truth.to(torch.int64).numpy())

    return macro_f1_multiclass(scores.argmax(dim=-1).numpy(),
                               truth.to(torch.int64).numpy(), head.width)


def validation_scores(
    model: BehaviourModel,
    batches: Sequence[dict],
    heads: dict[str, tuple[HeadSpec, ...]],
    weights: LossWeights,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """Validation loss, macro-F1, per-behaviour F1, and scalar MAE.

    Masked clips are dropped before scoring. Including them would score the model against a
    label the rater said could not be read, and at the S2 shift level - where the classroom
    footage is hardest to score - that would be a large and invisible share of the set.
    """
    model.eval()
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        behaviour: {head.field: [] for head in specs} for behaviour, specs in heads.items()}
    truths: dict[str, dict[str, list[torch.Tensor]]] = {
        behaviour: {head.field: [] for head in specs} for behaviour, specs in heads.items()}
    total_loss, seen = 0.0, 0

    with torch.no_grad():
        for batch in batches:
            output = model(batch["features"], batch["keypoints"])
            loss, _ = multitask_loss(output.raw, batch["targets"], heads,
                                     batch["scorable"], weights)
            total_loss += float(loss) * batch["features"].shape[0]
            seen += batch["features"].shape[0]

            for behaviour, specs in heads.items():
                keep = batch["scorable"][behaviour].to(torch.bool)
                if not bool(keep.any()):
                    continue
                for head in specs:
                    collected[behaviour][head.field].append(
                        output.raw[behaviour][head.field][keep])
                    truths[behaviour][head.field].append(
                        batch["targets"][behaviour][head.field][keep])

    per_behaviour: dict[str, float] = {}
    scalar_mae: dict[str, float] = {}
    for behaviour, specs in heads.items():
        scores: list[float] = []
        for head in specs:
            if head.is_nonscorable_flag or not collected[behaviour][head.field]:
                continue
            predicted = torch.cat(collected[behaviour][head.field])
            truth = torch.cat(truths[behaviour][head.field])

            if head.kind is HeadKind.SCALAR:
                scalar_mae[head.field] = float((predicted - truth).abs().mean())
                continue
            value = _macro_f1_for_head(head, predicted, truth)
            if value is not None:
                scores.append(value)
        if scores:
            per_behaviour[behaviour] = float(np.mean(scores))

    macro = float(np.mean(list(per_behaviour.values()))) if per_behaviour else 0.0
    return (total_loss / seen if seen else 0.0), macro, per_behaviour, scalar_mae


def _batches(dataset: ClipDataset, heads: dict[str, tuple[HeadSpec, ...]],
             batch_size: int, order: Sequence[int]) -> list[dict]:
    return [collate([dataset[i] for i in order[start:start + batch_size]], heads)
            for start in range(0, len(order), batch_size)]


def train(
    model: BehaviourModel,
    train_set: ClipDataset,
    val_set: ClipDataset,
    heads: dict[str, tuple[HeadSpec, ...]],
    *,
    plan: TrainingPlan,
    weights: LossWeights,
    seed: int,
    deterministic: bool = True,
) -> TrainingRun:
    """Train the Stage B head and return the history plus the best weights.

    **Determinism.** `seed_everything` is called once before anything is constructed, and the
    shuffle for each epoch comes from a `random.Random(seed + epoch)` rather than the global
    state, so a change elsewhere in the process cannot move the data order. That is what makes
    the acceptance test - two runs, same config and seed, identical validation metrics - a
    property of the code rather than of the machine it happened to run on.

    **The best weights are kept, not the last.** Early stopping without restoring the best
    checkpoint reports a number measured at the best epoch while returning the model from
    `patience` epochs later, which is a quiet mismatch between the figure in the thesis and the
    artefact that produced it.
    """
    if plan.mixed_precision:
        raise TrainingError(
            "behaviour.train.mixed_precision is true. BUILD-SPEC requires it to be measured "
            "before it is kept, and on a CPU build there is no autocast path that would make "
            "it faster. Measure it on the CUDA estate first and record the result.")
    if val_set.teachers() & train_set.teachers():
        raise TrainingError(
            f"teachers {sorted(val_set.teachers() & train_set.teachers())} appear in both the "
            f"training and validation sets. R2 forbids a teacher crossing partitions, and a "
            f"validation score measured on someone the model trained on is not a validation "
            f"score.")

    seed_everything(seed, deterministic=deterministic)
    optimiser = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate,
                                  weight_decay=plan.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda epoch: cosine_with_warmup(epoch, plan.epochs, plan.warmup_epochs))

    val_order = list(range(len(val_set)))
    records: list[EpochRecord] = []
    best_score, best_epoch, stale = -math.inf, 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(plan.epochs):
        model.train()
        order = list(range(len(train_set)))
        random.Random(seed + epoch).shuffle(order)

        running, seen = 0.0, 0
        for batch in _batches(train_set, heads, plan.batch_size, order):
            optimiser.zero_grad(set_to_none=True)
            output = model(batch["features"], batch["keypoints"])
            loss, _ = multitask_loss(output.raw, batch["targets"], heads,
                                     batch["scorable"], weights)
            loss.backward()
            if plan.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), plan.grad_clip_norm)
            optimiser.step()

            running += float(loss.detach()) * batch["features"].shape[0]
            seen += batch["features"].shape[0]

        val_loss, macro, per_behaviour, mae = validation_scores(
            model, _batches(val_set, heads, plan.batch_size, val_order), heads, weights)
        records.append(EpochRecord(
            epoch=epoch,
            train_loss=running / seen if seen else 0.0,
            val_loss=val_loss,
            val_macro_f1=macro,
            learning_rate=float(schedule.get_last_lr()[0]),
            per_behaviour_f1=per_behaviour,
            scalar_mae=mae))
        schedule.step()

        if macro > best_score + plan.min_delta:
            best_score, best_epoch, stale = macro, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= plan.patience:
                model.load_state_dict(best_state)
                return TrainingRun(tuple(records), best_epoch, best_score, True, best_state)

    model.load_state_dict(best_state)
    return TrainingRun(tuple(records), best_epoch, best_score, False, best_state)
