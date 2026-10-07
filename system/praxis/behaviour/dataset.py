# -*- coding: utf-8 -*-
"""Clip sampling and augmentation over cached backbone features.

**The tension this module resolves, and BUILD-SPEC does not.** Phase 4 Stage A caches the frozen
backbone's output once per clip, because the backbone never changes and recomputing it every
epoch is waste the measured 1.31 GB of headroom cannot afford. Phase 4 step 2 then asks for
"random crop, horizontal flip, colour jitter, temporal jitter". Three of those four act on
pixels, and a 2048-dimensional embedding cannot be flipped, cropped or recoloured after the
fact: a ResNet-50 activation is not equivariant to any of them. Applied naively, the
augmentation block in `configs/default.yaml` would be configuration that silently does nothing.

The resolution is to move the pixel augmentations into Stage A, where pixels still exist, and
materialise them as **cache variants**. Stage A writes one feature tensor per (clip, variant);
Stage B chooses among them. Temporal jitter stays in Stage B, because it acts on the frame axis
the cache keeps. See D75.

**The per-behaviour mask is what makes flipping safe, and it is the same mask non-scorability
uses.** A horizontal flip inverts left and right, so a flipped clip's B2 orientation label and
B3 zone label are no longer true of it, while its B1 gesture and B4 posture labels still are.
The model is multi-task and one forward pass answers all five behaviours, so a flipped clip
cannot simply be a training example: it has to be an example *for B1 and B4 only*, with B2, B3
and B5 masked out of the loss. `behaviour.augmentation.horizontal_flip.applies_to` is in the
config's `locked` list for exactly this reason, and it is load-bearing here rather than
advisory.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from praxis.behaviour.heads import HeadSpec, encode_labels
from praxis.vocabulary import BEHAVIOUR_IDS

IDENTITY = "identity"
HFLIP = "hflip"


class DatasetError(RuntimeError):
    """A clip cannot be turned into a training example, with the reason named."""


@dataclass(frozen=True)
class ClipVariant:
    """One cached rendering of a clip, and which behaviours it may supply targets for.

    `valid_for` is not a preference. A flipped clip carries a B3 zone label that is now wrong
    about it, and training on it would teach the model that the board is on the other side of
    the room.
    """

    name: str
    valid_for: tuple[str, ...]

    def supplies(self, behaviour: str) -> bool:
        return behaviour in self.valid_for


@dataclass(frozen=True)
class AugmentationPolicy:
    """What Stage A must cache and what Stage B may do, from `behaviour.augmentation`."""

    flip_enabled: bool
    flip_applies_to: tuple[str, ...]
    flip_probability: float
    temporal_jitter_frames: int
    colour_jitter_variants: int = 0
    behaviours: tuple[str, ...] = BEHAVIOUR_IDS

    def __post_init__(self) -> None:
        unknown = set(self.flip_applies_to) - set(self.behaviours)
        if unknown:
            raise DatasetError(
                f"horizontal_flip.applies_to names {sorted(unknown)}, which are not "
                f"behaviours. A typo here would silently disable the flip rather than widen "
                f"it, and the config key is in the locked list.")
        forbidden = {"B2", "B3"} & set(self.flip_applies_to)
        if forbidden:
            raise DatasetError(
                f"horizontal_flip.applies_to includes {sorted(forbidden)}. A flip inverts "
                f"orientation and position, so the B2 facing label and the B3 zone label stop "
                f"being true of the clip. BUILD-SPEC Phase 4 forbids this and the key is "
                f"locked.")

    @property
    def variants(self) -> tuple[ClipVariant, ...]:
        """Everything Stage A has to write for one clip.

        The identity variant supplies every behaviour. The flipped one supplies only those the
        config permits, which is what keeps a flip from corrupting orientation labels.
        """
        variants = [ClipVariant(IDENTITY, tuple(self.behaviours))]
        if self.flip_enabled and self.flip_applies_to:
            variants.append(ClipVariant(HFLIP, tuple(self.flip_applies_to)))
        for index in range(self.colour_jitter_variants):
            # Colour jitter does not change spatial semantics, so a jittered clip is a valid
            # example for every behaviour. It still has to be cached rather than applied here,
            # because by Stage B the pixels are gone.
            variants.append(ClipVariant(f"colour{index}", tuple(self.behaviours)))
        return tuple(variants)


def policy_from_config(config) -> AugmentationPolicy:
    augmentation = config.behaviour.augmentation
    if augmentation.colour_jitter.applied:
        raise DatasetError(
            "behaviour.augmentation.colour_jitter.applied is true, and the Stage A variant "
            "path that would make it real is not built. A colour jitter acts on pixels and the "
            "feature cache holds activations, so honouring this flag silently would mean "
            "training with no colour augmentation while the config says otherwise. Build the "
            "cache variants or set it false. See D75.")
    return AugmentationPolicy(
        flip_enabled=augmentation.horizontal_flip.enabled,
        flip_applies_to=tuple(augmentation.horizontal_flip.applies_to),
        flip_probability=augmentation.horizontal_flip.probability,
        temporal_jitter_frames=augmentation.temporal_jitter_frames,
        behaviours=tuple(config.behaviour.heads.presence_behaviours),
    )


@dataclass(frozen=True)
class ClipRecord:
    """One annotated clip, as the dataset reads it.

    `labels` is keyed by behaviour and holds the codebook field dict a rater produced.
    `nonscorable` is derived from those labels rather than passed alongside them, so the two
    cannot disagree.
    """

    clip_id: str
    session_id: str
    teacher_id: str
    features: dict[str, torch.Tensor]        # variant name -> (T, feature_dim)
    keypoints: torch.Tensor                  # (T, 17, 3)
    labels: dict[str, dict[str, Any]]        # behaviour -> codebook fields
    variants: tuple[str, ...] = (IDENTITY,)

    def is_nonscorable(self, behaviour: str) -> bool:
        return bool(self.labels[behaviour].get(f"{behaviour.lower()}_nonscorable", False))


@dataclass
class ClipSample:
    """What the model and the loss are handed for one clip."""

    clip_id: str
    variant: str
    features: torch.Tensor
    keypoints: torch.Tensor
    targets: dict[str, dict[str, float]]
    scorable: dict[str, bool] = field(default_factory=dict)


def _jittered_window(length: int, frames: int, jitter: int,
                     rng: random.Random) -> list[int]:
    """Frame indices for a temporally jittered window.

    Indices are clamped at the boundaries rather than wrapped. Wrapping would splice the end of
    a clip onto its beginning and present the result as eight continuous seconds, which for B3
    transitions - a count of boundary crossings - manufactures a crossing that never happened.
    Clamping repeats at most `jitter` frames at one edge, which understates motion slightly and
    invents nothing.
    """
    if jitter <= 0 and length == frames:
        return list(range(frames))
    offset = rng.randint(-jitter, jitter) if jitter > 0 else 0
    return [min(length - 1, max(0, offset + index)) for index in range(frames)]


def clips_per_session(records: Sequence[ClipRecord]) -> dict[str, int]:
    """How many clips each session contributes. Reported on the split manifest, so the shape of
    the training set is visible in the artefact rather than only in the data."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.session_id] = counts.get(record.session_id, 0) + 1
    return counts


def balance_by_session(records: Sequence[ClipRecord], cap: int | str | None,
                       *, seed: int) -> list[ClipRecord]:
    """Limit how many clips one session may contribute, so no room dominates training.

    Session length in this corpus runs from 59 seconds to 32 minutes, which at 8-second clips is
    7 against 243. Two sessions supply most of the training signal, and every clip from one
    session shares a teacher, a room, a camera and a light level: a model fitted on that learns
    the room. It is leakage-adjacent rather than leakage - nothing crosses a partition - and it
    corrupts the same thing, which is whether the measured accuracy says anything about a
    session the model has not seen. D84.

    `cap` is an integer, None for no cap, or "median" for the median count across sessions. The
    median is the principled setting: half the sessions are untouched and only the long tail is
    trimmed, so the cap adapts to the corpus instead of being a number somebody picked.

    Deterministic for R7: records are ordered by clip id and selected with a seeded shuffle, so
    the same corpus and seed keep the same clips.
    """
    if cap is None or not records:
        return list(records)

    by_session: dict[str, list[ClipRecord]] = {}
    for record in records:
        by_session.setdefault(record.session_id, []).append(record)

    if cap == "median":
        counts = sorted(len(group) for group in by_session.values())
        limit = counts[len(counts) // 2] if len(counts) % 2 else (
            (counts[len(counts) // 2 - 1] + counts[len(counts) // 2]) // 2)
    else:
        limit = int(cap)
    if limit < 1:
        raise DatasetError(
            f"max_clips_per_session resolved to {limit}; a session contributing no clips is a "
            f"session dropped from training without saying so")

    kept: list[ClipRecord] = []
    for session in sorted(by_session):
        group = sorted(by_session[session], key=lambda r: r.clip_id)
        if len(group) > limit:
            # Seeded per session, so adding a session later does not reshuffle the others.
            random.Random(f"{seed}:{session}").shuffle(group)
            group = sorted(group[:limit], key=lambda r: r.clip_id)
        kept.extend(group)
    return kept


class ClipDataset:
    """Cached clips as training examples, with the augmentation mask applied.

    Deliberately not a `torch.utils.data.Dataset` subclass. It satisfies the same protocol -
    `__len__` and `__getitem__` - so a DataLoader accepts it, while staying constructible and
    testable without torch's data machinery. `num_workers` is 2 on this estate and each worker
    is a full process at roughly 200 MB, so the collate path is exercised rarely and the
    indirection would cost more than it returns.
    """

    def __init__(
        self,
        records: Sequence[ClipRecord],
        heads: dict[str, tuple[HeadSpec, ...]],
        *,
        frames: int,
        policy: AugmentationPolicy,
        seed: int,
        train: bool = True,
        max_clips_per_session: int | str | None = None,
    ) -> None:
        if not records:
            raise DatasetError("no clips to sample from")
        # Training only. Capping an evaluation set would report accuracy and ECE on a subsample
        # of the held-out sessions while naming them in full, which is a different number than
        # the one the thesis claims to have measured. D84.
        self.records = (balance_by_session(records, max_clips_per_session, seed=seed)
                        if train else list(records))
        self.heads = heads
        self.frames = frames
        self.policy = policy
        self.train = train
        self.seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def teachers(self) -> set[str]:
        """Who is in this dataset. The caller checks it against the manifest partition; R2 is
        not something a dataset should be trusted to have got right on its own."""
        return {record.teacher_id for record in self.records}

    def _choose_variant(self, record: ClipRecord) -> str:
        """Which cached rendering to use for this example.

        At evaluation time, always the identity. An augmented evaluation set would measure the
        model on data the corpus does not contain, and the accuracy and ECE reported at S0
        would describe neither the clips nor the augmentation.
        """
        if not self.train:
            return IDENTITY
        flippable = [v.name for v in self.policy.variants
                     if v.name == HFLIP and v.name in record.variants]
        if flippable and self._rng.random() < self.policy.flip_probability:
            return HFLIP
        alternatives = [name for name in record.variants
                        if name.startswith("colour") ]
        if alternatives and self._rng.random() < 0.5:
            return self._rng.choice(alternatives)
        return IDENTITY

    def __getitem__(self, index: int) -> ClipSample:
        record = self.records[index]
        variant_name = self._choose_variant(record)
        if variant_name not in record.features:
            raise DatasetError(
                f"clip {record.clip_id} has no cached features for variant "
                f"{variant_name!r}. Stage A writes one tensor per variant; a missing one means "
                f"the cache was built under a different augmentation policy.")

        cached = record.features[variant_name]
        jitter = self.policy.temporal_jitter_frames if self.train else 0
        window = _jittered_window(cached.shape[0], self.frames, jitter, self._rng)
        picked = torch.tensor(window, dtype=torch.long)

        variant = next(v for v in self.policy.variants if v.name == variant_name)
        targets: dict[str, dict[str, float]] = {}
        scorable: dict[str, bool] = {}
        for behaviour, specs in self.heads.items():
            targets[behaviour] = encode_labels(specs, record.labels[behaviour])
            # Two independent reasons to mask, combined here and nowhere else: the rater could
            # not score this clip for this behaviour, or the augmentation makes the label
            # untrue of the rendering being shown.
            scorable[behaviour] = (not record.is_nonscorable(behaviour)
                                   and variant.supplies(behaviour))

        return ClipSample(
            clip_id=record.clip_id,
            variant=variant_name,
            features=cached.index_select(0, picked),
            keypoints=record.keypoints.index_select(0, picked),
            targets=targets,
            scorable=scorable,
        )


def collate(samples: Sequence[ClipSample],
            heads: dict[str, tuple[HeadSpec, ...]]) -> dict[str, Any]:
    """Stack samples into the tensors `multitask_loss` expects.

    Categorical targets are kept as long, everything else as float, because cross-entropy wants
    class indices and the other two losses want values.
    """
    from praxis.behaviour.heads import HeadKind

    features = torch.stack([s.features for s in samples])
    keypoints = torch.stack([s.keypoints for s in samples])

    targets: dict[str, dict[str, torch.Tensor]] = {}
    scorable: dict[str, torch.Tensor] = {}
    for behaviour, specs in heads.items():
        targets[behaviour] = {
            head.field: torch.tensor(
                [s.targets[behaviour][head.field] for s in samples],
                dtype=torch.long if head.kind is HeadKind.CATEGORICAL else torch.float32)
            for head in specs
        }
        scorable[behaviour] = torch.tensor([s.scorable[behaviour] for s in samples])

    return {"features": features, "keypoints": keypoints,
            "targets": targets, "scorable": scorable,
            "clip_ids": [s.clip_id for s in samples],
            "variants": [s.variant for s in samples]}


def dataset_from_config(records: Sequence[ClipRecord],
                        heads: dict[str, tuple[HeadSpec, ...]], config, *,
                        seed: int, train: bool) -> ClipDataset:
    """Build a dataset with every parameter the configuration decides.

    The one place `behaviour.train.max_clips_per_session` reaches a dataset. Constructing
    `ClipDataset` directly is still allowed and leaves the cap off, which is what the tests
    want; a caller that is training on the real corpus goes through here so that forgetting the
    cap takes an explicit decision rather than an omission. D84.
    """
    return ClipDataset(
        records, heads,
        frames=config.behaviour.clip.frames,
        policy=policy_from_config(config),
        seed=seed,
        train=train,
        max_clips_per_session=config.behaviour.train.max_clips_per_session)
