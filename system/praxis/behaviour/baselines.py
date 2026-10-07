# -*- coding: utf-8 -*-
"""The three baselines the frozen report must compare the deep model against.

BUILD-SPEC Phase 4 step 5 lists them and calls none of them optional:

    majority_class          predict the training majority, always
    keypoints_gbdt          keypoint summary statistics into gradient-boosted trees
    frame_averaged_resnet   the cached backbone features, mean-pooled, no temporal modelling

**Why all three and not just the weakest.** They answer different questions. Majority class says
what the class balance alone buys, which is the number a macro-F1 has to beat before it means
anything. The GBDT on keypoints says what pose gives without deep learning, and if it matches
the deep model then the TCN and the backbone are expensive decoration. Frame-averaged ResNet-50
removes only the temporal modelling, so the gap between it and the full model is the value of
the TCN specifically rather than of the whole architecture.

**Every baseline is fitted on the training partition only**, and the same non-scorable mask
applies. A baseline fitted on the evaluation set would beat the model for the wrong reason and
the comparison would be worthless - which is the sort of mistake that reads as a surprising
result rather than as a bug.

These predict *probabilities for the classification heads*, not the full codebook field set.
The comparison BUILD-SPEC asks for is macro-F1 and ECE against the deep model, and both are
defined on the classification fields; asking a majority-class predictor for a gesture count
would be inventing a baseline nobody specified.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from praxis.behaviour.heads import HeadKind, HeadSpec


class BaselineError(RuntimeError):
    """A baseline cannot be fitted or applied, with the reason named."""


@dataclass(frozen=True)
class LabelledClips:
    """What a baseline is fitted on: features, keypoints, targets and the scorable mask.

    Plain arrays rather than the torch dataset, because two of the three baselines have nothing
    to do with torch and importing it into a tree model would only make the dependency wider.
    """

    features: np.ndarray            # (N, T, D) cached backbone features
    keypoints: np.ndarray           # (N, T, K, 3)
    targets: dict[str, np.ndarray]  # field -> (N,)
    scorable: dict[str, np.ndarray]  # behaviour -> (N,) bool

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise BaselineError(
                f"features must be (N, T, D), got {self.features.shape}")
        if self.keypoints.shape[0] != self.features.shape[0]:
            raise BaselineError("features and keypoints describe different numbers of clips")

    def __len__(self) -> int:
        return int(self.features.shape[0])


class Baseline(Protocol):
    """Fit on the training partition, then score. Probabilities, never hard labels.

    Probabilities rather than labels because R4 is enforced by the evaluation return type:
    every accuracy figure is reported with its calibration, and a baseline that emitted only
    hard labels could not have an ECE computed for it and would silently drop out of the
    comparison the thesis needs.
    """

    name: str

    def fit(self, clips: LabelledClips, heads: dict[str, tuple[HeadSpec, ...]]) -> None:
        ...

    def predict_proba(self, clips: LabelledClips) -> dict[str, np.ndarray]:
        ...


def _classification_heads(heads: dict[str, tuple[HeadSpec, ...]]) -> list[tuple[str, HeadSpec]]:
    """Every head a baseline is expected to answer: classification, excluding the flag.

    The non-scorable flag is excluded for the same reason it is excluded from the training
    monitor: it measures abstention rather than recognition, and a baseline that abstains
    everywhere would otherwise score well.
    """
    return [(behaviour, head)
            for behaviour, specs in sorted(heads.items()) for head in specs
            if head.kind is not HeadKind.SCALAR and not head.is_nonscorable_flag]


def _mask_for(clips: LabelledClips, behaviour: str) -> np.ndarray:
    return np.asarray(clips.scorable[behaviour], dtype=bool)


class MajorityClass:
    """Predict the training base rate for every clip, always.

    The floor. A macro-F1 that does not clear this is a model that has learned the class
    balance, and on a corpus where a behaviour is rare that can look like a high accuracy.

    The base rate rather than a hard majority label, because R4 requires a calibration figure
    alongside every accuracy figure and a baseline emitting only 0 or 1 has no ECE worth
    reporting. Predicting the training prevalence is also the honest constant: it is exactly
    as confident as the training data justifies.
    """

    name = "majority_class"

    def __init__(self, seed: int = 20260911) -> None:
        self.seed = seed
        self._rates: dict[str, np.ndarray] = {}

    def fit(self, clips: LabelledClips, heads: dict[str, tuple[HeadSpec, ...]]) -> None:
        for behaviour, head in _classification_heads(heads):
            mask = _mask_for(clips, behaviour)
            values = np.asarray(clips.targets[head.field])[mask].astype(int)
            self._rates[head.field] = _constant_rate(head, values)

    def predict_proba(self, clips: LabelledClips) -> dict[str, np.ndarray]:
        if not self._rates:
            raise BaselineError("majority_class was not fitted")
        n = len(clips)
        return {field: _tile(rate, n) for field, rate in self._rates.items()}


def keypoint_features(keypoints: np.ndarray) -> np.ndarray:
    """Summary statistics over a clip's pose, as a flat feature vector per clip.

    Deliberately hand-built and few. The point of this baseline is "what does pose give without
    deep learning", so learning a representation of the keypoints would answer a different
    question. Per keypoint: mean and standard deviation of x and y, mean confidence, and total
    path length - which is the motion signal B3's mobility and B1's gesture both depend on.
    """
    if keypoints.ndim != 4 or keypoints.shape[-1] != 3:
        raise BaselineError(f"keypoints must be (N, T, K, 3), got {keypoints.shape}")

    clips, frames, joints = keypoints.shape[0], keypoints.shape[1], keypoints.shape[2]
    xy = keypoints[..., :2]
    confidence = keypoints[..., 2]

    if frames > 1:
        steps = np.diff(xy, axis=1)
        path = np.sqrt((steps ** 2).sum(axis=-1)).sum(axis=1)
    else:
        # A one-frame clip has no motion to measure. Zero is the true path length, not a
        # missing value, so it is recorded rather than dropped.
        path = np.zeros((clips, joints))

    return np.concatenate([
        xy.mean(axis=1).reshape(len(keypoints), -1),
        xy.std(axis=1).reshape(len(keypoints), -1),
        confidence.mean(axis=1),
        path,
    ], axis=1)


class _FittedPerHead:
    """Shared machinery for the two baselines that fit a scikit-learn model per head.

    `KeypointsGBDT` and `FrameAveragedResnet` differ in exactly two things: the design matrix
    they build from a clip, and the estimator they fit. Everything else - the per-head loop,
    the non-scorable mask, the single-class fallback, and widening a classifier's `classes_`
    back to the codebook's full level set - is identical, and two copies of it would be two
    places for the level-widening to go wrong in.
    """

    name = "unnamed"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._models: dict[str, object] = {}
        self._constant: dict[str, np.ndarray] = {}
        self._heads: dict[str, HeadSpec] = {}

    def design(self, clips: LabelledClips) -> np.ndarray:
        raise NotImplementedError

    def estimator(self):
        raise NotImplementedError

    def fit(self, clips: LabelledClips, heads: dict[str, tuple[HeadSpec, ...]]) -> None:
        design = self.design(clips)
        for behaviour, head in _classification_heads(heads):
            mask = _mask_for(clips, behaviour)
            y = np.asarray(clips.targets[head.field])[mask].astype(int)
            self._heads[head.field] = head

            # A head whose training labels are all one value cannot be fitted by a classifier,
            # and for a rare behaviour that is a real state rather than an error. The constant
            # the data supports is recorded, and it shows up in the reported ECE rather than
            # being hidden behind a crash or a skipped row.
            if y.size == 0 or np.unique(y).size < 2:
                self._constant[head.field] = _constant_rate(head, y)
                continue

            model = self.estimator()
            model.fit(design[mask], y)
            self._models[head.field] = model

    def predict_proba(self, clips: LabelledClips) -> dict[str, np.ndarray]:
        if not self._heads:
            raise BaselineError(f"{self.name} was not fitted")
        design = self.design(clips)
        n = len(clips)

        out: dict[str, np.ndarray] = {}
        for field, head in self._heads.items():
            if field in self._constant:
                out[field] = _tile(self._constant[field], n)
                continue
            out[field] = _widen(self._models[field], design, head, n)
        return out


def _constant_rate(head: HeadSpec, y: np.ndarray) -> np.ndarray:
    """The best constant prediction for a head with no usable training signal."""
    if head.kind is HeadKind.BINARY:
        return np.array([float(y.mean())]) if y.size else np.array([0.5])
    if y.size:
        return np.bincount(y, minlength=head.width)[:head.width] / y.size
    return np.full(head.width, 1.0 / head.width)


def _tile(rate: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(rate, n) if rate.size == 1 else np.tile(rate, (n, 1))


def _widen(model, design: np.ndarray, head: HeadSpec, n: int) -> np.ndarray:
    """A classifier's probabilities, widened back to the codebook's full level set.

    A classifier only knows the classes it saw. Returning its own column order would give an
    array whose width depends on which levels the training partition happened to contain, and
    the mismatch would surface as a shape error somewhere far from here - or worse, not at all,
    when the widths coincide and the columns mean different levels.
    """
    probabilities = model.predict_proba(design)
    if head.kind is HeadKind.BINARY:
        # `classes_` is [0, 1] for a fitted binary head, but not always in that order, so the
        # positive column is located rather than assumed.
        positive = int(np.where(model.classes_ == 1)[0][0])
        return probabilities[:, positive]

    widened = np.zeros((n, head.width))
    for column, level in enumerate(model.classes_):
        widened[:, int(level)] = probabilities[:, column]
    return widened


class KeypointsGBDT(_FittedPerHead):
    """Gradient-boosted trees on pose summary statistics. No deep learning.

    The question it answers: what does pose give without a network. If it matches the deep
    model then the TCN and the frozen backbone are expensive decoration, and the thesis should
    say so.

    `HistGradientBoostingClassifier` because it is the implementation scikit-learn maintains
    for speed, seeded so the baseline is as reproducible as the model it is compared against.
    R7 does not stop at the thing being measured.
    """

    name = "keypoints_gbdt"

    def __init__(self, seed: int = 20260911, max_iter: int = 100) -> None:
        super().__init__(seed)
        self.max_iter = max_iter

    def design(self, clips: LabelledClips) -> np.ndarray:
        return keypoint_features(clips.keypoints)

    def estimator(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=self.max_iter,
                                              random_state=self.seed)


class FrameAveragedResnet(_FittedPerHead):
    """The cached backbone features, mean-pooled over time, into logistic regression.

    No temporal modelling at all, which is the point: the gap between this and the full model
    is what the TCN and the attention pooling are worth. A mean over 64 frames dilutes a
    gesture by the ratio of its duration to the clip's, and if that turns out not to matter
    then the temporal stack is not earning its place.
    """

    name = "frame_averaged_resnet"

    def __init__(self, seed: int = 20260911, max_iter: int = 1000) -> None:
        super().__init__(seed)
        self.max_iter = max_iter

    def design(self, clips: LabelledClips) -> np.ndarray:
        return clips.features.mean(axis=1)

    def estimator(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=self.max_iter, random_state=self.seed)


REQUIRED_BASELINES: tuple[str, ...] = (
    "majority_class", "keypoints_gbdt", "frame_averaged_resnet")


def build_baselines(names: Sequence[str] = REQUIRED_BASELINES,
                    *, seed: int = 20260911) -> tuple[Baseline, ...]:
    """The baselines named in `behaviour.baselines`, refusing any that is not implemented.

    An unknown name raises rather than being skipped. BUILD-SPEC calls all three required, and
    a comparison table quietly missing a row is how "compared against all three baselines"
    becomes true of a document and false of the work.
    """
    available = {
        "majority_class": lambda: MajorityClass(),
        "keypoints_gbdt": lambda: KeypointsGBDT(seed=seed),
        "frame_averaged_resnet": lambda: FrameAveragedResnet(seed=seed),
    }
    unknown = [name for name in names if name not in available]
    if unknown:
        raise BaselineError(
            f"no baseline called {unknown}. BUILD-SPEC Phase 4 requires "
            f"{list(REQUIRED_BASELINES)}, and a missing row in the comparison table is how "
            f"a claim about all three stops being true.")
    return tuple(available[name]() for name in names)
