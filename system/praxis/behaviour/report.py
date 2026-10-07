# -*- coding: utf-8 -*-
"""The frozen Phase 4 report: the model against all three baselines, per codebook field.

BUILD-SPEC Phase 4 is done when "a frozen evaluation report exists for the in-domain
microteaching test set, expressed against the human agreement band, and H2 can be assessed",
and one of its acceptance tests is that "the deep model is compared against all three
baselines". Both are properties of this module.

**Scored per field, not per behaviour.** The model predicts the codebook's field set - that is
what `Detection` requires - and Phase 2 measures agreement per field, because Krippendorff's
alpha is computed on a field. A per-behaviour figure would have to collapse four to six fields
into one number by a rule nobody has stated, and then be compared against a human band measured
on a different quantity.

**R4 has no exception here.** Every accuracy in this report carries a `CalibrationReport`
beside it: binary fields through `evaluate_behaviour`, categorical fields through
`multiclass_calibration_report`. `FieldScore` cannot be constructed without one.

**Scalar fields are reported as error, not accuracy.** A gesture count has no accuracy and no
ECE, so it is reported as masked mean absolute error and kept out of the accuracy table rather
than given a number that looks comparable and is not.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from praxis.behaviour.baselines import REQUIRED_BASELINES, LabelledClips
from praxis.behaviour.heads import HeadKind, HeadSpec
from praxis.behaviour.scoring import macro_f1_multiclass
from praxis.confidence.metrics import CalibrationReport, multiclass_calibration_report
from praxis.evaluation.harness import (
    BehaviourEvaluation,
    evaluate_behaviour,
    human_comparison,
    require_band_statistic,
)


class ReportError(RuntimeError):
    """The comparison cannot be made, with the reason named."""


@dataclass(frozen=True)
class FieldScore:
    """One field's result for one system. Accuracy cannot exist here without calibration."""

    behaviour: str
    field: str
    kind: HeadKind
    n: int
    accuracy: float
    macro_f1: float
    calibration: CalibrationReport            # REQUIRED. R4 is enforced by this line
    human_band: tuple[float, float] | None = None
    human_band_statistic: str | None = None
    detail: BehaviourEvaluation | None = None

    def __post_init__(self) -> None:
        require_band_statistic(self.field, self.human_band, self.human_band_statistic)

    @property
    def ece(self) -> float:
        return self.calibration.ece

    @property
    def against_humans(self) -> dict[str, object] | None:
        """Accuracy and the human band side by side, with no verdict drawn from the pair.

        The band is a chance-corrected coefficient and accuracy is a raw proportion, so there is
        no defensible threshold between them; this field used to expose one as a boolean named
        `within_human_band`. None when Phase 2 measured no band for this field, which is an
        abstention rather than a failure (D48). D97.
        """
        return human_comparison(self.accuracy, self.human_band, self.human_band_statistic)


@dataclass(frozen=True)
class SystemScores:
    """One system - the model, or a baseline - across every field."""

    name: str
    fields: tuple[FieldScore, ...]
    scalar_mae: dict[str, float]

    def field(self, name: str) -> FieldScore:
        for score in self.fields:
            if score.field == name:
                return score
        raise KeyError(name)

    @property
    def macro_accuracy(self) -> float:
        return float(np.mean([f.accuracy for f in self.fields])) if self.fields else 0.0

    @property
    def macro_f1(self) -> float:
        return float(np.mean([f.macro_f1 for f in self.fields])) if self.fields else 0.0

    @property
    def macro_ece(self) -> float:
        """Mean ECE across fields. Available wherever macro_accuracy is. R4."""
        return float(np.mean([f.ece for f in self.fields])) if self.fields else 0.0


@dataclass(frozen=True)
class FrozenReport:
    """The comparison, with the provenance that makes it citable.

    `manifest_id` and `model_version` are not decoration. A result is a claim about a model
    trained on a particular partition, and a table without them cannot be traced back to
    either - which is what D73 made the split manifest append-only to protect.
    """

    label: str
    manifest_id: str
    model_version: str
    systems: tuple[SystemScores, ...]

    def system(self, name: str) -> SystemScores:
        for scores in self.systems:
            if scores.name == name:
                return scores
        raise KeyError(name)

    @property
    def model(self) -> SystemScores:
        return self.systems[0]

    def as_table(self) -> list[dict[str, object]]:
        """One row per system per field, with accuracy and ECE side by side. R4.

        The human band is given as its endpoints and the coefficient they came from. There is no
        verdict column: the band and accuracy are different quantities and the table states both
        rather than a comparison nobody can defend (D97).
        """
        return [
            {"system": system.name, "behaviour": score.behaviour, "field": score.field,
             "n": score.n, "accuracy": round(score.accuracy, 4),
             "macro_f1": round(score.macro_f1, 4), "ece": round(score.ece, 4),
             "human_band_low": (None if score.human_band is None
                                else round(score.human_band[0], 4)),
             "human_band_high": (None if score.human_band is None
                                 else round(score.human_band[1], 4)),
             "human_band_statistic": score.human_band_statistic}
            for system in self.systems for score in system.fields
        ]

    def summary(self) -> list[str]:
        return [f"{s.name}: acc={s.macro_accuracy:.3f}  macroF1={s.macro_f1:.3f}  "
                f"ECE={s.macro_ece:.4f}" for s in self.systems]


def _classification_heads(heads: dict[str, tuple[HeadSpec, ...]]):
    return [(behaviour, head)
            for behaviour, specs in sorted(heads.items()) for head in specs
            if head.kind is not HeadKind.SCALAR and not head.is_nonscorable_flag]


def score_system(
    name: str,
    probabilities: dict[str, np.ndarray],
    clips: LabelledClips,
    heads: dict[str, tuple[HeadSpec, ...]],
    *,
    human_bands: dict[str, tuple[float, float]] | None = None,
    human_band_statistic: str | None = None,
    scalar_mae: dict[str, float] | None = None,
    bootstrap: int = 1000,
    seed: int = 20260911,
) -> SystemScores:
    """One system's scores, with the non-scorable mask applied before anything is measured.

    Masked clips are dropped rather than scored, because a clip the rater could not read has no
    ground truth to be right or wrong about. Including them would move accuracy by an amount
    that depends on how the masked clips happen to be distributed, which at S2 is the largest
    share of the set.

    `human_band_statistic` names the coefficient `human_bands` was computed with. It is required
    whenever bands are supplied, and refused at construction if it is missing.
    """
    human_bands = human_bands or {}
    scores: list[FieldScore] = []

    for behaviour, head in _classification_heads(heads):
        if head.field not in probabilities:
            raise ReportError(
                f"{name} produced no prediction for {head.field}. Every classification field "
                f"is in the comparison, and a missing row is how a claim about all three "
                f"baselines stops being true.")

        mask = np.asarray(clips.scorable[behaviour], dtype=bool)
        truth = np.asarray(clips.targets[head.field])[mask].astype(int)
        predicted_proba = np.asarray(probabilities[head.field])[mask]
        if truth.size == 0:
            continue

        if head.kind is HeadKind.BINARY:
            # The harness's own path, so a binary field's numbers here are the same numbers
            # Phase 6 reports at each shift level.
            detail = evaluate_behaviour(
                head.field, predicted_proba, truth,
                human_band=human_bands.get(head.field),
                human_band_statistic=human_band_statistic, bootstrap=bootstrap, seed=seed)
            scores.append(FieldScore(
                behaviour=behaviour, field=head.field, kind=head.kind, n=detail.n,
                accuracy=detail.accuracy, macro_f1=detail.macro_f1,
                calibration=detail.calibration, human_band=detail.human_band,
                human_band_statistic=detail.human_band_statistic, detail=detail))
            continue

        calibration = multiclass_calibration_report(predicted_proba, truth)
        macro = macro_f1_multiclass(predicted_proba.argmax(axis=1), truth, head.width)
        scores.append(FieldScore(
            behaviour=behaviour, field=head.field, kind=head.kind, n=int(truth.size),
            accuracy=calibration.accuracy, macro_f1=macro if macro is not None else 0.0,
            calibration=calibration, human_band=human_bands.get(head.field),
            human_band_statistic=human_band_statistic))

    return SystemScores(name=name, fields=tuple(scores), scalar_mae=dict(scalar_mae or {}))


def compare(
    model_probabilities: dict[str, np.ndarray],
    baselines: Sequence[tuple[str, dict[str, np.ndarray]]],
    clips: LabelledClips,
    heads: dict[str, tuple[HeadSpec, ...]],
    *,
    manifest_id: str,
    model_version: str,
    label: str = "in-domain microteaching test",
    human_bands: dict[str, tuple[float, float]] | None = None,
    human_band_statistic: str | None = None,
    model_scalar_mae: dict[str, float] | None = None,
    required: Sequence[str] = REQUIRED_BASELINES,
    bootstrap: int = 1000,
    seed: int = 20260911,
) -> FrozenReport:
    """The frozen comparison. Refuses to exist without every required baseline.

    BUILD-SPEC calls all three required, and a comparison table quietly missing a row is how
    "compared against all three baselines" becomes true of a document and false of the work. So
    a missing baseline raises here rather than producing a shorter table.
    """
    supplied = [name for name, _ in baselines]
    missing = [name for name in required if name not in supplied]
    if missing:
        raise ReportError(
            f"the frozen report is missing {missing}. BUILD-SPEC Phase 4 requires the deep "
            f"model to be compared against {list(required)}, and a table without them cannot "
            f"support the claim that it was.")

    duplicated = sorted({name for name in supplied if supplied.count(name) > 1})
    if duplicated:
        raise ReportError(f"baseline named more than once: {duplicated}")

    systems = [score_system("model", model_probabilities, clips, heads,
                            human_bands=human_bands,
                            human_band_statistic=human_band_statistic,
                            scalar_mae=model_scalar_mae, bootstrap=bootstrap, seed=seed)]
    systems.extend(
        score_system(name, probabilities, clips, heads, human_bands=human_bands,
                     human_band_statistic=human_band_statistic,
                     bootstrap=bootstrap, seed=seed)
        for name, probabilities in baselines)

    return FrozenReport(label=label, manifest_id=manifest_id,
                        model_version=model_version, systems=tuple(systems))


def model_probabilities(model, features, keypoints,
                        heads: dict[str, tuple[HeadSpec, ...]]) -> dict[str, np.ndarray]:
    """The model's outputs as probabilities, in the same shape the baselines return.

    Sigmoid for binary heads and softmax for categorical ones, applied here rather than in the
    model, because the model's raw outputs are what the losses consume and applying an
    activation twice is the kind of error that shows up as a suspiciously flat calibration
    curve rather than as a crash.
    """
    import torch

    model.eval()
    with torch.no_grad():
        output = model(features, keypoints)

    out: dict[str, np.ndarray] = {}
    for behaviour, head in _classification_heads(heads):
        raw = output.raw[behaviour][head.field]
        if head.kind is HeadKind.BINARY:
            out[head.field] = torch.sigmoid(raw).numpy()
        else:
            out[head.field] = torch.softmax(raw, dim=-1).numpy()
    return out
