# -*- coding: utf-8 -*-
"""The frozen Phase 4 comparison.

Two acceptance tests from BUILD-SPEC live here: that the deep model is compared against all
three baselines, and that an evaluation result without a calibration field cannot exist. The
second is enforced by the type, so what is tested is that no path produces a score without one.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from praxis.annotation import CODEBOOK_V1
from praxis.behaviour.baselines import LabelledClips, build_baselines
from praxis.behaviour.heads import HeadKind, behaviour_heads
from praxis.behaviour.model import BehaviourModel, ModelShape
from praxis.behaviour.report import (
    FieldScore,
    ReportError,
    compare,
    model_probabilities,
    score_system,
)
from praxis.vocabulary import BEHAVIOUR_IDS

HEADS = {b: behaviour_heads(CODEBOOK_V1, b) for b in BEHAVIOUR_IDS}
FRAMES, DIM, JOINTS = 6, 5, 17
SHAPE = ModelShape(feature_dim=DIM, keypoints=JOINTS, keypoint_hidden=8, keypoint_out=4,
                   tcn_channels=8, tcn_kernel=3, tcn_dilations=(1, 2), tcn_dropout=0.0,
                   attention_dim=4)


def classification_fields():
    for behaviour, specs in HEADS.items():
        for head in specs:
            if head.kind is not HeadKind.SCALAR and not head.is_nonscorable_flag:
                yield behaviour, head


def clips(n: int = 24, *, seed: int = 0, unscorable: tuple[str, ...] = ()) -> LabelledClips:
    rng = np.random.default_rng(seed)
    positive = np.arange(n) % 2 == 0
    features = np.where(positive[:, None, None], 1.0, -1.0) + 0.05 * rng.normal(
        size=(n, FRAMES, DIM))
    keypoints = np.concatenate([
        np.where(positive[:, None, None, None], 0.8, 0.2)
        + 0.05 * rng.normal(size=(n, FRAMES, JOINTS, 2)),
        rng.uniform(0.5, 1.0, size=(n, FRAMES, JOINTS, 1))], axis=-1)

    targets = {}
    for _, head in classification_fields():
        targets[head.field] = (positive.astype(int) if head.kind is HeadKind.BINARY
                               else np.where(positive, 0, min(1, head.width - 1)))
    scorable = {b: np.full(n, b not in unscorable) for b in BEHAVIOUR_IDS}
    return LabelledClips(features, keypoints, targets, scorable)


def fitted_baselines(training: LabelledClips):
    built = build_baselines()
    for baseline in built:
        baseline.fit(training, HEADS)
    return built


def a_report(*, held_out=None, names=None, bootstrap: int = 0):
    training = clips(40, seed=1)
    held_out = held_out if held_out is not None else clips(16, seed=2)
    built = fitted_baselines(training)
    chosen = [b for b in built if names is None or b.name in names]

    torch.manual_seed(0)
    model = BehaviourModel(SHAPE, CODEBOOK_V1)
    probs = model_probabilities(
        model, torch.tensor(held_out.features, dtype=torch.float32),
        torch.tensor(held_out.keypoints, dtype=torch.float32), HEADS)

    return compare(
        probs,
        [(b.name, b.predict_proba(held_out)) for b in chosen],
        held_out, HEADS,
        manifest_id="01JMANIFEST0000000000000000",
        model_version="b-resnet50-tcn-v1",
        bootstrap=bootstrap)


class TestAllThreeBaselines:
    """BUILD-SPEC's acceptance test."""

    def test_the_report_carries_the_model_and_all_three_baselines(self) -> None:
        report = a_report()
        assert [s.name for s in report.systems] == [
            "model", "majority_class", "keypoints_gbdt", "frame_averaged_resnet"]
        assert report.model.name == "model"

    def test_a_missing_baseline_is_refused_not_omitted(self) -> None:
        """A table quietly missing a row is how a claim about all three stops being true."""
        with pytest.raises(ReportError, match="missing"):
            a_report(names={"majority_class", "keypoints_gbdt"})

    def test_a_duplicated_baseline_is_refused(self) -> None:
        held_out = clips(16, seed=2)
        built = fitted_baselines(clips(40, seed=1))
        entries = [(b.name, b.predict_proba(held_out)) for b in built]
        entries.append(entries[0])

        torch.manual_seed(0)
        model = BehaviourModel(SHAPE, CODEBOOK_V1)
        probs = model_probabilities(
            model, torch.tensor(held_out.features, dtype=torch.float32),
            torch.tensor(held_out.keypoints, dtype=torch.float32), HEADS)

        with pytest.raises(ReportError, match="more than once"):
            compare(probs, entries, held_out, HEADS, manifest_id="M", model_version="V")


class TestR4:
    """Calibration is reported wherever accuracy is. No exception for categorical fields."""

    def test_every_field_of_every_system_carries_a_calibration_report(self) -> None:
        report = a_report()
        for system in report.systems:
            assert system.fields, system.name
            for score in system.fields:
                assert score.calibration is not None, f"{system.name}/{score.field}"
                assert np.isfinite(score.ece), f"{system.name}/{score.field}"

    def test_a_field_score_cannot_be_built_without_calibration(self) -> None:
        """Enforced by the type, which is what makes R4 structural rather than procedural."""
        required = {f.name for f in dataclasses.fields(FieldScore) if
                    f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING}
        assert "calibration" in required

    def test_macro_ece_is_available_wherever_macro_accuracy_is(self) -> None:
        for system in a_report().systems:
            assert np.isfinite(system.macro_accuracy)
            assert np.isfinite(system.macro_ece)

    def test_the_table_puts_accuracy_and_ece_side_by_side(self) -> None:
        rows = a_report().as_table()
        assert rows
        for row in rows:
            assert "accuracy" in row and "ece" in row

    def test_categorical_fields_are_in_the_table_with_an_ece(self) -> None:
        """The five categorical fields are where R4 would quietly acquire an exception."""
        report = a_report()
        categorical = {head.field for _, head in classification_fields()
                       if head.kind is HeadKind.CATEGORICAL}
        assert categorical

        scored = {s.field for s in report.model.fields if s.kind is HeadKind.CATEGORICAL}
        assert scored == categorical
        for field in categorical:
            assert np.isfinite(report.model.field(field).ece)


class TestMasking:
    def test_unscorable_clips_are_dropped_before_anything_is_measured(self) -> None:
        """A clip the rater could not read has no ground truth to be right or wrong about."""
        held_out = clips(16, seed=2, unscorable=("B3",))
        report = a_report(held_out=held_out)

        b3 = [s for s in report.model.fields if s.behaviour == "B3"]
        assert b3 == [], "B3 is wholly unscorable here and must not be scored at all"
        assert any(s.behaviour == "B1" for s in report.model.fields)

    def test_the_n_reported_is_the_scorable_count(self) -> None:
        held_out = clips(16, seed=2)
        report = a_report(held_out=held_out)
        assert report.model.field("b1_present").n == 16


class TestHumanBand:
    def _probabilities(self, n: int) -> dict[str, np.ndarray]:
        return {head.field: (np.ones(n) * 0.9 if head.kind is HeadKind.BINARY
                             else np.full((n, head.width), 1.0 / head.width))
                for _, head in classification_fields()}

    def test_the_band_is_carried_and_reported_beside_accuracy(self) -> None:
        """D6 withdrew the absolute accuracy target, so the figure is stated against what two
        trained humans achieved rather than against a number the candidate does not control.
        D97 then settled that "against" means side by side: the band is a chance-corrected
        coefficient and accuracy is a proportion, so the two are reported and not compared."""
        held_out = clips(16, seed=2)
        scored = score_system("model", self._probabilities(16), held_out, HEADS,
                              human_bands={"b1_present": (0.4, 0.6)},
                              human_band_statistic="krippendorff_alpha", bootstrap=0)

        field = scored.field("b1_present")
        assert field.human_band == (0.4, 0.6)
        assert field.human_band_statistic == "krippendorff_alpha"
        assert field.against_humans["accuracy"] == field.accuracy
        assert field.against_humans["commensurable"] is False

    def test_a_band_with_no_named_coefficient_is_refused(self) -> None:
        """An interval of two bare floats does not say whether it is alpha, a weighted kappa or
        an ICC, and those have different scales. Unreadable is worse than absent."""
        held_out = clips(16, seed=2)
        with pytest.raises(ValueError, match="which coefficient produced it"):
            score_system("model", self._probabilities(16), held_out, HEADS,
                         human_bands={"b1_present": (0.4, 0.6)}, bootstrap=0)

    def test_a_field_with_no_measured_band_abstains_rather_than_failing(self) -> None:
        """D48: an unavailable check abstains, and abstention is its own verdict."""
        assert a_report().model.field("b1_present").against_humans is None

    def test_the_table_has_no_verdict_column(self) -> None:
        """A boolean column would be read as the result whatever the prose beside it said."""
        row = a_report().as_table()[0]
        assert "within_human_band" not in row
        assert {"human_band_low", "human_band_high", "human_band_statistic"} <= set(row)


class TestRefusals:
    def test_a_system_that_omits_a_field_is_refused(self) -> None:
        held_out = clips(8, seed=3)
        partial = {head.field: (np.full(8, 0.5) if head.kind is HeadKind.BINARY
                                else np.full((8, head.width), 1.0 / head.width))
                   for _, head in classification_fields()}
        partial.pop("b1_present")

        with pytest.raises(ReportError, match="no prediction for b1_present"):
            score_system("a_baseline", partial, held_out, HEADS, bootstrap=0)


def test_model_probabilities_are_in_the_unit_interval_and_normalised() -> None:
    """Applied here rather than in the model, because the losses consume raw outputs and a
    double-applied activation shows up as a suspiciously flat calibration curve."""
    held_out = clips(8, seed=4)
    torch.manual_seed(0)
    model = BehaviourModel(SHAPE, CODEBOOK_V1)
    probs = model_probabilities(
        model, torch.tensor(held_out.features, dtype=torch.float32),
        torch.tensor(held_out.keypoints, dtype=torch.float32), HEADS)

    for _, head in classification_fields():
        values = probs[head.field]
        assert ((values >= 0.0) & (values <= 1.0)).all(), head.field
        if head.kind is HeadKind.CATEGORICAL:
            assert np.allclose(values.sum(axis=1), 1.0, atol=1e-5), head.field


def test_the_report_records_the_partition_it_was_measured_on() -> None:
    """A result is a claim about a model trained on a particular split. A table without the
    manifest id cannot be traced to one, which is what D73 made the manifest immutable for."""
    report = a_report()
    assert report.manifest_id == "01JMANIFEST0000000000000000"
    assert report.model_version == "b-resnet50-tcn-v1"
