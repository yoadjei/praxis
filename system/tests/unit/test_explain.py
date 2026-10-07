# -*- coding: utf-8 -*-
"""Explanation and its fidelity checks.

BUILD-SPEC Phase 7's acceptance tests are all here:

- the sanity checks run end to end and produce a pass/fail per method;
- a deliberately broken explainer, returning constant maps, **fails** the parameter
  randomisation test - and if it passes, the test is wrong;
- the interface reads the method from configuration, so swapping requires no code change.

The constant explainer is a specified test double, not an introduced defect: Adebayo et al.'s
check is only meaningful if it can be shown to catch something, and a method insensitive to the
model is exactly what it is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from torch import nn

from praxis.annotation import CODEBOOK_V1
from praxis.behaviour.model import BehaviourModel, ModelShape
from praxis.explain import base
from praxis.explain.base import (
    FORBIDDEN,
    ClipInput,
    ExplainError,
    Explanation,
    normalise,
    select,
)
from praxis.explain.fidelity import (
    FidelityError,
    compare_maps,
    data_randomisation_test,
    deletion_insertion_auc,
    parameter_randomisation_test,
    reinitialise,
    run_fidelity,
)
from praxis.explain.gradcam import GradCAM, resolve_layer
from praxis.explain.intrinsic import IntrinsicExplainer

FRAMES, DIM, JOINTS, SIDE = 4, 16, 17, 16
SHAPE = ModelShape(feature_dim=DIM, keypoints=JOINTS, keypoint_hidden=8, keypoint_out=4,
                   tcn_channels=8, tcn_kernel=3, tcn_dilations=(1, 2), tcn_dropout=0.0,
                   attention_dim=4)


class TinyEncoder(nn.Module):
    """A stand-in frame encoder with a `layer4`, so Grad-CAM's mechanics can be tested
    without a ResNet-50 forward and backward pass per assertion."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.layer3 = nn.Conv2d(3, 8, 3, padding=1)
        self.layer4 = nn.Conv2d(8, 12, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(12, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        return self.fc(self.pool(x).flatten(1))


@dataclass
class TinyBackbone:
    """The `FrozenBackbone` surface Grad-CAM actually uses."""

    encoder: nn.Module

    @property
    def module(self) -> nn.Module:
        return self.encoder

    def embed_for_explanation(self, crops: torch.Tensor) -> torch.Tensor:
        return self.encoder(crops)


@dataclass
class ConstantExplainer:
    """The broken method BUILD-SPEC requires the checks to catch.

    Returns the same map whatever the model does, which is what an unfaithful method looks
    like: it describes the input, or nothing at all, rather than the computation.
    """

    name: str = field(default="constant", init=False)
    value: float = 0.5

    def explain(self, clip: ClipInput, behaviour: str, field_name: str) -> Explanation:
        maps = np.full((FRAMES, SIDE, SIDE), self.value)
        return Explanation(method=self.name, behaviour=behaviour, field=field_name,
                           temporal=maps.reshape(FRAMES, -1).mean(axis=1), spatial=maps)


@pytest.fixture
def model() -> BehaviourModel:
    torch.manual_seed(0)
    return BehaviourModel(SHAPE, CODEBOOK_V1)


@pytest.fixture
def backbone() -> TinyBackbone:
    torch.manual_seed(1)
    return TinyBackbone(TinyEncoder(DIM))


@pytest.fixture
def clip() -> ClipInput:
    torch.manual_seed(2)
    return ClipInput(keypoints=torch.randn(FRAMES, JOINTS, 3),
                     crops=torch.rand(FRAMES, 3, SIDE, SIDE),
                     features=torch.randn(FRAMES, DIM))


# ---------------------------------------------------------------------------
# The registry, which is what "swapping methods requires no code change" means.
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_the_method_is_chosen_by_name_from_configuration(self, config, model,
                                                             backbone) -> None:
        registry = dict(base._REGISTRY)
        try:
            base._REGISTRY.clear()
            base.register(GradCAM(backbone=backbone, model=model))
            base.register(IntrinsicExplainer(model=model))

            chosen = select(config.explain.method, fallback=config.explain.fallback)
            assert chosen.name == config.explain.method == "gradcam"
        finally:
            base._REGISTRY.clear()
            base._REGISTRY.update(registry)

    def test_withdrawing_a_method_falls_back_without_a_code_change(self, model) -> None:
        """BUILD-SPEC's decision rule: a method that fails a sanity check is removed from the
        interface and the intrinsic explanation is used instead."""
        registry = dict(base._REGISTRY)
        try:
            base._REGISTRY.clear()
            base.register(IntrinsicExplainer(model=model))
            assert select("gradcam", fallback="intrinsic").name == "intrinsic"
        finally:
            base._REGISTRY.clear()
            base._REGISTRY.update(registry)

    def test_a_config_naming_two_unavailable_methods_fails_loudly(self) -> None:
        """An interface with no explanation must say so rather than show a detection as
        though one had been produced."""
        registry = dict(base._REGISTRY)
        try:
            base._REGISTRY.clear()
            with pytest.raises(ExplainError, match="neither"):
                select("gradcam", fallback="intrinsic")
        finally:
            base._REGISTRY.clear()
            base._REGISTRY.update(registry)

    def test_guided_gradcam_cannot_be_registered_or_selected(self) -> None:
        """It failed Adebayo's parameter randomisation test while producing the best-looking
        maps. There is deliberately nowhere to put it."""
        assert "guided_gradcam" in FORBIDDEN
        with pytest.raises(ExplainError, match="forbidden"):
            select("guided_gradcam")

    def test_no_guided_method_is_implemented_anywhere_in_the_package(self) -> None:
        import pathlib
        package = pathlib.Path(base.__file__).parent
        for source in package.glob("*.py"):
            text = source.read_text(encoding="utf-8").lower()
            assert "def guided" not in text, source.name
            assert "class guidedbackprop" not in text, source.name


# ---------------------------------------------------------------------------
# The acceptance test: a broken explainer must fail.
# ---------------------------------------------------------------------------

def randomiser(module: nn.Module, layers: list[str]):
    """Cascading randomisation with an exact restore, as the check requires."""
    original = {k: v.detach().clone() for k, v in module.state_dict().items()}

    def randomise(name: str) -> None:
        reinitialise(resolve_layer(module, name))

    def restore() -> None:
        module.load_state_dict(original)

    return randomise, restore, layers


class TestParameterRandomisation:
    def test_a_constant_explainer_fails(self, model, backbone, clip) -> None:
        """BUILD-SPEC: "a deliberately broken explainer, returning constant maps, fails the
        parameter randomisation test. If it passes, the test is wrong." """
        randomise, restore, layers = randomiser(backbone.module, ["layer4", "layer3"])

        result = parameter_randomisation_test(
            ConstantExplainer(), clip, "B1", "b1_present",
            randomise=randomise, restore=restore, layers=layers, threshold=0.6)

        assert not result.passed
        assert result.worst_similarity == pytest.approx(1.0)
        assert "describing the input" in result.detail

    def test_gradcam_is_measured_rather_than_assumed(self, model, backbone, clip) -> None:
        """The point of Phase 7: Grad-CAM passing on Inception v3 over ImageNet is not
        evidence about a frozen encoder plus a TCN over classroom video. What is asserted here
        is that the check runs and returns a verdict, not what the verdict is - that needs the
        corpus, and claiming it from a synthetic fixture would be manufacturing a result."""
        explainer = GradCAM(backbone=backbone, model=model, target_layer="backbone.layer4")
        randomise, restore, layers = randomiser(backbone.module, ["layer4", "layer3"])

        result = parameter_randomisation_test(
            explainer, clip, "B1", "b1_present",
            randomise=randomise, restore=restore, layers=layers, threshold=0.6)

        assert isinstance(result.passed, bool)
        assert len(result.stages) == 2
        assert [s.layer for s in result.stages] == ["layer4", "layer3"]
        assert np.isfinite(result.worst_similarity)

    def test_the_model_is_restored_even_when_the_explainer_raises(self, backbone,
                                                                  clip) -> None:
        """A randomised model left behind would silently poison every later check."""
        before = {k: v.clone() for k, v in backbone.module.state_dict().items()}
        randomise, restore, layers = randomiser(backbone.module, ["layer4"])

        class Exploding:
            name = "exploding"

            def explain(self, *_args):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            parameter_randomisation_test(
                Exploding(), clip, "B1", "b1_present",
                randomise=randomise, restore=restore, layers=layers, threshold=0.6)

        for name, tensor in backbone.module.state_dict().items():
            assert torch.equal(tensor, before[name]), name

    def test_the_worst_stage_decides_not_the_last(self, model, backbone, clip) -> None:
        """A method that recovers at the final stage has still failed."""
        randomise, restore, layers = randomiser(backbone.module, ["layer4", "layer3"])
        result = parameter_randomisation_test(
            ConstantExplainer(), clip, "B1", "b1_present",
            randomise=randomise, restore=restore, layers=layers, threshold=0.6)

        assert result.worst_similarity == max(s.similarity.worst for s in result.stages)

    def test_a_check_with_no_layers_is_refused(self, model, backbone, clip) -> None:
        """It would pass unconditionally, which is the quiet way an invariant is deleted."""
        randomise, restore, _ = randomiser(backbone.module, [])
        with pytest.raises(FidelityError, match="nothing to randomise"):
            parameter_randomisation_test(
                ConstantExplainer(), clip, "B1", "b1_present",
                randomise=randomise, restore=restore, layers=[], threshold=0.6)


class TestDataRandomisation:
    def test_a_constant_explainer_fails_this_one_too(self, clip) -> None:
        result = data_randomisation_test(
            ConstantExplainer(), ConstantExplainer(), clip, "B1", "b1_present",
            threshold=0.6)
        assert not result.passed
        assert "whether the model learned the task or not" in result.detail

    def test_two_differently_trained_models_give_a_verdict(self, clip) -> None:
        torch.manual_seed(3)
        trained = IntrinsicExplainer(model=BehaviourModel(SHAPE, CODEBOOK_V1))
        torch.manual_seed(99)
        permuted = IntrinsicExplainer(model=BehaviourModel(SHAPE, CODEBOOK_V1))

        result = data_randomisation_test(trained, permuted, clip, "B1", "b1_present",
                                         threshold=0.6)
        assert result.check == "data_randomisation"
        assert isinstance(result.passed, bool)


class TestReport:
    def test_the_report_states_a_pass_or_fail_per_check(self, model, backbone,
                                                        clip) -> None:
        randomise, restore, layers = randomiser(backbone.module, ["layer4"])
        report = run_fidelity(
            ConstantExplainer(), clip, "B1", "b1_present",
            randomise=randomise, restore=restore, layers=layers, threshold=0.6,
            permuted_explainer=ConstantExplainer())

        assert report.complete
        assert not report.passed
        assert {c.check for c in report.sanity_checks} == {
            "parameter_randomisation", "data_randomisation"}
        assert all("FAIL" in line for line in report.summary())

    def test_a_check_that_did_not_run_is_not_a_pass(self, model, backbone, clip) -> None:
        """D48: an unavailable check abstains, and abstention is its own verdict. A partial
        evaluation must not be quotable as a clean one."""
        randomise, restore, layers = randomiser(backbone.module, ["layer4"])
        report = run_fidelity(
            GradCAM(backbone=backbone, model=model), clip, "B1", "b1_present",
            randomise=randomise, restore=restore, layers=layers, threshold=0.6)

        assert report.data_randomisation is None
        assert not report.complete
        assert any("NOT RUN" in line for line in report.summary())


# ---------------------------------------------------------------------------
# Similarity, which every verdict rests on.
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical_maps_are_maximally_similar(self) -> None:
        a = np.random.default_rng(0).random((8, 8))
        similarity = compare_maps(a, a)
        assert similarity.rank_correlation == pytest.approx(1.0)
        assert similarity.ssim == pytest.approx(1.0)

    def test_two_constant_maps_are_identical_not_undefined(self) -> None:
        """Spearman's rho is NaN at zero variance, and a NaN in the verdict would read as "no
        result" rather than as the perfect similarity it represents - which is exactly the
        case the broken-explainer acceptance test produces."""
        similarity = compare_maps(np.full((8, 8), 0.5), np.full((8, 8), 0.5))
        assert similarity.rank_correlation == 1.0
        assert similarity.worst == pytest.approx(1.0)

    def test_a_constant_map_against_a_varying_one_is_zero_not_nan(self) -> None:
        similarity = compare_maps(np.full((8, 8), 0.5),
                                  np.random.default_rng(1).random((8, 8)))
        assert np.isfinite(similarity.rank_correlation)
        assert similarity.rank_correlation == 0.0

    def test_the_worst_of_the_two_measures_decides(self) -> None:
        """Averaging would let a method hide a near-identical structural match behind an
        uncorrelated ranking."""
        similarity = compare_maps(np.full((8, 8), 0.5), np.full((8, 8), 0.5))
        assert similarity.worst == max(similarity.rank_correlation, similarity.ssim)

    def test_a_one_dimensional_attribution_can_still_be_compared(self) -> None:
        """The intrinsic method's temporal attribution is 1-D, and refusing to compare it
        would exempt the fallback from the checks."""
        a = np.linspace(0, 1, 16)
        assert compare_maps(a, a).ssim == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_sizes_are_refused(self) -> None:
        with pytest.raises(FidelityError, match="cannot be compared"):
            compare_maps(np.zeros(8), np.zeros(9))


class TestDeletionInsertion:
    def test_a_faithful_map_deletes_the_score_and_inserts_it_back(self) -> None:
        """A map ranking the elements that carry the prediction highest gives a low deletion
        AUC and a high insertion one."""
        subject = np.array([0.0, 0.0, 10.0, 0.0])
        baseline = np.zeros(4)
        explanation = Explanation(method="m", behaviour="B1", field="b1_present",
                                  temporal=np.array([0.0, 0.0, 1.0, 0.0]))

        deletion, insertion = deletion_insertion_auc(
            explanation, lambda x: float(x.sum()), baseline, subject, steps=5)
        assert deletion < insertion

    def test_a_misleading_map_scores_the_other_way_round(self) -> None:
        subject = np.array([0.0, 0.0, 10.0, 0.0])
        baseline = np.zeros(4)
        misleading = Explanation(method="m", behaviour="B1", field="b1_present",
                                 temporal=np.array([1.0, 1.0, 0.0, 1.0]))

        deletion, insertion = deletion_insertion_auc(
            misleading, lambda x: float(x.sum()), baseline, subject, steps=5)
        assert deletion > insertion

    def test_a_map_that_does_not_describe_the_input_is_refused(self) -> None:
        explanation = Explanation(method="m", behaviour="B1", field="b1_present",
                                  temporal=np.zeros(3))
        with pytest.raises(FidelityError, match="same elements"):
            deletion_insertion_auc(explanation, lambda x: 0.0,
                                   np.zeros(4), np.zeros(4), steps=5)

    def test_fewer_than_two_steps_is_not_a_curve(self) -> None:
        explanation = Explanation(method="m", behaviour="B1", field="b1_present",
                                  temporal=np.zeros(4))
        with pytest.raises(FidelityError, match="not a curve"):
            deletion_insertion_auc(explanation, lambda x: 0.0,
                                   np.zeros(4), np.zeros(4), steps=1)


# ---------------------------------------------------------------------------
# The methods themselves.
# ---------------------------------------------------------------------------

class TestGradCAM:
    def test_it_produces_one_heatmap_per_frame(self, model, backbone, clip) -> None:
        explanation = GradCAM(backbone=backbone, model=model).explain(
            clip, "B1", "b1_present")

        assert explanation.spatial.shape == (FRAMES, SIDE, SIDE)
        assert explanation.temporal.shape == (FRAMES,)
        assert np.isfinite(explanation.spatial).all()

    def test_the_map_is_non_negative(self, model, backbone, clip) -> None:
        """Grad-CAM applies a ReLU: the question is which regions raise the score, and a
        negative value rendered on a heatmap reads as evidence against, which it is not."""
        explanation = GradCAM(backbone=backbone, model=model).explain(
            clip, "B1", "b1_present")
        assert (explanation.spatial >= 0).all()

    def test_a_categorical_field_explains_the_level_that_was_predicted(self, model,
                                                                        backbone,
                                                                        clip) -> None:
        """Attributing a level the model did not choose would describe a decision that never
        happened."""
        explanation = GradCAM(backbone=backbone, model=model).explain(
            clip, "B2", "b2_dominant")
        assert explanation.spatial.shape[0] == FRAMES

    def test_it_refuses_to_work_from_cached_features(self, model, backbone) -> None:
        """A spatial explanation is a statement about pixels, and the cache has thrown them
        away. Returning a blank map would answer the question wrongly."""
        featureless = ClipInput(keypoints=torch.randn(FRAMES, JOINTS, 3),
                                features=torch.randn(FRAMES, DIM))
        with pytest.raises(ExplainError, match="needs the clip's pixels"):
            GradCAM(backbone=backbone, model=model).explain(
                featureless, "B1", "b1_present")

    def test_an_unknown_target_layer_is_refused_with_the_alternatives(self,
                                                                       backbone) -> None:
        """A silently wrong layer produces maps that look entirely reasonable."""
        with pytest.raises(ExplainError, match="no layer at"):
            resolve_layer(backbone.module, "backbone.layer9")

    def test_the_backbone_prefix_in_the_config_resolves(self, backbone, config) -> None:
        assert config.explain.gradcam.target_layer == "backbone.layer4"
        assert resolve_layer(backbone.module, "backbone.layer4") is backbone.module.layer4


class TestIntrinsic:
    def test_it_returns_the_attention_the_model_actually_used(self, model, clip) -> None:
        explanation = IntrinsicExplainer(model=model).explain(clip, "B1", "b1_present")

        assert explanation.temporal.shape == (FRAMES,)
        assert explanation.temporal.sum() == pytest.approx(1.0, abs=1e-5), (
            "the pooling weights are a distribution over frames; rescaling them would turn "
            "'the model weighted every frame about equally' into a false claim of focus")

    def test_it_attributes_to_keypoints(self, model, clip) -> None:
        explanation = IntrinsicExplainer(model=model).explain(clip, "B1", "b1_present")
        assert explanation.keypoint.shape == (FRAMES, JOINTS)
        assert np.isfinite(explanation.keypoint).all()

    def test_it_produces_no_spatial_map(self, model, clip) -> None:
        """Producing one would mean inventing spatial structure neither the attention nor the
        keypoint gradients contain - the 'looks plausible' failure Adebayo et al. warn of."""
        assert IntrinsicExplainer(model=model).explain(
            clip, "B1", "b1_present").spatial is None

    def test_it_needs_cached_features(self, model) -> None:
        pixels_only = ClipInput(keypoints=torch.randn(FRAMES, JOINTS, 3),
                                crops=torch.rand(FRAMES, 3, SIDE, SIDE))
        with pytest.raises(ExplainError, match="cached backbone features"):
            IntrinsicExplainer(model=model).explain(pixels_only, "B1", "b1_present")


class TestExplanationContract:
    def test_a_heatmap_must_cover_the_frames_it_is_shown_against(self) -> None:
        with pytest.raises(ExplainError, match="worse than no heatmap"):
            Explanation(method="m", behaviour="B1", field="b1_present",
                        temporal=np.zeros(4), spatial=np.zeros((3, 8, 8)))

    def test_normalise_of_a_constant_map_is_zero_not_nan(self) -> None:
        """A NaN would propagate into the similarity scores and read as a missing result
        rather than as the failure it is."""
        assert np.array_equal(normalise(np.full((4, 4), 7.0)), np.zeros((4, 4)))

    def test_the_comparable_map_prefers_the_spatial_one(self) -> None:
        explanation = Explanation(method="m", behaviour="B1", field="b1_present",
                                  temporal=np.zeros(2), spatial=np.ones((2, 3, 3)))
        assert explanation.comparable_map().size == 18
