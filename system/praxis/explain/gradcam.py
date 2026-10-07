# -*- coding: utf-8 -*-
"""Grad-CAM over the frame encoder.

Adebayo et al. (2018) found Grad-CAM **passes** their sanity checks on Inception v3 over
ImageNet, which is why BUILD-SPEC chooses it. Passing there does not transfer automatically to
a frozen ResNet-50 plus a TCN over classroom video, so `fidelity.py` runs the checks again on
this model and this data, and the decision rule is fixed in advance: fail either check and the
method is withdrawn from the interface.

**Why the backbone has to be in the graph here, when Stage B never puts it there.** A spatial
explanation is a statement about pixels, and the cached features have thrown the pixels away.
So the explanation pass runs the crops through the backbone with gradients flowing - the
parameters stay frozen, which is a fact about `requires_grad` on the weights, not about whether
gradients can flow through the activations. This is why `ClipInput.require_crops` exists: asking
for a spatial map from cached features is a question this method cannot answer, and returning a
blank one would answer it wrongly.

**The target layer is the last convolutional block**, `layer4` by default and configurable.
Grad-CAM's premise is that the final convolutional layer holds the best compromise between
semantics and spatial resolution; taking it earlier gives sharper maps of less meaningful
features, which is the "looks better" trap Phase 7 is written against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from praxis.behaviour.heads import HeadKind, HeadSpec
from praxis.explain.base import ClipInput, ExplainError, Explanation, normalise


def resolve_layer(module, path: str):
    """The submodule a dotted path names, refusing one that is not there.

    `backbone.layer4` in the config refers to the backbone's own `layer4`, so the `backbone.`
    prefix is stripped when the module handed in is already the backbone. A silently wrong
    layer would produce maps that look entirely reasonable, which is the whole problem.
    """
    remaining = path
    if remaining.startswith("backbone."):
        remaining = remaining[len("backbone."):]

    target = module
    for part in remaining.split("."):
        if not hasattr(target, part):
            available = [name for name, _ in target.named_children()]
            raise ExplainError(
                f"no layer at {path!r}: {type(target).__name__} has no {part!r}. "
                f"Available: {available}")
        target = getattr(target, part)
    return target


@dataclass
class GradCAM:
    """Grad-CAM for one behaviour field, over a clip's frames.

    Args:
        backbone: the frozen `FrozenBackbone`. Its parameters stay frozen; only activations
            carry gradients.
        model: the Stage B `BehaviourModel`, whose head score is differentiated.
        target_layer: dotted path into the backbone, from `explain.gradcam.target_layer`.
        normalise_map: whether to scale each map to [0, 1], from `explain.gradcam.normalise`.
    """

    name: str = field(default="gradcam", init=False)
    backbone: object = None
    model: object = None
    target_layer: str = "backbone.layer4"
    normalise_map: bool = True

    def _score(self, outputs, behaviour: str, head: HeadSpec):
        """The scalar this explanation is of.

        For a categorical field it is the logit of the predicted level, not of a fixed one: an
        explanation is an account of the prediction that was made, and attributing the score of
        a level the model did not choose would describe a decision that never happened.
        """
        raw = outputs.raw[behaviour][head.field]
        if head.kind is HeadKind.CATEGORICAL:
            return raw[0, raw[0].argmax()]
        return raw[0] if raw.ndim == 1 else raw[0, 0]

    def explain(self, clip: ClipInput, behaviour: str, field_name: str) -> Explanation:
        import torch

        if self.backbone is None or self.model is None:
            raise ExplainError("gradcam needs a backbone and a model")

        crops = clip.require_crops(self.name)
        head = self._head(behaviour, field_name)

        module = resolve_layer(self.backbone.module, self.target_layer)
        activations: list = []
        gradients: list = []

        forward_handle = module.register_forward_hook(
            lambda _m, _i, out: activations.append(out))
        backward_handle = module.register_full_backward_hook(
            lambda _m, _gi, grad_out: gradients.append(grad_out[0]))

        try:
            self.model.eval()
            # No `no_grad`: the parameters are frozen, and what is needed is the gradient with
            # respect to the activations, which frozen parameters do not prevent.
            features = self.backbone.embed_for_explanation(crops)
            outputs = self.model(features.unsqueeze(0),
                                 clip.keypoints.unsqueeze(0))
            score = self._score(outputs, behaviour, head)

            self.model.zero_grad(set_to_none=True)
            score.backward()
        finally:
            forward_handle.remove()
            backward_handle.remove()

        if not activations or not gradients:
            raise ExplainError(
                f"no activations reached {self.target_layer!r}; the layer is not on the path "
                f"the forward pass takes")

        # (T, C, H, W) for both. Channel weights are the globally averaged gradients, which is
        # Grad-CAM's definition: how much this channel's presence anywhere raises the score.
        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))

        maps = cam.detach().cpu().numpy()
        if self.normalise_map:
            maps = np.stack([normalise(frame) for frame in maps])

        return Explanation(
            method=self.name,
            behaviour=behaviour,
            field=field_name,
            # Per-frame evidence strength is the map's own mass. A frame the model drew nothing
            # from scores zero, which is what makes the temporal series comparable with the
            # intrinsic method's attention weights.
            temporal=maps.reshape(maps.shape[0], -1).mean(axis=1),
            spatial=maps,
        )

    def _head(self, behaviour: str, field_name: str) -> HeadSpec:
        for spec in self.model.head_specs[behaviour]:
            if spec.field == field_name:
                return spec
        raise ExplainError(f"{behaviour} has no field {field_name!r}")
