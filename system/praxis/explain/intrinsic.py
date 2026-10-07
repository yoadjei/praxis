# -*- coding: utf-8 -*-
"""The fallback explanation, faithful by construction.

BUILD-SPEC Phase 7 step 3: expose the temporal attention weights from the pooling layer and
per-keypoint gradients. These are **the model's actual computation**, not a post-hoc
reconstruction of it, which is why they cannot fail a faithfulness check in the way a post-hoc
method can: there is no separate explanation to disagree with the computation.

That is also the limit of the claim. Faithful does not mean informative. Attention over frames
says which frames the pooling layer weighted; it does not say the model was "looking at" the
gesture, and the thesis should not say so either. What it supports is the narrower statement
the routing gate needs: this clip's decision rested mostly on these seconds.

**No spatial map, and that is not a gap to be filled.** The attention is over time and the
keypoint gradients are over joints. Producing a pixel heatmap from either would mean inventing
spatial structure neither contains, which is exactly the "looks plausible" failure Adebayo et
al. warn about - they show an edge detector producing maps that resemble popular saliency
methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from praxis.behaviour.heads import HeadKind, HeadSpec
from praxis.explain.base import ClipInput, ExplainError, Explanation, normalise


@dataclass
class IntrinsicExplainer:
    """Attention weights and keypoint gradients, read from the model as it ran."""

    name: str = field(default="intrinsic", init=False)
    model: object = None
    normalise_map: bool = True

    def explain(self, clip: ClipInput, behaviour: str, field_name: str) -> Explanation:
        if self.model is None:
            raise ExplainError("intrinsic needs a model")

        features = clip.require_features(self.name).clone()
        keypoints = clip.keypoints.clone().requires_grad_(True)
        head = self._head(behaviour, field_name)

        self.model.eval()
        outputs = self.model(features.unsqueeze(0), keypoints.unsqueeze(0))

        raw = outputs.raw[behaviour][field_name]
        score = (raw[0, raw[0].argmax()] if head.kind is HeadKind.CATEGORICAL
                 else (raw[0] if raw.ndim == 1 else raw[0, 0]))

        self.model.zero_grad(set_to_none=True)
        score.backward()

        attention = outputs.attention[0].detach().cpu().numpy()
        if keypoints.grad is None:
            # Not zero-filled. A None gradient means the score did not depend on the pose at
            # all, which is a defect in the model or in this call, and a map of zeros would
            # render it as "the pose did not matter here" - a claim about the clip rather than
            # about a broken computation.
            raise ExplainError(
                f"no gradient reached the keypoints for {behaviour}/{field_name}, so the "
                f"score does not depend on the pose. The keypoint branch is disconnected.")

        # `keypoints` is the unbatched (T, K, 3) leaf, so its gradient is too; the batch axis
        # exists only inside the forward pass. Summed over x, y and confidence, because the
        # question is which joint mattered rather than which of its three channels, and
        # reporting them separately invites reading a confidence gradient as evidence about
        # the body.
        keypoint_attribution = keypoints.grad.abs().sum(dim=-1).detach().cpu().numpy()

        if self.normalise_map:
            keypoint_attribution = normalise(keypoint_attribution)

        return Explanation(
            method=self.name,
            behaviour=behaviour,
            field=field_name,
            # The attention weights themselves, not a normalised copy. They are already a
            # distribution over frames, and rescaling them to fill [0, 1] would turn "the model
            # weighted every frame about equally" into "the model concentrated on frame 12".
            temporal=attention,
            keypoint=keypoint_attribution,
        )

    def _head(self, behaviour: str, field_name: str) -> HeadSpec:
        for spec in self.model.head_specs[behaviour]:
            if spec.field == field_name:
                return spec
        raise ExplainError(f"{behaviour} has no field {field_name!r}")
