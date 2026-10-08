# -*- coding: utf-8 -*-
"""The frozen frame encoder: teacher crops in, per-frame embeddings out.

BUILD-SPEC Phase 4 Stage A. A ResNet-50 pretrained on ImageNet, frozen, in eval mode, under
`no_grad`, producing a 2048-dimensional embedding per frame. Stage B trains on the cached
output, which is what makes an ensemble affordable on this estate (D2, D13).

**Nothing here downloads anything.** `torchvision.models.resnet50(weights=...)` fetches the
checkpoint on a cache miss, and it honours neither `YOLO_OFFLINE` nor `HF_HUB_OFFLINE` - the
two variables `prepare_environment` sets, which cover ultralytics and huggingface and were
quietly assumed to cover this. They do not. So the architecture is always built with
`weights=None` and the parameters are loaded from a file that must already exist;
`scripts/vendor_weights.py` is the deliberate operator step that puts it there. D72.

**The weights are pinned by version, never by alias.** `DEFAULT` currently resolves to
IMAGENET1K_V2 and has already moved once for this architecture. A run pinned to an alias would
change every cached feature in the corpus after a routine dependency upgrade, with the config
hash that R7 rests on unchanged. `BackboneSection.weights_name_a_version` refuses the alias;
this module records what was actually loaded so the manifest can carry it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The classifier head the pretrained checkpoint carries. It is discarded: this encoder emits
# the penultimate activations, and ImageNet's thousand classes are not a vocabulary this
# project has any use for.
CLASSIFIER_ATTRIBUTE = "fc"


def version_of(arch: str, weights_name: str) -> str:
    """The one spelling of a backbone's identity, for everything that records or compares it.

    `resnet50/IMAGENET1K_V1` rather than `resnet50`. The feature cache stores this against every
    clip and refuses a mismatch on read, and it has to be able to name the configured backbone
    without loading the weights to ask one - so the composition lives here rather than only on
    the instance, and the two cannot drift.
    """
    return f"{arch}/{weights_name}"


def declared_version(config) -> str:
    """What `build_backbone(config)` would call itself, without building it."""
    return version_of(config.behaviour.backbone.arch, config.behaviour.backbone.weights)


class BackboneError(RuntimeError):
    """The frame encoder could not be built, with the reason distinguished."""


class WeightsMissing(BackboneError):
    """No file at the configured path. Vendoring is a deliberate operator step."""


class WeightsIncompatible(BackboneError):
    """A file exists and does not load into the configured architecture."""


@dataclass
class FrozenBackbone:
    """ResNet-50 over teacher crops, frozen, on CPU or CUDA.

    Construction is eager rather than lazy. The failure this guards against - absent or wrong
    weights - should surface when the pipeline is assembled, not forty minutes into a feature
    extraction run that has already written half a session to the cache.
    """

    weights_path: Path
    arch: str
    weights_name: str
    output_dim: int
    normalise_mean: tuple[float, float, float]
    normalise_std: tuple[float, float, float]
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.weights_path.is_file():
            raise WeightsMissing(
                f"no backbone weights at {self.weights_path}. Run "
                f"`python scripts/vendor_weights.py --only backbone` once on a networked "
                f"machine and copy the result to the configured model_weights directory; "
                f"runtime never fetches.")

        import torch
        from torchvision.models import get_model

        # weights=None is the invariant, not a default. Any other value reaches the network on
        # a cache miss, inside a call that looks local.
        model = get_model(self.arch, weights=None)

        try:
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except Exception as exc:
            raise WeightsIncompatible(
                f"{self.weights_path.name} does not load into {self.arch}: {exc}") from exc

        classifier = getattr(model, CLASSIFIER_ATTRIBUTE, None)
        if classifier is None or classifier.in_features != self.output_dim:
            got = None if classifier is None else classifier.in_features
            raise WeightsIncompatible(
                f"{self.arch} has a {got}-dimensional penultimate layer and the config "
                f"declares output_dim {self.output_dim}. Every cached feature would be the "
                f"wrong width, and nothing downstream checks.")

        # Replacing the head with an identity is what makes the forward pass emit the
        # penultimate activations. Reaching into the module graph to pull them out with a hook
        # would do the same thing less legibly and would break on any architecture change.
        setattr(model, CLASSIFIER_ATTRIBUTE, torch.nn.Identity())

        model.eval()
        for parameter in model.parameters():
            # Belt and braces alongside `no_grad` in `embed`. D2 makes frozen a property of the
            # estate rather than of one call site, and a caller that builds its own autograd
            # context should still not be able to train this.
            parameter.requires_grad_(False)

        self._model = model.to(self.device)
        self._mean = torch.tensor(self.normalise_mean).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(self.normalise_std).view(1, 3, 1, 1).to(self.device)

    @property
    def model_version(self) -> str:
        """What the manifest records. The alias problem in one string.

        `resnet50/IMAGENET1K_V1` rather than `resnet50`, because the architecture alone does
        not identify the features and two runs differing only in this would otherwise be
        indistinguishable in the artefact record.
        """
        return version_of(self.arch, self.weights_name)

    @property
    def module(self):
        """The wrapped network, for attaching hooks to.

        Exposed because Grad-CAM has to register a forward and a backward hook on a named
        convolutional block, and reaching into a private attribute from another package is how
        that block's name ends up depending on an implementation detail nobody documented.
        """
        return self._model

    def _normalise(self, crops):
        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError(
                f"expected a batch of (N, 3, H, W) RGB crops, got {tuple(crops.shape)}")
        return (crops.to(self.device).float() - self._mean) / self._std

    def embed(self, crops):
        """Per-frame embeddings for a batch of teacher crops.

        Args:
            crops: float tensor of shape (N, 3, H, W) with values in [0, 1], already cropped
                and resized to the configured crop size. Normalisation is applied here rather
                than by the caller, so that it cannot be applied twice or skipped.

        Returns:
            A (N, output_dim) float tensor on the configured device.
        """
        import torch

        with torch.no_grad():
            return self._model(self._normalise(crops))

    def embed_for_explanation(self, crops):
        """The same forward pass, with the graph retained.

        Grad-CAM needs the gradient of a head's score with respect to a convolutional layer's
        activations. The parameters stay frozen - that is `requires_grad=False` on the weights,
        set in `__post_init__` - and frozen parameters do not stop gradients flowing *through*
        activations, which is what this is for.

        Separate from `embed` rather than a flag on it, because Stage A must never build a
        graph over 64 frames per clip: that is the allocation the measured 1.31 GB of headroom
        cannot take, and a caller passing the wrong flag would discover it as an out-of-memory
        kill partway through a corpus.
        """
        return self._model(self._normalise(crops))


def build_backbone(config, *, device: str | None = None) -> FrozenBackbone:
    """The frame encoder described by a loaded config.

    The single place that maps configuration onto the encoder, so that the weights path, the
    normalisation and the declared output width cannot drift apart across call sites.
    """
    backbone = config.behaviour.backbone
    return FrozenBackbone(
        weights_path=Path(config.paths.model_weights) / backbone.weights_file,
        arch=backbone.arch,
        weights_name=backbone.weights,
        output_dim=backbone.output_dim,
        normalise_mean=backbone.normalise_mean,
        normalise_std=backbone.normalise_std,
        device=device or config.run.device,
    )
