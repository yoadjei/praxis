# -*- coding: utf-8 -*-
"""Stage B: the trainable head over cached backbone features.

BUILD-SPEC Phase 4. The backbone is frozen and its output cached once (`backbone.py`), so this
is the only part that trains:

    cached features (T, 2048)  +  keypoints (T, 17, 3)
        |                             |
        |                    [keypoint branch] MLP -> (T, 128)
        +--------- concatenate -------+   -> (T, 2176)
                        |
              [temporal] 3-layer dilated TCN, kernel 3, dilations 1/2/4, residual
                        |
              [attention pooling] learned attention over time -> clip embedding
                        |
              [heads] one per codebook field, laid out by `heads.py`

**The TCN is causal only in the sense of being dilated, not in the sense of being one-sided.**
A clip is eight seconds of already-recorded video scored as a whole; there is no streaming
constraint, and restricting the receptive field to the past would discard half the context for
no reason. Padding is therefore symmetric and the output length matches the input.

**Attention pooling rather than a mean.** A gesture occupies a fraction of a clip and a mean
over 64 frames dilutes it by the ratio of its duration to the clip's. The attention weights are
also the natural intrinsic explanation Phase 7 falls back to when Grad-CAM fails its sanity
checks, which is why they are returned rather than consumed internally.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from praxis.annotation.codebook import Codebook
from praxis.behaviour.heads import HeadKind, HeadSpec, behaviour_heads
from praxis.vocabulary import BEHAVIOUR_IDS


@dataclass(frozen=True)
class ModelShape:
    """Everything about the network that a config decides, in one object.

    Built by `shape_from_config` so that the model can be constructed in a test without a
    config file, and so that the config is read in exactly one place.
    """

    feature_dim: int
    keypoints: int
    keypoint_hidden: int
    keypoint_out: int
    tcn_channels: int
    tcn_kernel: int
    tcn_dilations: tuple[int, ...]
    tcn_dropout: float
    attention_dim: int
    behaviours: tuple[str, ...] = BEHAVIOUR_IDS


def shape_from_config(config) -> ModelShape:
    behaviour = config.behaviour
    return ModelShape(
        feature_dim=behaviour.backbone.output_dim,
        keypoints=behaviour.keypoint_branch.input_keypoints,
        keypoint_hidden=behaviour.keypoint_branch.hidden_dim,
        keypoint_out=behaviour.keypoint_branch.output_dim,
        tcn_channels=behaviour.temporal.channels,
        tcn_kernel=behaviour.temporal.kernel_size,
        tcn_dilations=tuple(behaviour.temporal.dilations),
        tcn_dropout=behaviour.temporal.dropout,
        attention_dim=behaviour.pooling.attention_dim,
        behaviours=tuple(behaviour.heads.presence_behaviours),
    )


class KeypointBranch(nn.Module):
    """Per-frame pose to a dense embedding.

    The (17, 3) keypoint tensor is flattened per frame rather than treated as a graph. A GCN
    over the COCO skeleton is the obvious alternative and is deliberately not used: it adds a
    topology the pose estimator's confidence channel already makes unreliable - an occluded
    wrist is a zero-confidence node, not an absent one - and BUILD-SPEC specifies an MLP.
    """

    def __init__(self, keypoints: int, hidden: int, out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(keypoints * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
            nn.ReLU(inplace=True),
        )

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        batch, frames = keypoints.shape[0], keypoints.shape[1]
        return self.net(keypoints.reshape(batch, frames, -1))


class TemporalBlock(nn.Module):
    """One dilated residual convolution over time.

    The residual connection is what lets three layers stack without the gradient vanishing
    through the dilations, and the 1x1 projection exists only for the first block, where the
    input width (2176) differs from the channel width.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        # Symmetric padding keeps the output length equal to the input length, so the dilations
        # can stack without the clip shortening under them.
        padding = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel,
                              padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.project = (nn.Conv1d(in_channels, out_channels, 1)
                        if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.activation(self.norm(self.conv(x))))
        # An even kernel with symmetric padding returns one extra step; trimmed rather than
        # left to broadcast against the residual, which would silently misalign time.
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        return out + self.project(x)


class AttentionPooling(nn.Module):
    """Learned attention over time, returning the weights as well as the embedding.

    The weights are returned because Phase 7 needs them: when Grad-CAM fails the Adebayo
    sanity checks, `explain.fallback` is "intrinsic", and attention over frames is the
    intrinsic signal this architecture has. A pooling layer that discarded them would make the
    fallback impossible to implement without retraining.
    """

    def __init__(self, channels: int, attention_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("btc,bt->bc", sequence, weights)
        return pooled, weights


class BehaviourHeads(nn.Module):
    """One linear head per codebook field, for one behaviour.

    A `ModuleDict` keyed by field name rather than a single wide layer that callers slice. The
    slicing version works and has been the source of the same bug twice in projects of this
    shape: a field added to the codebook shifts every offset after it, and the model keeps
    running while reporting one field's value under another's name.
    """

    def __init__(self, channels: int, heads: tuple[HeadSpec, ...]) -> None:
        super().__init__()
        self.specs = heads
        self.layers = nn.ModuleDict(
            {head.field: nn.Linear(channels, head.width) for head in heads})

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for head in self.specs:
            raw = self.layers[head.field](pooled)
            outputs[head.field] = raw if head.kind is HeadKind.CATEGORICAL else raw.squeeze(-1)
        return outputs


@dataclass(frozen=True)
class BehaviourOutput:
    """What one forward pass produces.

    `attention` travels with the predictions rather than being recoverable by a second pass,
    because a second pass under dropout would not give the same weights and the explanation
    would then not describe the prediction it is shown beside.
    """

    raw: dict[str, dict[str, torch.Tensor]]
    attention: torch.Tensor

    def logits(self, behaviour: str) -> dict[str, torch.Tensor]:
        return self.raw[behaviour]


class BehaviourModel(nn.Module):
    """The Stage B network: cached features and keypoints in, codebook fields out."""

    def __init__(self, shape: ModelShape, codebook: Codebook) -> None:
        super().__init__()
        self.shape = shape
        self.keypoint_branch = KeypointBranch(
            shape.keypoints, shape.keypoint_hidden, shape.keypoint_out)

        width = shape.feature_dim + shape.keypoint_out
        blocks = []
        for dilation in shape.tcn_dilations:
            blocks.append(TemporalBlock(width, shape.tcn_channels, shape.tcn_kernel,
                                        dilation, shape.tcn_dropout))
            width = shape.tcn_channels
        self.temporal = nn.ModuleList(blocks)

        self.pooling = AttentionPooling(shape.tcn_channels, shape.attention_dim)

        self.head_specs: dict[str, tuple[HeadSpec, ...]] = {
            behaviour: behaviour_heads(codebook, behaviour)  # type: ignore[arg-type]
            for behaviour in shape.behaviours
        }
        self.heads = nn.ModuleDict({
            behaviour: BehaviourHeads(shape.tcn_channels, specs)
            for behaviour, specs in self.head_specs.items()
        })

    def forward(self, features: torch.Tensor, keypoints: torch.Tensor) -> BehaviourOutput:
        """
        Args:
            features: (B, T, feature_dim) cached backbone embeddings.
            keypoints: (B, T, keypoints, 3) pose, x/y/confidence.

        Returns:
            A BehaviourOutput carrying raw head outputs per behaviour and the attention
            weights over time.
        """
        if features.ndim != 3 or features.shape[-1] != self.shape.feature_dim:
            raise ValueError(
                f"expected features of (B, T, {self.shape.feature_dim}), got "
                f"{tuple(features.shape)}")
        if keypoints.shape[:2] != features.shape[:2]:
            raise ValueError(
                f"features cover {tuple(features.shape[:2])} and keypoints "
                f"{tuple(keypoints.shape[:2])}; a clip and its pose must be the same length, "
                f"or every frame after the first mismatch is scored against another's pose")

        pose = self.keypoint_branch(keypoints)
        sequence = torch.cat([features, pose], dim=-1).transpose(1, 2)
        for block in self.temporal:
            sequence = block(sequence)

        pooled, attention = self.pooling(sequence.transpose(1, 2))
        return BehaviourOutput(
            raw={behaviour: self.heads[behaviour](pooled) for behaviour in self.head_specs},
            attention=attention,
        )

    def predict(self, features: torch.Tensor,
                keypoints: torch.Tensor) -> dict[str, list[dict[str, object]]]:
        """Decoded codebook values, one dict per clip per behaviour.

        Probabilities rather than logits reach `HeadSpec.decode`, because the decoder's binary
        threshold is stated in probability space. Categorical heads are handed their scores
        unchanged; argmax is invariant to the softmax, so normalising would cost time and
        change nothing.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, keypoints)

        decoded: dict[str, list[dict[str, object]]] = {}
        for behaviour, specs in self.head_specs.items():
            per_clip: list[dict[str, object]] = []
            for index in range(features.shape[0]):
                values: dict[str, object] = {}
                for head in specs:
                    raw = output.raw[behaviour][head.field][index]
                    if head.kind is HeadKind.CATEGORICAL:
                        values[head.field] = head.decode(raw.tolist())
                    elif head.kind is HeadKind.BINARY:
                        values[head.field] = head.decode(float(torch.sigmoid(raw)))
                    else:
                        values[head.field] = head.decode(float(raw))
                per_clip.append(values)
            decoded[behaviour] = per_clip
        return decoded


def build_model(config, codebook: Codebook) -> BehaviourModel:
    """The network described by a loaded config."""
    return BehaviourModel(shape_from_config(config), codebook)


__all__ = [
    "AttentionPooling",
    "BehaviourHeads",
    "BehaviourModel",
    "BehaviourOutput",
    "KeypointBranch",
    "ModelShape",
    "TemporalBlock",
    "build_model",
    "shape_from_config",
]
