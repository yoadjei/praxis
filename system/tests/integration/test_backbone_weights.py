# -*- coding: utf-8 -*-
"""The frozen frame encoder, and the weights it refuses to fetch.

The same conditional enforcement `test_pose_weights.py` uses, for the same reason: the weights
are vendored by `scripts/vendor_weights.py` and are not in this repository, so when the file is
absent the assertion is that the runtime refuses and names the operator step, and when it is
present the same test runs the model. No skip and no xfail - "the backbone test did not run"
and "the backbone test passed" must never look alike.

The R6 half of this file is not conditional. Whether or not the weights are vendored, asking
torchvision for pretrained weights reaches the network, and `FrozenBackbone` must not.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest
import torch

from praxis.behaviour.backbone import (
    FrozenBackbone,
    WeightsIncompatible,
    WeightsMissing,
    build_backbone,
)


def configured_weights(config) -> Path:
    return Path(config.paths.model_weights) / config.behaviour.backbone.weights_file


@pytest.fixture
def no_network(monkeypatch):
    """Every outbound socket refused, for the duration of one test.

    Blunter than checking for a `urllib` import and that is the point: it catches a fetch made
    by any library at any depth, which is the only kind R6 has actually been threatened by.
    """
    def refuse(*args, **kwargs):
        raise OSError("R6: the inference path attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return refuse


def test_absent_weights_are_refused_with_the_operator_step_named(config, tmp_path) -> None:
    """Not "downloaded on demand", which is how R6 dies quietly."""
    with pytest.raises(WeightsMissing, match="vendor_weights"):
        FrozenBackbone(
            weights_path=tmp_path / "resnet50.pt",
            arch="resnet50",
            weights_name="IMAGENET1K_V1",
            output_dim=2048,
            normalise_mean=(0.485, 0.456, 0.406),
            normalise_std=(0.229, 0.224, 0.225),
        )


def test_a_file_that_is_not_a_state_dict_is_refused_as_incompatible(config, tmp_path) -> None:
    """The distinction matters to whoever reads the failure: missing means run the script,
    incompatible means the copy is wrong or the architecture moved."""
    impostor = tmp_path / "resnet50.pt"
    torch.save({"not": "a resnet"}, impostor)

    with pytest.raises(WeightsIncompatible, match="does not load into resnet50"):
        FrozenBackbone(
            weights_path=impostor,
            arch="resnet50",
            weights_name="IMAGENET1K_V1",
            output_dim=2048,
            normalise_mean=(0.485, 0.456, 0.406),
            normalise_std=(0.229, 0.224, 0.225),
        )


def test_asking_torchvision_for_pretrained_weights_reaches_the_network(
    monkeypatch, tmp_path, no_network
) -> None:
    """D72, demonstrated rather than asserted.

    `prepare_environment` sets `YOLO_OFFLINE` and `HF_HUB_OFFLINE`, which cover ultralytics and
    huggingface. Neither covers torchvision, and the assumption that the two together meant
    "offline" is what this test exists to disprove. TORCH_HOME is redirected at an empty
    directory so the result cannot depend on what a previous run happened to leave in the hub
    cache.

    If torchvision ever starts honouring an offline switch, this test fails - and the right
    response is to record that it changed, not to delete the test. `FrozenBackbone` would still
    be correct, because it never asks in the first place.
    """
    from torchvision.models import get_model

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty-hub"))

    with pytest.raises(Exception) as fetched:
        get_model("resnet50", weights="IMAGENET1K_V1")
    assert "R6" in str(fetched.value) or "urlopen" in str(fetched.value).lower(), (
        f"expected the fetch to be refused by the blocked socket, got {fetched.value!r}")


def test_the_backbone_builds_with_the_network_down(config, no_network) -> None:
    """R6. The encoder is constructed with every socket refused.

    This is the claim that matters: on the college's air-gapped machine, building the frame
    encoder from vendored weights completes. Conditional only on the weights being present,
    and present-but-wrong fails rather than degrading.
    """
    weights = configured_weights(config)
    if not weights.is_file():
        with pytest.raises(WeightsMissing, match="vendor_weights"):
            build_backbone(config)
        return

    backbone = build_backbone(config)
    assert backbone.model_version == f"{config.behaviour.backbone.arch}/IMAGENET1K_V1", (
        "the manifest records the weights version, not just the architecture; two runs "
        "differing only in the recipe would otherwise be indistinguishable")


def test_the_vendored_backbone_embeds_at_the_declared_width(config) -> None:
    """Conditional enforcement, as in test_pose_weights.

    Weights absent: assert the runtime refuses and names the operator step, then stop. What
    this does NOT establish is that the encoder produces useful features; that needs the
    corpus. What it does establish is the contract Stage A's cache is written against.
    """
    weights = configured_weights(config)
    if not weights.is_file():
        with pytest.raises(WeightsMissing, match="vendor_weights"):
            build_backbone(config)
        return

    backbone = build_backbone(config)
    crop_size = config.behaviour.clip.crop_size
    crops = torch.rand(2, 3, crop_size, crop_size)

    embeddings = backbone.embed(crops)
    assert embeddings.shape == (2, config.behaviour.backbone.output_dim)
    assert torch.isfinite(embeddings).all()
    assert not embeddings.requires_grad, (
        "D2: the backbone is frozen, and an embedding carrying grad means Stage A built a "
        "graph over 64 frames per clip, which does not fit in the measured headroom")


def test_the_encoder_refuses_a_width_the_config_does_not_declare(config) -> None:
    """A mismatch between the architecture and `output_dim` is caught at construction.

    Every cached feature would otherwise be the wrong width, and nothing downstream checks: the
    TCN accepts whatever it is handed and the failure surfaces, if at all, as poor accuracy
    weeks later.
    """
    weights = configured_weights(config)
    if not weights.is_file():
        with pytest.raises(WeightsMissing, match="vendor_weights"):
            build_backbone(config)
        return

    with pytest.raises(WeightsIncompatible, match="output_dim"):
        FrozenBackbone(
            weights_path=weights,
            arch=config.behaviour.backbone.arch,
            weights_name=config.behaviour.backbone.weights,
            output_dim=512,
            normalise_mean=config.behaviour.backbone.normalise_mean,
            normalise_std=config.behaviour.backbone.normalise_std,
        )


def test_normalisation_is_applied_by_the_encoder_not_the_caller(config) -> None:
    """The same crops normalised twice would be a silent accuracy loss.

    `embed` owns the normalisation so there is one place it can happen. This checks it happens
    at all: a batch at the channel means must embed differently from a batch of zeros.
    """
    weights = configured_weights(config)
    if not weights.is_file():
        with pytest.raises(WeightsMissing, match="vendor_weights"):
            build_backbone(config)
        return

    backbone = build_backbone(config)
    crop_size = config.behaviour.clip.crop_size
    mean = torch.tensor(config.behaviour.backbone.normalise_mean).view(1, 3, 1, 1)

    at_the_mean = backbone.embed(mean.expand(1, 3, crop_size, crop_size).clone())
    at_zero = backbone.embed(torch.zeros(1, 3, crop_size, crop_size))
    assert not torch.allclose(at_the_mean, at_zero), (
        "a batch at the channel means and a batch of zeros embed identically, so the "
        "normalisation is not being applied")
