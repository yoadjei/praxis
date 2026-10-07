# -*- coding: utf-8 -*-
"""The ONNX pose estimator, and the weights it refuses to fetch.

The weights are vendored by `scripts/vendor_weights.py` and are not in this repository. That
makes the estimator's *execution* unverifiable here, and this file says so out loud instead of
pretending otherwise: when the file is absent, the assertion is that the runtime refuses and
names the operator step; when it is present, the same test runs the model. No skip and no xfail,
because an unenforced rule made invisible is the failure mode R1 to R7 exist to prevent - and
"the weights test did not run" and "the weights test passed" must never look alike.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from praxis.preprocess.pose import (
    CPU_ONLY_PROVIDERS,
    KEYPOINT_COUNT,
    POSE_INPUT_SIZE,
    Detection,
    OnnxPoseEstimator,
    WeightsInvalid,
    WeightsMissing,
)


def configured_weights(config) -> Path:
    return Path(config.paths.model_weights) / config.preprocess.pose.weights_file


def build(config, weights: Path) -> OnnxPoseEstimator:
    pose = config.preprocess.pose
    return OnnxPoseEstimator(
        weights_path=weights, min_person_confidence=pose.min_person_confidence,
        min_keypoint_confidence=pose.min_keypoint_confidence,
        max_persons_per_frame=pose.max_persons_per_frame,
        nms_iou_threshold=pose.nms_iou_threshold)


def test_absent_weights_are_refused_with_the_operator_step_named(config, tmp_path) -> None:
    """Not "downloaded on demand", which is how R6 dies quietly."""
    with pytest.raises(WeightsMissing, match="vendor_weights"):
        build(config, tmp_path / "yolov8m-pose.onnx")


def test_a_file_that_is_not_an_export_is_refused_as_invalid_not_missing(config,
                                                                        tmp_path) -> None:
    """The distinction matters to whoever reads the failure: one means run the script, the
    other means the copy is corrupt."""
    impostor = tmp_path / "yolov8m-pose.onnx"
    impostor.write_bytes(b"not an onnx graph")

    with pytest.raises(WeightsInvalid, match="onnxruntime refused"):
        build(config, impostor)


def test_the_provider_pin_is_load_bearing() -> None:
    """D43. onnxruntime offers an Azure provider and it is available on this machine, so a
    session constructed without naming its providers would have a network-capable one in its
    list. The runtime names CPU and only CPU."""
    import onnxruntime as ort

    assert CPU_ONLY_PROVIDERS == ("CPUExecutionProvider",)
    assert "AzureExecutionProvider" in ort.get_available_providers(), (
        "if this ever stops being true the pin is still correct, but this test no longer "
        "demonstrates why it is needed")
    assert "AzureExecutionProvider" not in CPU_ONLY_PROVIDERS


def test_the_vendored_weights_run_if_they_are_vendored(config) -> None:
    """Conditional enforcement, the same shape the database invariants use.

    Weights absent: assert that the runtime refuses and names the operator step, then stop.
    Absence is an environment state, not a defect - the weights are produced by
    `scripts/vendor_weights.py` and are deliberately not in the repository - so this is the
    stated weaker check, and it still runs and still asserts. What it does NOT establish is
    that `OnnxPoseEstimator._decode` reads a real YOLOv8-pose output correctly; nothing in this
    repository can establish that without the file.

    Weights present: load the real model and run a frame through it. A blank frame, because
    what is verified is the contract - shapes, coordinate space, the confidence floor - and not
    the model's accuracy, which needs the corpus and a human. Present-but-wrong fails here
    rather than degrading, because `build` raises `WeightsInvalid` or `WeightsIncompatible`.
    """
    weights = configured_weights(config)
    if not weights.is_file():
        with pytest.raises(WeightsMissing, match="vendor_weights"):
            build(config, weights)
        return

    estimator = build(config, weights)
    assert estimator.input_size == POSE_INPUT_SIZE
    assert estimator.model_version == weights.stem

    detections = estimator.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert isinstance(detections, list)
    assert len(detections) <= config.preprocess.pose.max_persons_per_frame
    for detection in detections:
        assert isinstance(detection, Detection)
        assert detection.keypoints.shape == (KEYPOINT_COUNT, 3)
        assert detection.confidence >= config.preprocess.pose.min_person_confidence
        x1, y1, x2, y2 = detection.bbox
        assert x2 > x1 and y2 > y1, "boxes come back in the frame's own pixel space"
