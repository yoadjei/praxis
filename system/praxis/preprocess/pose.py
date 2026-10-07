# -*- coding: utf-8 -*-
"""Pose estimation: frames in, per-person keypoints out.

`PoseEstimator` is a protocol rather than a class hierarchy because the pipeline's correctness
does not depend on which detector produced a detection. Tracking, teacher identification, blur
and the learner aggregates all consume `Detection`, so they are testable against a detector
whose output is known exactly, and the ONNX model is one implementation of the same protocol
rather than a special case the rest of the code knows about.

**Nothing here imports ultralytics.** Importing its tracker resolves DNS and pip-installs
missing extras during the import itself, so no environment variable set afterwards prevents it.
D57.

Execution providers are pinned. onnxruntime 1.30 offers `AzureExecutionProvider`, and the
default provider list includes it, so a session constructed without naming its providers is one
`.pt` path away from a network call inside the inference path. D43.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# COCO 17-keypoint order, which is what YOLOv8-pose emits and what the .npz stores.
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KEYPOINT_COUNT = len(COCO_KEYPOINTS)
FACE_KEYPOINTS = (0, 1, 2, 3, 4)          # nose, eyes, ears: what the blur box is built from

# R6. The only provider this system runs on. Named rather than defaulted.
CPU_ONLY_PROVIDERS = ("CPUExecutionProvider",)

# The square the frame is letterboxed into, and the `imgsz` the export must be produced at. One
# constant because the two have to agree: an export at a different size letterboxes into the
# wrong canvas and every coordinate comes back scaled. `scripts/vendor_weights.py` reads it.
POSE_INPUT_SIZE = 640


class PoseError(RuntimeError):
    """Pose estimation could not be performed, with the reason distinguished."""


class WeightsMissing(PoseError):
    """No file at the configured path. Vendoring is a deliberate operator step."""


class WeightsInvalid(PoseError):
    """A file exists and onnxruntime will not load it."""


class WeightsIncompatible(PoseError):
    """It loads and its shapes are not a 17-keypoint pose model."""


@dataclass(frozen=True)
class Detection:
    """One person in one frame.

    `keypoints` is (17, 3) as x, y, confidence in pixels. `bbox` is x1, y1, x2, y2 in pixels.
    Both are absolute rather than normalised, because blur writes into the same pixel space and
    a normalisation applied twice is silent corruption.
    """

    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray
    confidence: float

    def __post_init__(self) -> None:
        if self.keypoints.shape != (KEYPOINT_COUNT, 3):
            raise ValueError(
                f"expected {KEYPOINT_COUNT} keypoints of (x, y, confidence), got "
                f"{self.keypoints.shape}")

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def face_box(self, dilate: float) -> tuple[float, float, float, float] | None:
        """The region the blur covers, from the visible face keypoints.

        None when no facial keypoint is visible, which is a real state: a teacher facing the
        board has no face in frame. The caller decides what that means; this does not invent a
        box, because a blur applied to a guessed location is worse than a recorded absence.
        """
        visible = [self.keypoints[i] for i in FACE_KEYPOINTS if self.keypoints[i, 2] > 0]
        if not visible:
            return None

        xs = [point[0] for point in visible]
        ys = [point[1] for point in visible]
        # A face is wider than the eye-to-ear span and taller than the eye-to-nose span, so the
        # extent is grown to a square around its centre before the configured dilation.
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 or 1.0
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        reach = half * dilate
        return cx - reach, cy - reach, cx + reach, cy + reach


class PoseEstimator(Protocol):
    """Whatever turns a frame into detections."""

    model_version: str

    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


@dataclass
class OnnxPoseEstimator:
    """YOLOv8-pose exported to ONNX, run on CPU.

    The weights are vendored; `scripts/vendor_weights.py` is the deliberate operator step that
    produces them and is never invoked from here. D58.
    """

    weights_path: Path
    min_person_confidence: float
    min_keypoint_confidence: float
    max_persons_per_frame: int
    nms_iou_threshold: float
    input_size: int = POSE_INPUT_SIZE

    def __post_init__(self) -> None:
        if not self.weights_path.is_file():
            raise WeightsMissing(
                f"no pose weights at {self.weights_path}. Run "
                f"`python scripts/vendor_weights.py` once on a networked machine and copy the "
                f"result to the configured model_weights directory; runtime never fetches.")

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self._session = ort.InferenceSession(
                str(self.weights_path), sess_options=options,
                providers=list(CPU_ONLY_PROVIDERS))
        except Exception as exc:
            raise WeightsInvalid(
                f"{self.weights_path} exists and onnxruntime refused it: {exc}") from exc

        self._input = self._session.get_inputs()[0]
        outputs = self._session.get_outputs()
        # YOLOv8-pose emits (1, 56, N): 4 box + 1 score + 17*3 keypoints.
        expected_channels = 5 + KEYPOINT_COUNT * 3
        shape = outputs[0].shape
        if len(shape) != 3 or (isinstance(shape[1], int) and shape[1] != expected_channels):
            raise WeightsIncompatible(
                f"{self.weights_path} outputs {shape}; a {KEYPOINT_COUNT}-keypoint pose model "
                f"emits {expected_channels} channels. This is the wrong export.")

    @property
    def model_version(self) -> str:
        return self.weights_path.stem

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detections in the frame's own pixel coordinates."""
        tensor, scale, pad = _letterbox(frame, self.input_size)
        raw = self._session.run(None, {self._input.name: tensor})[0]
        return _decode(raw, scale, pad, self.min_person_confidence,
                       self.min_keypoint_confidence, self.max_persons_per_frame,
                       self.nms_iou_threshold)


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize preserving aspect ratio and pad, returning what is needed to invert it."""
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    new_h, new_w = round(height * scale), round(width * scale)

    import cv2

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), scale, (float(left), float(top))


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two `(n, 4)` and `(m, 4)` arrays of xyxy boxes.

    The single definition of box overlap in this system. `preprocess.tracking.iou_matrix` and
    the suppression below both call it, because two implementations of one rule is how the two
    drift apart and only one of them gets the fix (L20).
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - overlap
    return np.where(union > 0, overlap / union, 0.0).astype(np.float32)


def _suppress(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float,
              limit: int) -> list[int]:
    """Greedy non-maximum suppression, returning kept row indices in descending score order.

    Exact rather than approximate: each surviving box is compared against the boxes still in
    contention, not against a precomputed matrix that a previous removal has invalidated.
    """
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while order.size and len(kept) < limit:
        best = int(order[0])
        kept.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        overlap = box_iou_matrix(boxes[best][None, :], boxes[rest])[0]
        order = rest[overlap < iou_threshold]
    return kept


def _decode(raw: np.ndarray, scale: float, pad: tuple[float, float], min_person: float,
            min_keypoint: float, max_persons: int, nms_iou: float) -> list[Detection]:
    """(1, 56, N) into detections, with padding removed and scale undone.

    YOLOv8's ONNX graph emits one prediction per anchor - 8400 of them at 640 - and performs no
    suppression itself, so a single teacher clears the confidence floor dozens of times over.
    Suppression therefore happens **here**, before the cap, and the ordering is the substance of
    D77: capping first keeps `max_persons` copies of the most confident body and discards the
    real people further down the list.
    """
    predictions = raw[0].T                                  # (N, 56)
    keep = predictions[:, 4] >= min_person
    predictions = predictions[keep]
    if predictions.size == 0:
        return []

    # Suppression runs in letterboxed coordinates. IoU is invariant under the uniform scale and
    # translation that undoing the letterbox applies, so the surviving set is identical either
    # way and this avoids transforming rows that are about to be thrown away.
    centre_x, centre_y, width, height = (predictions[:, 0], predictions[:, 1],
                                         predictions[:, 2], predictions[:, 3])
    boxes = np.stack([centre_x - width / 2, centre_y - height / 2,
                      centre_x + width / 2, centre_y + height / 2], axis=1)
    predictions = predictions[_suppress(boxes, predictions[:, 4], nms_iou, max_persons)]

    left, top = pad
    detections = []
    for row in predictions:
        cx, cy, w, h = row[:4]
        x1 = (cx - w / 2 - left) / scale
        y1 = (cy - h / 2 - top) / scale
        x2 = (cx + w / 2 - left) / scale
        y2 = (cy + h / 2 - top) / scale

        keypoints = row[5:].reshape(KEYPOINT_COUNT, 3).astype(np.float32).copy()
        keypoints[:, 0] = (keypoints[:, 0] - left) / scale
        keypoints[:, 1] = (keypoints[:, 1] - top) / scale
        # A keypoint below threshold is marked unobserved rather than dropped: the array is
        # fixed-shape and a zero confidence is how absence is recorded.
        keypoints[keypoints[:, 2] < min_keypoint] = 0.0

        detections.append(Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                                    keypoints=keypoints, confidence=float(row[4])))
    return detections
