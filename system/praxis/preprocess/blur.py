# -*- coding: utf-8 -*-
"""Face blur, and the deletion of the original.

BUILD-SPEC: "Gaussian blur over every detected face box, teacher included, with a dilation
margin. Write the blurred video. Delete the original in the same transaction." The ordering in
Phase 3 is stated to be a privacy requirement rather than an optimisation: pose runs first
because it needs the unblurred faces, and nothing else ever does.

`blur_all_faces: true` includes the teacher deliberately. The teacher consented to being
recorded and analysed, not to their face persisting in a file that outlives the study, and a
pipeline that blurred only the people who did not consent would encode consent in pixels.

Deletion is verified rather than assumed. `delete_original_after_blur` and `verify_deletion` are
both marked `NEVER set false` in the configuration, and `verify_deletion` exists because
`unlink()` returning without raising is not evidence on Windows, where a file held open
elsewhere can survive the call.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from praxis.preprocess.pose import Detection


class BlurError(RuntimeError):
    """The blurred artefact could not be produced, or the original could not be removed."""


class OriginalSurvived(BlurError):
    """The unblurred file is still on disk after the pipeline claimed to remove it.

    Its own type because this is the one failure in Phase 3 that is a privacy incident rather
    than a processing error, and it must never be caught alongside the others.
    """


@dataclass(frozen=True)
class BlurPolicy:
    """Mirrors `preprocess.blur`."""

    kernel_fraction: float
    dilate_face_box: float
    blur_all_faces: bool
    delete_original_after_blur: bool
    verify_deletion: bool

    @classmethod
    def from_config(cls, blur) -> BlurPolicy:
        policy = cls(kernel_fraction=blur.kernel_fraction,
                     dilate_face_box=blur.dilate_face_box,
                     blur_all_faces=blur.blur_all_faces,
                     delete_original_after_blur=blur.delete_original_after_blur,
                     verify_deletion=blur.verify_deletion)
        if not policy.blur_all_faces:
            raise BlurError(
                "preprocess.blur.blur_all_faces is false. Blurring only some faces would make "
                "the pipeline encode who consented, and the teacher's face is not exempt.")
        if not policy.delete_original_after_blur:
            raise BlurError(
                "preprocess.blur.delete_original_after_blur is false. The original unblurred "
                "video must never persist; the configuration marks this NEVER set false.")
        return policy


def kernel_for(box: tuple[float, float, float, float], fraction: float) -> int:
    """An odd kernel sized to the shorter side of the box.

    Scaled to the face rather than fixed, because a face near the camera and one at the back of
    the room need very different kernels to be equally unrecognisable, and a fixed kernel would
    leave the distant faces legible.
    """
    x1, y1, x2, y2 = box
    shorter = min(x2 - x1, y2 - y1)
    size = max(3, round(shorter * fraction))
    return size + 1 if size % 2 == 0 else size


def blur_frame(frame: np.ndarray, detections: list[Detection],
               policy: BlurPolicy) -> tuple[np.ndarray, int]:
    """Return the frame with every visible face blurred, and how many were blurred.

    The count is returned rather than logged because the pipeline records it per session: a
    session where the number of blurred faces is far below the number of detected people is a
    signal that faces were not visible, not that privacy was satisfied.
    """
    import cv2

    output = frame.copy()
    height, width = frame.shape[:2]
    blurred = 0

    for detection in detections:
        box = detection.face_box(policy.dilate_face_box)
        if box is None:
            continue

        x1 = max(0, int(np.floor(box[0])))
        y1 = max(0, int(np.floor(box[1])))
        x2 = min(width, int(np.ceil(box[2])))
        y2 = min(height, int(np.ceil(box[3])))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        region = output[y1:y2, x1:x2]
        kernel = kernel_for((x1, y1, x2, y2), policy.kernel_fraction)
        output[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
        blurred += 1

    return output, blurred


def remove_original(path: Path, policy: BlurPolicy) -> None:
    """Delete the unblurred source and prove it is gone.

    Raises `OriginalSurvived` rather than returning a status, because a caller that could
    continue past this would be a caller that leaves unblurred video on disk.
    """
    if not policy.delete_original_after_blur:
        raise BlurError("deletion is disabled; this policy should not have been constructed")

    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise OriginalSurvived(
            f"the original at {path} could not be deleted: {exc}. It is still on disk and no "
            f"session may be marked preprocessed while it is.") from exc

    if policy.verify_deletion and path.exists():
        raise OriginalSurvived(
            f"{path} still exists after unlink. On Windows a file held open by another handle "
            f"survives deletion silently, which is why this is checked rather than assumed.")
