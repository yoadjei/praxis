# -*- coding: utf-8 -*-
"""The pose artefact: what Phase 4 reads, and the provenance that makes it reproducible.

SCHEMA.md section 4 fixes the shape: `.npz` holding `keypoints (F,P,17,3)` and `track_ids
(F,P)`, with `frame_count`, `sampled_fps`, `model_version` and `sha256` in the row that points
at it. Pose is bulk numeric data and stays out of Postgres; the database holds the pointer and
the hash.

`P` is the per-session maximum number of simultaneous detections, not a global constant. The
array is padded, and padding is distinguishable from a detection with zero confidence: an
absent person has `track_id` -1 and all-zero keypoints, and `person_count` records how many of
the P slots were real in each frame. Without that a consumer counting non-zero rows would
silently treat a padded slot as an undetected person.

R7: the artefact records the config hash and the model version it was produced under, so a
Phase 4 feature cache keyed on them invalidates rather than silently reusing pose from a
different preprocessing configuration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from praxis.preprocess.pose import KEYPOINT_COUNT

NO_TRACK = -1
ARTIFACT_VERSION = 1


class ArtifactError(RuntimeError):
    """The artefact could not be written, or does not describe what it claims."""


@dataclass(frozen=True)
class PoseArtifact:
    """The row that points at the file, plus what the file contains."""

    session_id: str
    relative_path: str
    frame_count: int
    sampled_fps: float
    model_version: str
    sha256: str
    max_persons: int
    config_sha256: str

    def as_row(self) -> dict[str, object]:
        """Exactly the columns `pose_artifacts` declares; the rest is provenance in the file."""
        return {"session_id": self.session_id, "relative_path": self.relative_path,
                "frame_count": self.frame_count, "sampled_fps": self.sampled_fps,
                "model_version": self.model_version, "sha256": self.sha256}


def pack(frames: list[tuple[list, list[int | None]]]) -> tuple[np.ndarray, np.ndarray,
                                                               np.ndarray]:
    """Per-frame detections into the fixed-shape arrays the artefact stores.

    Returns keypoints (F,P,17,3), track_ids (F,P) and person_count (F,). Detections are written
    in the order they arrived, which is descending confidence out of the decoder, so slot 0 is
    the most confident person in every frame rather than an arbitrary one.
    """
    frame_count = len(frames)
    max_persons = max((len(detections) for detections, _ in frames), default=0)
    if frame_count == 0:
        raise ArtifactError("no frames were processed; there is nothing to write")

    keypoints = np.zeros((frame_count, max_persons, KEYPOINT_COUNT, 3), dtype=np.float32)
    track_ids = np.full((frame_count, max_persons), NO_TRACK, dtype=np.int32)
    person_count = np.zeros(frame_count, dtype=np.int32)

    for f, (detections, assigned) in enumerate(frames):
        person_count[f] = len(detections)
        for p, detection in enumerate(detections):
            keypoints[f, p] = detection.keypoints
            if assigned[p] is not None:
                track_ids[f, p] = assigned[p]

    return keypoints, track_ids, person_count


def write(path: Path, *, session_id: str, keypoints: np.ndarray, track_ids: np.ndarray,
          person_count: np.ndarray, sampled_fps: float, model_version: str,
          config_sha256: str, frame_width: int, frame_height: int,
          teacher_track_id: int | None) -> PoseArtifact:
    """Write the `.npz` and hash it.

    The hash is taken from the bytes on disk rather than from the arrays in memory, because it
    is checked later against the file, and a hash of something else would verify nothing.
    """
    if keypoints.shape[:2] != track_ids.shape:
        raise ArtifactError(
            f"keypoints {keypoints.shape} and track_ids {track_ids.shape} disagree on how many "
            f"frames or people this session has")
    if keypoints.shape[0] != person_count.shape[0]:
        raise ArtifactError("person_count does not cover every frame")

    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "artifact_version": ARTIFACT_VERSION, "session_id": session_id,
        "sampled_fps": sampled_fps, "model_version": model_version,
        "config_sha256": config_sha256, "frame_width": frame_width,
        "frame_height": frame_height,
        # Recorded so a consumer can select the teacher without a database round trip. It is
        # the *proposal*; confirmation lives in teacher_tracks and is what Phase 4 must check.
        "proposed_teacher_track_id": -1 if teacher_track_id is None else teacher_track_id,
    }

    with open(path, "wb") as handle:
        np.savez_compressed(handle, keypoints=keypoints, track_ids=track_ids,
                            person_count=person_count,
                            provenance=np.array(json.dumps(provenance)))

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PoseArtifact(
        session_id=session_id, relative_path=path.name, frame_count=int(keypoints.shape[0]),
        sampled_fps=sampled_fps, model_version=model_version, sha256=digest,
        max_persons=int(keypoints.shape[1]), config_sha256=config_sha256)


@dataclass(frozen=True)
class LoadedPose:
    keypoints: np.ndarray
    track_ids: np.ndarray
    person_count: np.ndarray
    provenance: dict

    def track(self, track_id: int) -> np.ndarray:
        """Keypoints for one track as (F, 17, 3), zero where that track is absent.

        This is what Phase 4 calls to build its `(64, 17, 3)` tensor for the teacher. Returning
        zeros for absent frames rather than a shorter array keeps the time axis aligned with the
        clip's frames, which a temporal model requires.
        """
        frames, _ = self.track_ids.shape
        out = np.zeros((frames, KEYPOINT_COUNT, 3), dtype=np.float32)
        rows, cols = np.nonzero(self.track_ids == track_id)
        out[rows] = self.keypoints[rows, cols]
        return out


def load(path: Path, expected_sha256: str | None = None) -> LoadedPose:
    """Read an artefact back, refusing one whose bytes do not match the recorded hash."""
    if not path.is_file():
        raise ArtifactError(f"no pose artefact at {path}")

    if expected_sha256 is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ArtifactError(
                f"{path} hashes to {actual[:12]} and pose_artifacts records "
                f"{expected_sha256[:12]}. The file has changed since it was written.")

    with np.load(path, allow_pickle=False) as data:
        return LoadedPose(
            keypoints=data["keypoints"], track_ids=data["track_ids"],
            person_count=data["person_count"],
            provenance=json.loads(str(data["provenance"])))


def config_digest(config) -> str:
    """A hash of the preprocessing settings that change what pose comes out.

    Only the preprocessing block, so an unrelated configuration edit does not invalidate a
    cache that is still valid. R7.
    """
    payload = json.dumps(asdict(config) if hasattr(config, "__dataclass_fields__")
                         else config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
