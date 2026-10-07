# -*- coding: utf-8 -*-
"""Phase 3, in the order BUILD-SPEC fixes and for the reason it gives.

    1. pose over sampled frames      - needs the unblurred faces, and nothing else does
    2. tracking                      - detections become identities
    3. teacher heuristic             - a proposal; a human confirms separately
    4. learner aggregates            - computed while identities still exist
    5. identifiers discarded         - R1, enforced here rather than by convention
    6. blur and delete the original  - the original never outlives the job

"Order is a privacy requirement, not an optimisation." Steps 1 and 6 are the pair that matters:
pose reads faces, so it must run before they are destroyed, and the original must be destroyed
before the job is allowed to report success.

Nothing here decides which track is the teacher. `TeacherProposal` is written to
`teacher_tracks` with `confirmed_by` null, and every downstream phase filters on the confirmed
column, so a session nobody has reviewed cannot be trained on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from praxis.preprocess import aggregates, artifacts
from praxis.preprocess import blur as blur_module
from praxis.preprocess import teacher as teacher_module
from praxis.preprocess.pose import Detection, PoseEstimator
from praxis.preprocess.tracking import ByteTrackConfig, ByteTracker, Track
from praxis.preprocess.zones import CameraSetup


class PreprocessError(RuntimeError):
    """The session could not be preprocessed."""


class VideoUnreadable(PreprocessError):
    """The media cannot be decoded. Distinct from a pose failure on a readable frame."""


class FrameProcessingFailed(PreprocessError):
    """Frames were lost: one failed pose estimation, or the file ran out before it claimed to.

    Raised rather than skipped. BUILD-SPEC gives no permission to drop frames, and a pipeline
    that silently skipped them would produce a pose artefact whose frame count disagrees with
    the video's, which every downstream temporal model would then misalign.
    """


@dataclass(frozen=True)
class PreprocessResult:
    artifact: artifacts.PoseArtifact
    proposal: teacher_module.TeacherProposal
    learner_bins: list[aggregates.LearnerBin]
    blurred_path: Path
    original_deleted: bool
    frames_processed: int
    faces_blurred: int
    zones_available: bool


def sample_step(source_fps: float, target_fps: float) -> float:
    """Source frames per sampled frame. The k-th sample is source frame `int(k * step)`.

    Expressed as a step rather than a precomputed list of indices so the decoder does not have
    to trust the container's frame count before it starts. Deterministic for R7: the same video
    and the same configuration select the same frames on every run.
    """
    if source_fps <= 0 or target_fps <= 0:
        raise VideoUnreadable(
            f"cannot sample at {target_fps} fps from a {source_fps} fps source")
    return 1.0 if target_fps >= source_fps else source_fps / target_fps


def run(*, session_id: str, source: Path, blurred_destination: Path, pose_output: Path,
        estimator: PoseEstimator, config, setup: CameraSetup | None,
        learner_bin_seconds: float = 60.0) -> PreprocessResult:
    """Preprocess one session. On success the original at `source` no longer exists."""
    import cv2

    policy = blur_module.BlurPolicy.from_config(config.preprocess.blur)
    tracker = ByteTracker(ByteTrackConfig.from_config(config.preprocess.tracking))

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoUnreadable(f"{source} could not be opened for decoding")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0 or declared_frames <= 0:
        capture.release()
        raise VideoUnreadable(
            f"{source} reports {width}x{height} and {declared_frames} frames; it is malformed")

    step = sample_step(source_fps, config.preprocess.sample_fps)
    blurred_destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(blurred_destination),
                             cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise PreprocessError(f"could not open {blurred_destination} for writing")

    sampled: list[tuple[list[Detection], list[int | None]]] = []
    tracked_frames: list[tuple[int, list[Detection], list[int | None]]] = []
    positions: dict[int, list[tuple[float, float]]] = {}
    areas: dict[int, list[float]] = {}
    faces_blurred = 0
    sampled_index = 0
    decoded = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = decoded
            decoded += 1

            if frame_index == 0:
                actual_height, actual_width = frame.shape[:2]
                if (actual_width, actual_height) != (width, height):
                    # The writer was sized from the declared geometry and every blurred frame is
                    # about to be written through it, so a disagreement here corrupts the whole
                    # output silently. It arises when the decoder applies a container rotation
                    # that `CAP_PROP_FRAME_WIDTH` does not report - behaviour that has differed
                    # between OpenCV builds - so this is checked rather than assumed. D80.
                    raise VideoUnreadable(
                        f"{source} declares {width}x{height} but decodes to "
                        f"{actual_width}x{actual_height}. The container's rotation is being "
                        f"applied to the frames and not to the reported size, so every "
                        f"coordinate this pipeline produced would be transposed.")

            if frame_index >= int(sampled_index * step):
                try:
                    detections = estimator.detect(frame)
                except Exception as exc:
                    raise FrameProcessingFailed(
                        f"pose estimation failed on frame {frame_index}: {exc}") from exc

                assigned = tracker.update(detections, sampled_index)
                sampled.append((detections, assigned))
                tracked_frames.append((sampled_index, detections, assigned))

                for detection, track_id in zip(detections, assigned, strict=True):
                    if track_id is not None:
                        positions.setdefault(track_id, []).append(detection.centroid)
                        areas.setdefault(track_id, []).append(detection.area)

                # Blur uses this frame's own detections. Frames between samples carry the
                # previous sample's boxes, which is why the sampling rate is also a privacy
                # parameter: at 8 fps a face moves little between samples.
                last_detections = detections
                sampled_index += 1
            else:
                last_detections = sampled[-1][0] if sampled else []

            blurred, count = blur_module.blur_frame(frame, last_detections, policy)
            faces_blurred += count
            writer.write(blurred)
    finally:
        capture.release()
        writer.release()

    # Decoding ran to exhaustion rather than to the declared count, so a container that
    # understates its length loses nothing. Overstating it by more than the one frame that
    # metadata rounding accounts for means the file is truncated, and a pose artefact covering
    # part of a lesson would be silently misaligned with the lesson.
    if decoded + 1 < declared_frames:
        raise FrameProcessingFailed(
            f"{source} declares {declared_frames} frames and stopped decoding after {decoded}. "
            f"The file is truncated; preprocessing part of a session is not an outcome this "
            f"pipeline offers.")

    if not sampled:
        raise PreprocessError(f"{source} yielded no sampled frames at "
                              f"{config.preprocess.sample_fps} fps")

    tracks: list[Track] = tracker.finished()
    signals = teacher_module.measure(
        tracks, frame_count=len(sampled), frame_width=width, frame_height=height,
        setup=setup, track_positions=positions, track_areas=areas)
    proposal = teacher_module.propose(signals, config.preprocess.teacher_id,
                                      zones_available=setup is not None)

    # Aggregates are computed while identities still exist, then the identities go no further.
    learner_bins = aggregates.summarise(
        tracked_frames, teacher_track_id=proposal.track_id, bin_seconds=learner_bin_seconds,
        sampled_fps=config.preprocess.sample_fps, frame_width=width, frame_height=height,
        setup=setup)

    keypoints, track_ids, person_count = artifacts.pack(sampled)
    artifact = artifacts.write(
        pose_output, session_id=session_id, keypoints=keypoints, track_ids=track_ids,
        person_count=person_count, sampled_fps=config.preprocess.sample_fps,
        model_version=estimator.model_version,
        config_sha256=artifacts.config_digest(config.preprocess),
        frame_width=width, frame_height=height, teacher_track_id=proposal.track_id)

    # Last, and only once the blurred artefact is closed and the pose is on disk. A failure
    # before this point leaves the original intact, which is recoverable; a failure after it
    # would have destroyed the source with nothing to show for it.
    blur_module.remove_original(source, policy)

    return PreprocessResult(
        artifact=artifact, proposal=proposal, learner_bins=learner_bins,
        blurred_path=blurred_destination, original_deleted=not source.exists(),
        frames_processed=len(sampled), faces_blurred=faces_blurred,
        zones_available=setup is not None)


def teacher_keypoints(artifact_path: Path, track_id: int,
                      expected_sha256: str | None = None) -> np.ndarray:
    """(F, 17, 3) for the confirmed teacher, for Phase 4 to slice into clips.

    Takes the track id from the caller rather than reading the proposal out of the file, so
    that a caller who has not consulted `teacher_tracks.confirmed_by` cannot accidentally train
    on an unconfirmed guess.
    """
    return artifacts.load(artifact_path, expected_sha256).track(track_id)
