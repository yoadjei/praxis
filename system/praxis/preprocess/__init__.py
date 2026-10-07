# -*- coding: utf-8 -*-
"""Phase 3: video becomes keypoints, zones, and a blurred artifact.

Nothing in this package imports ultralytics. Its tracker resolves DNS and pip-installs missing
extras while being imported, which is a network call inside the inference path. R6, D57.
"""
from praxis.preprocess.aggregates import LearnerBin, hands_per_minute, summarise
from praxis.preprocess.artifacts import PoseArtifact, load, pack, write
from praxis.preprocess.blur import BlurPolicy, OriginalSurvived, blur_frame, remove_original
from praxis.preprocess.pipeline import (
                                        FrameProcessingFailed,
                                        PreprocessError,
                                        PreprocessResult,
                                        VideoUnreadable,
                                        run,
                                        sample_step,
                                        teacher_keypoints,
)
from praxis.preprocess.pose import (
                                        COCO_KEYPOINTS,
                                        CPU_ONLY_PROVIDERS,
                                        POSE_INPUT_SIZE,
                                        Detection,
                                        OnnxPoseEstimator,
                                        PoseEstimator,
                                        WeightsIncompatible,
                                        WeightsInvalid,
                                        WeightsMissing,
                                        box_iou_matrix,
)
from praxis.preprocess.store import (
                                        PersistError,
                                        confirm_teacher,
                                        confirmed_teacher_track,
                                        persist,
                                        save_camera_setup,
)
from praxis.preprocess.teacher import TeacherProposal, TrackSignals, propose
from praxis.preprocess.tracking import ByteTrackConfig, ByteTracker, Track, TrackState
from praxis.preprocess.zones import CameraSetup, Polygon

__all__ = [
    "COCO_KEYPOINTS", "CPU_ONLY_PROVIDERS", "POSE_INPUT_SIZE", "BlurPolicy", "ByteTrackConfig",
    "ByteTracker", "CameraSetup", "Detection", "FrameProcessingFailed", "LearnerBin",
    "OnnxPoseEstimator", "OriginalSurvived", "PersistError", "Polygon", "PoseArtifact",
    "PoseEstimator", "PreprocessError", "PreprocessResult", "TeacherProposal", "Track",
    "TrackSignals", "TrackState", "VideoUnreadable", "WeightsIncompatible", "WeightsInvalid",
    "WeightsMissing", "blur_frame", "box_iou_matrix", "confirm_teacher",
    "confirmed_teacher_track", "hands_per_minute", "load", "pack", "persist", "propose",
    "remove_original", "run", "sample_step", "save_camera_setup", "summarise",
    "teacher_keypoints", "write",
]
