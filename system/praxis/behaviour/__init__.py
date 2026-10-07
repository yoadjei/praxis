# -*- coding: utf-8 -*-
"""Phase 4: cached backbone features become behaviour predictions.

Two stages, because the backbone is frozen and the measured 1.31 GB of headroom does not fit
backpropagation through ResNet-50 over 64 frames at any batch size. Stage A (`backbone`) runs
once and caches; Stage B (`model`, `train`) trains on the cache and is cheap enough that an
ensemble is nearly free, which is what Phase 5 needs.

Nothing here downloads anything. `torchvision` honours neither `YOLO_OFFLINE` nor
`HF_HUB_OFFLINE`, so `FrozenBackbone` always builds with `weights=None` and loads a vendored
file. R6, D72.
"""
from praxis.behaviour.backbone import (
    BackboneError,
    FrozenBackbone,
    WeightsIncompatible,
    WeightsMissing,
    build_backbone,
)
from praxis.behaviour.baselines import (
    REQUIRED_BASELINES,
    Baseline,
    BaselineError,
    FrameAveragedResnet,
    KeypointsGBDT,
    LabelledClips,
    MajorityClass,
    build_baselines,
    keypoint_features,
)
from praxis.behaviour.dataset import (
    AugmentationPolicy,
    ClipDataset,
    ClipRecord,
    ClipSample,
    ClipVariant,
    DatasetError,
    balance_by_session,
    clips_per_session,
    collate,
    dataset_from_config,
    policy_from_config,
)
from praxis.behaviour.heads import (
    HeadKind,
    HeadSpec,
    behaviour_heads,
    decode_prediction,
    encode_labels,
    output_width,
)
from praxis.behaviour.losses import (
    LossWeights,
    behaviour_loss,
    categorical_loss,
    focal_loss,
    masked_mse,
    multitask_loss,
)
from praxis.behaviour.model import (
    BehaviourModel,
    BehaviourOutput,
    ModelShape,
    build_model,
    shape_from_config,
)
from praxis.behaviour.report import (
    FieldScore,
    FrozenReport,
    ReportError,
    SystemScores,
    compare,
    model_probabilities,
    score_system,
)
from praxis.behaviour.scoring import macro_f1_binary, macro_f1_multiclass
from praxis.behaviour.train import (
    EpochRecord,
    TrainingError,
    TrainingPlan,
    TrainingRun,
    cosine_with_warmup,
    plan_from_config,
    train,
    validation_scores,
    weights_from_config,
)

__all__ = [
    "REQUIRED_BASELINES",
    "AugmentationPolicy",
    "BackboneError",
    "Baseline",
    "BaselineError",
    "BehaviourModel",
    "BehaviourOutput",
    "ClipDataset",
    "ClipRecord",
    "ClipSample",
    "ClipVariant",
    "DatasetError",
    "EpochRecord",
    "FieldScore",
    "FrameAveragedResnet",
    "FrozenBackbone",
    "FrozenReport",
    "HeadKind",
    "HeadSpec",
    "KeypointsGBDT",
    "LabelledClips",
    "LossWeights",
    "MajorityClass",
    "ModelShape",
    "ReportError",
    "SystemScores",
    "TrainingError",
    "TrainingPlan",
    "TrainingRun",
    "WeightsIncompatible",
    "WeightsMissing",
    "balance_by_session",
    "behaviour_heads",
    "behaviour_loss",
    "build_backbone",
    "build_baselines",
    "build_model",
    "categorical_loss",
    "clips_per_session",
    "collate",
    "compare",
    "cosine_with_warmup",
    "dataset_from_config",
    "decode_prediction",
    "encode_labels",
    "focal_loss",
    "keypoint_features",
    "macro_f1_binary",
    "macro_f1_multiclass",
    "masked_mse",
    "model_probabilities",
    "multitask_loss",
    "output_width",
    "plan_from_config",
    "policy_from_config",
    "score_system",
    "shape_from_config",
    "train",
    "validation_scores",
    "weights_from_config",
]
