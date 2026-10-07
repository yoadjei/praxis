# -*- coding: utf-8 -*-
"""Annotation: the codebook's authority, and the human agreement band.

Phase 2 comes before any modelling. Training data does not exist yet, and inter-rater
agreement is a reported thesis result rather than a preprocessing step: without it there is
no ground truth and no baseline for a model number to mean anything against.

    from praxis.annotation import report
    table = report(annotations, raters=roster)
    table.observed_order()      # against the order CODEBOOK.md §7 registers in advance
"""
from praxis.annotation.clips import ClipRef, clip_plan, parse_clip_id
from praxis.annotation.codebook import (
    ACTIVE_CODEBOOK,
    CODEBOOK_V1,
    EXPECTED_AGREEMENT_ORDER,
    BehaviourSpec,
    Codebook,
    CodebookError,
    FieldSpec,
    ScaleType,
)
from praxis.annotation.irr import (
    DEFAULT_ALPHA_GATE,
    AgreementReport,
    Annotation,
    ArtefactChecks,
    BehaviourAgreement,
    FieldAgreement,
    RangeCheck,
    Rater,
    report,
)
from praxis.annotation.server import (
    AnnotationRefused,
    Assignment,
    CalibrationOutcome,
    Disagreement,
    accept_annotation,
    build_assignments,
    calibration_round,
    disagreements,
)
from praxis.annotation.statistics import (
    IccResult,
    Interval,
    bootstrap_interval,
    icc_2k,
    krippendorff_alpha,
    mean_pairwise_kappa,
    quadratic_weighted_kappa,
)

__all__ = [
    "ACTIVE_CODEBOOK",
    "CODEBOOK_V1",
    "DEFAULT_ALPHA_GATE",
    "EXPECTED_AGREEMENT_ORDER",
    "AgreementReport",
    "Annotation",
    "AnnotationRefused",
    "ArtefactChecks",
    "Assignment",
    "BehaviourAgreement",
    "BehaviourSpec",
    "CalibrationOutcome",
    "ClipRef",
    "Codebook",
    "CodebookError",
    "Disagreement",
    "FieldAgreement",
    "FieldSpec",
    "IccResult",
    "Interval",
    "RangeCheck",
    "Rater",
    "ScaleType",
    "accept_annotation",
    "bootstrap_interval",
    "build_assignments",
    "calibration_round",
    "clip_plan",
    "disagreements",
    "icc_2k",
    "krippendorff_alpha",
    "mean_pairwise_kappa",
    "parse_clip_id",
    "quadratic_weighted_kappa",
    "report",
]
