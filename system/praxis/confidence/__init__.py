# -*- coding: utf-8 -*-
"""Confidence: what the model believes, and whether that belief is checkable.

This is the construct that separates this system from the ones it is compared against. Rule R3
forbids a bare label leaving the inference layer, and R4 requires calibration wherever accuracy
is reported, so nothing here is an optional diagnostic.

Three methods, and the ordering between them is the thesis's central technical argument.
Temperature scaling (Guo et al., 2017) is the **baseline**, expected to hold on i.i.d. data and
to fail under shift. Deep ensembles (Ovadia et al., 2019) are the **production method**, because
marginalising over models is what held up in their comparison. MC dropout is the cheap middle
option and is never a substitute for the ensemble.

    from praxis.confidence import calibration_report, fit_temperature, combine

`ood.py` belongs to Phase 6 and is not built yet, so an out-of-distribution score is supplied
to `to_confidence_state` by the caller rather than manufactured here.
"""
from praxis.confidence.ensemble import (
    IDENTICAL_TOLERANCE,
    SHARED_PREFIXES,
    DiversityReport,
    EnsembleError,
    EnsemblePrediction,
    assert_members_differ,
    check_members_share_device,
    combine,
    mc_dropout_combine,
    mutual_information,
    pairwise_weight_distance,
)
from praxis.confidence.metrics import (
    DEFAULT_BINS,
    MAX_BINARY_ENTROPY,
    BinningMode,
    CalibrationReport,
    ReliabilityBin,
    binary_entropy,
    brier_score,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    reliability_diagram,
)
from praxis.confidence.temperature import (
    Temperature,
    TemperatureScalingFailed,
    apply_temperature,
    fit_per_behaviour,
    fit_temperature,
    probabilities_to_logits,
)

__all__ = [
    "DEFAULT_BINS",
    "IDENTICAL_TOLERANCE",
    "MAX_BINARY_ENTROPY",
    "SHARED_PREFIXES",
    "BinningMode",
    "CalibrationReport",
    "DiversityReport",
    "EnsembleError",
    "EnsemblePrediction",
    "ReliabilityBin",
    "Temperature",
    "TemperatureScalingFailed",
    "apply_temperature",
    "assert_members_differ",
    "binary_entropy",
    "brier_score",
    "calibration_report",
    "check_members_share_device",
    "combine",
    "expected_calibration_error",
    "fit_per_behaviour",
    "fit_temperature",
    "maximum_calibration_error",
    "mc_dropout_combine",
    "mutual_information",
    "negative_log_likelihood",
    "pairwise_weight_distance",
    "probabilities_to_logits",
    "reliability_bins",
    "reliability_diagram",
]
