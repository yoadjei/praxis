# -*- coding: utf-8 -*-
"""Phase 7: why a detection was made, and whether that account is honest.

Grad-CAM is the method and the intrinsic attention the fallback, and which one the interface
uses is decided by `explain.method` rather than by code. A method that fails either of the
Adebayo et al. (2018) sanity checks **on this model and this data** is withdrawn from the
interface, and the failure is reported in the thesis rather than buried.

Guided Grad-CAM is forbidden and has no implementation here. It fails the parameter
randomisation test while producing the most convincing pictures of the methods Adebayo et al.
examined, which is the reason for the prohibition rather than an argument against it.
"""
from praxis.explain.base import (
    FORBIDDEN,
    ClipInput,
    Explainer,
    ExplainError,
    Explanation,
    available,
    normalise,
    register,
    select,
)
from praxis.explain.fidelity import (
    CheckResult,
    FidelityError,
    FidelityReport,
    RandomisationStage,
    Similarity,
    compare_maps,
    data_randomisation_test,
    deletion_insertion_auc,
    parameter_randomisation_test,
    reinitialise,
    run_fidelity,
)
from praxis.explain.gradcam import GradCAM, resolve_layer
from praxis.explain.intrinsic import IntrinsicExplainer

__all__ = [
    "FORBIDDEN",
    "CheckResult",
    "ClipInput",
    "ExplainError",
    "Explainer",
    "Explanation",
    "FidelityError",
    "FidelityReport",
    "GradCAM",
    "IntrinsicExplainer",
    "RandomisationStage",
    "Similarity",
    "available",
    "compare_maps",
    "data_randomisation_test",
    "deletion_insertion_auc",
    "normalise",
    "parameter_randomisation_test",
    "register",
    "reinitialise",
    "resolve_layer",
    "run_fidelity",
    "select",
]
