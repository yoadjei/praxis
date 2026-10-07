# -*- coding: utf-8 -*-
"""The interfaces between stages.

BUILD-SPEC §4: never pass a bare dict between stages. Every one of these is frozen and sets
`extra="forbid"`, so a stage cannot smuggle an undeclared field past the boundary and a typo
is an error at the boundary rather than a missing value three stages later.
"""
from praxis.contracts.adjudication import MIN_RATIONALE_CHARS, Action, Adjudication, Arm
from praxis.contracts.confidence import (
    CalibrationMethod,
    ConfidenceState,
    ValidatedDomain,
)
from praxis.contracts.detection import (
    BEHAVIOUR_NAMES,
    BehaviourId,
    Detection,
    GateOutcome,
    SuppressionReason,
)
from praxis.contracts.manifest import ArtefactRef, RunManifest, SplitManifest
from praxis.contracts.session import Domain, Session

__all__ = [
    "BEHAVIOUR_NAMES",
    "MIN_RATIONALE_CHARS",
    "Action",
    "Adjudication",
    "Arm",
    "ArtefactRef",
    "BehaviourId",
    "CalibrationMethod",
    "ConfidenceState",
    "Detection",
    "Domain",
    "GateOutcome",
    "RunManifest",
    "Session",
    "SplitManifest",
    "SuppressionReason",
    "ValidatedDomain",
]
