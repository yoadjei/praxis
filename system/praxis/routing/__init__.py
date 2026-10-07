# -*- coding: utf-8 -*-
"""The routing gate and the cognitive forcing function.

`gate.py` decides what a reviewer is shown. `reveal.py` decides when they are shown it, and
records what they did in the meantime, which is the reliance study's raw data.
"""
from praxis.routing.gate import (
    GateDecision,
    GateError,
    GatePolicy,
    RoutingSummary,
    decide,
    route,
    route_all,
    summarise,
)
from praxis.routing.reveal import (
    RevealError,
    RevealPolicy,
    ReviewItem,
    ReviewSession,
    evidence_only,
)

__all__ = [
    "GateDecision",
    "GateError",
    "GatePolicy",
    "RevealError",
    "RevealPolicy",
    "ReviewItem",
    "ReviewSession",
    "RoutingSummary",
    "decide",
    "evidence_only",
    "route",
    "route_all",
    "summarise",
]
