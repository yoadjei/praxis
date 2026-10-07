# -*- coding: utf-8 -*-
"""Phases 6, 7 and 8: shift analysis, explanation fidelity, and routing decisions.

Phase 6 - Shift: domain shift, OOD and attribution
  Evaluates the model at each shift level present, builds the shift table, runs OOD
  evaluation, chooses an abstention threshold, and runs attribution regression.

Phase 7 - Explain: explanation and its honesty
  Selects an explainer by configured method, produces explanations for a sample of
  clips, and runs fidelity checks to catch broken explainers.

Phase 8 - Routing: routing gate, review and audit
  Applies the routing gate to scored detections, summarises what was suppressed,
  and appends the decision to the audit trail.
"""
from __future__ import annotations

from praxis.phases import PhaseContext, PhaseResult, abstain, register
from praxis.phases.chain import ArtefactMissing, latest_run


@register("6", "Domain shift, OOD and attribution", needs=("5",))
def shift(context: PhaseContext) -> PhaseResult:
    """Evaluate model at each shift level, measure drop, and attribute it."""
    engine = context.engine()
    if engine is None:
        return abstain(
            "phase 6 reads predictions from the database",
            "DATABASE_URL or PRAXIS_TEST_DSN")

    try:
        phase_5_run = latest_run(context.config, "5")
    except ArtefactMissing:
        phase_5_run = None

    if phase_5_run is None:
        return abstain(
            "no completed run of phase 5 was found under the configured run outputs directory",
            "phase 5 complete (run: python scripts/run_phase.py --phase 5)")

    # Phase 6 is not yet fully implemented: it needs to discover and use the actual
    # artefacts phase 5 produced, query predictions from the database, evaluate at
    # each shift level, and run attribution. Until those are built this phase abstains.
    return abstain(
        "phase 6 implementation is incomplete; phase 5 has run but phase 6 has not yet "
        "wired the predictions, evaluations, and attribution code",
        "full phase 6 implementation")


@register("7", "Explanation and its honesty", needs=("4",))
def explain(context: PhaseContext) -> PhaseResult:
    """Select an explainer, produce sample explanations, and check their honesty."""
    try:
        phase_4_run = latest_run(context.config, "4")
    except ArtefactMissing:
        phase_4_run = None

    if phase_4_run is None:
        return abstain(
            "no completed run of phase 4 was found under the configured run outputs directory",
            "phase 4 complete (run: python scripts/run_phase.py --phase 4)")

    # Phase 7 is not yet fully implemented: it needs to load the trained model,
    # select an explainer, produce explanations for a sample of clips, and run
    # fidelity checks. Until those are built this phase abstains.
    return abstain(
        "phase 7 implementation is incomplete; phase 4 has run but phase 7 has not yet "
        "wired the explainer selection, explanation generation, and fidelity checks",
        "full phase 7 implementation")


@register("8", "Routing gate, review and audit", needs=("5",))
def routing(context: PhaseContext) -> PhaseResult:
    """Apply routing gate to predictions and record decisions to audit trail."""
    engine = context.engine()
    if engine is None:
        return abstain(
            "phase 8 reads scored detections and writes to the audit trail",
            "DATABASE_URL or PRAXIS_TEST_DSN")

    try:
        phase_5_run = latest_run(context.config, "5")
    except ArtefactMissing:
        phase_5_run = None

    if phase_5_run is None:
        return abstain(
            "no completed run of phase 5 was found under the configured run outputs directory",
            "phase 5 complete (run: python scripts/run_phase.py --phase 5)")

    # Phase 8 is not yet fully implemented: it needs to load predictions from
    # the database, apply the routing gate, and append decisions to the audit trail.
    # Until those are built this phase abstains.
    return abstain(
        "phase 8 implementation is incomplete; phase 5 has run but phase 8 has not yet "
        "wired the gate application, decision recording, and audit trail append",
        "full phase 8 implementation")
