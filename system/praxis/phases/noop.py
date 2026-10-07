# -*- coding: utf-8 -*-
"""The phase that does nothing, which is the only phase that can test the harness.

Phase 0's acceptance test needs a run that exercises freezing the config, seeding, resolving the
device and writing a manifest, without depending on any of the work those things support.
`tests/test_invariants.py::test_run_is_deterministic` runs this twice and compares the manifests
field by field, so its summary must stay byte-identical between two runs of the same config.
"""
from __future__ import annotations

from praxis.phases import PhaseContext, PhaseResult, register


@register("noop", "Nothing, so that the harness itself can be tested")
def noop(context: PhaseContext) -> PhaseResult:
    return PhaseResult(summary={
        "phase": "noop",
        "behaviours": sorted(context.config.behaviour.heads.presence_behaviours),
    })
