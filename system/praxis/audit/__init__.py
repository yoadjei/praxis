# -*- coding: utf-8 -*-
"""The audit trail. Append-only, hash-chained, and replayable.

Rule R5 is a database guarantee first and a Python one second. `chain.py` provides the part a
superuser cannot quietly defeat; `migrations/versions/0001_append_only_audit.py` provides the
part that stops everything below one.
"""
from praxis.audit.chain import AuditLog, AuditRecord, ChainBreak, verify
from praxis.audit.events import EVENTS, AuditError, EventSpec, spec_for
from praxis.audit.replay import AdjudicationState, SystemState, replay

__all__ = [
    "EVENTS",
    "AdjudicationState",
    "AuditError",
    "AuditLog",
    "AuditRecord",
    "ChainBreak",
    "EventSpec",
    "SystemState",
    "replay",
    "spec_for",
    "verify",
]
