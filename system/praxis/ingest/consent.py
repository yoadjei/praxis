# -*- coding: utf-8 -*-
"""Consent, checked before a single byte is written.

API.md section 3: "Consent is checked *before* a single byte is written to disk. A refusal must
leave the system byte-identical to its prior state." That ordering is the whole reason this
module is separate from the rest of ingest and takes only a connection and an identifier: it has
to be callable before a stream is opened.

The liveness predicate is not reimplemented here. `active_consents` is a view, and asking it is
what stops a second definition of "withdrawn or expired" drifting away from the first. This
module adds only the check the view cannot make, which is whether the consent covers the domain
of the session being offered.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import Connection, select

from praxis.contracts.session import Domain
from praxis.db.schema import active_consents, consent_records
from praxis.ingest.errors import consent_invalid

# Act 843 s.27(1) requires the purpose and the recipients to be stated, and `scope` is what
# records which kind of recording the subject agreed to. "both" covers either domain.
COVERAGE: dict[str, frozenset[str]] = {
    "microteaching": frozenset({"microteaching"}),
    "classroom": frozenset({"classroom"}),
    "both": frozenset({"microteaching", "classroom"}),
}


@dataclass(frozen=True)
class ActiveConsent:
    consent_id: str
    college_id: str
    teacher_id: str | None
    subject_type: str
    scope: str
    granted_on: date
    expires_on: date | None


def resolve(connection: Connection, consent_id: str | None, domain: Domain) -> ActiveConsent:
    """The consent this upload will be recorded under, or a refusal saying precisely why.

    Every branch returns 422 with a distinguishable detail. A single "consent invalid" would
    make the four cases indistinguishable to the operator, who can fix three of them.
    """
    if not consent_id:
        raise consent_invalid("consent_id is required; ingest stores nothing without one")

    row = connection.execute(
        select(active_consents).where(active_consents.c.consent_id == consent_id)
    ).mappings().first()

    if row is None:
        raise consent_invalid(_why_not_active(connection, consent_id))

    if domain not in COVERAGE.get(row["scope"], frozenset()):
        raise consent_invalid(
            f"consent scope {row['scope']!r} does not cover session domain {domain!r}")

    return ActiveConsent(
        consent_id=row["consent_id"], college_id=row["college_id"],
        teacher_id=row["teacher_id"], subject_type=row["subject_type"], scope=row["scope"],
        granted_on=row["granted_on"], expires_on=row["expires_on"])


def _why_not_active(connection: Connection, consent_id: str) -> str:
    """Absent from the view means unknown, withdrawn or expired, and the operator's next step
    differs in each case, so the record itself is consulted to say which."""
    row = connection.execute(
        select(consent_records.c.withdrawn_on, consent_records.c.expires_on)
        .where(consent_records.c.consent_id == consent_id)
    ).mappings().first()

    if row is None:
        return f"consent {consent_id!r} does not exist"
    if row["withdrawn_on"] is not None:
        return f"consent {consent_id!r} was withdrawn on {row['withdrawn_on']}"
    if row["expires_on"] is not None:
        return f"consent {consent_id!r} expired on {row['expires_on']}"
    # The view and the record disagree, which should be impossible while the view is what
    # 0002 creates. Reported rather than swallowed, because a silently absent consent would
    # read to the operator as "unknown" and send them to create a duplicate.
    return (f"consent {consent_id!r} exists and is neither withdrawn nor expired, yet "
            f"active_consents does not return it; the view definition has changed")
