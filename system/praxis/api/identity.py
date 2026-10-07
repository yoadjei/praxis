# -*- coding: utf-8 -*-
"""Who is calling, as claimed, and whether their role may do this.

One implementation, shared by every route that needs it. It lived inside the ingest router until
a second endpoint needed the same three roles, and L20 is explicit that two implementations of
one rule is a recurring defect source: the copy would have been the one that kept accepting a
role after the list changed.

**Nothing verifies the claim yet.** Phase 9 adds `users`, token issue and verification. Until
then the header is recorded and trusted, which is a dated gap rather than an oversight, and the
reason every caller refuses a request without one: a session ingested or a teacher confirmed
with no actor recorded is a hole in the audit trail that cannot be filled in retrospectively.
"""
from __future__ import annotations

from fastapi import HTTPException

from praxis.ids import is_ulid

# API.md sections 3 and 4 give the same three roles to ingest and to teacher confirmation.
ALLOWED_ROLES = ("supervisor", "researcher", "admin")


def actor(authorization: str | None, *, action: str) -> tuple[str | None, str | None]:
    """The claimed `(user_id, role)`, or a refusal naming what was being attempted.

    `action` reaches the 403 body so a reviewer denied one endpoint is told which one, rather
    than reading a message about uploading while trying to confirm a track.

    A user id that is present and is not a ULID is refused here. `audit_log.actor_user_id` is
    `CHAR(26)` and blank-pads anything shorter, which would make every subsequent verification
    of the chain report a break (see `praxis.audit.write._storable_actor`). The audit layer
    refuses it too; catching it here is what turns a 500 out of the writer into a 401 that
    tells the caller what is wrong with their token.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail={
            "type": "/errors/unauthenticated",
            "detail": "a bearer token is required; it is recorded but not yet verified"})

    claimed = authorization.split(" ", 1)[1].strip()
    role, _, user_id = claimed.partition(":")
    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=403, detail={
            "type": "/errors/role-forbidden",
            "detail": f"{action} is limited to {', '.join(ALLOWED_ROLES)}"})

    # Absent is allowed and means the actor is unknown, which stores faithfully as null. A
    # malformed one is not: it would be recorded as a person who cannot exist.
    if user_id and not is_ulid(user_id):
        raise HTTPException(status_code=401, detail={
            "type": "/errors/unauthenticated",
            "detail": f"the token names user {user_id!r}, which is not a ULID. The audit "
                      f"trail stores an actor in a fixed-width column and a shorter value "
                      f"would be padded, breaking every later verification of the chain."})
    return (user_id or None), role
