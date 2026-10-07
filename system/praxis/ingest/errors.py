# -*- coding: utf-8 -*-
"""Why an upload was refused, in a form both a person and an endpoint can use.

BUILD-SPEC Phase 1: "a deliberately corrupt mp4 fails the gate with a specific reason, not a
stack trace". That means the reason has to be a value carried deliberately, not a message
scraped out of an exception, because the endpoint has to turn it into a status code and the
audit trail has to record it as a field somebody can count later.

API.md returns RFC 7807 style bodies, so each refusal carries the `type` slug that document
uses. The status lives here rather than in the route: whether a missing consent record is a 422
is a fact about the rule, and a second route that ingested media would have to make the same
choice again.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Refusal:
    """One reason an upload did not become a session."""

    slug: str
    status: int
    detail: str

    def as_problem(self) -> dict[str, object]:
        """The response body API.md section 3 specifies."""
        return {"type": f"/errors/{self.slug}", "detail": self.detail, "status": self.status}


class IngestRefused(Exception):
    """Raised where the refusal is decided, carrying what the caller must report."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(refusal.detail)
        self.refusal = refusal


def refuse(slug: str, status: int, detail: str) -> IngestRefused:
    return IngestRefused(Refusal(slug=slug, status=status, detail=detail))


# --- consent, which is checked before a single byte is written --------------------------

def consent_invalid(detail: str) -> IngestRefused:
    return refuse("consent-invalid", 422, detail)


# --- the media itself -------------------------------------------------------------------

def unreadable_media(detail: str) -> IngestRefused:
    """ffprobe could not parse the file.

    422 and nothing stored, because `media_objects.duration_s`, `width`, `height` and `fps` are
    all NOT NULL and an unparseable file supplies none of them. There is no row that could be
    written, so this is the one gate outcome that is a refusal rather than a verdict. D45.
    """
    return refuse("media-unreadable", 422, detail)


def unsupported_container(extension: str, accepted: tuple[str, ...]) -> IngestRefused:
    return refuse("media-unsupported", 415,
                  f"{extension!r} is not accepted; this endpoint takes "
                  f"{' or '.join(accepted)}")


def too_large(limit_bytes: int) -> IngestRefused:
    return refuse("media-too-large", 413,
                  f"the upload exceeds the {limit_bytes} byte limit and was discarded")


def duplicate_session(media_sha256: str) -> IngestRefused:
    """The same file, the same teacher, the same day. D51."""
    return refuse("session-duplicate", 409,
                  f"a session already exists for {media_sha256[:12]} with this teacher on this "
                  f"date")


def media_volume_unreachable(detail: str) -> IngestRefused:
    """The external volume is not mounted. 503, because retrying later is the right advice."""
    return refuse("media-volume-unreachable", 503, detail)
