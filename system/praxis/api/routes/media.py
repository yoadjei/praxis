# -*- coding: utf-8 -*-
"""GET /api/v1/media/* - video and single frames, for annotation and teacher confirmation.

The annotation tool seeks through blurred videos to find clips. This endpoint serves the blurred
copy for a session, enforcing D18 (faces blurred, original deleted, only the blurred is
reachable) and R1 (no identity leakage). It refuses, with a distinct reason, when: the session
does not exist; is_blurred is false; blurred_relative_path is null; or the file is not on disk.

`/frame` serves one decoded still from that same file, for any caller that needs to show a
person what a session looks like without downloading it. Resolving the session id to a path, and
proving that path stays inside the media root, is `praxis.api.locate`'s job rather than this
module's: the tracks router needs the same resolution, and a second copy of it is how one of the
two ends up with the join and without the containment check.

HTTP Range requests are what let the player seek to a clip without downloading everything
before it, and they are handled by starlette's FileResponse rather than here. It already
implements the whole of RFC 7233 - 206 with Content-Range, 416 with the real size, If-Range,
multipart ranges, and a body suppressed for HEAD - and this module used to carry a second
implementation of the same rules. That duplicate is what made a HEAD request stream the entire
file, which on a thirty-minute session is gigabytes sent to answer a question about headers
(L20: two implementations of one rule is a recurring defect source).

One deviation is inherited and accepted: starlette answers a Range header it cannot parse with
400, where RFC 7233 says an unintelligible Range should be ignored and the whole entity served.
No browser sends one, and reimplementing range handling to win that point is how the HEAD bug
got here in the first place.

Identity never appears: no teacher id, no college code, no filename. The session id in error
messages is already public to the caller.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import FileResponse, Response

from praxis.api.locate import blurred_video
from praxis.ingest.errors import IngestRefused
from praxis.preprocess.frames import jpeg_at

router = APIRouter(prefix="/api/v1", tags=["media"])


def _serve_video(request: Request, session_id: str) -> FileResponse:
    """The blurred video, in full or as a range.

    Handed to starlette, which reads the request's Range header out of the scope and does the
    rest: the whole file, a 206 window, a 416 with the true size, or headers alone for a HEAD.
    No `filename=` is passed, and that is load-bearing rather than an omission: it is the one
    thing that would put a Content-Disposition naming the stored file into every response, and
    the path is derived from the media hash (R1, D76).
    """
    return FileResponse(blurred_video(request, session_id), media_type="video/mp4")


@router.get("/media/{session_id}/video", response_model=None)
def get_video(request: Request, session_id: str) -> FileResponse:
    """GET /api/v1/media/{session_id}/video - Serve the blurred video for a session."""
    return _serve_video(request, session_id)


@router.head("/media/{session_id}/video", response_model=None)
def head_video(request: Request, session_id: str) -> FileResponse:
    """HEAD /api/v1/media/{session_id}/video - Return headers only, no body.

    HEAD returns the same headers as GET but with no response body. A HEAD request
    supports Range headers just like GET, allowing clients to check content length
    and range support without downloading the file.
    """
    return _serve_video(request, session_id)


# A reviewer identifying the teacher looks at a handful of stills per session, and each is
# decoded on demand rather than written to disk at preprocessing time. Caching them would mean
# a second copy of footage D18 exists to reduce to one, and an `-ss` seek costs the same on a
# thirty-minute file as on a two-minute one, so there is nothing to amortise.
@router.get("/media/{session_id}/frame", response_model=None)
def get_frame(
    request: Request,
    session_id: str,
    at_seconds: Annotated[float, Query(ge=0, description="offset into the session")] = 0.0,
    width: Annotated[int, Query(ge=64, le=1920, description="output width in pixels")] = 640,
) -> Response:
    """GET /api/v1/media/{session_id}/frame - one still out of the blurred video.

    Bounded on both query parameters rather than passed through: `width` reaches an ffmpeg
    scale filter, and an unbounded one is an invitation to ask for a 40000-pixel frame. A
    negative offset would be clamped silently by the sampler, so it is refused here instead,
    where the caller can be told.

    Raises:
        404 if the session, its blurred copy or the frame at that offset is not available.
            A timestamp past the end of the file is a request for something that is not there
            rather than a server fault, which is why it is not a 400.
    """
    path = blurred_video(request, session_id)
    try:
        image = jpeg_at(request.app.state.ffmpeg, path, at_seconds=at_seconds, width=width)
    except IngestRefused as refused:
        raise HTTPException(status_code=refused.refusal.status,
                            detail=refused.refusal.as_problem()) from refused

    # No Content-Disposition, for the same reason the video has none: the stored filename is
    # derived from the media hash and naming it in a response header would publish it (R1, D76).
    return Response(content=image, media_type="image/jpeg")
