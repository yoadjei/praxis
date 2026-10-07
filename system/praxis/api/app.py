# -*- coding: utf-8 -*-
"""The FastAPI application.

Everything a route needs is put on `app.state` by the factory rather than imported at module
scope. That is what lets a test build an app against a scratch database and a temporary media
root without patching anything, and it is the same reason R7 wants a run configured from one
place: a route that reached for a global engine would be configured by whatever imported first.

The API layer is the one part of the system excluded from R6's network scan, because it is a
network service by definition. R6 is about the model never reaching out, not about the server
never listening, and `tests/test_invariants.py` says so explicitly.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from praxis.api.routes import annotation as annotation_routes
from praxis.api.routes import dashboard as dashboard_routes
from praxis.api.routes import ingest as ingest_routes
from praxis.api.routes import media as media_routes
from praxis.api.routes import tracks as tracks_routes
from praxis.config import load_config
from praxis.config_schema import PraxisConfig
from praxis.db.engine import build_engine
from praxis.tasks import select_queue
from praxis.tools import resolve


def create_app(*, config: PraxisConfig | None = None, engine=None,
               media_root: Path | None = None,
               enqueue: Callable[[str], None] | None = None) -> FastAPI:
    """Build the application. Every dependency is injectable, and each has a working default."""
    config = config or load_config()
    app = FastAPI(title="PRAXIS", version="1", docs_url="/api/v1/docs",
                  openapi_url="/api/v1/openapi.json")

    app.state.config = config
    app.state.engine = engine if engine is not None else build_engine()
    app.state.media_root = media_root or config.require_media_root()
    app.state.enqueue = enqueue if enqueue is not None else select_queue()
    app.state.ffprobe = resolve("ffprobe", config.tools.ffprobe)
    app.state.ffmpeg = resolve("ffmpeg", config.tools.ffmpeg)

    app.include_router(ingest_routes.router)
    app.include_router(annotation_routes.router)
    app.include_router(dashboard_routes.router)
    app.include_router(media_routes.router)
    app.include_router(tracks_routes.router)

    @app.exception_handler(StarletteHTTPException)
    def problem_response(_request, exc: StarletteHTTPException) -> JSONResponse:
        """RFC 7807 style bodies, and never a stack trace.

        BUILD-SPEC Phase 1 requires a corrupt file to fail "with a specific reason, not a stack
        trace". The refusals already carry a slug and a detail; this is what stops anything else
        leaking a traceback into a response on the way out.
        """
        detail = exc.detail
        if not isinstance(detail, dict):
            detail = {"type": "/errors/request-failed", "detail": str(detail)}
        return JSONResponse(status_code=exc.status_code,
                            content={**detail, "status": exc.status_code})

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
