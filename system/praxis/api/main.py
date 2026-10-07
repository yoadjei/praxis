# -*- coding: utf-8 -*-
"""Uvicorn entry point for the PRAXIS API.

Both the factory and this module-level `app` are needed, and for different reasons.
`create_app()` is how a test builds an application against a scratch database and a temporary
media root without patching anything. `app` is how uvicorn starts one without arguments, which
is what the container command names. Keeping the entry point here and the factory in `app.py`
means neither has to compromise for the other.

Importing this module opens a database connection and resolves the media root, because
`create_app()` does. That is correct for an entry point and is why tests import the factory
instead: a module whose import had no side effects would be a module that had not configured
anything.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from praxis.api.app import create_app

SERVE_FRONTEND = "PRAXIS_SERVE_FRONTEND"
TRUTHY = frozenset({"1", "true", "yes", "on"})

app = create_app()

# Mounted after the routers, never before. StaticFiles at "/" with html=True answers everything,
# so mounting it first would swallow /api/v1 and the failure would look like a routing bug.
_bundle = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if os.environ.get(SERVE_FRONTEND, "").strip().lower() in TRUTHY:
    if not (_bundle / "index.html").is_file():
        # Raised, not skipped. Asking for the frontend and getting an API that serves JSON at
        # every path is the kind of failure an operator debugs in the browser for an hour; the
        # cause is one missing build step and it should say so at startup.
        raise RuntimeError(
            f"{SERVE_FRONTEND} is set but no built frontend is at {_bundle}. "
            f"Run `npm ci && npm run build` in frontend/, or unset {SERVE_FRONTEND} to serve "
            f"the API alone.")
    app.mount("/", StaticFiles(directory=_bundle, html=True), name="frontend")
