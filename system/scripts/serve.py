# -*- coding: utf-8 -*-
"""Serve the API and the built frontend locally, in one command, from any shell.

    python scripts/serve.py
    python scripts/serve.py --port 8080 --no-frontend
    python scripts/serve.py --start-database

This exists because the alternative is three environment variables set by hand before uvicorn,
and that went wrong in every shell it was tried in. The incantation was written out in bash,
pasted into PowerShell, and `export` is not a PowerShell command: the variables silently did not
exist, and the traceback that followed was `DatabaseNotConfigured` from deep inside
`create_app()`, which reads as a broken application rather than as a shell mismatch. A launcher
is immune to that, because the environment it configures is its own.

Nothing here is a second source of configuration. The database address comes from
`dev_services.dsn()`, the same function `dev_services.py dsn` prints, so the cluster is named in
one place. Anything already set in the environment is left exactly as it is, so this is a
convenience for a developer machine and never overrides a real deployment: run it with
DATABASE_URL set and that is the database it serves.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.db.engine import KEYWORD_VAR, URL_VAR
from scripts.dev_services import PGDATA, PORT, dsn, start

# D17 keeps project data off C:, and dev_services.py already hardcodes B:/praxis/pgdata with an
# environment override. The media root follows that precedent rather than inventing a second
# convention for where a developer machine keeps things.
MEDIA_ROOT = os.environ.get("PRAXIS_MEDIA_ROOT") or "B:/praxis/media"
MEDIA_VAR = "PRAXIS_MEDIA_ROOT"
SERVE_FRONTEND = "PRAXIS_SERVE_FRONTEND"
BUNDLE = REPO_ROOT / "frontend" / "dist" / "index.html"


def database_is_running() -> bool:
    """Whether the project's cluster answers, without importing pg_ctl's failure modes."""
    try:
        import psycopg
    except ImportError:
        return False
    try:
        with psycopg.connect(dsn(), connect_timeout=5):
            return True
    except psycopg.Error:
        return False


def configure(serve_frontend: bool) -> list[str]:
    """Set what the application reads, leaving anything already set alone. Returns a report."""
    notes = []

    if os.environ.get(URL_VAR) or os.environ.get(KEYWORD_VAR):
        notes.append(f"database   from the environment ({URL_VAR} or {KEYWORD_VAR})")
    else:
        # KEYWORD_VAR rather than URL_VAR: `dsn()` returns libpq keywords, and `database_url()`
        # converts those itself. Writing a URL here would mean formatting one in a second place.
        os.environ[KEYWORD_VAR] = dsn()
        notes.append(f"database   local cluster on port {PORT}")

    if os.environ.get(MEDIA_VAR):
        notes.append(f"media      from the environment: {os.environ[MEDIA_VAR]}")
    else:
        os.environ[MEDIA_VAR] = MEDIA_ROOT
        notes.append(f"media      {MEDIA_ROOT}")

    if serve_frontend:
        os.environ[SERVE_FRONTEND] = "1"
        notes.append(f"frontend   served at / from {BUNDLE.parent}")
    else:
        os.environ.pop(SERVE_FRONTEND, None)
        notes.append("frontend   not served; the API answers JSON at every path")

    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback by default; R6 keeps this off the network")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--no-frontend", action="store_true",
                        help="serve the API alone, without the built bundle")
    parser.add_argument("--start-database", action="store_true",
                        help="start the local cluster first if it is not already up")
    args = parser.parse_args(argv)

    serve_frontend = not args.no_frontend
    if serve_frontend and not BUNDLE.is_file():
        print(f"error: no built frontend at {BUNDLE.parent}. Run `npm ci && npm run build` in "
              f"frontend/, or pass --no-frontend to serve the API alone.", file=sys.stderr)
        return 2

    using_own_database = not (os.environ.get(URL_VAR) or os.environ.get(KEYWORD_VAR))
    if using_own_database and not database_is_running():
        if not args.start_database:
            print(f"error: the project's cluster is not answering on port {PORT} "
                  f"({PGDATA}). Start it with `python scripts/dev_services.py start`, or pass "
                  f"--start-database, or set {URL_VAR} to serve a different database.",
                  file=sys.stderr)
            return 2
        if start() != 0:
            return 2

    for note in configure(serve_frontend):
        print(f"  {note}")
    print(f"  serving    http://{args.host}:{args.port}")
    print("  stop with  Ctrl+C")

    # Imported here, after the environment is configured. `praxis.api.main` builds the
    # application at import time, so importing it any earlier would read the environment this
    # function has not finished writing - which is the failure this script exists to remove.
    import uvicorn

    uvicorn.run("praxis.api.main:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
