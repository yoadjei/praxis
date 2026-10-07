# -*- coding: utf-8 -*-
"""Connecting to the database, and agreeing on where the address comes from.

One fact in two notations. Alembic and SQLAlchemy want a URL; psycopg, psql and `pg_isready`
want libpq keywords. Rather than let each caller invent its own variable, both are accepted and
converted here: `DATABASE_URL` wins, `PRAXIS_TEST_DSN` is the fallback, and nothing else is
consulted. R7 does not care which notation a run used, only that the run can say which database
it meant.

No pooling configuration is invented here. A Celery worker and a test process want different
pools, and guessing on behalf of a worker that does not exist yet would be a number nobody
chose.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import quote, unquote, urlsplit

from sqlalchemy import Connection, Engine, create_engine

URL_VAR = "DATABASE_URL"
KEYWORD_VAR = "PRAXIS_TEST_DSN"
DRIVER = "postgresql+psycopg"


class DatabaseNotConfigured(RuntimeError):
    """Neither variable is set. Phases that do not touch the database never ask."""


def keywords_to_url(dsn: str) -> str:
    """`host=h port=5433 dbname=d user=u password=p` into a SQLAlchemy URL."""
    parts = dict(piece.split("=", 1) for piece in dsn.split() if "=" in piece)
    missing = {"host", "port", "dbname", "user"} - set(parts)
    if missing:
        raise ValueError(f"{KEYWORD_VAR} is missing {', '.join(sorted(missing))}")
    password = f":{quote(parts['password'], safe='')}" if parts.get("password") else ""
    return (f"{DRIVER}://{quote(parts['user'], safe='')}{password}"
            f"@{parts['host']}:{parts['port']}/{parts['dbname']}")


def url_to_keywords(url: str) -> str:
    """The inverse, for handing an address to psycopg or to a command-line tool."""
    split = urlsplit(url)
    if not split.hostname or not split.path.lstrip("/"):
        raise ValueError(f"{url!r} names no host or no database")
    pieces = [f"host={split.hostname}", f"port={split.port or 5432}",
              f"dbname={split.path.lstrip('/')}"]
    if split.username:
        pieces.append(f"user={unquote(split.username)}")
    if split.password:
        pieces.append(f"password={unquote(split.password)}")
    return " ".join(pieces)


def database_url(required: bool = True) -> str | None:
    """The configured address as a SQLAlchemy URL, or None when none is configured."""
    url = os.environ.get(URL_VAR, "").strip()
    if not url:
        keywords = os.environ.get(KEYWORD_VAR, "").strip()
        url = keywords_to_url(keywords) if keywords else ""

    if url:
        return url
    if required:
        raise DatabaseNotConfigured(
            f"set {URL_VAR} to a SQLAlchemy URL or {KEYWORD_VAR} to libpq keywords. "
            f"`python scripts/dev_services.py dsn` prints one for the local cluster.")
    return None


def build_engine(url: str | None = None, **options) -> Engine:
    """An engine with autocommit off. Every write in this system names its own transaction."""
    return create_engine(url or database_url(), future=True, **options)


@contextmanager
def transaction(engine: Engine, serializable: bool = False) -> Iterator[Connection]:
    """One explicit transaction, committed on success and rolled back on anything else.

    `serializable` is for the audit chain, which cannot tolerate two writers computing the same
    predecessor. It is off by default because paying for it on reads would teach people to turn
    it off in the one place it matters.
    """
    connection = engine.connect()
    if serializable:
        connection = connection.execution_options(isolation_level="SERIALIZABLE")
    try:
        with connection.begin():
            yield connection
    finally:
        connection.close()
