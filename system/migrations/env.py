# -*- coding: utf-8 -*-
"""Alembic environment.

The database URL comes from the environment, never from alembic.ini, so a password cannot be
committed by accident.

SCHEMA.md is the source of truth and migrations are written to match it. Autogenerate is
deliberately not wired to a metadata object yet: the append-only triggers, the REVOKE grants,
and the array-overlap CHECK that enforces R2 are all things autogenerate does not see and
would happily drop on the next revision.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    config.set_main_option("sqlalchemy.url", DATABASE_URL)

target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection, for review before it is applied."""
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("DATABASE_URL is unset; alembic has no database to target")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    if not config.get_main_option("sqlalchemy.url"):
        raise RuntimeError("DATABASE_URL is unset; alembic has no database to target")
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
