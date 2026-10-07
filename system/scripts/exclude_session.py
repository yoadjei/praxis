# -*- coding: utf-8 -*-
"""Record that a session will not be annotated, on a stated ground. Or undo that.

    python scripts/exclude_session.py --list
    python scripts/exclude_session.py <session_id> --reason "..." --actor <ULID>
    python scripts/exclude_session.py <session_id> --reinstate --reason "..." --actor <ULID>

A session nobody has confirmed a teacher track for and a session a researcher has decided not
to use look identical to `scripts/plan_annotation.py`: both are simply absent from the plan.
This is how the second is said out loud. The ground is required, it goes in the row, and the
decision goes on the audit chain, so a reader six months later can ask why a third of the corpus
is not being labelled and get an answer rather than an absence.

Dry by default. `--write` is the only thing that changes a row.

The actor must be a ULID: `audit_log.actor_user_id` is CHAR(26) and blank-pads anything shorter,
which would make every later verification of the chain report a break (D93).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select

from praxis.db.engine import build_engine
from praxis.db.schema import media_objects, sessions
from praxis.ids import is_ulid
from praxis.ingest.exclusion import ExclusionError, exclude, excluded, reinstate


class _DryRun(Exception):
    """Rolls back after the write has been built, so a dry run exercises the real path rather
    than a parallel one that predicts it. A prediction that never runs the UPDATE cannot
    discover a constraint violation."""


def show(engine) -> int:
    with engine.begin() as connection:
        current = excluded(connection)
        rows = connection.execute(
            select(sessions.c.session_id, sessions.c.excluded_at, media_objects.c.duration_s)
            .join(media_objects, media_objects.c.media_sha256 == sessions.c.media_sha256)
            .order_by(sessions.c.session_id)).all()

    total = sum(float(r.duration_s) for r in rows)
    out = sum(float(r.duration_s) for r in rows if r.excluded_at is not None)
    print(f"{len(rows)} session(s), {total / 60:.1f} minutes total")
    if not current:
        print("none excluded.")
        return 0

    print(f"{len(current)} excluded, {out / 60:.1f} minutes ({out / total:.0%} of the corpus):")
    for session_id, reason in current.items():
        print(f"  {session_id}")
        print(f"    {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_id", nargs="?", help="the session to exclude or reinstate")
    parser.add_argument("--reason", help="why; required for a change, and never defaulted")
    parser.add_argument("--actor", help="the deciding person's user id (a ULID)")
    parser.add_argument("--role", default="researcher", help="recorded against the audit event")
    parser.add_argument("--reinstate", action="store_true", help="undo an exclusion")
    parser.add_argument("--write", action="store_true", help="apply it; without this, nothing")
    parser.add_argument("--list", action="store_true", help="show what is excluded and exit")
    args = parser.parse_args(argv)

    engine = build_engine()
    if args.list or not args.session_id:
        return show(engine)

    if not args.reason:
        print("error: --reason is required. An exclusion with no stated ground is "
              "indistinguishable from unfinished work.", file=sys.stderr)
        return 2
    if not args.actor or not is_ulid(args.actor):
        print("error: --actor must be a ULID. The audit trail stores the deciding person in a "
              "fixed-width column and a shorter value would be padded, breaking every later "
              "verification of the chain (D93).", file=sys.stderr)
        return 2

    verb = "reinstate" if args.reinstate else "exclude"
    try:
        with engine.begin() as connection:
            if args.reinstate:
                reinstate(connection, session_id=args.session_id, reason=args.reason,
                          reinstated_by=args.actor, actor_role=args.role)
            else:
                exclude(connection, session_id=args.session_id, reason=args.reason,
                        excluded_by=args.actor, actor_role=args.role)
            print(f"{verb}d {args.session_id}")
            print(f"  ground: {args.reason}")
            print(f"  by:     {args.actor} ({args.role})")
            if not args.write:
                raise _DryRun
    except _DryRun:
        print("\nnothing was written. Pass --write to apply.")
        return 0
    except ExclusionError as refused:
        print(f"error: {refused}", file=sys.stderr)
        return 1

    print("\nwritten, and recorded on the audit chain as session.excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
