# -*- coding: utf-8 -*-
"""Walk the audit chain and report the first break.

SCHEMA.md §9: "This makes silent tampering by a superuser detectable: `scripts/verify_audit_
chain.py` walks the chain and reports the first break."

    python scripts/verify_audit_chain.py --from-json B:/praxis/runs/<id>/audit.json
    python scripts/verify_audit_chain.py --from-db

Two sources, one check. `--from-json` reads an export, which is the artefact a reader of the
thesis could be handed and the only source available before the database existed. `--from-db`
selects the same columns ordered by `audit_id` and hands them to the same `verify`, which is
what this file said would happen once the database landed. It has landed: psycopg is installed
and `audit_log` carries the rows, so the promise is kept here rather than left as a comment.

`verify` is not reimplemented for the second source, and `load_from_db` returns the same
`AuditRecord` list that `load` does, so neither source can drift into checking something
slightly different from the other. That is the point of the arrangement.

Exit codes: 0 intact, 1 broken, 2 could not be read. A broken chain is a finding, so it exits
non-zero and prints where rather than raising a traceback.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.audit.chain import AuditRecord, verify

# The line that carries the head hash. Named, because scripts/verify_end_to_end.py reads this
# script's output to fill the "audit chain" field of its report, and it was taking the last line
# of stdout: for an intact chain that is the head, but for an empty one the last line is the
# sentence saying so, and the report printed that sentence where a hash belongs. One prefix,
# imported by the reader, rather than a format known separately in two places.
HEAD_PREFIX = "head "


def load(path: Path) -> list[AuditRecord]:
    """Rows as exported: a JSON list in the order the log was written."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} holds a {type(rows).__name__}, not a list of audit rows")

    records = []
    for index, row in enumerate(rows):
        try:
            records.append(AuditRecord(
                occurred_at=datetime.fromisoformat(row["occurred_at"].replace("Z", "+00:00")),
                event_type=row["event_type"], entity_type=row["entity_type"],
                entity_id=row["entity_id"], payload=row["payload"],
                actor_user_id=row.get("actor_user_id"), actor_role=row.get("actor_role"),
                prev_hash=row.get("prev_hash") or "", row_hash=row["row_hash"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"row {index} is not a readable audit row: {exc}") from exc
    return records


def load_from_db(url: str | None = None) -> list[AuditRecord]:
    """The same rows from `audit_log`, in the order the log was written.

    `praxis.audit.write.read_chain` already selects them ordered by `audit_id` and builds the
    records, so it is called rather than repeated. The ordering column is load-bearing -
    `audit_id` is the identity, so it is insertion order and therefore the order each
    `prev_hash` was computed against, where `occurred_at` is supplied per row and two rows
    written in one transaction can share a timestamp - and a second copy of that decision here
    is one that could be changed without the first one moving.
    """
    # Imported inside the function, not at the top, so that `--from-json` keeps working on a
    # machine with neither sqlalchemy nor psycopg. The export is the artefact a reader of the
    # thesis is handed, and it should stay checkable without a database to check it against.
    #
    # Inside the `try` as well, which is the whole point and was wrong at first. On exactly the
    # machine the paragraph above describes, `--from-db` raised ImportError through main() and
    # the process exited 1 - and 1 is this script's code for CHAIN BROKEN. A missing library was
    # reported as a broken audit chain, which is the most damaging thing it could have said.
    try:
        from sqlalchemy.exc import SQLAlchemyError

        from praxis.audit.write import read_chain
        from praxis.db import DatabaseNotConfigured, build_engine

        engine = build_engine(url)
        with engine.connect() as connection:
            return read_chain(connection)
    # ImportError first, and the order is load-bearing: the clause below names
    # `DatabaseNotConfigured`, which the import binds, and that name does not exist when the
    # import is what failed. Python only evaluates a handler it reaches, so ImportError being
    # matched first means the unbound name is never looked at.
    except ImportError as exc:
        raise ValueError(
            f"the audit log could not be read: {exc}. Reading from the database needs "
            f"sqlalchemy and psycopg; an exported chain needs neither, so --from-json still "
            f"works here.") from exc
    except (DatabaseNotConfigured, SQLAlchemyError) as exc:
        # ValueError, because to this script an unreachable or unconfigured database is a source
        # it could not read: exit code 2, and distinct from a chain it read and found broken.
        # Not `except Exception`: a TypeError out of read_chain is a defect in this project, not
        # a database that could not be read, and reporting it as the latter would hide it.
        raise ValueError(f"the audit log could not be read: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-json", type=Path,
                        help="an exported audit_log, in written order")
    source.add_argument("--from-db", action="store_true",
                        help="the live audit_log, ordered by audit_id; reads DATABASE_URL")
    args = parser.parse_args()

    origin = str(args.from_json) if args.from_json else "audit_log"
    try:
        records = load(args.from_json) if args.from_json else load_from_db()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not records:
        print(f"{origin}: empty. An empty chain is intact and proves nothing.")
        return 0

    break_found = verify(records)
    if break_found is None:
        print(f"{origin}: {len(records)} rows, chain intact.")
        print(f"{HEAD_PREFIX}{records[-1].row_hash}")
        return 0

    print(f"{origin}: CHAIN BROKEN at row {break_found.index} of {len(records)}.",
          file=sys.stderr)
    print(break_found.describe(), file=sys.stderr)
    print(f"Rows 0 to {break_found.index - 1} verify. Everything from {break_found.index} on "
          f"is unverifiable, not necessarily altered.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
