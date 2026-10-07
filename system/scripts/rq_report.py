# -*- coding: utf-8 -*-
"""What this system can currently say about each research question, and what it cannot.

    python scripts/rq_report.py
    python scripts/rq_report.py --json
    python scripts/rq_report.py --no-database        # the code check alone

Two different facts, kept apart on purpose. **Does the code that would answer this question
exist**, which `praxis.research.resolve` decides by importing every evidence source the question
declares. And **does the corpus contain what that code needs**, which this script measures
against the live database rather than reading off a sentence somebody wrote. A module that
imports is not a result, and the one place this system conflated the two is why phase 10 reports
`seeded_errors_available: True` on the strength of an unrelated module loading cleanly.

Nothing here runs a phase, trains anything or writes a row. It reports, and where it cannot
measure it says which of the two reasons applies: no database configured, or the corpus has
nothing to count. D48, one level up from a phase.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.research import QUESTIONS, resolve

WIDTH = 92

# Per question, the counts that decide whether its evidence has anything to work on. Each is a
# SQL scalar and a sentence saying what a zero means, so a reader does not have to know the
# schema to read the row. Kept here rather than in `praxis.research` because these are questions
# about one deployment's corpus, and the declarations there are about the system.
MEASURES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "RQ2": (
        ("annotations", "SELECT count(*) FROM annotations",
         "no labelled clip exists, so neither the human band nor any model figure can be "
         "computed"),
        ("distinct raters", "SELECT count(DISTINCT rater_id) FROM annotations",
         "agreement needs two raters on the same clip; below two, phase 2 abstains"),
        ("teachers with sessions", "SELECT count(DISTINCT teacher_id) FROM sessions",
         "R2 forbids a teacher in more than one partition, so this is the ceiling on "
         "teacher-disjoint splits"),
        ("confirmed teacher tracks",
         "SELECT count(*) FROM teacher_tracks WHERE confirmed_by IS NOT NULL",
         "every downstream phase filters on a confirmed track, so zero means no clip can be "
         "cut for annotation"),
        # Reported beside the confirmations because the two zeros mean opposite things. No
        # ranking is a system that cannot ask the question; a ranking and no confirmation is a
        # system waiting on a person, which is a different thing to go and do.
        ("sessions awaiting a reviewer",
         "SELECT count(*) FROM teacher_tracks WHERE candidates->>'source' <> 'unrecorded'",
         "no session has a stored candidate ranking, so the confirmation screen has nothing to "
         "show and scripts/reemit_candidates.py has not been run"),
    ),
    "RQ3": (
        ("practicum sessions", "SELECT count(*) FROM sessions WHERE domain = 'classroom'",
         "the revised question studies teaching-practicum classrooms throughout; at zero "
         "there is no development domain and nothing to vary the setting of"),
        ("microteaching sessions", "SELECT count(*) FROM sessions WHERE domain = "
         "'microteaching'",
         "pilot footage from the earlier design's setting, not development-domain data for "
         "this question"),
        ("placement schools registered",
         "SELECT count(*) FROM schools",
         "S1 holds one school out while its college stays in training, so the ladder needs at "
         "least two schools under a college that also trains, and a third under the held-out "
         "college. D98"),
        ("sessions placed at a school",
         "SELECT count(*) FROM sessions WHERE school_id IS NOT NULL",
         "a practicum session with no school cannot be put on the ladder; the database refuses "
         "one through a_classroom_session_names_its_school, so this counts the practicum "
         "footage that exists at all"),
        ("colleges contributing sessions",
         "SELECT count(DISTINCT college_id) FROM sessions",
         "S2 holds a college out entirely, so the ladder needs at least two, and O5 asks for "
         "at least two anyway"),
        ("sessions with camera zones", "SELECT count(*) FROM sessions WHERE setup_id IS NOT "
         "NULL", "B3 is non-scorable without marked zones, and B3 carries the largest "
         "expected variation across settings"),
    ),
    "RQ5": (
        ("adjudications", "SELECT count(*) FROM adjudications",
         "no reviewer decision has been recorded"),
        ("seeded-error probes storable",
         "SELECT count(*) FROM information_schema.columns WHERE table_name = 'adjudications' "
         "AND column_name = 'was_seeded_error'",
         "the contract carries the flag and the table has no column for it, so a probe "
         "outcome cannot be persisted"),
        ("seeded_errors table",
         "SELECT count(*) FROM information_schema.tables WHERE table_name = 'seeded_errors'",
         "specified in docs/SCHEMA.md and created by no migration"),
    ),
}


def measured(dsn: str) -> dict[str, list[dict[str, object]]]:
    """Run each question's counts. Raises rather than degrading: a database that is configured
    and unreachable is a different state from one that was never configured, and reporting them
    alike is how a broken deployment comes to look like an empty one."""
    import psycopg

    results: dict[str, list[dict[str, object]]] = {}
    with psycopg.connect(dsn) as connection:
        for question_id, measures in MEASURES.items():
            rows = []
            for label, sql, meaning in measures:
                count = connection.execute(sql).fetchone()[0]
                rows.append({"label": label, "count": int(count), "means": meaning})
            results[question_id] = rows
    return results


def database_dsn(allow: bool) -> str | None:
    if not allow:
        return None
    keywords = os.environ.get("PRAXIS_TEST_DSN")
    return keywords or None


def render(report: list[dict[str, object]],
           counts: dict[str, list[dict[str, object]]] | None) -> None:
    for row in report:
        question = next(q for q in QUESTIONS if q.id == row["id"])
        print("=" * WIDTH)
        print(f"{row['id']}" + (f"   {', '.join(row['hypotheses'])}"
                                if row["hypotheses"] else "   no hypothesis keyed to it"))
        print("-" * WIDTH)
        for line in _wrapped(question.question):
            print(f"  {line}")
        print()

        if not row["implemented"]:
            print("  CODE: no evidence source is declared, because none exists.")
        elif row["missing"]:
            print("  ANSWERABLE BY THIS SYSTEM: no. Declared evidence that is not there:")
            for path in row["missing"]:
                print(f"    missing  {path}")
        else:
            print("  CODE: every declared evidence source resolves.")
            for item in row["evidence"]:
                print(f"    {item['state']:>8}  {item['produced_by']}")
                print(f"              {item['what']}")

        if counts is not None and row["id"] in counts:
            print()
            print("  CORPUS:")
            for measure in counts[row["id"]]:
                mark = " " if measure["count"] else "!"
                print(f"    {mark} {measure['label']:<30} {measure['count']}")
                if not measure["count"]:
                    for line in _wrapped(measure["means"], indent=38):
                        print(f"      {line}")

        if row["blocked_on"]:
            print()
            print("  WAITING ON:")
            for item in row["blocked_on"]:
                lines = _wrapped(item, indent=6)
                print(f"    - {lines[0]}")
                for line in lines[1:]:
                    print(f"      {line}")
        print()


def _wrapped(text: str, indent: int = 2) -> list[str]:
    limit = WIDTH - indent
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > limit:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="machine-readable, for a report to embed")
    parser.add_argument("--no-database", action="store_true",
                        help="check the code only, without measuring the corpus")
    args = parser.parse_args(argv)

    report = resolve()
    dsn = database_dsn(not args.no_database)
    counts = measured(dsn) if dsn else None

    if args.json:
        print(json.dumps({"questions": report, "corpus": counts}, indent=2))
        return 0

    render(report, counts)
    print("=" * WIDTH)
    implemented = sum(1 for row in report if row["implemented"] and not row["missing"])
    print(f"{implemented} of {len(report)} questions have every declared evidence source "
          f"present.")
    if counts is None:
        print("The corpus was not measured: set PRAXIS_TEST_DSN, or pass --no-database to say "
              "so deliberately.")
    print("Evidence that exists is not a result. Nothing here reports a measured finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
