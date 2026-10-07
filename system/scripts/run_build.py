# -*- coding: utf-8 -*-
"""Run the build: every phase, in order, each as its own run.

    python scripts/run_build.py
    python scripts/run_build.py --from 1 --to 3
    python scripts/run_build.py --dry-run

Each phase is a separate `run_phase.py` invocation and therefore a separate run identifier,
frozen config and manifest. That is deliberate: R7 is a property of a *run*, and a single
process that ran eleven phases would produce one manifest describing eleven different things.
Chaining them here costs a subprocess per phase and keeps the unit of reproducibility the same
size as the unit of work.

**An abstention does not stop the build.** Phase 4 cannot train without labels, and the honest
outcome is that phases 4 to 8 and 10 abstain while 1, 3, 9 and 11 do real work. A failure does
stop it, because a later phase reading a broken phase's output would report numbers built on
it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.config import ConfigError, load_config
from praxis.ids import new_ulid
from praxis.phases import BUILD_ORDER, load_all

RUNNER = REPO_ROOT / "scripts" / "run_phase.py"
RAN, FAILED, USAGE, ABSTAINED = 0, 1, 2, 3
VERDICTS = {RAN: "ran", FAILED: "failed", USAGE: "usage error", ABSTAINED: "abstained"}


def selected(first: str | None, last: str | None) -> list[str]:
    """The phases to run, bounded by --from and --to, in build order."""
    registered = load_all()
    keys = [key for key in BUILD_ORDER if key in registered]
    start = keys.index(first) if first else 0
    stop = keys.index(last) + 1 if last else len(keys)
    if start > stop - 1:
        raise SystemExit(f"error: --from {first} comes after --to {last}")
    return keys[start:stop]


def run_one(key: str, config_path: Path) -> tuple[int, str]:
    """One phase, as its own process. Returns its exit code and the last line it printed."""
    done = subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(config_path), "--phase", key],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=REPO_ROOT)
    tail = [line for line in (done.stdout + done.stderr).splitlines() if line.strip()]
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    return done.returncode, tail[-1] if tail else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run every PRAXIS build phase in order.")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--from", dest="first", help="first phase to run")
    parser.add_argument("--to", dest="last", help="last phase to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the phases that would run and exit")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE

    keys = selected(args.first, args.last)
    registered = load_all()

    if args.dry_run:
        for key in keys:
            print(f"  {key:>4}  {registered[key].title}")
        return RAN

    build_id = new_ulid()
    started = datetime.now(UTC)
    rows: list[dict[str, object]] = []
    worst = RAN

    print(f"build {build_id}: {len(keys)} phases from {args.config}")
    for key in keys:
        print(f"\n--- phase {key}: {registered[key].title}")
        code, last_line = run_one(key, args.config)
        rows.append({"phase": key, "title": registered[key].title,
                     "verdict": VERDICTS.get(code, f"exit {code}"), "exit_code": code,
                     "last_line": last_line})
        if code in (FAILED, USAGE):
            worst = code
            print(f"\nbuild stopped: phase {key} did not complete", file=sys.stderr)
            break
        if code == ABSTAINED and worst == RAN:
            worst = ABSTAINED

    summary = {
        "build_id": build_id,
        "config_path": str(args.config).replace("\\", "/"),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "verdict": VERDICTS.get(worst, f"exit {worst}"),
        "phases": rows,
    }
    destination = config.paths.run_outputs / f"build-{build_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    ran = sum(1 for r in rows if r["exit_code"] == RAN)
    held = sum(1 for r in rows if r["exit_code"] == ABSTAINED)
    print(f"\nbuild {build_id}: {ran} ran, {held} abstained, "
          f"{len(rows) - ran - held} failed  -> {destination}")
    for row in rows:
        print(f"  {row['phase']:>4}  {row['verdict']:<9}  {row['title']}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
