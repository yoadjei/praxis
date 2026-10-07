# -*- coding: utf-8 -*-
"""Start, stop and inspect the project's local PostgreSQL cluster.

The cluster is project-owned: its own data directory on B:, its own port, bound to loopback,
created without administrator rights and never touching a system service. See D44 for why a
separate cluster rather than the default one.

R7 is the reason this is a script and not a paragraph in a README. A run that can only be
reproduced by someone who remembers the port and the data directory is not reproducible. Two
sessions have now lost time to `pg_ctl start` appearing to hang while the server came up
perfectly well behind it; `run` below explains what actually causes that and why a timeout
does not end it.

    python scripts/dev_services.py status
    python scripts/dev_services.py start
    python scripts/dev_services.py stop
    python scripts/dev_services.py dsn                     # bash: eval "$(... dsn)"
    python scripts/dev_services.py dsn --shell powershell  # PowerShell: ... dsn | iex

`dsn` emits an assignment for a shell to evaluate, so it has to know which shell. It emitted
`export` unconditionally, which is a statement about bash that nothing here checked, and on this
project's own development machine the shell is PowerShell: `export` is not a command there, the
variable was never set, and what the operator then saw was `DatabaseNotConfigured` raised inside
`create_app()` - a message about the application, for a fault in the shell. bash stays the
default so that every existing `eval "$(...)"` keeps working.

To serve the API without setting anything by hand, see scripts/serve.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PGDATA = Path(os.environ.get("PRAXIS_PGDATA", "B:/praxis/pgdata"))
PGLOG = Path(os.environ.get("PRAXIS_PGLOG", "B:/praxis/pglog")) / "postgresql.log"
PORT = os.environ.get("PRAXIS_PGPORT", "5433")
DATABASE = "praxis"
SUPERUSER = "praxis"

# Development only, on a loopback-bound cluster that holds synthetic data. Real deployments
# read the DSN from the environment; nothing here is used outside a developer machine.
SUPERUSER_PASSWORD = "praxis_dev_local"

SEARCH_ROOTS = (Path(r"C:/Program Files/PostgreSQL"),
                Path(r"C:/Program Files (x86)/PostgreSQL"))


def pg_ctl() -> Path:
    """Find pg_ctl. PostgreSQL installs off PATH on Windows, which cost a session once."""
    on_path = shutil.which("pg_ctl")
    if on_path:
        return Path(on_path)

    found = sorted((exe for root in SEARCH_ROOTS if root.is_dir()
                    for exe in root.glob("*/bin/pg_ctl.exe")), reverse=True)
    if not found:
        raise SystemExit(
            f"pg_ctl is not on PATH and not under {' or '.join(map(str, SEARCH_ROOTS))}. "
            f"Install PostgreSQL or put its bin directory on PATH.")
    return found[0]


def run(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    """pg_ctl, with stdin closed and pipes withheld from anything that leaves a server behind.

    `capture=False` matters and is not tidiness. `pg_ctl start` hands its stdout and stderr to
    the postgres it spawns, and that process outlives it by design, so a pipe here is never
    closed and the read waits forever. `timeout` does not rescue it either: the kill path
    drains the same pipes. Everything pg_ctl would have printed goes to `-l` regardless.
    """
    streams = ({"capture_output": True, "text": True} if capture
               else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})
    return subprocess.run([str(pg_ctl()), *args], stdin=subprocess.DEVNULL,
                          timeout=120, **streams)


def dsn(user: str = SUPERUSER, password: str = SUPERUSER_PASSWORD) -> str:
    return f"host=127.0.0.1 port={PORT} dbname={DATABASE} user={user} password={password}"


def status() -> int:
    result = run("status", "-D", str(PGDATA))
    print(result.stdout.strip() or result.stderr.strip())
    return result.returncode


def start() -> int:
    if run("status", "-D", str(PGDATA)).returncode == 0:
        print(f"already running on port {PORT}")
        return 0

    PGLOG.parent.mkdir(parents=True, exist_ok=True)
    result = run("start", "-D", str(PGDATA), "-l", str(PGLOG), "-w", "-t", "60",
                 "-o", f"-p {PORT} -c listen_addresses=127.0.0.1", capture=False)
    if result.returncode == 0:
        print(f"started on port {PORT}, logging to {PGLOG}")
        return 0

    print(f"pg_ctl start failed; last lines of {PGLOG}:", file=sys.stderr)
    if PGLOG.exists():
        print("\n".join(PGLOG.read_text(encoding="utf-8",
                                        errors="replace").splitlines()[-15:]), file=sys.stderr)
    return result.returncode


def stop() -> int:
    result = run("stop", "-D", str(PGDATA), "-m", "fast", "-w", "-t", "60")
    print(result.stdout.strip() or result.stderr.strip())
    return result.returncode


# One assignment per shell, keyed by the name the shell knows. cmd's `set` takes no quotes at
# all: quoting there makes them part of the value, which is why this is a table and not a flag
# spliced into one format string.
ASSIGNMENT = {
    "bash": 'export PRAXIS_TEST_DSN="{value}"',
    "powershell": '$env:PRAXIS_TEST_DSN = "{value}"',
    "cmd": "set PRAXIS_TEST_DSN={value}",
}


def print_dsn(shell: str) -> int:
    print(ASSIGNMENT[shell].format(value=dsn()))
    return 0


def main() -> int:
    actions = {"status": status, "start": start, "stop": stop, "dsn": None}
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=sorted(actions))
    parser.add_argument("--shell", choices=sorted(ASSIGNMENT), default="bash",
                        help="which shell the `dsn` assignment is written for (default bash)")
    args = parser.parse_args()

    if args.action == "dsn":
        return print_dsn(args.shell)
    return actions[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
