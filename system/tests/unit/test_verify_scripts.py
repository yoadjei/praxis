# -*- coding: utf-8 -*-
"""The two verification scripts, and the three ways they misreported their own results.

scripts/verify_audit_chain.py exits 0 for an intact chain, 1 for a broken one and 2 for a source
it could not read. scripts/verify_end_to_end.py runs it and puts the answer in its report. Both
were wrong in ways that a passing suite and a green run could not show, because both were wrong
about what they *said* rather than about what they did:

  - `--from-db` imported sqlalchemy outside its own try block, so on a machine without it the
    ImportError escaped and the process exited 1. Exit 1 is this script's code for CHAIN BROKEN.
    A missing library was reported as a broken audit trail, which is the most damaging sentence
    it could have produced, and R5 is the invariant it exists to carry.
  - `check_audit_chain` took the last line of stdout as the head hash. For an intact chain that
    is the head; for an empty one the last line is the sentence explaining that it is empty, and
    the report printed that sentence in the field labelled "audit chain".
  - and the exit code was never asserted for an empty chain at all.

Each test here fails against the version that shipped.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.verify_audit_chain import HEAD_PREFIX
from scripts.verify_end_to_end import reported_head

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHAIN_SCRIPT = REPO_ROOT / "scripts" / "verify_audit_chain.py"

INTACT = "audit_log: 4 rows, chain intact.\nhead " + "a" * 64
EMPTY = "audit_log: empty. An empty chain is intact and proves nothing."
BROKEN_PREFIX_LOOKALIKE = "audit_log: 4 rows, chain intact.\nheading for trouble " + "b" * 64


class TestReportedHead:
    def test_an_intact_chain_reports_its_head(self) -> None:
        assert reported_head(INTACT) == HEAD_PREFIX + "a" * 64

    def test_an_empty_chain_does_not_report_a_sentence_as_a_head(self) -> None:
        """The defect. An empty chain is not a failure - the verifier exits 0 and D48 makes an
        unavailable check abstain - but it must not occupy the same field as a hash."""
        answer = reported_head(EMPTY)
        assert answer != EMPTY, (
            "the explanatory sentence was returned as the chain head, which is what the report "
            "then printed under 'audit chain'")
        assert "empty" in answer
        assert not answer.startswith(HEAD_PREFIX), (
            "an empty chain has no head, so the answer must not be shaped like one")

    def test_only_the_head_line_is_read_and_not_whatever_resembles_it(self) -> None:
        """Keyed on the prefix the other script prints, imported rather than spelled out."""
        assert reported_head(BROKEN_PREFIX_LOOKALIKE) != BROKEN_PREFIX_LOOKALIKE
        assert "empty" in reported_head(BROKEN_PREFIX_LOOKALIKE)

    def test_the_prefix_is_the_one_the_verifier_actually_prints(self) -> None:
        """The join. If verify_audit_chain stopped printing this prefix, reported_head would
        silently answer "no head" for every intact chain, and the report would say the audit log
        was empty for a run that had written to it."""
        source = CHAIN_SCRIPT.read_text(encoding="utf-8")
        assert 'print(f"{HEAD_PREFIX}' in source, (
            "verify_audit_chain.py no longer prints the head through HEAD_PREFIX, so the two "
            "scripts no longer agree on the line that carries it")


def run_chain_script(*argv: str, blocking: str | None = None,
                     env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """The script in a subprocess, optionally with one import made to fail.

    A subprocess, because the thing under test is the exit code, and because blocking an import
    inside the running interpreter would unload a module the rest of the suite needs.
    """
    command = [sys.executable]
    if blocking:
        # A meta_path hook that refuses one top-level package and its submodules, installed
        # before the script is imported. This is how "a machine without sqlalchemy" is
        # reproduced on a machine that has it.
        command += ["-c", (
            "import sys\n"
            "class Refuse:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            f"        if name == {blocking!r} or name.startswith({blocking!r} + '.'):\n"
            "            raise ImportError('No module named ' + repr(name))\n"
            "        return None\n"
            f"for loaded in [m for m in sys.modules if m == {blocking!r} "
            f"or m.startswith({blocking!r} + '.')]:\n"
            "    del sys.modules[loaded]\n"
            "sys.meta_path.insert(0, Refuse())\n"
            f"sys.argv = ['verify_audit_chain.py'] + {list(argv)!r}\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "from scripts.verify_audit_chain import main\n"
            "raise SystemExit(main())\n")]
    else:
        command += [str(CHAIN_SCRIPT), *argv]

    return subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT,
                          stdin=subprocess.DEVNULL, timeout=120,
                          env={**_clean_environment(), **(env or {})})


def _clean_environment() -> dict[str, str]:
    """The ambient environment without a database address, so these tests do not depend on
    whether the suite was run with one configured."""
    import os

    return {key: value for key, value in os.environ.items()
            if key not in {"DATABASE_URL", "PRAXIS_TEST_DSN"}}


class TestChainScriptExitCodes:
    """0 intact, 1 broken, 2 could not be read. Nothing else, and never the wrong one."""

    def test_a_missing_sqlalchemy_is_a_source_it_could_not_read_not_a_broken_chain(
        self
    ) -> None:
        """The whole finding, in one assertion: exit 2, not exit 1."""
        done = run_chain_script("--from-db", blocking="sqlalchemy")

        assert done.returncode == 2, (
            f"exit {done.returncode} for a missing library. 1 means CHAIN BROKEN, which would "
            f"report the audit trail as tampered with because a package was absent.\n"
            f"{done.stdout}{done.stderr}")
        assert "could not be read" in done.stderr
        assert "Traceback" not in done.stderr, (
            "the documented behaviour is a message, not a traceback")
        assert "--from-json" in done.stderr, (
            "the message should say which source still works without a database")

    def test_no_configured_database_is_also_exit_two(self) -> None:
        done = run_chain_script("--from-db")
        assert done.returncode == 2, f"{done.stdout}{done.stderr}"
        assert "could not be read" in done.stderr
        assert "Traceback" not in done.stderr

    def test_an_unreadable_export_is_exit_two(self, tmp_path) -> None:
        bad = tmp_path / "not-a-chain.json"
        bad.write_text('{"this": "is an object, not a list of rows"}', encoding="utf-8")
        done = run_chain_script("--from-json", str(bad))
        assert done.returncode == 2, f"{done.stdout}{done.stderr}"
        assert "Traceback" not in done.stderr

    def test_an_empty_export_is_intact_and_says_it_proves_nothing(self, tmp_path) -> None:
        """D48: the honest answer to "is this chain intact" for an empty chain is yes, and the
        honest thing to add is that it establishes nothing."""
        empty = tmp_path / "empty.json"
        empty.write_text("[]", encoding="utf-8")
        done = run_chain_script("--from-json", str(empty))
        assert done.returncode == 0, f"{done.stdout}{done.stderr}"
        assert "proves nothing" in done.stdout
        assert HEAD_PREFIX not in done.stdout, (
            "an empty chain printed something shaped like a head hash")

    def test_the_two_sources_are_mutually_exclusive_and_one_is_required(self) -> None:
        neither = run_chain_script()
        assert neither.returncode == 2
        assert "one of the arguments" in neither.stderr

        both = run_chain_script("--from-db", "--from-json", "x.json")
        assert both.returncode == 2
        assert "not allowed with argument" in both.stderr
