# -*- coding: utf-8 -*-
"""The local launcher configures the variables the application actually reads.

scripts/serve.py exists because setting three environment variables by hand before uvicorn went
wrong in every shell it was attempted in, most recently by pasting bash into PowerShell, where
`export` is not a command: the variables were never set and the operator was shown
`DatabaseNotConfigured` from inside `create_app()`.

Replacing that with a launcher only helps if the launcher names the same variables the
application consults, and a launcher is exactly the kind of file that can drift from them
silently - it writes names into the environment and nothing checks that anything reads them.
That is L28, which was found when docker-compose.yml set REDIS_URL and `broker_url()` read
PRAXIS_BROKER_URL. So the join is asserted here rather than assumed.

`praxis.api.main` cannot be imported to read its constant: it builds the application at import
time, which needs a database, and avoiding exactly that ordering is why serve.py defers its own
import. The name is read out of the source with `ast` instead, which needs neither.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from praxis.db.engine import KEYWORD_VAR, URL_VAR
from scripts import serve
from scripts.dev_services import ASSIGNMENT, dsn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
API_MAIN = REPO_ROOT / "praxis" / "api" / "main.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

MANAGED = (URL_VAR, KEYWORD_VAR, serve.MEDIA_VAR, serve.SERVE_FRONTEND)


@pytest.fixture
def clean_environment(monkeypatch):
    """None of the variables the launcher manages, so `configure` takes its own paths.

    The snapshot-and-restore is the point, and `monkeypatch` alone could not do it. `delenv` on
    a variable that was already absent records nothing to undo, and `serve.configure` assigns
    `os.environ[KEYWORD_VAR]` itself - inside production code, where monkeypatch cannot see it.
    So the variable survived the test, pointing at the **research** database, and every later
    test in the session read it as its own DSN. With `pytest-randomly` deciding the order, that
    surfaced as `test_every_session_is_accounted_for_in_the_chain` failing on some runs and not
    others.

    What it found there was real - two synthetic sessions with no `session.ingested` row - so
    the leak made a true report about a database this file has no business opening at all.
    """
    saved = {name: os.environ.get(name) for name in MANAGED}
    for name in MANAGED:
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def module_constant(path: Path, name: str):
    """A module-level constant, read from the source without importing it.

    `frozenset({...})` is a call rather than a literal, so `literal_eval` refuses it; the one
    argument inside is a literal and is what gets evaluated. That covers every constant this
    file needs and nothing more - a constant built by real computation is not readable this way
    and should not be guessed at.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets)):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and len(
                value.args) == 1:
            return ast.literal_eval(value.args[0])
        return ast.literal_eval(value)
    raise AssertionError(f"{path.name} has no module-level {name}")


class TestTheJoin:
    """Every name the launcher writes is a name something else reads."""

    def test_the_frontend_variable_is_the_one_the_entry_point_reads(self) -> None:
        assert module_constant(API_MAIN, "SERVE_FRONTEND") == serve.SERVE_FRONTEND, (
            "scripts/serve.py sets a different variable from the one praxis/api/main.py "
            "consults, so --no-frontend and its opposite would both do nothing")

    def test_the_media_variable_is_the_one_the_config_interpolates(self) -> None:
        text = DEFAULT_CONFIG.read_text(encoding="utf-8")
        assert "${" + serve.MEDIA_VAR + "}" in text, (
            f"configs/default.yaml does not interpolate ${{{serve.MEDIA_VAR}}}, so setting it "
            f"would not reach config.require_media_root()")

    def test_the_database_variables_come_from_the_module_that_reads_them(self) -> None:
        """Not a spelling check: serve.py imports these, so this asserts it keeps importing
        them rather than hardcoding a copy that could be renamed out from under it."""
        source = (REPO_ROOT / "scripts" / "serve.py").read_text(encoding="utf-8")
        assert "from praxis.db.engine import KEYWORD_VAR, URL_VAR" in source
        assert '"DATABASE_URL"' not in source and '"PRAXIS_TEST_DSN"' not in source, (
            "serve.py spells a database variable out instead of importing the constant")


class TestConfigure:
    def test_it_supplies_the_local_cluster_when_nothing_is_configured(
        self, clean_environment
    ) -> None:
        serve.configure(serve_frontend=False)
        assert os.environ[KEYWORD_VAR] == dsn(), (
            "the launcher built its own address instead of using dev_services.dsn(), which is "
            "two places naming one cluster")

    def test_it_does_not_override_a_configured_database(self, clean_environment) -> None:
        """Run with DATABASE_URL set and that is the database served. A launcher that helpfully
        replaced it would point a real deployment at a developer's machine."""
        monkeypatch = clean_environment
        monkeypatch.setenv(URL_VAR, "postgresql+psycopg://somebody@elsewhere:5432/real")

        serve.configure(serve_frontend=False)

        assert os.environ[URL_VAR] == "postgresql+psycopg://somebody@elsewhere:5432/real"
        assert KEYWORD_VAR not in os.environ, (
            f"{KEYWORD_VAR} was set alongside {URL_VAR}; database_url() prefers {URL_VAR}, so "
            f"this is a second address that silently does nothing")

    def test_it_does_not_override_a_configured_media_root(self, clean_environment) -> None:
        clean_environment.setenv(serve.MEDIA_VAR, "Z:/somewhere/else")
        serve.configure(serve_frontend=False)
        assert os.environ[serve.MEDIA_VAR] == "Z:/somewhere/else"

    def test_declining_the_frontend_unsets_rather_than_falsifies_it(
        self, clean_environment
    ) -> None:
        """`main.py` tests the value against a truthy set, so "0" would work; removing it is
        still right, because a stale "1" inherited from the shell must not survive
        --no-frontend."""
        clean_environment.setenv(serve.SERVE_FRONTEND, "1")
        serve.configure(serve_frontend=False)
        assert serve.SERVE_FRONTEND not in os.environ

    def test_asking_for_the_frontend_sets_what_the_entry_point_accepts(
        self, clean_environment
    ) -> None:
        serve.configure(serve_frontend=True)
        # Compared against the entry point's own set, read out of its source, and the way it
        # compares - it strips and lower-cases before testing membership. A launcher that wrote
        # "yes" would be fine; one that wrote "on-please" would not, and only main.py knows.
        accepted = module_constant(API_MAIN, "TRUTHY")
        assert os.environ[serve.SERVE_FRONTEND].strip().lower() in accepted


class TestDsnForEachShell:
    """`dsn` emits an assignment for a shell to evaluate, so it has to name the right shell."""

    def test_every_offered_shell_produces_an_assignment_carrying_the_address(self) -> None:
        for shell, template in ASSIGNMENT.items():
            line = template.format(value=dsn())
            assert "PRAXIS_TEST_DSN" in line, f"{shell} assignment names no variable"
            assert dsn() in line, f"{shell} assignment does not carry the address"

    def test_powershell_and_cmd_use_their_own_syntax_and_not_bash(self) -> None:
        assert ASSIGNMENT["bash"].startswith("export ")
        assert ASSIGNMENT["powershell"].startswith("$env:")
        assert ASSIGNMENT["cmd"].startswith("set ")

    def test_cmd_is_unquoted_because_cmd_would_keep_the_quotes(self) -> None:
        """`set X="a b"` makes the quotes part of the value in cmd, and the DSN contains
        spaces, so a quoted assignment there produces an address libpq cannot parse."""
        assert '"' not in ASSIGNMENT["cmd"]

    def test_bash_remains_the_default_so_existing_eval_lines_keep_working(self) -> None:
        """Every script, document and habit in this project uses `eval "$(... dsn)"`."""
        source = (REPO_ROOT / "scripts" / "dev_services.py").read_text(encoding="utf-8")
        assert 'default="bash"' in source
