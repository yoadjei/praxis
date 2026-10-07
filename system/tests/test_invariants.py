# -*- coding: utf-8 -*-
"""R1 to R7. These must never be skipped.

Violating any one of these invalidates the research, so each has a test here and each runs on
every commit. BUILD-SPEC Phase 0 anticipates that most would fail against stubs at this stage.
They do not, because they are written to **enforce everything that currently exists and tighten
automatically as modules appear**, rather than to assert a finished system.

That is a deliberate choice over the alternatives. A permanently failing suite teaches everyone
to ignore red. `skip` and `xfail` make an unenforced rule invisible, which is the exact failure
mode R1 to R7 exist to prevent. So each test states what it checks *today* and what it will
check once the phase that makes more checkable lands, and `test_invariant_coverage_is_declared`
asserts that record stays honest.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from praxis.contracts import ConfidenceState, Detection, SplitManifest
from praxis.ids import new_ulid
from tests.conftest import database_dsn, example_prediction

# What each rule can be enforced against now, and what tightens it later. Kept beside the
# tests so that a rule cannot quietly sit unenforced: the last test asserts this is accurate.
COVERAGE = {
    "R1": ("no learner-track table or learner identifier exists anywhere", "Phase 3"),
    "R2": ("SplitManifest raises on any overlap between partitions", "Phase 4"),
    "R3": ("a Detection needs a ConfidenceState, and a suppressed one serialises with no "
           "prediction key, asserted on the JSON for all five behaviours", "Phase 9"),
    "R4": ("no accuracy-only surface exists in code or in the API spec", "Phase 4"),
    "R5": ("no migration grants UPDATE or DELETE on the append-only tables, the hash chain "
           "detects an edit, a deletion or a reordering, and - whenever PRAXIS_TEST_DSN "
           "names a database - PostgreSQL itself refuses an UPDATE and a DELETE against "
           "the superuser, and every session in that database has a session.ingested row "
           "so the chain accounts for the corpus it sits beside",
           "Phase 9, when the DSN stops being optional"),
    "R6": ("the inference path imports nothing that can reach a network", "Phase 11"),
    "R7": ("two noop runs of one config agree on everything but id and time", "Phase 4"),
}

REPO_ROOT = Path(__file__).resolve().parent.parent

# Anything that can open a socket. urllib and socket are stdlib, which is precisely why they
# need naming: they are the ones that arrive without a line in requirements.txt.
NETWORK_MODULES = {
    "socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx", "aiohttp",
    "ftplib", "telnetlib", "smtplib", "xmlrpc", "webbrowser",
    "boto3", "botocore", "google", "azure", "openai", "anthropic",
    "wandb", "mlflow", "comet_ml", "neptune", "sentry_sdk", "posthog", "segment",
}

# Packages that run at inference time. The API layer is excluded: it is a network
# service by definition, and R6 is about the model never reaching out, not about the
# server never listening.
INFERENCE_PACKAGES = (
    "behaviour", "confidence", "contracts", "explain", "preprocess", "routing")


# ---------------------------------------------------------------------------
# R1. Only the teacher is classified.
# ---------------------------------------------------------------------------

def test_no_learner_tracks_persisted(repo_root: Path, python_sources: list[Path]) -> None:
    """R1. There is deliberately no place to store a learner identity.

    SCHEMA.md §4 makes this structural rather than procedural: no `learner_tracks` table is
    defined, and `learner_aggregates` carries no identifier column. Enforced by the absence of
    somewhere to put one, which is why this test reads the DDL and the models rather than
    querying a database.

    BUILD-SPEC's Phase 3 acceptance test asks the same question of a live database after a real
    preprocessing run; that one is
    `test_no_learner_tracks_are_persisted_after_preprocessing`, in
    `tests/integration/test_preprocess_store.py`. This one holds whether or not a database is
    configured, so the invariant is never unchecked.
    """
    ddl = (repo_root / "docs" / "SCHEMA.md").read_text(encoding="utf-8")
    sources = "\n".join(p.read_text(encoding="utf-8") for p in python_sources)

    offending = re.findall(r"CREATE TABLE\s+(\w*learner\w*track\w*|\w*track\w*learner\w*)",
                           ddl, re.IGNORECASE)
    assert not offending, f"R1: a learner-track table is defined: {offending}"

    aggregates = re.search(r"CREATE TABLE learner_aggregates\s*\((.*?)\);", ddl, re.DOTALL)
    assert aggregates, "R1: learner_aggregates is not defined; R1 has nowhere to be enforced"
    body = aggregates.group(1)
    for forbidden in (
            "learner_id", "pupil_id", "student_id", "track_id", "face_id", "person_id"):
        assert forbidden not in body, (
            f"R1: learner_aggregates carries {forbidden!r}. Learner evidence is anonymous "
            f"aggregate counts only.")

    for forbidden in ("learner_tracks", "pupil_tracks", "student_tracks"):
        assert forbidden not in sources, f"R1: {forbidden!r} appears in praxis/"


# ---------------------------------------------------------------------------
# R2. No teacher appears in more than one data partition.
# ---------------------------------------------------------------------------

def _manifest(**overrides) -> dict:
    base = dict(
        manifest_id=new_ulid(), created_at=datetime.now(UTC),
        config_sha256="a" * 64, train_teachers=["T1", "T2"], val_teachers=["T3"],
        test_teachers=["T4"])
    base.update(overrides)
    return base


def test_splits_are_teacher_disjoint() -> None:
    """R2. Teacher-disjointness is enforced in code, never by convention."""
    SplitManifest(**_manifest())  # the clean case must construct

    contaminated = [
        ("train and val", _manifest(val_teachers=["T2", "T3"])),
        ("train and test", _manifest(test_teachers=["T1"])),
        ("val and test", _manifest(val_teachers=["T3"], test_teachers=["T3"])),
    ]
    for _label, payload in contaminated:
        with pytest.raises(ValueError, match="R2 violated"):
            SplitManifest(**payload)

    with pytest.raises(ValueError, match="duplicates"):
        SplitManifest(**_manifest(train_teachers=["T1", "T1"]))


# ---------------------------------------------------------------------------
# R3. Every model output carries a confidence state.
# ---------------------------------------------------------------------------

def test_no_bare_predictions(detection: Detection) -> None:
    """R3. A bare label may not leave the inference layer.

    Two halves. The contract cannot be built without a confidence state, and a suppressed
    detection's value is absent from the serialised payload rather than nulled or hidden.
    The second is asserted against JSON text, not a Python object, because that is where the
    rule actually has to hold.
    """
    # Narrowed to the validation error naming the missing field, not a bare `Exception`. A bare
    # one passes when the constructor fails for any reason at all - a renamed field, a changed
    # signature, an unrelated typo in this test - and R3 would then be reported as enforced on a
    # run where nothing about confidence was checked. The assertion has to be about `confidence`
    # specifically or it is not about R3.
    with pytest.raises(ValidationError, match="confidence"):
        Detection(
            detection_id=new_ulid(), session_id=new_ulid(), behaviour="B1",
            t_start_s=0.0, t_end_s=8.0, predicted=example_prediction("B1"),
            evidence_ref="/e", model_version="v1", gate_outcome="present")

    flagged = ConfidenceState(
        raw_prob=0.5, calibrated_prob=0.5, method="ensemble", epistemic=0.31,
        ood_score=0.94, ood_flag=True, in_validated_domain="unknown")
    suppressed = detection.model_copy(update={
        "confidence": flagged, "gate_outcome": "suppress",
        "suppression_reason": "out_of_distribution"})

    raw = json.dumps(suppressed.to_payload())
    assert '"predicted"' not in raw, (
        "R3: a suppressed detection leaked its prediction into the payload")
    assert '"calibrated_prob"' not in raw, "R3: a suppressed detection leaked its probability"
    assert '"suppression_reason"' in raw, "R3: suppression must state its reason"
    for field_name in suppressed.predicted:
        assert f'"{field_name}"' not in raw, (
            f"R3: {field_name} leaked, so omitting the enclosing key was not enough")

    assert '"predicted"' in json.dumps(detection.to_payload()), (
        "a presented detection must still carry its prediction, or the test above proves "
        "nothing")


# ---------------------------------------------------------------------------
# R4. Calibration is reported wherever accuracy is reported.
# ---------------------------------------------------------------------------

def test_eval_returns_calibration(repo_root: Path) -> None:
    """R4. There is no way to obtain accuracy without obtaining calibration with it.

    Enforced by the return type rather than by inspection. `BehaviourEvaluation.calibration`
    has no default, so an accuracy-bearing result cannot be constructed without it, and every
    route by which a number leaves the module carries ECE beside accuracy.

    This test used to search `harness.py` for the string "ece" and return early if the file was
    missing or empty. Deleting the harness would have turned R4 green by absence, which is the
    D56 failure exactly: an invariant guarded by a passing assertion is a lie.
    """
    api = (repo_root / "docs" / "API.md").read_text(encoding="utf-8")
    assert "accuracy alone" in api, (
        "R4: API.md no longer states that accuracy cannot be requested alone")

    harness = repo_root / "praxis" / "evaluation" / "harness.py"
    assert harness.exists() and harness.read_text(encoding="utf-8").strip(), (
        "R4: praxis/evaluation/harness.py is missing or empty. R4 is enforced by that module's "
        "return type, so its absence is an unenforced invariant and not a passing one.")

    # Imported here rather than at module scope: this file is collected before any phase module
    # is guaranteed to import cleanly, and a collection error in the invariant file would take
    # all seven rules down at once rather than failing the one that broke.
    import numpy as np

    from praxis.confidence.metrics import CalibrationReport
    from praxis.evaluation.harness import (
        BehaviourEvaluation,
        EvaluationResult,
        confusion_structure,
    )

    # Constructing a result without calibration must be impossible, not merely discouraged.
    common = dict(behaviour="B1 gesture production", n=10, accuracy=0.8, macro_f1=0.78,
                  confusion=confusion_structure(np.array([1, 0]), np.array([1, 0])),
                  recalls=())
    with pytest.raises(TypeError):
        BehaviourEvaluation(**common)                            # type: ignore[arg-type]

    report = CalibrationReport(
        n=10, mode="positive_class", ece=0.04, mce=0.09, brier=0.12, nll=0.34,
        accuracy=0.8, base_rate=0.5, mean_confidence=0.82, mean_outcome=0.8, bins=())
    evaluation = BehaviourEvaluation(calibration=report, **common)

    assert evaluation.ece == report.ece, "R4: the per-behaviour ECE is not the reported one"
    assert "ECE" in evaluation.summary(), (
        "R4: a behaviour summary states accuracy without stating ECE beside it")

    result = EvaluationResult(behaviours=(evaluation,))
    assert result.macro_ece == pytest.approx(report.ece), (
        "R4: macro accuracy is exposed without a macro ECE alongside it")

    row = result.as_table()[0]
    assert "accuracy" in row and "ece" in row, (
        "R4: the exported table carries accuracy without carrying calibration with it")


# ---------------------------------------------------------------------------
# R5. The audit trail is append-only.
# ---------------------------------------------------------------------------

def test_audit_trail_immutable(repo_root: Path) -> None:
    """R5. No update, no delete, ever. Corrections are new rows.

    Three layers, and this test reaches two of them. At the database it is a REVOKE plus a
    trigger, and SCHEMA.md §12 requires CI to grep migrations for anything granting the
    privilege back — that grep is here. Neither layer stops a superuser, so the hash chain in
    `praxis/audit/chain.py` makes an edit *detectable* instead, and `tests/unit/test_audit.py`
    asserts it catches an edit, a deletion and a reordering.

    It tightens once psycopg and PostgreSQL are installed, to an actual UPDATE and DELETE
    against a live table asserting the database raises rather than the application.
    """
    schema = (repo_root / "docs" / "SCHEMA.md").read_text(encoding="utf-8")
    for required in ("forbid_mutation", "audit_log_no_update", "adjudications_no_update"):
        assert required in schema, f"R5: SCHEMA.md no longer defines {required}"
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON audit_log, adjudications" in schema, (
        "R5: the REVOKE that stops the application role is gone from SCHEMA.md")

    grant = re.compile(r"GRANT\s+[^;]*\b(UPDATE|DELETE|TRUNCATE)\b[^;]*\bON\b[^;]*"
                       r"\b(audit_log|adjudications)\b", re.IGNORECASE | re.DOTALL)
    for migration in (repo_root / "migrations").rglob("*.py"):
        text = migration.read_text(encoding="utf-8")
        assert not grant.search(text), (
            f"R5: {migration.name} grants a mutating privilege on an append-only table")

    # Grepping for what must be absent passes on an empty directory, so the migration that
    # installs both layers must be present and must still install them.
    migrations = "\n".join(p.read_text(encoding="utf-8")
                           for p in (repo_root / "migrations" / "versions").rglob("*.py"))
    for required in ("forbid_mutation", "audit_log_no_update", "adjudications_no_update",
                     "REVOKE UPDATE, DELETE, TRUNCATE ON audit_log, adjudications"):
        assert required in migrations, (
            f"R5: no migration installs {required!r}; the trail is append-only only on paper")

    # TRUNCATE fires no row triggers, so the UPDATE/DELETE triggers above never see it. Verified
    # against the running cluster: with both of them installed, `TRUNCATE audit_log` emptied the
    # table. The REVOKE stops praxis_app and nobody else. D53.
    for table in ("audit_log", "adjudications"):
        assert re.search(rf"BEFORE TRUNCATE ON {table}\b", migrations), (
            f"R5: nothing stops TRUNCATE on {table}. A row-level trigger cannot see it, so the "
            f"trail can be emptied in one statement by any role holding the privilege.")

    # A trigger can only be created on a table that exists. The first version of this migration
    # protected `adjudications` without creating it, so `alembic upgrade` would have failed on a
    # fresh database and the protection would have existed only in the file.
    created = set(re.findall(r"CREATE TABLE\s+(\w+)", migrations, re.IGNORECASE))
    triggered = set(re.findall(r"ON\s+(\w+)\s*\n\s*FOR EACH (?:ROW|STATEMENT)",
                               migrations, re.IGNORECASE))
    assert triggered, "R5: no append-only trigger is installed by any migration"
    assert triggered <= created, (
        f"R5: {sorted(triggered - created)} are protected by a trigger that no migration "
        f"creates a table for, so the migration cannot run and the protection is fictional")

    # The chain is the layer a superuser cannot defeat, and it has to be reachable from here.
    from praxis.audit import AuditLog, verify

    log = AuditLog()
    log.append("consent.withdrawn", "T1", {"teacher_id": "T1", "scope": "all"},
               datetime(2026, 9, 13, 9, 0, tzinfo=UTC))
    log.append("media.deleted", "M1", {"media_id": "M1", "reason": "withdrawal"},
               datetime(2026, 9, 13, 9, 1, tzinfo=UTC))
    assert log.verify() is None
    assert verify(list(log)[1:]) is not None, (
        "R5: removing the first row left a chain that still verifies")

    adjudication_source = (repo_root / "praxis" / "contracts" / "adjudication.py").read_text(
        encoding="utf-8")
    assert "frozen=True" in adjudication_source, (
        "R5: Adjudication is no longer immutable in Python either")

    # And the real thing, whenever a database is configured. Conditional enforcement rather
    # than a skip: `database_dsn` raises if PRAXIS_TEST_DSN is set but unreachable, so "not
    # configured" and "configured and broken" cannot be confused for each other.
    dsn = database_dsn()
    if dsn is None:
        # D48: an unavailable check abstains, and the abstention is its own verdict. Warned and
        # not passed over, because everything above this line tests the Python half while R5 is
        # the invariant that says "at the database level" - so a green run that never reached a
        # database must not read as a green R5. CI supplies one; this fires locally.
        warnings.warn(
            "R5 abstained on its database half: PRAXIS_TEST_DSN is unset, so the trigger that "
            "refuses an UPDATE on the audit table was never exercised. Start the development "
            "cluster with scripts/dev_services.py start to enforce it.", stacklevel=2)
    else:
        _assert_database_refuses_mutation(dsn)


def test_every_session_is_accounted_for_in_the_chain() -> None:
    """R5, the half about what the trail is for: a chain that does not account for the state.

    An append-only log nobody can edit is still worthless if rows reach the tables beside it
    without passing through it. `praxis/audit/replay.py` reconstructs the system's state from
    the log alone, so a session the log never recorded is one that reconstruction cannot
    produce, and the divergence then shows up as a mismatch nobody has a rule for rather than
    as the insertion it was.

    This is the shape test fixtures take when they are pointed at the research database rather
    than a scratch one: rows inserted behind `praxis.ingest.service`, which is the only writer
    that appends `session.ingested`. `scripts/verify_end_to_end.py` creates and drops its own
    database precisely so that cannot happen.
    """
    dsn = database_dsn()
    if dsn is None:
        warnings.warn(
            "R5 abstained on chain accountability: PRAXIS_TEST_DSN is unset, so whether "
            "every session in the database was recorded by the audit trail was not checked. "
            "Start the development cluster with scripts/dev_services.py start to enforce it.",
            stacklevel=2)
        return

    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.sessions'), to_regclass('public.audit_log')")
        sessions_table, audit_table = cur.fetchone()
        if not (sessions_table and audit_table):
            warnings.warn(
                f"R5 abstained on chain accountability: "
                f"{dsn.split('dbname=')[-1].split()[0]!r} has no sessions or audit_log table, "
                f"so there was nothing to reconcile. Apply the migrations before running "
                f"against it.", stacklevel=2)
            return

        cur.execute("""
            SELECT s.session_id FROM sessions s
            WHERE NOT EXISTS (
                SELECT 1 FROM audit_log a
                WHERE a.event_type = 'session.ingested' AND a.entity_id = s.session_id)
            ORDER BY s.session_id
        """)
        unaccounted = [row[0] for row in cur.fetchall()]

    assert not unaccounted, (
        f"R5: {len(unaccounted)} session(s) exist with no session.ingested row in the audit "
        f"trail, so the chain does not account for the corpus and replay cannot reproduce it: "
        f"{', '.join(unaccounted)}. Every real session is written by praxis.ingest.service, "
        f"which appends the event in the same transaction; rows without one were inserted "
        f"around it. Synthetic work belongs in a scratch database.")


def _assert_database_refuses_mutation(dsn: str) -> None:
    """The real thing: an UPDATE and a DELETE that PostgreSQL itself must refuse.

    BUILD-SPEC: "an UPDATE or DELETE against the audit table raises at the database level, not
    in application code." Run as the connecting user, which in the development cluster is a
    superuser — the point being that the trigger stops even them, which is the only reason the
    trail is worth anything against someone with database access.
    """
    import psycopg

    from praxis.audit.write import append
    from praxis.db import build_engine, keywords_to_url, transaction

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.audit_log'), to_regclass('public.adjudications')")
            audit_table, adjudications_table = cur.fetchone()
        assert audit_table and adjudications_table, (
            "R5: the append-only tables do not exist in the database "
            f"{dsn.split('dbname=')[-1].split()[0]!r}. Apply "
            "migrations/versions/0001_append_only_audit.py before running against it.")

        # Appended through the real writer, not hand-inserted. An earlier version of this test
        # wrote a row with row_hash="b" * 64 and committed it, which is a permanent break in a
        # chain nobody can repair, in a table nobody can clean. The test that proves the trail
        # cannot be tampered with must not be the thing that tampers with it. D54.
        engine = build_engine(keywords_to_url(dsn))
        with transaction(engine) as connection:
            record = append(connection, "media.deleted", f"M-{new_ulid()}",
                            {"media_id": new_ulid(), "reason": "R5 enforcement check"})

        with conn.cursor() as cur:
            cur.execute(
                "SELECT audit_id FROM audit_log WHERE row_hash = %s", (record.row_hash,))
            audit_id = cur.fetchone()[0]

        for statement, label in (
                ("UPDATE audit_log SET entity_id = 'tampered' WHERE audit_id = %s", "UPDATE"),
                ("DELETE FROM audit_log WHERE audit_id = %s", "DELETE")):
            with pytest.raises(psycopg.errors.RaiseException) as raised, conn.cursor() as cur:
                cur.execute(statement, (audit_id,))
            assert "append-only table" in str(raised.value), (
                f"R5: the database rejected the {label} for the wrong reason: {raised.value}")
            conn.rollback()

        with conn.cursor() as cur:
            cur.execute("SELECT entity_id FROM audit_log WHERE audit_id = %s", (audit_id,))
            row = cur.fetchone()
        assert row is not None and row[0] == record.entity_id, (
            "R5: the row was altered or removed despite the trigger raising")



# ---------------------------------------------------------------------------
# R6. No network access at inference time.
# ---------------------------------------------------------------------------

def _imported_roots(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_offline_inference(repo_root: Path) -> None:
    """R6. The system must run fully offline on an air-gapped machine.

    Checked statically here and by actually disabling the network at Phase 11. The static
    half matters on its own: an import that only reaches out on a cache miss would pass a
    runtime test on a warm machine and fail on the college's.
    """
    for package in INFERENCE_PACKAGES:
        for source in (repo_root / "praxis" / package).rglob("*.py"):
            offending = _imported_roots(source) & NETWORK_MODULES
            assert not offending, (
                f"R6: {source.relative_to(repo_root)} imports {sorted(offending)}, which can "
                f"reach a network. The inference path runs air-gapped.")

    requirements = repo_root / "requirements.txt"
    if requirements.exists():
        text = requirements.read_text(encoding="utf-8").lower()
        for forbidden in ("boto3", "google-cloud", "azure-", "wandb", "mlflow", "sentry-sdk"):
            assert forbidden not in text, f"R6: {forbidden} is a dependency that phones home"

    config_source = (repo_root / "praxis" / "config.py").read_text(encoding="utf-8")
    assert "YOLO_OFFLINE" in config_source, (
        "R6: Ultralytics gates every network path on ONLINE, which is a live DNS probe unless "
        "YOLO_OFFLINE is set. This test previously required ULTRALYTICS_OFFLINE, which the "
        "package does not read, so R6 had a hole guarded by a passing assertion. D56.")

    # The inference path must not import ultralytics at all. Importing its ByteTrack resolves
    # DNS and pip-installs missing extras during the import itself, so no environment variable
    # set afterwards can help. D57.
    for package in INFERENCE_PACKAGES:
        for source in (repo_root / "praxis" / package).rglob("*.py"):
            assert "ultralytics" not in _imported_roots(source), (
                f"R6: {source.relative_to(repo_root)} imports ultralytics, which reaches the "
                f"network while being imported.")


# ---------------------------------------------------------------------------
# R7. Every run is reproducible from one config file.
# ---------------------------------------------------------------------------

def test_run_is_deterministic(repo_root: Path) -> None:
    """R7. Same config plus same data equals same outputs.

    Bit-exact on CPU, which is where every reported number is produced, so this guarantee is
    unconditional rather than tolerance-bounded. Only the run identifier and the timestamps
    may differ; anything else is a determinism failure.
    """
    from praxis.config import load_config

    run_outputs = load_config(repo_root / "configs" / "default.yaml").paths.run_outputs

    manifests = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "run_phase.py"),
             "--config", str(repo_root / "configs" / "default.yaml"), "--phase", "noop"],
            cwd=repo_root, capture_output=True, text=True,
            # pytest replaces sys.stdin with an object that has no OS-level handle, and on
            # Windows subprocess still tries to make that handle inheritable. Without this the
            # test passes alone and fails in a full run, which is the worst way to fail.
            stdin=subprocess.DEVNULL)
        assert completed.returncode == 0, f"R7: the noop phase failed:\n{completed.stderr}"
        run_id = completed.stdout.split()[1]
        manifests.append(json.loads(
            (run_outputs / run_id / "manifest.json").read_text(encoding="utf-8")))

    first, second = manifests
    assert first["run_id"] != second["run_id"], "R7: two runs reused one identifier"

    volatile = {"run_id", "started_at", "finished_at"}
    stable = [{k: v for k, v in m.items() if k not in volatile} for m in manifests]
    assert stable[0] == stable[1], (
        "R7: two runs of one config disagree on "
        f"{[k for k in stable[0] if stable[0][k] != stable[1][k]]}")


def test_cuda_run_is_stable(repo_root: Path) -> None:
    """R7's CUDA half: tolerance determinism, atol=1e-4, training only.

    No CUDA device is present on this machine, and the honest form of this test is therefore
    not to skip but to assert the consequence: **no run may claim to be a CUDA run.** That is
    what would actually go wrong — a number reported as GPU-produced when it was not.
    """
    import torch

    if torch.cuda.is_available():
        from praxis.config import load_config
        tolerance = load_config(repo_root / "configs" / "default.yaml").run.cuda_tolerance
        assert tolerance > 0, "R7: cuda_tolerance must be positive"
        return

    from praxis.config import load_config
    from praxis.phases import chain
    config = load_config(repo_root / "configs" / "default.yaml")

    # Asked of `chain`, not of a glob over manifest.json. A glob also picks up directories that
    # were abandoned half-written, which carry no device because they never finished, and the
    # KeyError that produced said nothing about CUDA. `all_completed_runs` is the one definition
    # of a run this project has, so a run it returns is one that finished and must account for
    # the device it finished on.
    for run in chain.all_completed_runs(config):
        manifest = json.loads((run.directory / "manifest.json").read_text(encoding="utf-8"))
        assert "device_used" in manifest, (
            f"R7: {run.directory} recorded a completed run without naming the device it ran "
            f"on. A number that cannot be attributed to a device cannot be reproduced on one.")
        assert manifest["device_used"] != "cuda", (
            f"R7: {run.directory} claims a CUDA run, but this machine has no CUDA device. "
            f"A number is only ever attributed to the device that produced it.")


# The distribution a module is imported under, where the two names differ.
IMPORT_TO_DISTRIBUTION = {
    "cv2": "opencv-python-headless",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
    "jose": "python-jose",
    "multipart": "python-multipart",
}


def test_every_import_is_a_declared_dependency(repo_root: Path,
                                               python_sources: list[Path]) -> None:
    """R7. A run is reproducible from one config **and the environment the lock describes**.

    This closes a gap that is invisible locally by construction. The development machine has
    packages installed that `requirements.lock` does not list - `lime`, `quantus` and
    `captum` among them - so a module importing one would work here and fail in CI, which
    installs the lock and nothing else. With no git remote, CI has never run, so the first
    notice would arrive at the worst possible moment.

    Checked against the lock rather than against what is importable, because what is
    importable is exactly the thing that cannot be trusted here.
    """
    stdlib = set(sys.stdlib_module_names)
    lock = (repo_root / "requirements.lock").read_text(encoding="utf-8").lower()
    declared = {line.split("==")[0].strip().replace("_", "-")
                for line in lock.splitlines()
                if line and not line[0].isspace() and "==" in line}

    undeclared: dict[str, list[str]] = {}
    for source in python_sources:
        if "tests" in source.parts or "scripts" in source.parts:
            continue
        for root in _imported_roots(source) - stdlib - {"praxis"}:
            distribution = IMPORT_TO_DISTRIBUTION.get(root, root).lower().replace("_", "-")
            if distribution not in declared:
                undeclared.setdefault(root, []).append(
                    str(source.relative_to(repo_root)))

    assert not undeclared, (
        f"R7: {sorted(undeclared)} are imported by praxis but absent from requirements.lock, "
        f"so the environment these tests passed in is not the one CI installs. "
        f"Offending files: {undeclared}")


# ---------------------------------------------------------------------------
# The record of what is enforced, kept honest.
# ---------------------------------------------------------------------------

def test_invariant_coverage_is_declared() -> None:
    """Every rule has a test, and every test says what it does not yet cover.

    Without this, an invariant could be narrowed to something trivially true and nobody would
    notice, which is the quiet version of deleting it.
    """
    tests = {name for name in globals() if name.startswith("test_")}
    expected = {
        "R1": "test_no_learner_tracks_persisted",
        "R2": "test_splits_are_teacher_disjoint",
        "R3": "test_no_bare_predictions",
        "R4": "test_eval_returns_calibration",
        "R5": "test_audit_trail_immutable",
        "R6": "test_offline_inference",
        "R7": "test_run_is_deterministic",
    }
    assert set(COVERAGE) == set(expected), "a rule lost its coverage entry"
    for rule, test_name in expected.items():
        assert test_name in tests, f"{rule} has no test named {test_name}"
        enforced, tightens_at = COVERAGE[rule]
        assert enforced and tightens_at, f"{rule} declares no coverage"
