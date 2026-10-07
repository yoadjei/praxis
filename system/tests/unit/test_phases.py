# -*- coding: utf-8 -*-
"""Tests for the phase registry, artefact checksumming, and phase context.

The phase registry is the central record of what a build does: what each phase
is for, what order to run them in, and what each phase depends on. It must be
maintained carefully so a build that claims to run phases 1-11 actually runs them
in the right order, and so every phase that runs gets recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from praxis.config import load_config
from praxis.contracts.manifest import ArtefactRef
from praxis.phases import (
    BUILD_ORDER,
    Abstention,
    PhaseContext,
    PhaseError,
    PhaseResult,
    RegisteredPhase,
    abstain,
    get,
    load_all,
    ordered,
    register,
)
from praxis.phases.artefacts import artefact, sha256_of, write_json


class TestPhaseRegistry:
    """Tests for the phase registry: registration, lookup, and build order."""

    def test_every_phase_in_build_order_is_registered(self) -> None:
        """Every phase named in BUILD_ORDER must exist in the registry.

        A phase that claims to be in the build order but is not registered would
        make run_build crash when it tried to fetch it. This catches that case.
        """
        registry = load_all()
        for phase_key in BUILD_ORDER:
            assert phase_key in registry, (
                f"phase {phase_key!r} is in BUILD_ORDER but not registered"
            )

    def test_every_registered_numbered_phase_is_in_build_order(self) -> None:
        """Every registered phase with a numeric key must be in BUILD_ORDER.

        A phase that exists and is registered but is absent from BUILD_ORDER
        would never run in a build, even if explicitly requested. Silent
        silences like this are the exact failure mode R1-R7 are meant to prevent.
        """
        registry = load_all()
        for key in registry:
            if key.isdigit():
                assert key in BUILD_ORDER, (
                    f"phase {key!r} is registered but not in BUILD_ORDER; "
                    f"it would never run in a build"
                )

    def test_ordered_returns_phases_in_numeric_order_not_string_order(
        self
    ) -> None:
        """ordered() must return phases in numeric order, not string order.

        As a string, "10" < "2", which is the exact bug the BUILD_ORDER tuple
        exists to prevent. A build that ran phases out of order would look fine
        at first glance but would produce wrong numbers. This test explicitly
        asserts numeric ordering of the numbered phases.
        """
        phases = ordered()
        keys = [p.key for p in phases if p.key.isdigit()]

        assert keys == list(BUILD_ORDER), (
            f"ordered() returned phases out of order: {keys} != {list(BUILD_ORDER)}"
        )

        numeric_keys = [int(k) for k in keys]
        assert numeric_keys == sorted(numeric_keys), (
            f"numeric phases are not in numeric order: {numeric_keys}"
        )

    def test_register_raises_on_duplicate_key(self) -> None:
        """register() must raise PhaseError when a key is registered twice.

        The second registration would silently overwrite the first in a dict,
        which means the original phase is lost to the registry and would not
        run. Raise instead to catch the mistake.
        """
        with patch.dict("praxis.phases._REGISTRY", {}, clear=True):

            def phase_one(context):
                pass

            def phase_two(context):
                pass

            register("test_phase", "Test Phase")(phase_one)

            with pytest.raises(PhaseError, match="registered twice"):
                register("test_phase", "Another Title")(phase_two)

    def test_get_raises_on_unknown_key(self) -> None:
        """get() must raise PhaseError on an unknown key, listing available phases.

        The error message must include the available phases so the operator
        knows what went wrong and what the correct key should be.
        """
        with pytest.raises(PhaseError, match="unknown phase"):
            get("no_such_phase_xyz")

    def test_get_error_message_lists_available_phases(self) -> None:
        """When get() raises, the error message must list available phases.

        So a caller can see what options they have.
        """
        try:
            get("no_such_phase_xyz")
        except PhaseError as exc:
            error_msg = str(exc)
            assert "available:" in error_msg, (
                f"get() error did not list available phases: {error_msg}"
            )
            registry = load_all()
            for key in registry:
                assert key in error_msg, (
                    f"available phase {key!r} not mentioned in error: {error_msg}"
                )


class TestAbstention:
    """Tests for Abstention and the abstain() shorthand."""

    def test_abstain_returns_phase_result_with_ran_false(self) -> None:
        """abstain() must return a PhaseResult with ran == False."""
        result = abstain("no input")
        assert result.ran is False

    def test_abstain_includes_reason_in_result(self) -> None:
        """abstain() result must carry the reason as an Abstention."""
        reason = "database is not configured"
        result = abstain(reason)
        assert result.abstained is not None
        assert result.abstained.reason == reason

    def test_abstain_includes_missing_in_result(self) -> None:
        """abstain() result must include missing items when provided."""
        result = abstain("no labels", "training data", "annotations")
        assert result.abstained is not None
        assert result.abstained.missing == ("training data", "annotations")

    def test_abstain_as_json_includes_reason(self) -> None:
        """When a result is abstained, as_json() must include the reason."""
        reason = "feature cache not found"
        result = abstain(reason, "phase 3 output")
        payload = result.as_json()
        assert payload["ran"] is False
        assert payload["abstained"]["reason"] == reason

    def test_abstain_as_json_includes_missing(self) -> None:
        """When a result is abstained, as_json() must include what is missing."""
        result = abstain("incomplete", "poses", "face blurs")
        payload = result.as_json()
        assert "abstained" in payload
        assert set(payload["abstained"]["missing"]) == {"poses", "face blurs"}


class TestPhaseResult:
    """Tests for PhaseResult and its JSON serialization."""

    def test_phase_result_ran_is_true_when_no_abstention(self) -> None:
        """PhaseResult.ran must be True when abstained is None."""
        result = PhaseResult()
        assert result.ran is True

    def test_phase_result_ran_is_false_when_abstained(self) -> None:
        """PhaseResult.ran must be False when abstained is not None."""
        abstention = Abstention(reason="test", missing=())
        result = PhaseResult(abstained=abstention)
        assert result.ran is False

    def test_phase_result_as_json_omits_abstained_when_none(self) -> None:
        """as_json() must omit the 'abstained' key when not abstained."""
        result = PhaseResult(summary={"status": "ok"})
        payload = result.as_json()
        assert "abstained" not in payload
        assert payload["ran"] is True

    def test_phase_result_as_json_includes_abstained_when_present(self) -> None:
        """as_json() must include 'abstained' when the result abstained."""
        result = abstain("no data", "sessions")
        payload = result.as_json()
        assert "abstained" in payload
        assert payload["abstained"]["reason"] == "no data"

    def test_phase_result_as_json_includes_summary(self) -> None:
        """as_json() must include summary fields in the output."""
        summary = {"phase": 1, "sessions_ingested": 42, "clips_kept": 120}
        result = PhaseResult(summary=summary)
        payload = result.as_json()
        assert payload["phase"] == 1
        assert payload["sessions_ingested"] == 42
        assert payload["clips_kept"] == 120

    def test_phase_result_as_json_includes_artefacts(self) -> None:
        """as_json() must include artefacts when present."""
        ref = ArtefactRef(
            name="weights",
            relative_path="model.pt",
            sha256="a" * 64,
            bytes=1024,
            produced_on_device="cpu",
        )
        result = PhaseResult(artefacts=(ref,))
        payload = result.as_json()
        assert "artefacts" in payload
        assert len(payload["artefacts"]) == 1
        assert payload["artefacts"][0]["name"] == "weights"


class TestPhaseContext:
    """Tests for PhaseContext: seeding, determinism, and output writing."""

    def test_phase_context_seed_from_config_run(self, config, tmp_path) -> None:
        """PhaseContext.seed must return config.run.seed, not from elsewhere."""
        context = PhaseContext(
            config=config,
            run_id="test_run_001",
            output_dir=tmp_path,
            device="cpu",
        )
        assert context.seed == config.run.seed

    def test_phase_context_deterministic_from_config_run(
        self, config, tmp_path
    ) -> None:
        """PhaseContext.deterministic must return config.run.deterministic."""
        context = PhaseContext(
            config=config,
            run_id="test_run_001",
            output_dir=tmp_path,
            device="cpu",
        )
        assert context.deterministic == config.run.deterministic

    def test_phase_context_write_creates_json_file(self, config, tmp_path) -> None:
        """PhaseContext.write() must create a JSON file in output_dir."""
        context = PhaseContext(
            config=config,
            run_id="test_run_001",
            output_dir=tmp_path,
            device="cpu",
        )
        payload = {"phase": 1, "clips": 50}
        path = context.write("results.json", payload)

        assert path.exists()
        assert path.parent == tmp_path
        assert path.name == "results.json"

    def test_phase_context_write_returns_path(self, config, tmp_path) -> None:
        """PhaseContext.write() must return the path to the written file."""
        context = PhaseContext(
            config=config,
            run_id="test_run_001",
            output_dir=tmp_path,
            device="cpu",
        )
        payload = {"status": "done"}
        result_path = context.write("output.json", payload)

        assert isinstance(result_path, Path)
        assert result_path.is_file()

    def test_phase_context_write_produces_valid_json(
        self, config, tmp_path
    ) -> None:
        """PhaseContext.write() must produce valid JSON that can be parsed back."""
        context = PhaseContext(
            config=config,
            run_id="test_run_001",
            output_dir=tmp_path,
            device="cpu",
        )
        payload = {"key": "value", "number": 42}
        path = context.write("test.json", payload)

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == payload

    def test_phase_context_artefact_records_relative_path(
        self, config, tmp_path
    ) -> None:
        """PhaseContext.artefact() must record path relative to output_dir.

        An absolute path in the manifest would name the machine rather than the
        file, breaking reproducibility in a bundle moved to another machine.
        """
        context = PhaseContext(
            config=config,
            run_id="test_run_001",
            output_dir=tmp_path,
            device="cpu",
        )

        file_dir = tmp_path / "subdir"
        file_dir.mkdir()
        file_path = file_dir / "artefact.txt"
        file_path.write_text("content")

        ref = context.artefact("test_artefact", file_path)

        assert ref.relative_path == "subdir/artefact.txt"
        assert not ref.relative_path.startswith("/")
        assert not ref.relative_path.startswith("C:")

    def test_phase_context_artefact_not_absolute_path(
        self, config, tmp_path
    ) -> None:
        """PhaseContext.artefact() must never record absolute paths."""
        context = PhaseContext(
            config=config,
            run_id="test_run_001",
            output_dir=tmp_path,
            device="cpu",
        )

        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        ref = context.artefact("test", file_path)

        recorded = ref.relative_path
        assert not Path(recorded).is_absolute()

    def test_phase_context_engine_returns_none_when_not_configured(
        self, tmp_path
    ) -> None:
        """PhaseContext.engine() must return None when database is not configured."""
        config = load_config("configs/default.yaml")

        context = PhaseContext(
            config=config,
            run_id="test_run_001",
            output_dir=tmp_path,
            device="cpu",
        )

        with patch("praxis.db.engine.database_url", return_value=None):
            engine = context.engine()
            assert engine is None


class TestArtefactSha256:
    """Tests for sha256_of: checksum computation."""

    def test_sha256_of_agrees_with_hashlib(self, tmp_path) -> None:
        """sha256_of() must compute the same hash as hashlib.sha256.

        The artefact manifest records SHA-256 of every file so that R7's
        determinism test can compare hashes between runs. The hash must be
        computed deterministically: two runs that read the same bytes must
        compute the same hash.
        """
        test_file = tmp_path / "test.txt"
        content = b"test content for checksum"
        test_file.write_bytes(content)

        computed = sha256_of(test_file)

        expected = hashlib.sha256(content).hexdigest()
        assert computed == expected

    def test_sha256_of_works_with_large_files(self, tmp_path) -> None:
        """sha256_of() must handle files larger than the chunk size."""
        test_file = tmp_path / "large.bin"
        chunk_size = 1024 * 1024

        large_content = b"x" * (chunk_size * 3 + 500)
        test_file.write_bytes(large_content)

        computed = sha256_of(test_file)
        expected = hashlib.sha256(large_content).hexdigest()

        assert computed == expected

    def test_sha256_of_is_lowercase_hex(self, tmp_path) -> None:
        """sha256_of() must return lowercase hex, not uppercase."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        digest = sha256_of(test_file)

        assert digest == digest.lower()
        assert all(c in "0123456789abcdef" for c in digest)
        assert len(digest) == 64


class TestWriteJson:
    """Tests for write_json: deterministic JSON serialization."""

    def test_write_json_creates_file(self, tmp_path) -> None:
        """write_json() must create a file at the specified path."""
        output_path = tmp_path / "output.json"
        payload = {"key": "value"}

        result = write_json(output_path, payload)

        assert output_path.exists()
        assert result == output_path

    def test_write_json_creates_parent_directories(self, tmp_path) -> None:
        """write_json() must create parent directories if they do not exist."""
        output_path = tmp_path / "deep" / "nested" / "output.json"
        payload = {"key": "value"}

        result = write_json(output_path, payload)

        assert result.exists()
        assert result.parent.is_dir()

    def test_write_json_is_deterministic_with_dict_order(self, tmp_path) -> None:
        """write_json() must produce byte-identical output for same dict.

        R7 compares artefact hashes between runs to detect non-determinism. If
        write_json() serialised dicts in insertion order, an honest rerun with
        dict keys inserted in a different order would produce a different hash,
        making the run look like it was not deterministic. sort_keys=True
        prevents this.
        """
        output1 = tmp_path / "out1.json"
        output2 = tmp_path / "out2.json"

        payload = {"z": 1, "a": 2, "m": 3}
        write_json(output1, payload)

        payload_reordered = {"a": 2, "m": 3, "z": 1}
        write_json(output2, payload_reordered)

        content1 = output1.read_bytes()
        content2 = output2.read_bytes()

        assert content1 == content2, (
            "write_json() produced different bytes for the same dict in "
            "different insertion orders; R7 determinism test would fail"
        )

    def test_write_json_includes_trailing_newline(self, tmp_path) -> None:
        """write_json() must include a trailing newline for POSIX compliance."""
        output_path = tmp_path / "output.json"
        payload = {"key": "value"}

        write_json(output_path, payload)

        content = output_path.read_text(encoding="utf-8")
        assert content.endswith("\n"), (
            "write_json() output must end with newline"
        )

    def test_write_json_uses_utf8_encoding(self, tmp_path) -> None:
        """write_json() must use UTF-8 encoding."""
        output_path = tmp_path / "output.json"
        payload = {"message": "hello world"}

        write_json(output_path, payload)

        content_utf8 = output_path.read_text(encoding="utf-8")
        assert "hello world" in content_utf8

    def test_write_json_sorts_keys(self, tmp_path) -> None:
        """write_json() must sort keys in the output."""
        output_path = tmp_path / "output.json"
        payload = {"zebra": 1, "apple": 2, "middle": 3}

        write_json(output_path, payload)

        content = output_path.read_text(encoding="utf-8")
        apple_pos = content.find('"apple"')
        middle_pos = content.find('"middle"')
        zebra_pos = content.find('"zebra"')

        assert apple_pos < middle_pos < zebra_pos, (
            "write_json() did not sort keys alphabetically"
        )

    def test_write_json_two_runs_produce_identical_bytes(self, tmp_path) -> None:
        """Two calls to write_json() with the same payload produce identical bytes.

        This is the test that validates R7's determinism requirement at the
        artefact level. If this test fails, the whole determinism contract fails.
        """
        payload = {
            "z_last": [1, 2, 3],
            "a_first": {"nested": "value"},
            "m_middle": 42,
        }

        out1 = tmp_path / "run1.json"
        out2 = tmp_path / "run2.json"

        write_json(out1, payload)
        write_json(out2, payload)

        bytes1 = out1.read_bytes()
        bytes2 = out2.read_bytes()

        assert bytes1 == bytes2, (
            "write_json() produced different bytes on two calls with the same "
            "payload; R7 determinism validation would fail"
        )


class TestArtefactFunction:
    """Tests for artefact(): creation of ArtefactRef objects."""

    def test_artefact_includes_name(self, tmp_path) -> None:
        """artefact() must include the provided name in the result."""
        file_path = tmp_path / "weights.pt"
        file_path.write_text("model")

        ref = artefact("model_weights", file_path)

        assert ref.name == "model_weights"

    def test_artefact_includes_device_info(self, tmp_path) -> None:
        """artefact() must record the device that produced it."""
        file_path = tmp_path / "weights.pt"
        file_path.write_text("model")

        ref = artefact(
            "weights", file_path, device="cuda", device_model="A100"
        )

        assert ref.produced_on_device == "cuda"
        assert ref.produced_on_device_model == "A100"

    def test_artefact_computes_sha256(self, tmp_path) -> None:
        """artefact() must compute and include the file's SHA-256."""
        file_path = tmp_path / "data.txt"
        file_path.write_text("test content")

        ref = artefact("data", file_path)

        expected = hashlib.sha256(b"test content").hexdigest()
        assert ref.sha256 == expected

    def test_artefact_records_file_size(self, tmp_path) -> None:
        """artefact() must record the file size in bytes."""
        file_path = tmp_path / "data.txt"
        content = b"test content"
        file_path.write_bytes(content)

        ref = artefact("data", file_path)

        assert ref.bytes == len(content)

    def test_artefact_with_relative_to_records_relative_path(
        self, tmp_path
    ) -> None:
        """artefact() with relative_to must record path relative to that dir."""
        base = tmp_path / "run_output"
        base.mkdir()
        subdir = base / "phase_3"
        subdir.mkdir()
        file_path = subdir / "pose.json"
        file_path.write_text("{}")

        ref = artefact("poses", file_path, relative_to=base)

        assert ref.relative_path == "phase_3/pose.json"

    def test_artefact_without_relative_to_records_absolute_path(
        self, tmp_path
    ) -> None:
        """artefact() without relative_to records absolute path."""
        file_path = tmp_path / "weights.pt"
        file_path.write_text("model")

        ref = artefact("weights", file_path)

        assert Path(ref.relative_path).is_absolute()


class TestRegisteredPhase:
    """Tests for RegisteredPhase: the phase registry entries."""

    def test_registered_phase_has_key(self) -> None:
        """RegisteredPhase must have a key."""
        def dummy(context):
            pass

        phase = RegisteredPhase(
            key="test", title="Test Phase", run=dummy, needs=()
        )
        assert phase.key == "test"

    def test_registered_phase_has_title(self) -> None:
        """RegisteredPhase must have a descriptive title."""
        def dummy(context):
            pass

        phase = RegisteredPhase(
            key="test", title="Test Phase", run=dummy
        )
        assert phase.title == "Test Phase"

    def test_registered_phase_has_callable_run(self) -> None:
        """RegisteredPhase.run must be callable."""
        def dummy(context):
            pass

        phase = RegisteredPhase(
            key="test", title="Test", run=dummy
        )
        assert callable(phase.run)

    def test_registered_phase_needs_is_tuple_of_strings(self) -> None:
        """RegisteredPhase.needs must be a tuple of phase keys."""
        def dummy(context):
            pass

        phase = RegisteredPhase(
            key="4", title="Model training", run=dummy, needs=("2", "3")
        )
        assert phase.needs == ("2", "3")
        assert isinstance(phase.needs, tuple)

    def test_every_registered_phase_run_is_callable(self) -> None:
        """Every phase in the registry must have a callable run function."""
        registry = load_all()
        for key, phase in registry.items():
            assert callable(phase.run), (
                f"phase {key!r} run is not callable"
            )

    def test_every_registered_phase_needs_is_tuple(self) -> None:
        """Every phase in the registry must have needs as a tuple."""
        registry = load_all()
        for key, phase in registry.items():
            assert isinstance(phase.needs, tuple), (
                f"phase {key!r} needs is not a tuple: {type(phase.needs)}"
            )
            for dep in phase.needs:
                assert isinstance(dep, str), (
                    f"phase {key!r} dependency {dep!r} is not a string"
                )


class TestPhaseErrorMessages:
    """Tests for PhaseError and its messages."""

    def test_phase_error_is_runtime_error(self) -> None:
        """PhaseError must be a subclass of RuntimeError."""
        assert issubclass(PhaseError, RuntimeError)

    def test_register_error_names_duplicate_phase(self) -> None:
        """register() error must name the phase that was duplicated."""
        with patch.dict("praxis.phases._REGISTRY", {}, clear=True):

            def phase_a(context):
                pass

            def phase_b(context):
                pass

            register("dup", "First")(phase_a)

            try:
                register("dup", "Second")(phase_b)
            except PhaseError as exc:
                error_msg = str(exc)
                assert "dup" in error_msg
                assert "registered twice" in error_msg


class TestBuildOrder:
    """Tests for BUILD_ORDER: the complete list of phases to run."""

    def test_build_order_is_tuple(self) -> None:
        """BUILD_ORDER must be a tuple of strings."""
        assert isinstance(BUILD_ORDER, tuple)
        assert all(isinstance(k, str) for k in BUILD_ORDER)

    def test_build_order_is_not_empty(self) -> None:
        """BUILD_ORDER must not be empty."""
        assert len(BUILD_ORDER) > 0

    def test_build_order_has_numeric_phases(self) -> None:
        """BUILD_ORDER must contain numeric phase keys."""
        numeric_keys = [k for k in BUILD_ORDER if k.isdigit()]
        assert len(numeric_keys) > 0

    def test_build_order_numeric_phases_are_sorted(self) -> None:
        """Numeric phases in BUILD_ORDER must be in ascending order."""
        numeric_keys = [k for k in BUILD_ORDER if k.isdigit()]
        numeric_values = [int(k) for k in numeric_keys]
        assert numeric_values == sorted(numeric_values)
