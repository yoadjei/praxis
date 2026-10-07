# -*- coding: utf-8 -*-
"""Tests for praxis/phases/chain.py: the integrity guarantee between phases."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from praxis.config_schema import PraxisConfig
from praxis.contracts.manifest import ArtefactRef
from praxis.phases.artefacts import sha256_of
from praxis.phases.chain import (
    ArtefactCorrupt,
    ArtefactMissing,
    CompletedRun,
    completed_runs,
    latest_run,
    require,
)


def _make_run_directory(
    tmp_path: Path,
    run_id: str,
    phase: str,
    config_sha256: str,
    ran: bool,
    artefacts: list[ArtefactRef] | None = None,
) -> Path:
    """Create a fake run directory with manifest.json and result.json.

    Matches the structure that scripts/run_phase.py writes. The run_id, phase and config_sha256
    are recorded in the manifest, while ran goes in result.json. When ran=False, the result
    simulates an abstained run that left a manifest behind.
    """
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "phase": phase,
        "config_sha256": config_sha256,
        "artefacts": [a.model_dump(mode="json") for a in (artefacts or [])],
        "finished_at": "2026-10-01T12:00:00+00:00",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = {"ran": ran}
    if not ran:
        result["abstained"] = {"reason": "test abstention", "missing": []}
    (run_dir / "result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")

    return run_dir


def test_completed_runs_returns_only_runs_that_ran(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """Only runs where result.json has ran=true are returned by completed_runs.

    An abstained run or a failed run left a manifest behind, and treating them as usable is how
    a later phase reads a half-written file. This test would fail if we returned an abstained
    run: the mutation would be to remove the `result.get("ran")` check.
    """
    root = tmp_path / "runs"
    root.mkdir()
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": root})})

    _make_run_directory(root, "run001", "1", "abc123def456", ran=True)
    _make_run_directory(root, "run002", "1", "abc123def456", ran=False)

    runs = completed_runs(test_config, "1")

    assert len(runs) == 1
    assert runs[0].run_id == "run001"


def test_completed_runs_is_newest_first_ordered_by_run_id(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """completed_runs returns runs newest first, ordered by run_id not mtime.

    ULIDs have their timestamp in the first 48 bits, so lexicographic ordering sorts by time.
    This test would fail if we sorted by mtime or returned in arbitrary order: the mutation
    would be to remove the reverse=True or sort by something else.
    """
    root = tmp_path / "runs"
    root.mkdir()
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": root})})

    # Build three runs with deterministic IDs (not real ULIDs, but lexicographically ordered).
    # Construct in non-sorted order to prove we sort, not just iterate in creation order.
    _make_run_directory(root, "01arkvqvjf92ts69r", "2", "abc", ran=True)
    _make_run_directory(root, "01awp3cfqg0006y1e", "2", "abc", ran=True)
    _make_run_directory(root, "01ap3cfqg0006y1e", "2", "abc", ran=True)

    runs = completed_runs(test_config, "2")

    assert len(runs) == 3
    assert [r.run_id for r in runs] == [
        "01awp3cfqg0006y1e",
        "01arkvqvjf92ts69r",
        "01ap3cfqg0006y1e",
    ]


def test_completed_runs_ignores_directory_with_no_manifest(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """A directory with no manifest.json is ignored rather than raising.

    A directory being written right now must not break the lookup. This test would fail if we
    raised an exception: the mutation would be to remove the `if pair is None` check.
    """
    root = tmp_path / "runs"
    root.mkdir()
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": root})})

    _make_run_directory(root, "run001", "1", "abc", ran=True)
    incomplete_dir = root / "run002"
    incomplete_dir.mkdir()

    runs = completed_runs(test_config, "1")

    assert len(runs) == 1
    assert runs[0].run_id == "run001"


def test_completed_runs_ignores_truncated_json(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """Truncated or invalid JSON in manifest.json or result.json is ignored.

    A directory being written right now, or one truncated by a crash, must not break the
    lookup. This test would fail if we raised a JSONDecodeError: the mutation would be to
    remove the try-except around json.loads.
    """
    root = tmp_path / "runs"
    root.mkdir()
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": root})})

    _make_run_directory(root, "run001", "1", "abc", ran=True)
    bad_json_dir = root / "run002"
    bad_json_dir.mkdir()
    (bad_json_dir / "manifest.json").write_text('{"run_id": "run002"', encoding="utf-8")
    (bad_json_dir / "result.json").write_text('{"ran": true}', encoding="utf-8")

    runs = completed_runs(test_config, "1")

    assert len(runs) == 1
    assert runs[0].run_id == "run001"


def test_completed_runs_returns_empty_list_when_run_outputs_does_not_exist(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """completed_runs returns [] when the run_outputs directory does not exist.

    This test would fail if we raised an exception: the mutation would be to remove the
    `if not root.is_dir()` check.
    """
    nonexistent = tmp_path / "does_not_exist"
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": nonexistent})})

    runs = completed_runs(test_config, "1")

    assert runs == []


def test_latest_run_returns_none_when_no_run_exists(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """latest_run returns None when no run of that phase exists.

    This test would fail if we returned an abstained run or raised an exception: the mutations
    would be to remove the None check or return something else.
    """
    root = tmp_path / "runs"
    root.mkdir()
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": root})})

    _make_run_directory(root, "run001", "1", "abc", ran=True)

    result = latest_run(test_config, "2")

    assert result is None


def test_latest_run_returns_newest_run_of_phase(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """latest_run returns the most recent run of the given phase.

    This test would fail if we returned an older run or the wrong phase: the mutations would be
    to remove the `phase` filter or return the first run instead of iterating for the newest.
    """
    root = tmp_path / "runs"
    root.mkdir()
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": root})})

    _make_run_directory(root, "01arkvqvjf92ts69r", "1", "abc", ran=True)
    _make_run_directory(root, "01awp3cfqg0006y1e", "1", "abc", ran=True)
    _make_run_directory(root, "01ap3cfqg0006y1e", "2", "abc", ran=True)

    result = latest_run(test_config, "1")

    assert result is not None
    assert result.run_id == "01awp3cfqg0006y1e"
    assert result.phase == "1"


def test_latest_run_filters_by_same_config_as(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """latest_run with same_config_as returns only a run with that config hash.

    This test would fail if we ignored the same_config_as parameter: the mutation would be to
    remove the `same_config_as` check.
    """
    root = tmp_path / "runs"
    root.mkdir()
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": root})})

    _make_run_directory(root, "run001", "3", "config_hash_aaa", ran=True)
    _make_run_directory(root, "run002", "3", "config_hash_bbb", ran=True)

    result = latest_run(test_config, "3", same_config_as="config_hash_aaa")

    assert result is not None
    assert result.run_id == "run001"
    assert result.config_sha256 == "config_hash_aaa"


def test_require_raises_artefact_missing_when_no_run_exists(
    tmp_path: Path,
    config: PraxisConfig,
) -> None:
    """require() raises ArtefactMissing and its message names the command to produce a run.

    This test would fail if we returned None or raised a different exception: the mutations
    would be to remove the raise or to return something. The message must name the phase so the
    operator can act on it.
    """
    root = tmp_path / "runs"
    root.mkdir()
    test_config = config.model_copy(update={"paths": config.paths.model_copy(
        update={"run_outputs": root})})

    with pytest.raises(ArtefactMissing) as exc_info:
        require(test_config, "4")

    error_msg = str(exc_info.value)
    assert "phase 4" in error_msg
    assert "run_phase.py" in error_msg or "scripts/run_phase.py" in error_msg


def test_completed_run_artefact_returns_matching_artefact(tmp_path: Path) -> None:
    """CompletedRun.artefact returns an ArtefactRef when the name is present.

    This test would fail if we returned None or the wrong artefact: the mutations would be to
    return something else or to skip the name check.
    """
    art1 = ArtefactRef(
        name="weights",
        relative_path="weights.pt",
        sha256="a" * 64,
        bytes=1000,
        produced_on_device="cpu",
    )
    art2 = ArtefactRef(
        name="metrics",
        relative_path="metrics.json",
        sha256="b" * 64,
        bytes=500,
        produced_on_device="cpu",
    )

    run = CompletedRun(
        run_id="test_run",
        phase="5",
        directory=tmp_path,
        config_sha256="abc" * 21 + "def",
        finished_at="2026-10-01T12:00:00",
        artefacts=(art1, art2),
    )

    result = run.artefact("metrics")

    assert result.name == "metrics"
    assert result.relative_path == "metrics.json"


def test_completed_run_artefact_raises_missing_for_unknown_name(tmp_path: Path) -> None:
    """CompletedRun.artefact raises ArtefactMissing and lists available names.

    This test would fail if we returned None or raised a different exception: the mutations
    would be to return something or raise KeyError. The message lists what was available so a
    caller knows what to look for.
    """
    art = ArtefactRef(
        name="weights",
        relative_path="weights.pt",
        sha256="a" * 64,
        bytes=1000,
        produced_on_device="cpu",
    )

    run = CompletedRun(
        run_id="test_run",
        phase="5",
        directory=tmp_path,
        config_sha256="abc" * 21 + "def",
        finished_at="2026-10-01T12:00:00",
        artefacts=(art,),
    )

    with pytest.raises(ArtefactMissing) as exc_info:
        run.artefact("nonexistent")

    error_msg = str(exc_info.value)
    assert "nonexistent" in error_msg
    assert "weights" in error_msg


def test_completed_run_artefact_lists_available_when_empty(tmp_path: Path) -> None:
    """CompletedRun.artefact lists 'none' when the run has no artefacts.

    This test would fail if we crashed or returned a confusing message: the mutation would be
    to remove the fallback message.
    """
    run = CompletedRun(
        run_id="test_run",
        phase="5",
        directory=tmp_path,
        config_sha256="abc" * 21 + "def",
        finished_at="2026-10-01T12:00:00",
        artefacts=(),
    )

    with pytest.raises(ArtefactMissing) as exc_info:
        run.artefact("anything")

    error_msg = str(exc_info.value)
    assert "none" in error_msg


def test_completed_run_path_to_returns_file_when_present_and_hash_matches(
    tmp_path: Path,
) -> None:
    """path_to returns the resolved path when the file is present and its hash matches.

    This test would fail if we returned None or the wrong path: the mutations would be to
    return something else or to skip the file check.
    """
    artefacts_dir = tmp_path / "artefacts"
    artefacts_dir.mkdir()
    weights_file = artefacts_dir / "weights.pt"
    weights_file.write_bytes(b"fake weights data")
    actual_hash = sha256_of(weights_file)

    art = ArtefactRef(
        name="weights",
        relative_path="artefacts/weights.pt",
        sha256=actual_hash,
        bytes=len(b"fake weights data"),
        produced_on_device="cpu",
    )

    run = CompletedRun(
        run_id="test_run",
        phase="5",
        directory=tmp_path,
        config_sha256="abc" * 21 + "def",
        finished_at="2026-10-01T12:00:00",
        artefacts=(art,),
    )

    result = run.path_to("weights")

    assert result.is_file()
    assert result.resolve() == weights_file.resolve()


def test_completed_run_path_to_raises_corrupt_when_file_modified(
    tmp_path: Path,
) -> None:
    """path_to raises ArtefactCorrupt when the file bytes no longer match the manifest hash.

    This is the single most valuable test: it asserts a phase cannot silently compute numbers
    from bytes nobody described. This test would fail if we returned the path or raised a
    different exception: the mutations would be to remove the hash check or return before
    raising. We modify the file after recording its hash, then verify the check catches it.
    """
    artefacts_dir = tmp_path / "artefacts"
    artefacts_dir.mkdir()
    weights_file = artefacts_dir / "weights.pt"
    weights_file.write_bytes(b"original data")
    original_hash = sha256_of(weights_file)

    art = ArtefactRef(
        name="weights",
        relative_path="artefacts/weights.pt",
        sha256=original_hash,
        bytes=len(b"original data"),
        produced_on_device="cpu",
    )

    run = CompletedRun(
        run_id="test_run",
        phase="5",
        directory=tmp_path,
        config_sha256="abc" * 21 + "def",
        finished_at="2026-10-01T12:00:00",
        artefacts=(art,),
    )

    # Modify the file after recording its hash.
    weights_file.write_bytes(b"modified data")

    with pytest.raises(ArtefactCorrupt) as exc_info:
        run.path_to("weights")

    error_msg = str(exc_info.value)
    assert "weights" in error_msg
    assert "changed" in error_msg


def test_completed_run_path_to_raises_missing_when_file_not_on_disk(
    tmp_path: Path,
) -> None:
    """path_to raises ArtefactMissing when the manifest names a file that is not on disk.

    This test would fail if we returned None or raised a different exception: the mutations
    would be to remove the file existence check or to raise ArtefactCorrupt instead.
    """
    art = ArtefactRef(
        name="weights",
        relative_path="nonexistent/weights.pt",
        sha256="a" * 64,
        bytes=1000,
        produced_on_device="cpu",
    )

    run = CompletedRun(
        run_id="test_run",
        phase="5",
        directory=tmp_path,
        config_sha256="abc" * 21 + "def",
        finished_at="2026-10-01T12:00:00",
        artefacts=(art,),
    )

    with pytest.raises(ArtefactMissing) as exc_info:
        run.path_to("weights")

    error_msg = str(exc_info.value)
    assert "weights" in error_msg
    assert "no file is there" in error_msg or "not" in error_msg
