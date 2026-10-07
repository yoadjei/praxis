# -*- coding: utf-8 -*-
"""Finding what an earlier phase produced, and proving it is what it claimed.

Phases run as separate processes, each writing its own manifest, so phase 6 has no handle on
phase 5's objects and must find them on disk. This is the half of R7 that the manifest type
promised and nothing implemented: a run that cannot say which bytes it read is not reproducible
however carefully its config was frozen.

Two rules the lookup follows.

**The newest run wins, and only if it ran.** A run that abstained or failed left a manifest
behind too, and treating those alike is how a build comes to read a half-written weight file.
The verdict is taken from `result.json`, not inferred from the artefacts being present.

**A checksum mismatch is an error, not a warning.** The manifest records the SHA-256 of
every artefact. If the file on disk no longer matches, something edited it between phases, and
continuing would report numbers computed from bytes nobody has described. L20 applies: this is
the one place that decides an artefact is trustworthy, so no caller has to remember to verify.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from praxis.config_schema import PraxisConfig
from praxis.contracts.manifest import ArtefactRef
from praxis.phases.artefacts import sha256_of


class ArtefactMissing(RuntimeError):
    """An artefact an earlier phase should have produced is not there.

    Raised rather than returned where the caller has already established the run exists, which
    means the manifest and the directory disagree - a different problem from "that phase has not
    run", and one no abstention should absorb.
    """


class ArtefactCorrupt(RuntimeError):
    """The bytes on disk are not the bytes the manifest describes."""


@dataclass(frozen=True)
class CompletedRun:
    """One earlier run of one phase, and what it left behind."""

    run_id: str
    phase: str
    directory: Path
    config_sha256: str
    finished_at: str | None
    artefacts: tuple[ArtefactRef, ...]

    def artefact(self, name: str) -> ArtefactRef:
        for candidate in self.artefacts:
            if candidate.name == name:
                return candidate
        available = ", ".join(sorted(a.name for a in self.artefacts)) or "none"
        raise ArtefactMissing(
            f"run {self.run_id} (phase {self.phase}) has no artefact named {name!r}; "
            f"it recorded: {available}")

    def path_to(self, name: str) -> Path:
        """The artefact's absolute path, verified against its recorded hash."""
        reference = self.artefact(name)
        path = (self.directory / reference.relative_path).resolve()
        if not path.is_file():
            raise ArtefactMissing(
                f"{name!r} is in run {self.run_id}'s manifest at {reference.relative_path} "
                f"but no file is there")
        actual = sha256_of(path)
        if actual != reference.sha256:
            raise ArtefactCorrupt(
                f"{name!r} from run {self.run_id} hashes to {actual[:12]} and its manifest "
                f"says {reference.sha256[:12]}; the file changed after the run that made it")
        return path


# What a manifest must name before the directory counts as a run a later phase may consume.
# `config_sha256` is the load-bearing one: under R7 a run is identified by the config that
# produced it, so a directory that cannot say which config it came from is not a run this system
# can build on, however complete it otherwise looks. Requiring it here is also what keeps a
# hand-written or fabricated directory from being handed to a phase as a genuine input.
REQUIRED_MANIFEST_FIELDS = ("run_id", "phase", "config_sha256")


def _read(directory: Path) -> tuple[dict, dict] | None:
    """A run directory's manifest and result, or None if it is not a complete run directory."""
    manifest_file, result_file = directory / "manifest.json", directory / "result.json"
    if not (manifest_file.is_file() and result_file.is_file()):
        return None
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A directory being written right now, or one truncated by a crash. Skipped rather than
        # raised: the caller wants the newest *usable* run, and an unreadable one is not it.
        return None
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        return None
    if any(field not in manifest for field in REQUIRED_MANIFEST_FIELDS):
        return None
    return (manifest, result)


def all_completed_runs(config: PraxisConfig) -> list[CompletedRun]:
    """Every run in the output directory that actually ran, of any phase, newest first.

    Ordered by the run identifier rather than by file modification time. A ULID's first 48
    bits are its millisecond timestamp, so sorting the identifiers sorts the runs, and unlike
    mtime it does not change when a directory is copied to another machine.

    This is the one place that decides what counts as a run: both files present, both parseable,
    and `ran` true. Anything else in the directory is a half-written or abandoned attempt. Other
    code asks here rather than globbing for manifests, because a second definition of "a run"
    would drift from this one and the two would disagree about an abstention (L20).
    """
    root = config.paths.run_outputs
    if not root.is_dir():
        return []

    found: list[CompletedRun] = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        pair = _read(directory)
        if pair is None:
            continue
        manifest, result = pair
        if not result.get("ran"):
            continue
        found.append(CompletedRun(
            run_id=str(manifest.get("run_id", directory.name)),
            phase=str(manifest.get("phase", "")),
            directory=directory,
            config_sha256=str(manifest.get("config_sha256", "")),
            finished_at=manifest.get("finished_at"),
            artefacts=tuple(ArtefactRef(**a) for a in manifest.get("artefacts", []))))
    return found


def completed_runs(config: PraxisConfig, phase: str) -> list[CompletedRun]:
    """Every run of `phase` that actually ran, newest first."""
    return [run for run in all_completed_runs(config) if run.phase == phase]


def latest_run(config: PraxisConfig, phase: str, *,
               same_config_as: str | None = None) -> CompletedRun | None:
    """The most recent run of `phase` that ran, or None.

    `same_config_as` restricts the search to runs of one config hash. Off by default, because a
    config legitimately changes between phases - adding a manifest path before phase 1 does not
    invalidate a phase 3 run - but a phase comparing numbers across runs should pass it, since
    two results computed under different settings are not comparable whatever their provenance.
    """
    for run in completed_runs(config, phase):
        if same_config_as is None or run.config_sha256 == same_config_as:
            return run
    return None


def require(config: PraxisConfig, phase: str) -> CompletedRun:
    """The most recent run of `phase`, or an `ArtefactMissing` naming how to produce one."""
    run = latest_run(config, phase)
    if run is None:
        raise ArtefactMissing(
            f"no completed run of phase {phase} under {config.paths.run_outputs}; "
            f"run it with: python scripts/run_phase.py --phase {phase}")
    return run
