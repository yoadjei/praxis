# -*- coding: utf-8 -*-
"""Run one build phase under a frozen configuration.

This is the only entry point that executes a phase, because R7 requires every run to be
reproducible from one config file and that is only true if there is one place a run can start.
It freezes a byte copy of the config, records its SHA-256, seeds everything, resolves the
device, and writes a manifest describing exactly what ran.

    python scripts/run_phase.py --list
    python scripts/run_phase.py --config configs/default.yaml --phase 3

Two runs of the same config must produce identical manifests but for the run identifier and
the timestamps. `tests/test_invariants.py::test_run_is_deterministic` asserts it.

Exit codes are three-valued on purpose. 0 means the phase ran, 3 means it abstained because an
input was missing, and 1 means it failed. A chain runner needs to tell the second from the
third: an abstention is a verdict and a failure is a defect, and collapsing them to "non-zero"
is how a build that did nothing comes to look like a build that broke.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.config import (
    ConfigError,
    PraxisConfig,
    config_sha256,
    load_config,
    prepare_environment,
    resolve_device,
    seed_everything,
)
from praxis.contracts.manifest import RunManifest
from praxis.ids import new_ulid
from praxis.phases import (
    PhaseContext,
    PhaseError,
    PhaseResult,
    ordered,
)
from praxis.phases import get as get_phase

RAN, FAILED, USAGE, ABSTAINED = 0, 1, 2, 3


def freeze_config(source: Path, run_id: str, config: PraxisConfig) -> Path:
    """Copy the config beside the run's outputs, byte for byte.

    A copy, not a re-serialisation: the SHA-256 recorded in the manifest is of the file, and a
    round trip through the YAML writer would change the bytes while claiming the same hash.
    """
    destination = config.paths.run_outputs / run_id
    destination.mkdir(parents=True, exist_ok=True)
    frozen = destination / "config.yaml"
    shutil.copyfile(source, frozen)
    return frozen


def describe_phases() -> str:
    """What `--list` prints: the build, in the order it runs, with what each waits on."""
    lines = ["phases, in build order:"]
    for phase in ordered():
        waits = f"  (needs {', '.join(phase.needs)})" if phase.needs else ""
        lines.append(f"  {phase.key:>4}  {phase.title}{waits}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one PRAXIS build phase.")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--phase", help="a phase number, or noop")
    parser.add_argument("--list", action="store_true",
                        help="list the phases and exit without running anything")
    args = parser.parse_args(argv)

    if args.list:
        print(describe_phases())
        return RAN
    if not args.phase:
        parser.error("--phase is required unless --list is given")

    try:
        config = load_config(args.config)
        overrides = prepare_environment(config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE

    try:
        registered = get_phase(args.phase)
    except PhaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE

    import torch

    run_id = new_ulid()
    device_used, device_model = resolve_device(config.run.device)
    seed_everything(config.run.seed, config.run.deterministic)

    frozen = freeze_config(args.config, run_id, config)
    output_dir = frozen.parent
    manifest = RunManifest(
        run_id=run_id,
        phase=args.phase,
        config_path=str(args.config).replace("\\", "/"),
        config_sha256=config_sha256(frozen),
        seed=config.run.seed,
        deterministic=config.run.deterministic,
        device_requested=config.run.device,
        device_used=device_used,
        device_model=device_model,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        platform=platform.platform(terse=True),
        num_workers=config.run.num_workers,
        environment_overrides=overrides,
        started_at=datetime.now(UTC),
    )

    if manifest.fell_back_to_cpu:
        print("note: cuda was requested and is unavailable; this run is a CPU run and the "
              "manifest records it as one", file=sys.stderr)

    context = PhaseContext(
        config=config, run_id=run_id, output_dir=output_dir,
        device=device_used, device_model=device_model)

    status = RAN
    try:
        result = registered.run(context)
    except PhaseError as exc:
        # A phase that ran and is wrong. Distinct from an abstention, and the manifest is still
        # written: a failed run that left no record of what it tried is not reproducible either.
        result = PhaseResult(summary={"error": str(exc)})
        status = FAILED
        print(f"error: phase {args.phase} failed: {exc}", file=sys.stderr)

    manifest.artefacts = list(result.artefacts)
    manifest.finished_at = datetime.now(UTC)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (output_dir / "result.json").write_text(
        json.dumps(result.as_json(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")

    if status == RAN and result.abstained is not None:
        status = ABSTAINED
        missing = ", ".join(result.abstained.missing)
        print(f"abstained: {result.abstained.reason}"
              + (f"\n  missing: {missing}" if missing else ""), file=sys.stderr)

    verdict = {RAN: "ran", ABSTAINED: "abstained", FAILED: "failed"}[status]
    print(f"run {run_id}  phase={args.phase}  {verdict}  device={device_used}  -> {output_dir}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
