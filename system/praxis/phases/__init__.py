# -*- coding: utf-8 -*-
"""The phase registry: the one place that says what a build phase is.

R7 requires every run to be reproducible from one config, and `scripts/run_phase.py` is the
only entry point that executes one. This module is what it executes. A phase is a function
over a `PhaseContext` returning a `PhaseResult` and nothing else: the orchestration lives
here, the work stays in the subsystem that owns it, and no phase reaches for a global engine
or a global config.

**A phase that cannot run abstains.** It does not raise and it does not quietly succeed. D48:
an unavailable check abstains and the abstention is its own verdict. There are no human
annotations yet, so most of the modelling phases abstain today, and a build that reports seven
abstentions with their reasons has told the truth about itself. The alternative - a phase that
returns an empty result and a zero - is how a thesis ends up with a table of numbers that mean
nothing.

Registration is an explicit decorator rather than a scan of the package. Twelve phases is a
number a person can read, and an auto-discovered registry would silently lose a phase whose
module failed to import.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from praxis.config_schema import PraxisConfig
from praxis.contracts.manifest import ArtefactRef, Device
from praxis.phases.artefacts import artefact, write_json

if TYPE_CHECKING:  # pragma: no cover - import cost only paid by type checkers
    from sqlalchemy.engine import Engine


class PhaseError(RuntimeError):
    """A phase failed for a reason that is not an abstention.

    The distinction is the point. An abstention means "this could not run and here is what is
    missing"; a `PhaseError` means "this ran and is wrong". Collapsing them would let a broken
    phase hide behind a missing input.
    """


@dataclass(frozen=True)
class Abstention:
    """Why a phase did not run, in a form a build summary can print.

    `missing` names the things that would have to exist, so the reader's next action is obvious
    rather than inferred.
    """

    reason: str
    missing: tuple[str, ...] = ()

    def as_json(self) -> dict[str, object]:
        return {"reason": self.reason, "missing": list(self.missing)}


@dataclass(frozen=True)
class PhaseResult:
    """What a phase hands back: a summary, the files it wrote, and whether it ran at all."""

    summary: Mapping[str, object] = field(default_factory=dict)
    artefacts: tuple[ArtefactRef, ...] = ()
    abstained: Abstention | None = None

    @property
    def ran(self) -> bool:
        return self.abstained is None

    def as_json(self) -> dict[str, object]:
        payload: dict[str, object] = {"ran": self.ran, **dict(self.summary)}
        if self.abstained is not None:
            payload["abstained"] = self.abstained.as_json()
        if self.artefacts:
            payload["artefacts"] = [a.model_dump(mode="json") for a in self.artefacts]
        return payload


def abstain(reason: str, *missing: str) -> PhaseResult:
    """Shorthand, because every phase needs it and a phase-local spelling would drift (L20)."""
    return PhaseResult(abstained=Abstention(reason=reason, missing=tuple(missing)))


@dataclass
class PhaseContext:
    """Everything a phase is given. Assembled once by the runner, never reached for globally.

    Not frozen, solely so the engine can be built on first use: most phases need no database and
    opening one to run `noop` would make the cheapest phase depend on the heaviest service.
    """

    config: PraxisConfig
    run_id: str
    output_dir: Path
    device: Device = "cpu"
    device_model: str | None = None
    _engine: Engine | None = None
    _engine_resolved: bool = False

    @property
    def seed(self) -> int:
        return self.config.run.seed

    @property
    def deterministic(self) -> bool:
        return self.config.run.deterministic

    def engine(self) -> Engine | None:
        """The configured database, or None when none is configured.

        None rather than a raise, so a phase can abstain with a reason the operator can act on.
        A *broken* database still raises, from `database_url`, because "not configured" and
        "configured and unreachable" must not look alike - the same rule the test suite follows.
        """
        if not self._engine_resolved:
            from praxis.db.engine import build_engine, database_url

            url = database_url(required=False)
            self._engine = build_engine(url) if url else None
            self._engine_resolved = True
        return self._engine

    def artefact(self, name: str, path: Path) -> ArtefactRef:
        """Checksum a file this phase wrote, recorded relative to the run's output directory."""
        return artefact(name, path, relative_to=self.output_dir,
                        device=self.device, device_model=self.device_model)

    def write(self, filename: str, payload: object) -> Path:
        """Write one of this phase's outputs into the run directory."""
        return write_json(self.output_dir / filename, payload)


Phase = Callable[[PhaseContext], PhaseResult]


@dataclass(frozen=True)
class RegisteredPhase:
    """A phase and what it is for, so `--list` explains the build without reading the code."""

    key: str
    title: str
    run: Phase
    needs: tuple[str, ...] = ()


_REGISTRY: dict[str, RegisteredPhase] = {}

# The order a build runs in. Declared rather than derived from the keys, because "10" sorts
# before "2" as a string, and a build that ran phase 10 before phase 2 would still look ordered.
BUILD_ORDER: tuple[str, ...] = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")


def register(key: str, title: str, *, needs: Sequence[str] = ()) -> Callable[[Phase], Phase]:
    """Add a phase to the registry under its BUILD-SPEC number."""

    def decorate(function: Phase) -> Phase:
        if key in _REGISTRY:
            raise PhaseError(f"phase {key!r} is registered twice")
        _REGISTRY[key] = RegisteredPhase(
            key=key, title=title, run=function, needs=tuple(needs))
        return function

    return decorate


def load_all() -> dict[str, RegisteredPhase]:
    """Import every phase module, then hand back the registry.

    Imported here rather than at module scope so that `praxis.phases` can be imported for its
    types alone - by the runner, or by a test - without pulling torch, onnxruntime and
    SQLAlchemy in behind it.
    """
    from praxis.phases import (  # noqa: F401
        analysis,
        corpus,
        modelling,
        noop,
        operations,
    )

    return dict(_REGISTRY)


def get(key: str) -> RegisteredPhase:
    """Look up one phase, with a message that lists the alternatives when it is not there."""
    registry = load_all()
    if key not in registry:
        raise PhaseError(
            f"unknown phase {key!r}; available: {', '.join(sorted(registry, key=_sort_key))}")
    return registry[key]


def _sort_key(key: str) -> tuple[int, str]:
    return (int(key), "") if key.isdigit() else (99, key)


def ordered() -> list[RegisteredPhase]:
    """Every registered phase in build order, with anything unnumbered last."""
    registry = load_all()
    numbered = [registry[k] for k in BUILD_ORDER if k in registry]
    extra = sorted((p for k, p in registry.items() if k not in BUILD_ORDER),
                   key=lambda p: p.key)
    return numbered + extra
