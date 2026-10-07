# -*- coding: utf-8 -*-
"""Checksumming what a phase produced.

`RunManifest.artefacts` existed as a type before anything populated it. The reason it matters
is in its own docstring: a Kaggle or Colab session has no persistent filesystem, so the
machine that produced a weight file will not exist again and the file has to carry proof of
what it is. The same applies locally the moment one phase consumes another's output - a build
that cannot say which bytes it read cannot be reproduced from one config, whatever R7 claims.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from praxis.contracts.manifest import ArtefactRef, Device

CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    """Streamed, because a pose artefact or a weight file does not fit comfortably in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def artefact(name: str, path: Path, *, relative_to: Path | None = None,
             device: Device = "cpu", device_model: str | None = None) -> ArtefactRef:
    """Describe a file a phase wrote, with its hash and the device that produced it.

    `relative_to` is the run's output directory in the ordinary case. The path recorded is
    relative so a bundle can be moved and still verify; an absolute path would name a machine
    rather than a file, and naming the machine is what `produced_on_device` is for.
    """
    resolved = path.resolve()
    if relative_to is not None:
        try:
            recorded = resolved.relative_to(relative_to.resolve()).as_posix()
        except ValueError:
            recorded = resolved.as_posix()
    else:
        recorded = resolved.as_posix()

    return ArtefactRef(
        name=name,
        relative_path=recorded,
        sha256=sha256_of(resolved),
        bytes=resolved.stat().st_size,
        produced_on_device=device,
        produced_on_device_model=device_model)


def write_json(path: Path, payload: object) -> Path:
    """One JSON writer for every phase, so two phases cannot disagree about separators.

    `sort_keys` and the trailing newline are not tidiness: the artefact's SHA-256 goes in the
    manifest, and R7's determinism test compares those hashes between two runs of the same
    config. A dict that serialised in insertion order would make an honest run look like a
    different one.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path
