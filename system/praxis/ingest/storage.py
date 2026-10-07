# -*- coding: utf-8 -*-
"""Content-addressed media storage.

The hash is only known once the last byte has been read, so the final path is not known while
the file is being written. The sequence that follows from that: write to a temporary file under
the media root, hash while writing, then `os.replace` into `media/<sha[:2]>/<sha>.mp4`.

The temporary file lives under the media root rather than in the system temp directory, and that
is load-bearing. `os.replace` is atomic only within one filesystem; across volumes it degrades
to a copy, which at 20 GiB is neither atomic nor acceptable. D46.

Every failure path removes the temporary file. "A rejected upload leaves the system in exactly
the state it was in before" is the phase's definition of done, and a 20 GiB orphan on a volume
that a later phase reads by scanning is a slow way to break that.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from praxis.ingest.errors import media_volume_unreachable, too_large, unsupported_container

MEDIA_SUBDIR = "media"
TEMP_SUBDIR = ".ingest_tmp"


@dataclass(frozen=True)
class StoredMedia:
    """Where the bytes ended up, and whether this upload is the one that put them there."""

    media_sha256: str
    relative_path: str
    bytes_written: int
    deduplicated: bool

    def absolute(self, media_root: Path) -> Path:
        return media_root / self.relative_path


def relative_path_for(media_sha256: str, extension: str) -> str:
    """`media/ab/abcdef...mp4`. Fanned out by the first two characters so no directory holds
    more entries than a filesystem is comfortable listing."""
    return f"{MEDIA_SUBDIR}/{media_sha256[:2]}/{media_sha256}{extension}"


def normalised_extension(filename: str, accepted: Iterable[str]) -> str:
    """`.mp4` for an accepted container, or a refusal naming what is taken instead."""
    accepted = tuple(accepted)
    suffix = Path(filename).suffix.lower()
    if suffix.lstrip(".") not in accepted:
        raise unsupported_container(suffix or filename, accepted)
    return suffix


@contextmanager
def _temporary_file(media_root: Path) -> Iterator[Path]:
    """A temp file on the same filesystem as the destination, removed however we leave."""
    temp_dir = media_root / TEMP_SUBDIR
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise media_volume_unreachable(
            f"the media volume at {media_root} is not writable: {exc}") from exc

    path = temp_dir / f"{os.getpid()}-{os.urandom(8).hex()}.part"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def store(chunks: Iterable[bytes], media_root: Path, filename: str, *,
          accepted_containers: Iterable[str], max_bytes: int, chunk_bytes: int = 8 << 20,
          ) -> StoredMedia:
    """Consume `chunks`, hashing as they arrive, and place the result content-addressed.

    `chunk_bytes` is not used to read - the caller decides how the stream is sliced - but it is
    the size the digest is updated in, so it stays a configured number rather than an implicit
    one.

    The size limit is enforced while writing rather than from a declared content length, because
    a declared length is a claim by the client. Exceeding it stops the write immediately, so a
    caller cannot spend 20 GiB of disk proving they were over the limit.
    """
    extension = normalised_extension(filename, accepted_containers)
    digest = hashlib.sha256()
    written = 0

    with _temporary_file(media_root) as temp_path:
        try:
            with open(temp_path, "wb") as handle:
                for chunk in chunks:
                    written += len(chunk)
                    if written > max_bytes:
                        raise too_large(max_bytes)
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise media_volume_unreachable(
                f"writing to the media volume at {media_root} failed after {written} bytes: "
                f"{exc}") from exc

        media_sha256 = digest.hexdigest()
        relative = relative_path_for(media_sha256, extension)
        destination = media_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Identical bytes already stored. Keeping the existing file rather than replacing it
        # means a reader holding it open is never pulled out from under, and the content is by
        # definition the same.
        if destination.exists():
            return StoredMedia(media_sha256, relative, written, deduplicated=True)

        os.replace(temp_path, destination)

    return StoredMedia(media_sha256, relative, written, deduplicated=False)


def read_in_chunks(path: Path, chunk_bytes: int) -> Iterator[bytes]:
    """For callers holding a file rather than a stream, chiefly tests and re-ingest."""
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            yield chunk


def orphans(media_root: Path) -> list[Path]:
    """Temporary files left by a process that died mid-write.

    Not cleaned automatically. Deleting a `.part` file belonging to an upload still in flight
    would be worse than leaving it, and the reconciliation job that can tell the difference is
    Phase 11 work. This makes the debt visible rather than fixing it. D46.
    """
    temp_dir = media_root / TEMP_SUBDIR
    return sorted(temp_dir.glob("*.part")) if temp_dir.is_dir() else []
