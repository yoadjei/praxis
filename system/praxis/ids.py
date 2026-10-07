# -*- coding: utf-8 -*-
"""ULID generation and validation.

SCHEMA.md picks ULIDs for every primary key: sortable by creation time, needing no
coordination, and unlike a serial integer they leak no row count. The spec's §3 tree gives
them no home, so they get one here rather than being scattered.

Written out rather than pulled from a package because it is forty lines, because §2 forbids
adding dependencies casually, and because the encoding has to be stable for the lifetime of
the database. See D14.
"""
import secrets
import time

# Crockford base32. I, L, O and U are absent so a transcribed identifier cannot be misread.
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
LENGTH = 26
_TIMESTAMP_BITS = 48
_RANDOM_BITS = 80
_DECODE = {char: index for index, char in enumerate(ALPHABET)}


class ULIDError(ValueError):
    """Raised when a string is presented as a ULID and is not one."""


def new_ulid(timestamp_ms: int | None = None) -> str:
    """A fresh ULID: 48 bits of millisecond timestamp, 80 bits of randomness.

    `timestamp_ms` exists for tests that need a fixed prefix. It is never passed in
    production, where the clock is the point.
    """
    moment = time.time_ns() // 1_000_000 if timestamp_ms is None else timestamp_ms
    if not 0 <= moment < (1 << _TIMESTAMP_BITS):
        raise ULIDError(f"timestamp {moment} does not fit in {_TIMESTAMP_BITS} bits")
    return _encode((moment << _RANDOM_BITS) | secrets.randbits(_RANDOM_BITS))


def is_ulid(value: str) -> bool:
    """Whether `value` is a well-formed ULID. Cheap enough for a Pydantic validator."""
    return (
        isinstance(value, str)
        and len(value) == LENGTH
        and value[0] <= "7"  # 26 base32 chars hold 130 bits; the top 2 must be zero
        and all(char in _DECODE for char in value)
    )


def timestamp_ms(value: str) -> int:
    """The millisecond timestamp encoded in a ULID."""
    if not is_ulid(value):
        raise ULIDError(f"not a ULID: {value!r}")
    return _decode(value) >> _RANDOM_BITS


def _encode(value: int) -> str:
    characters = []
    for _ in range(LENGTH):
        value, remainder = divmod(value, 32)
        characters.append(ALPHABET[remainder])
    return "".join(reversed(characters))


def _decode(value: str) -> int:
    result = 0
    for char in value:
        result = result * 32 + _DECODE[char]
    return result
