"""External binaries the pipeline shells out to.

Resolved from the configuration, never from PATH alone. PATH is machine state, and R7 says a
run is reproducible from one config; "whichever ffmpeg happened to be first" is not in the
config. A bare command name is still allowed, because Colab and Kaggle images ship one and
pinning an absolute Linux path would be the same mistake in the other direction — but the
resolved path and its version are returned together so a run manifest can record what it
actually used rather than what it asked for.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ffmpeg builds routinely enable https, srt, rist, ssh and zmq. Passed a URL it will fetch it,
# which is a network call from inside the inference path. Every invocation that opens a file
# the pipeline did not create must carry this, so a filename that is secretly a URL fails
# instead of downloading. R6.
FILE_ONLY_PROTOCOLS = ("-protocol_whitelist", "file,crypto,data")

# ffmpeg reads stdin for keyboard control. Under a worker with no console that is a hang, not
# an error, so every invocation says it is not interactive. **ffprobe does not accept this
# flag** and fails with "Option not found", which is why it is keyed by tool rather than added
# to everything; every caller also closes stdin at the subprocess level, so this is the second
# of two defences rather than the only one.
NO_STDIN = ("-nostdin",)
ACCEPTS_NO_STDIN = frozenset({"ffmpeg"})

def _version_line(name: str) -> re.Pattern[str]:
    """Keyed on the name asked for: ffprobe pointed at ffmpeg must fail, not pass."""
    return re.compile(rf"^{re.escape(name)} version (\S+)")


class ToolError(RuntimeError):
    """A declared binary is missing, unreadable, or not the thing it claims to be."""


@dataclass(frozen=True)
class Tool:
    name: str
    path: Path
    version: str

    def command(self, *args: str, file_only: bool = True) -> list[str]:
        """Argv for one invocation. `file_only` defaults on so forgetting it is a choice."""
        prefix = list(NO_STDIN) if self.name in ACCEPTS_NO_STDIN else []
        if file_only:
            prefix += list(FILE_ONLY_PROTOCOLS)
        return [str(self.path), *prefix, *map(str, args)]


def resolve(name: str, declared: str | Path) -> Tool:
    """Locate a binary and read its version back, or say precisely what is wrong."""
    candidate = Path(declared)
    found = str(candidate) if candidate.is_file() else shutil.which(str(declared))
    if found is None:
        raise ToolError(
            f"{name} is configured as {declared!r} and no such file is on disk or on PATH. "
            f"Install it and set tools.{name} to the binary, or unset it for the phases that "
            f"do not need video.")

    try:
        result = subprocess.run([found, "-version"], capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=30)
    except OSError as exc:
        raise ToolError(f"{name} at {found} could not be executed: {exc}") from exc

    match = _version_line(name).match(result.stdout.strip())
    if match is None:
        raise ToolError(
            f"{found} does not identify itself as {name}; its first line was "
            f"{result.stdout.strip().splitlines()[:1]}")
    return Tool(name=name, path=Path(found), version=match.group(1))
