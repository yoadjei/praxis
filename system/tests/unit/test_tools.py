"""Resolving external binaries.

Two things matter here and neither is "does ffmpeg exist". The first is that the configured
binary is the one that runs and its version is read back, so a run manifest records the build
rather than the intention. The second is R6: this ffmpeg is compiled with https, srt, rist,
ssh and zmq, so an input path that is secretly a URL would be fetched from inside the
inference path unless every invocation says otherwise.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from praxis.tools import FILE_ONLY_PROTOCOLS, NO_STDIN, Tool, ToolError, resolve


@pytest.fixture(scope="module")
def ffmpeg(config) -> Tool:
    return resolve("ffmpeg", config.tools.ffmpeg)


def test_the_configured_ffmpeg_resolves(config, ffmpeg: Tool) -> None:
    """Declared in the config and absent from the machine is a broken run, not a warning."""
    assert ffmpeg.path.is_file()
    assert ffmpeg.version.startswith("9."), (
        f"configs/default.yaml pins ffmpeg 9.0.1 and {ffmpeg.path} reports {ffmpeg.version}")
    assert resolve("ffprobe", config.tools.ffprobe).version == ffmpeg.version


def test_a_missing_binary_names_the_key_that_declared_it() -> None:
    with pytest.raises(ToolError, match=r"tools\.ffmpeg"):
        resolve("ffmpeg", "B:/praxis/tools/nothing-is-here/ffmpeg.exe")


def test_a_binary_that_is_not_what_it_claims_is_refused(config) -> None:
    """ffprobe pointed at ffmpeg must fail. Both answer -version, so only the name separates
    them, and a probe that is really an encoder would hang waiting for an output file."""
    with pytest.raises(ToolError, match="does not identify itself as ffprobe"):
        resolve("ffprobe", config.tools.ffmpeg)

    with pytest.raises(ToolError, match="does not identify itself as ffmpeg"):
        resolve("ffmpeg", sys.executable)


def test_a_bare_command_name_is_resolved_on_path() -> None:
    """Colab and Kaggle ship ffmpeg on PATH; pinning an absolute Linux path would be the same
    mistake as trusting PATH here. The resolved location is still recorded."""
    stem = "python" if sys.platform == "win32" else "python3"
    with pytest.raises(ToolError, match="does not identify itself"):
        resolve("ffmpeg", stem)          # found on PATH, then rejected on identity, not absence


# ---------------------------------------------------------------------------
# R6. The whitelist is on unless someone types otherwise
# ---------------------------------------------------------------------------

def test_file_only_protocols_are_the_default(ffmpeg: Tool) -> None:
    argv = ffmpeg.command("-i", "clip.mp4")
    assert argv[1:4] == [*NO_STDIN, *FILE_ONLY_PROTOCOLS], "the whitelist precedes the input"
    assert "https" not in argv[3] and "http" not in argv[3]


def test_ffprobe_is_not_given_a_flag_it_rejects(config) -> None:
    """`-nostdin` is an ffmpeg option. Closing stdin at the subprocess level covers ffprobe."""
    argv = resolve("ffprobe", config.tools.ffprobe).command("-i", "clip.mp4")
    assert NO_STDIN[0] not in argv
    assert argv[1:3] == list(FILE_ONLY_PROTOCOLS), "R6 still applies to ffprobe"


def test_every_tool_can_actually_run_the_argv_it_builds(config) -> None:
    """The check that was missing. `command()` was asserted about and never executed for
    ffprobe, which does not accept `-nostdin` and failed with "Option not found" on every
    invocation. A test that only inspects the list it built tests nothing about the binary."""
    for name in ("ffmpeg", "ffprobe"):
        tool = resolve(name, getattr(config.tools, name))
        result = subprocess.run(tool.command("-hide_banner", "-loglevel", "error"),
                                capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=60)
        assert "Option not found" not in result.stderr, (
            f"{name} rejects an option that command() adds: {result.stderr.strip()[:160]}")


def test_opting_out_has_to_be_written_down(ffmpeg: Tool) -> None:
    assert FILE_ONLY_PROTOCOLS[0] not in ffmpeg.command("-i", "clip.mp4", file_only=False)


def test_the_whitelist_actually_refuses_a_url(ffmpeg: Tool) -> None:
    """The real check. A flag that looked right but did nothing would pass every test above."""
    result = subprocess.run(
        ffmpeg.command("-v", "error", "-i", "https://example.com/nope.mp4", "-f", "null", "-"),
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)

    assert result.returncode != 0
    assert "not on whitelist" in result.stderr, (
        f"ffmpeg did not refuse the URL on protocol grounds: {result.stderr[:200]}")
    assert "Connection" not in result.stderr and "resolve" not in result.stderr, (
        "R6: the refusal must happen before a socket is opened, not after it fails")
