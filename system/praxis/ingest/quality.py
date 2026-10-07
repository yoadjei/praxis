# -*- coding: utf-8 -*-
"""The quality gate: six checks, each with its own verdict, and one verdict for the session.

`evaluate` is pure. It takes measurements and returns judgements, so the rules can be tested
without a video file, and `measure` is the only part that runs a binary. Keeping the two apart
also means the thresholds are the only thing a reader has to follow to know why a session was
marked down.

A check fails when the session cannot do the job it exists for, which is recognising teacher
behaviour from the video. It warns when something is degraded but the session is still usable.
That is why a 480p file fails and a silent one warns: pose extraction cannot work without the
pixels, while audio is used for learner-response events only, and losing those costs one
downstream feature rather than the session.

`abstain` is the fourth verdict and it means the check did not run. It never raises the overall
verdict and it is never reported as `pass`, because a check that reports success without running
is indistinguishable, in every aggregate anyone computes, from one that ran. D48.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from praxis.config_schema import QualityGate
from praxis.ingest.probe import MediaMetadata, frame_rate_jitter, mean_luminance, probe
from praxis.tools import Tool

CheckVerdict = Literal["pass", "warn", "fail", "abstain"]
SessionVerdict = Literal["pass", "warn", "fail"]

# Abstention is deliberately absent: it says nothing about the session, only about the check.
SEVERITY: dict[CheckVerdict, int] = {"abstain": 0, "pass": 0, "warn": 1, "fail": 2}


@dataclass(frozen=True)
class Check:
    name: str
    verdict: CheckVerdict
    observed: Any
    expected: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "observed": self.observed, "expected": self.expected,
                **({"detail": self.detail} if self.detail else {})}


@dataclass(frozen=True)
class QualityReport:
    verdict: SessionVerdict
    checks: tuple[Check, ...]

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.verdict == "fail")

    @property
    def abstentions(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.verdict == "abstain")

    def as_detail(self) -> dict[str, Any]:
        """The JSONB stored in `sessions.quality_detail`."""
        return {check.name: check.as_dict() for check in self.checks}

    def reason(self) -> str:
        """One line naming what went wrong, for the audit trail and for a person."""
        if not self.failures:
            return "all checks passed" if self.verdict == "pass" else "passed with warnings"
        return "; ".join(f"{c.name}: {c.observed} (expected {c.expected})"
                         for c in self.failures)


def _verdict(ok: bool, warn_instead: bool = False) -> CheckVerdict:
    return "pass" if ok else ("warn" if warn_instead else "fail")


def evaluate(metadata: MediaMetadata, gate: QualityGate, *, fps_jitter: float,
             mean_luma: float, person_fraction: float | None) -> QualityReport:
    """Six checks against the configured thresholds. No I/O.

    `person_fraction` is None when no detector is available, which is the state Phase 1 ships
    in: the weights are not vendored. That becomes an abstention rather than a pass.
    """
    minutes = f"{gate.min_duration_s / 60:.0f} to {gate.max_duration_s / 60:.0f} minutes"
    checks = [
        Check("duration", _verdict(gate.min_duration_s <= metadata.duration_s
                                   <= gate.max_duration_s),
              round(metadata.duration_s, 2), minutes),

        # Frame area and shorter edge, not width and height. A width threshold asks a different
        # question of a portrait recording than of a landscape one, and half this corpus is
        # portrait: 576x1024 is the same amount of teacher as 1024x576 and the pose model
        # letterboxes both into the same square. What actually degrades detection is too few
        # pixels, and too thin a frame once letterboxed. D81.
        Check("resolution",
              _verdict(metadata.pixels >= gate.min_frame_pixels
                       and metadata.shorter_edge >= gate.min_shorter_edge),
              f"{metadata.width}x{metadata.height} ({metadata.orientation})",
              f"at least {gate.min_frame_pixels} px of frame and a shorter edge of "
              f"{gate.min_shorter_edge}"),

        # Recorded, never failed. Orientation is a property of how the corpus was captured, and
        # the Phase 6 attribution analysis needs it as a covariate; refusing on it would discard
        # the variation the thesis is about.
        Check("orientation", "pass", metadata.orientation,
              "recorded as a covariate, not gated"),

        Check("frame_rate", _verdict(metadata.fps >= gate.min_fps),
              round(metadata.fps, 3), f"at least {gate.min_fps} fps"),

        # Jitter warns rather than fails: the pipeline resamples to 8 fps anyway, so an unstable
        # source costs precision in clip boundaries rather than making the session unusable.
        Check("frame_rate_stability",
              _verdict(fps_jitter <= gate.max_fps_jitter, warn_instead=True),
              round(fps_jitter, 3), f"standard deviation at most {gate.max_fps_jitter} fps"),

        Check("luminance", _verdict(gate.min_mean_luminance <= mean_luma
                                    <= gate.max_mean_luminance),
              round(mean_luma, 1),
              f"mean Y between {gate.min_mean_luminance} and {gate.max_mean_luminance}"),
    ]

    if gate.require_audio:
        # Audio drives learner-response events only. Its absence costs one downstream feature,
        # not the session, so it warns.
        checks.append(Check("audio", _verdict(metadata.has_audio, warn_instead=True),
                            metadata.has_audio, "an audio stream"))

    checks.append(_person_check(gate, person_fraction))

    worst = max(SEVERITY[check.verdict] for check in checks)
    if worst == 2:
        verdict: SessionVerdict = "fail"
    elif worst == 1:
        verdict = "fail" if gate.fail_on_warn else "warn"
    else:
        verdict = "pass"

    return QualityReport(verdict=verdict, checks=tuple(checks))


def _person_check(gate: QualityGate, person_fraction: float | None) -> Check:
    expected = (f"a person in at least {gate.min_person_present_fraction:.0%} of "
                f"{gate.person_detection_sample_frames} sampled frames")
    if person_fraction is None:
        return Check("person_present", "abstain", None, expected,
                     detail="no detector weights are vendored, so this check did not run; "
                            "it is enforced from Phase 3 onward. D48")
    return Check("person_present",
                 _verdict(person_fraction >= gate.min_person_present_fraction),
                 round(person_fraction, 3), expected)


def measure(path: Path, gate: QualityGate, ffprobe: Tool, ffmpeg: Tool,
            person_fraction: float | None = None) -> tuple[MediaMetadata, QualityReport]:
    """Read the file and judge it. Raises `IngestRefused` when it cannot be read at all."""
    metadata = probe(ffprobe, path)
    report = evaluate(
        metadata, gate,
        fps_jitter=frame_rate_jitter(ffprobe, path),
        mean_luma=mean_luminance(ffmpeg, path, gate.person_detection_sample_frames,
                                 metadata.duration_s),
        person_fraction=person_fraction)
    return metadata, report
