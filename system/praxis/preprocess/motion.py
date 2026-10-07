# -*- coding: utf-8 -*-
"""Did the camera move, and may this session's room be marked with zones at all.

`zones.py` stores one polygon set per camera setup and normalises it so the setup survives a
change of recording resolution. It does not survive the camera moving: a polygon is a fixed map
from frame coordinates to room regions, and if the frame stops looking at the same room the map
is wrong for the rest of the session. The codebook already says so from the other direction - B3
is non-scorable when "the camera moved" - but nothing measured it, so "the camera moved" was a
judgement a rater made per clip with no number behind it.

**The tolerance is derived, not chosen.** `zones.OVERLAP_GRID` is 64, and its docstring states
that one cell of that grid is the finest zone geometry this project will distinguish: an overlap
thinner than one cell is not reported. So motion keeping a room point within one cell of where
it started cannot displace it beyond the resolution at which zones are defined in the first
place. One cell of the frame diagonal is therefore the tolerance, and it comes from the existing
constant rather than from a number picked to suit the corpus.

Measured on the blurred media, because D18 deletes the original and the blurred file is what
survives. Face blurring rewrites small regions and leaves the background, which is what a global
translation estimate reads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from praxis.preprocess.zones import OVERLAP_GRID

# Width the sampled frames are scaled to before correlating. One pixel at 384 wide is about a
# quarter of the tolerance on the shortest frame in the corpus, so the verdict is not decided by
# the measurement's own resolution.
SAMPLE_WIDTH = 384

# Enough samples to span a session without one seek per frame. D78: a 32-minute 1080p file
# cannot be walked frame by frame inside any sane timeout, and these are input seeks.
MIN_SAMPLES = 60
MAX_SAMPLES = 240


@dataclass(frozen=True)
class CameraMotion:
    """How far the frame drifted, against the tolerance zones are defined at.

    `max_drift` is the measurement that decides the verdict: the furthest a sample's content sat
    from the first sample's, as a fraction of the frame diagonal. Drift rather than per-step
    jitter, because a camera that shakes and returns still points at the same room, while one
    that pans away does not, and it is the panning that invalidates a marked zone.

    `p95_step` is reported alongside and decides nothing. It is the diagnostic separating
    a handheld recording from a tripod that was nudged once, which is a thesis observation about
    unconstrained camera placement rather than an input to this verdict.
    """

    samples: int
    max_drift: float
    p95_step: float
    tolerance: float = 1.0 / OVERLAP_GRID

    @property
    def is_static(self) -> bool:
        return self.samples >= 2 and self.max_drift <= self.tolerance

    @property
    def reason(self) -> str:
        if self.samples < 2:
            return (
                f"only {self.samples} frame(s) could be decoded, so motion was not "
                f"measured and the camera cannot be called static")
        verdict = "within" if self.is_static else "beyond"
        return (
            f"the frame drifted at most {self.max_drift:.1%} of its diagonal over "
            f"{self.samples} samples, {verdict} the {self.tolerance:.1%} that zone geometry "
            f"is defined to, and moved {self.p95_step:.1%} between samples at the 95th "
            f"percentile")

    def as_json(self) -> dict[str, object]:
        return {"samples": self.samples, "max_drift": self.max_drift,
                "p95_step": self.p95_step, "tolerance": self.tolerance,
                "is_static": self.is_static, "reason": self.reason}


def sample_count(duration_s: float) -> int:
    """About one sample a second, bounded at both ends."""
    if duration_s <= 0:
        return MIN_SAMPLES
    return max(MIN_SAMPLES, min(MAX_SAMPLES, int(duration_s)))


def _prepared(frame: np.ndarray) -> np.ndarray:
    """Mean removed and a Hann window applied, which is what makes the correlation readable.

    Without the window the FFT treats the frame as tiling the plane, and the discontinuity
    at the edges puts a cross of energy through the correlation surface that can outrank the
    true peak on a low-contrast frame. Removing the mean drops the DC term, which otherwise
    peaks at zero shift whatever the content did.
    """
    rows = np.hanning(frame.shape[0])[:, None]
    columns = np.hanning(frame.shape[1])[None, :]
    return (frame.astype(np.float64) - float(frame.mean())) * rows * columns


def _spectrum(frame: np.ndarray) -> np.ndarray:
    return np.fft.rfft2(_prepared(frame))


def _wrap(index: int, length: int) -> int:
    """A correlation peak past the midpoint is a negative shift, not a large positive one."""
    return index - length if index > length // 2 else index


def translation(first: np.ndarray, second: np.ndarray) -> tuple[int, int]:
    """Rows and columns the content moved between two frames of the same shape.

    Phase correlation. For `second(x) = first(x - d)` the cross-power spectrum normalised to
    unit magnitude is `exp(+2*pi*i*k*d/N)`, whose inverse transform is an impulse at `-d`, so
    the peak is negated to recover `d`.
    """
    if first.shape != second.shape:
        raise ValueError(f"cannot correlate a {first.shape} frame against a {second.shape} one")
    return _translation(_spectrum(first), _spectrum(second), first.shape)


def _translation(first: np.ndarray, second: np.ndarray,
                 shape: tuple[int, ...]) -> tuple[int, int]:
    cross = first * np.conj(second)
    magnitude = np.abs(cross)
    normalised = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 0)
    surface = np.fft.irfft2(normalised, s=shape)
    row, column = np.unravel_index(int(np.argmax(surface)), surface.shape)
    return -_wrap(int(row), shape[0]), -_wrap(int(column), shape[1])


def measure(frames: list[np.ndarray]) -> CameraMotion:
    """Drift from the first frame and step-to-step movement, over frames already sampled.

    Each frame is correlated against the first directly rather than by summing the steps, so the
    drift figure carries no accumulated error from the steps it did not take.
    """
    if len(frames) < 2:
        return CameraMotion(samples=len(frames), max_drift=0.0, p95_step=0.0)

    shape = frames[0].shape
    diagonal = float(np.hypot(*shape))
    reference = _spectrum(frames[0])

    drifts, steps = [], []
    previous = reference
    for frame in frames[1:]:
        current = _spectrum(frame)
        drifts.append(float(np.hypot(*_translation(reference, current, shape))) / diagonal)
        steps.append(float(np.hypot(*_translation(previous, current, shape))) / diagonal)
        previous = current

    return CameraMotion(samples=len(frames), max_drift=max(drifts),
                        p95_step=float(np.percentile(steps, 95)))
