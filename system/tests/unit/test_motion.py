# -*- coding: utf-8 -*-
"""Camera motion, against frames whose shift is known because the test applied it.

Synthetic frames validate the estimator's contract, not its behaviour on real footage: a
deterministic noise field is ideal content for phase correlation and real video is not. What
these tests fix is the sign convention, the drift-versus-jitter distinction the verdict rests
on, and the fact that the tolerance is derived from `zones.OVERLAP_GRID` rather than written
down twice.
"""
from __future__ import annotations

import numpy as np
import pytest

from praxis.preprocess import motion
from praxis.preprocess.zones import OVERLAP_GRID

SHAPE = (128, 96)


def textured(seed: int = 7) -> np.ndarray:
    """A frame with broadband detail, so the correlation peak is unambiguous."""
    return np.random.default_rng(seed).integers(0, 256, SHAPE, dtype=np.uint8)


def shifted(frame: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """`np.roll` by d gives `second[x] = first[x - d]`, which is the convention under test."""
    return np.roll(frame, (rows, columns), axis=(0, 1))


class TestTranslation:
    def test_identical_frames_have_not_moved(self):
        frame = textured()
        assert motion.translation(frame, frame) == (0, 0)

    @pytest.mark.parametrize("rows,columns", [(5, 0), (0, 7), (3, 4), (-6, -2), (11, -9)])
    def test_recovers_the_shift_that_was_applied(self, rows, columns):
        frame = textured()
        assert motion.translation(frame, shifted(frame, rows, columns)) == (rows, columns)

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="cannot correlate"):
            motion.translation(textured(), textured()[:64])

    def test_a_flat_frame_does_not_divide_by_zero(self):
        flat = np.zeros(SHAPE, dtype=np.uint8)
        assert motion.translation(flat, flat) == (0, 0)


class TestVerdict:
    def test_a_still_camera_is_static(self):
        frames = [textured() for _ in range(10)]
        result = motion.measure(frames)
        assert result.is_static
        assert result.max_drift == 0.0

    def test_a_pan_beyond_one_zone_cell_is_not_static(self):
        frame = textured()
        drift = int(np.hypot(*SHAPE) / OVERLAP_GRID) + 2
        frames = [frame] + [shifted(frame, 0, column)
                            for column in range(1, drift + 1)]
        result = motion.measure(frames)
        assert not result.is_static
        assert result.max_drift > result.tolerance

    def test_jitter_that_returns_is_still_static(self):
        """The design decision: shake does not invalidate a zone, but panning away does."""
        frame = textured()
        frames = [shifted(frame, 0, column % 2) for column in range(20)]
        result = motion.measure(frames)
        assert result.is_static
        assert result.p95_step > 0.0

    def test_drift_is_measured_from_the_first_frame_not_summed(self):
        """A camera that leaves and returns has drifted by its furthest point, not by zero."""
        frame = textured()
        far = int(np.hypot(*SHAPE) / OVERLAP_GRID) + 4
        frames = [frame, shifted(frame, 0, far), frame]
        result = motion.measure(frames)
        assert not result.is_static

    def test_one_frame_is_not_a_measurement(self):
        result = motion.measure([textured()])
        assert not result.is_static
        assert "1 frame(s) could be decoded" in result.reason

    def test_no_frames_is_not_a_measurement(self):
        result = motion.measure([])
        assert not result.is_static
        assert result.samples == 0


class TestTolerance:
    def test_the_tolerance_is_one_cell_of_the_zone_overlap_grid(self):
        """Derived, not chosen. If `zones.OVERLAP_GRID` moves, this moves with it."""
        assert motion.CameraMotion(samples=2, max_drift=0.0,
                                   p95_step=0.0).tolerance == 1.0 / OVERLAP_GRID

    def test_a_drift_equal_to_the_tolerance_is_still_static(self):
        """The boundary is inclusive, so a session sitting exactly at the zone resolution is not
        refused for a difference the zone geometry cannot express."""
        edge = motion.CameraMotion(samples=2, max_drift=1.0 / OVERLAP_GRID, p95_step=0.0)
        assert edge.is_static

    def test_the_reason_names_both_numbers_and_the_tolerance(self):
        result = motion.measure([textured(), shifted(textured(), 0, 1)])
        assert "of its diagonal" in result.reason
        assert "95th percentile" in result.reason
        assert "zone geometry is defined to" in result.reason


class TestSampleCount:
    @pytest.mark.parametrize("duration,expected", [
        (0.0, motion.MIN_SAMPLES),
        (-1.0, motion.MIN_SAMPLES),
        (10.0, motion.MIN_SAMPLES),
        (120.0, 120),
        (1947.0, motion.MAX_SAMPLES),
    ])
    def test_about_one_a_second_bounded_at_both_ends(self, duration, expected):
        assert motion.sample_count(duration) == expected
