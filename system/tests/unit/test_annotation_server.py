"""Unit tests for the annotation server domain logic.

Tests cover: clip arithmetic including the dropped tail and float drift, determinism of
build_assignments under a fixed seed, spread pattern verification, 422 refusal on unknown
labels, codebook version stamping, disagreement detection, and calibration rounds.
"""
from __future__ import annotations

import pytest

from praxis.annotation import (
    CODEBOOK_V1,
    Annotation,
    AnnotationRefused,
    CalibrationOutcome,
    ClipRef,
    accept_annotation,
    build_assignments,
    calibration_round,
    clip_plan,
    disagreements,
    parse_clip_id,
)

# =============================================================================
# CLIP ARITHMETIC TESTS
# =============================================================================


class TestClipPlan:
    """Tests for clip_plan: the 8-second stepper with dropped tail."""

    def test_clip_plan_basic(self) -> None:
        """A 40-second session yields exactly 5 clips of 8 seconds each."""
        clips = clip_plan("session_1", 40.0, clip_seconds=8.0)
        assert len(clips) == 5
        for i, clip in enumerate(clips):
            assert clip.session_id == "session_1"
            assert clip.clip_index == i
            assert clip.start_seconds == i * 8.0
            assert clip.end_seconds == (i + 1) * 8.0

    def test_clip_plan_tail_kept(self) -> None:
        """A remainder >= min_tail_seconds is kept as a final clip."""
        # 60 seconds: 7 full clips (56s) + 4-second tail. With min_tail=4.0, tail is kept.
        clips = clip_plan("session_2", 60.0, clip_seconds=8.0, min_tail_seconds=4.0)
        assert len(clips) == 8
        assert clips[-1].start_seconds == 56.0
        assert clips[-1].end_seconds == 60.0

    def test_clip_plan_tail_dropped(self) -> None:
        """A remainder < min_tail_seconds is dropped, not padded."""
        # 50 seconds: 6 full clips + 2-second tail, dropped because 2 < 4.
        clips = clip_plan("session_3", 50.0, clip_seconds=8.0, min_tail_seconds=4.0)
        assert len(clips) == 6
        # Last full clip is 5, ending at 48.
        assert clips[-1].clip_index == 5
        assert clips[-1].end_seconds == 48.0

    def test_clip_plan_short_session(self) -> None:
        """A session shorter than min_tail_seconds yields no clips."""
        clips = clip_plan("session_4", 3.0, clip_seconds=8.0, min_tail_seconds=4.0)
        assert len(clips) == 0

    def test_clip_plan_empty(self) -> None:
        """A session of zero length yields no clips."""
        clips = clip_plan("session_5", 0.0)
        assert len(clips) == 0

    def test_clip_plan_negative_duration(self) -> None:
        """A negative duration yields no clips, not an error."""
        clips = clip_plan("session_6", -10.0)
        assert len(clips) == 0

    def test_clip_id_format(self) -> None:
        """The clip_id property formats correctly."""
        clips = clip_plan("session_abc", 16.0)
        assert clips[0].clip_id == "session_abc:00000"
        assert clips[1].clip_id == "session_abc:00001"

    def test_parse_clip_id_round_trips_every_clip_in_a_plan(self) -> None:
        """parse_clip_id is the exact inverse of ClipRef.clip_id.

        The two have to stay inverses because assignments are stored as session_id plus
        clip_index while annotations carry the joined clip_id, and matching a label to the
        assignment it answers crosses that boundary. If the pair ever drifts apart, labels
        stop finding their assignments and disappear from every round-filtered query.
        """
        for clip in clip_plan("session_abc", 40.0):
            assert parse_clip_id(clip.clip_id) == (clip.session_id, clip.clip_index)

    def test_parse_clip_id_keeps_a_session_id_containing_a_colon(self) -> None:
        """The split is on the LAST colon, so a colon in the session id survives it."""
        assert parse_clip_id("tenant:a:00007") == ("tenant:a", 7)

    @pytest.mark.parametrize(
        "malformed",
        ["session_abc", "session_abc:", ":00001", "session_abc:abc", ""],
    )
    def test_parse_clip_id_refuses_what_it_did_not_build(self, malformed: str) -> None:
        """A malformed clip id raises rather than returning a plausible wrong answer.

        Returning None here would file an annotation with no assignment, which is the exact
        silent-orphan failure the linkage exists to prevent.
        """
        with pytest.raises(ValueError, match="not a clip id"):
            parse_clip_id(malformed)

    def test_clip_plan_determinism_no_float_drift(self) -> None:
        """Calling clip_plan twice yields byte-identical results, no drift."""
        clips1 = clip_plan("s1", 47.3, clip_seconds=8.0, min_tail_seconds=4.0)
        clips2 = clip_plan("s1", 47.3, clip_seconds=8.0, min_tail_seconds=4.0)
        assert clips1 == clips2
        # The tail should be preserved (47.3 - 40 = 7.3 >= 4.0).
        assert len(clips1) == 6
        assert clips1[-1].end_seconds == 47.3

    def test_clip_plan_sum_equals_duration(self) -> None:
        """The sum of clip lengths matches the session duration when tail is kept.

        When the tail is dropped, the sum is less by the amount of the dropped tail.
        """
        # Use a duration where the tail will be kept (>= min_tail_seconds).
        duration = 124.0  # 15 full clips (120s) + 4-second tail
        clips = clip_plan("s", duration, clip_seconds=8.0, min_tail_seconds=4.0)
        total = sum(clip.end_seconds - clip.start_seconds for clip in clips)
        assert abs(total - duration) < 1e-10


# =============================================================================
# ASSIGNMENT BUILDING TESTS
# =============================================================================


class TestBuildAssignments:
    """Tests for build_assignments: spread, determinism, no duplicates."""

    def test_build_assignments_basic(self) -> None:
        """Basic assignment creation."""
        clips = (
            ClipRef("s1", 0, 0.0, 8.0),
            ClipRef("s2", 0, 0.0, 8.0),
        )
        raters = ("alice", "bob", "charlie")
        assignments = build_assignments(
            clips,
            raters,
            round_name="calibration-1",
            raters_per_clip=2,
            seed=42,
        )
        # 2 clips * 5 behaviours * 2 raters = 20 assignments.
        assert len(assignments) == 2 * 5 * 2

    def test_build_assignments_deterministic(self) -> None:
        """Same seed, same assignments: the same rater rates the same clip for the same
        behaviour on every run. That is what `seed` is for and what R7 requires.

        Compared on the assignment itself - rater, clip, behaviour, round - and not on
        `assignment_id`, which is a fresh ULID per call exactly like `session_id`,
        `annotation_id` and every other surrogate key in this system. Two runs that agree on
        who rates what have produced the same plan; requiring their surrogate keys to collide
        as well would be asserting something R7 does not ask for and nothing else relies on.
        """
        clips = (
            ClipRef("s1", 0, 0.0, 8.0),
            ClipRef("s2", 0, 0.0, 8.0),
        )
        raters = ("alice", "bob", "charlie")
        kwargs = {"round_name": "calibration-1", "raters_per_clip": 2, "seed": 42}
        assignments1 = build_assignments(clips, raters, **kwargs)
        assignments2 = build_assignments(clips, raters, **kwargs)

        def plan(assignments):
            return [(a.rater_id, a.clip, a.behaviour, a.round_name) for a in assignments]

        assert plan(assignments1) == plan(assignments2)
        # And the surrogate keys really are distinct, so the comparison above is not
        # accidentally passing because the ids happened to match.
        ids1 = {a.assignment_id for a in assignments1}
        ids2 = {a.assignment_id for a in assignments2}
        assert len(ids1) == len(assignments1), "assignment ids are unique within a run"
        assert ids1.isdisjoint(ids2)

    def test_build_assignments_different_seed_different_result(self) -> None:
        """A different seed selects different raters.

        Compared on the plan rather than on the objects. Whole-object comparison would pass
        here whatever the seed did, because `assignment_id` is a fresh ULID on every call - the
        test would be unfailable, and an unfailable test hides the thing it was written to
        catch. Enough clips are used that agreement by chance is negligible.
        """
        clips = tuple(ClipRef("s1", i, i * 8.0, i * 8.0 + 8.0) for i in range(12))
        raters = ("alice", "bob", "charlie")

        def plan(seed):
            return [(a.rater_id, a.clip, a.behaviour) for a in build_assignments(
                clips, raters, round_name="calibration-1", raters_per_clip=2, seed=seed)]

        assert plan(42) != plan(99)
        assert plan(42) == plan(42), "and the same seed still reproduces itself"

    def test_build_assignments_spread_across_sessions(self) -> None:
        """Assignments spread across sessions, not concentrated in one.

        With 3 raters and raters_per_clip=2, the next clip should start with a different
        pair (or at least, the algorithm should rotate through raters).
        """
        clips = (
            ClipRef("s1", 0, 0.0, 8.0),
            ClipRef("s2", 0, 0.0, 8.0),
            ClipRef("s3", 0, 0.0, 8.0),
        )
        raters = ("alice", "bob", "charlie")
        assignments = build_assignments(
            clips, raters, round_name="calibration-1", raters_per_clip=2, seed=42
        )

        # For the first behaviour (B1), extract rater assignments per clip.
        b1_assigns = [a for a in assignments if a.behaviour == "B1"]
        clip1_raters = {a.rater_id for a in b1_assigns if a.clip.clip_index == 0}
        clip2_raters = {a.rater_id for a in b1_assigns if a.clip.clip_index == 1}
        clip3_raters = {a.rater_id for a in b1_assigns if a.clip.clip_index == 2}

        # Not all three clips should have the exact same pair of raters.
        # (This is not a guarantee, but with rotation it is highly likely.)
        distinct_pairs = {
            frozenset(clip1_raters),
            frozenset(clip2_raters),
            frozenset(clip3_raters),
        }
        assert len(distinct_pairs) >= 2

    def test_build_assignments_no_duplicates(self) -> None:
        """No rater is assigned the same clip and behaviour twice."""
        clips = (
            ClipRef("s1", 0, 0.0, 8.0),
            ClipRef("s2", 0, 0.0, 8.0),
        )
        raters = ("alice", "bob")
        assignments = build_assignments(
            clips, raters, round_name="calibration-1", raters_per_clip=2, seed=42
        )

        # Check uniqueness of (rater_id, clip_id, behaviour) triples.
        seen: set[tuple[str, str, str]] = set()
        for a in assignments:
            key = (a.rater_id, a.clip.clip_id, a.behaviour)
            assert key not in seen, f"duplicate assignment: {key}"
            seen.add(key)

    def test_build_assignments_empty_clips(self) -> None:
        """Empty clips list returns empty assignments."""
        assignments = build_assignments(
            (),
            ("alice", "bob"),
            round_name="calibration-1",
            raters_per_clip=1,
            seed=42,
        )
        assert assignments == ()

    def test_build_assignments_raters_per_clip_exceeds_rater_count(self) -> None:
        """raters_per_clip > len(raters) raises ValueError."""
        clips = (ClipRef("s1", 0, 0.0, 8.0),)
        raters = ("alice", "bob")
        with pytest.raises(ValueError, match="raters_per_clip"):
            build_assignments(
                clips, raters, round_name="calibration-1", raters_per_clip=5, seed=42
            )

    def test_build_assignments_no_raters(self) -> None:
        """Empty raters list raises ValueError."""
        clips = (ClipRef("s1", 0, 0.0, 8.0),)
        with pytest.raises(ValueError, match="at least one rater"):
            build_assignments(clips, (), round_name="calibration-1", raters_per_clip=1, seed=42)


# =============================================================================
# ACCEPT ANNOTATION TESTS
# =============================================================================


class TestAcceptAnnotation:
    """Tests for accept_annotation: codebook version stamping and validation."""

    def test_accept_annotation_valid(self) -> None:
        """A valid label is accepted and stamped with codebook version."""
        annotation = accept_annotation(
            CODEBOOK_V1,
            clip_id="s1:00000",
            rater_id="alice",
            behaviour="B1",
            labels={"b1_present": True, "b1_count": 3, "b1_amplitude": 2},
        )
        assert annotation.clip_id == "s1:00000"
        assert annotation.rater_id == "alice"
        assert annotation.behaviour == "B1"
        assert annotation.codebook_version == CODEBOOK_V1.version
        assert annotation.labels == {"b1_present": True, "b1_count": 3, "b1_amplitude": 2}

    def test_accept_annotation_unknown_field_raises_422(self) -> None:
        """An unknown field raises AnnotationRefused with http_status 422."""
        with pytest.raises(AnnotationRefused) as exc_info:
            accept_annotation(
                CODEBOOK_V1,
                clip_id="s1:00000",
                rater_id="alice",
                behaviour="B1",
                labels={"b1_present": True, "unknown_field": 999},
            )
        assert exc_info.value.http_status == 422
        assert "unknown_field" in exc_info.value.reason

    def test_accept_annotation_invalid_value_raises_422(self) -> None:
        """A value outside the field's domain raises AnnotationRefused with 422."""
        with pytest.raises(AnnotationRefused) as exc_info:
            accept_annotation(
                CODEBOOK_V1,
                clip_id="s1:00000",
                rater_id="alice",
                behaviour="B1",
                labels={"b1_present": "maybe"},  # boolean field
            )
        assert exc_info.value.http_status == 422

    def test_accept_annotation_codebook_version_stamped(self) -> None:
        """The annotation records the codebook version it was made under."""
        annotation = accept_annotation(
            CODEBOOK_V1,
            clip_id="s1:00000",
            rater_id="alice",
            behaviour="B1",
            labels={"b1_present": True, "b1_count": 1, "b1_amplitude": 1},
        )
        assert annotation.codebook_version == "v1.0-draft"

    def test_accept_annotation_nonscorable_flag(self) -> None:
        """The nonscorable flag is carried through."""
        annotation = accept_annotation(
            CODEBOOK_V1,
            clip_id="s1:00000",
            rater_id="alice",
            behaviour="B1",
            labels={"b1_nonscorable": True},
            is_nonscorable=True,
        )
        assert annotation.is_nonscorable is True

    def test_accept_annotation_rater_confidence(self) -> None:
        """Rater confidence is recorded."""
        annotation = accept_annotation(
            CODEBOOK_V1,
            clip_id="s1:00000",
            rater_id="alice",
            behaviour="B1",
            labels={"b1_present": True, "b1_count": 1, "b1_amplitude": 1},
            rater_confidence="probable",
        )
        assert annotation.rater_confidence == "probable"

    def test_accept_annotation_college_id(self) -> None:
        """Session college_id is recorded for familiarity checks."""
        annotation = accept_annotation(
            CODEBOOK_V1,
            clip_id="s1:00000",
            rater_id="alice",
            behaviour="B1",
            labels={"b1_present": True, "b1_count": 1, "b1_amplitude": 1},
            session_college_id="college_a",
        )
        assert annotation.session_college_id == "college_a"


# =============================================================================
# DISAGREEMENT TESTS
# =============================================================================


class TestDisagreements:
    """Tests for the disagreements function."""

    def test_disagreements_none(self) -> None:
        """Perfect agreement yields no disagreements."""
        annotations = (
            Annotation(
                clip_id="c1",
                rater_id="alice",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 3},
            ),
            Annotation(
                clip_id="c1",
                rater_id="bob",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 3},
            ),
        )
        result = disagreements(annotations)
        assert len(result) == 0

    def test_disagreements_detected(self) -> None:
        """Disagreements are detected."""
        annotations = (
            Annotation(
                clip_id="c1",
                rater_id="alice",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 3},
            ),
            Annotation(
                clip_id="c1",
                rater_id="bob",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 5},
            ),
        )
        result = disagreements(annotations)
        assert len(result) == 1
        d = result[0]
        assert d.clip_id == "c1"
        assert d.behaviour == "B1"
        assert d.field == "b1_count"
        assert d.by_rater == {"alice": 3, "bob": 5}
        assert d.distinct_values == 2

    def test_disagreements_multiple_fields(self) -> None:
        """Multiple fields can disagree on the same clip."""
        annotations = (
            Annotation(
                clip_id="c1",
                rater_id="alice",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 3},
            ),
            Annotation(
                clip_id="c1",
                rater_id="bob",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": False, "b1_count": 0},
            ),
        )
        result = disagreements(annotations)
        assert len(result) == 2
        fields = {d.field for d in result}
        assert fields == {"b1_present", "b1_count"}


# =============================================================================
# CALIBRATION ROUND TESTS
# =============================================================================


class TestCalibrationRound:
    """Tests for calibration_round: agreement + disagreements + gate."""

    def test_calibration_round_basic(self) -> None:
        """A calibration round computes agreement, disagreements, and gate."""
        annotations = (
            Annotation(
                clip_id="c1",
                rater_id="alice",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 3, "b1_amplitude": 2},
            ),
            Annotation(
                clip_id="c1",
                rater_id="bob",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 3, "b1_amplitude": 2},
            ),
            Annotation(
                clip_id="c2",
                rater_id="alice",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": False, "b1_count": 0, "b1_amplitude": 1},
            ),
            Annotation(
                clip_id="c2",
                rater_id="bob",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": False, "b1_count": 0, "b1_amplitude": 1},
            ),
        )
        outcome = calibration_round(
            annotations, raters=("alice", "bob"), round_name="calibration-1"
        )

        assert isinstance(outcome, CalibrationOutcome)
        assert outcome.round_name == "calibration-1"
        assert outcome.n_clips == 2
        assert outcome.n_raters == 2
        assert outcome.agreement is not None
        assert isinstance(outcome.disagreements, tuple)
        assert outcome.gate_value is not None
        assert isinstance(outcome.gate_passed, bool)

    def test_calibration_round_gate_passed(self) -> None:
        """A round with high agreement passes the gate."""
        # Create perfect agreement (all raters agree on everything).
        annotations_alice = tuple(
            Annotation(
                clip_id=f"c{i}",
                rater_id="alice",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": i + 1, "b1_amplitude": 2},
            )
            for i in range(5)
        )
        annotations_bob = tuple(
            Annotation(
                clip_id=f"c{i}",
                rater_id="bob",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": i + 1, "b1_amplitude": 2},
            )
            for i in range(5)
        )
        annotations = annotations_alice + annotations_bob

        outcome = calibration_round(
            annotations, raters=("alice", "bob"), round_name="calibration-1"
        )
        # Perfect agreement should pass the gate.
        assert outcome.gate_passed is True

    def test_calibration_round_gate_failed(self) -> None:
        """A round with low agreement fails the gate."""
        # Create perfect disagreement.
        annotations_alice = tuple(
            Annotation(
                clip_id=f"c{i}",
                rater_id="alice",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": True, "b1_count": 1, "b1_amplitude": 1},
            )
            for i in range(3)
        )
        annotations_bob = tuple(
            Annotation(
                clip_id=f"c{i}",
                rater_id="bob",
                behaviour="B1",
                codebook_version="v1.0-draft",
                labels={"b1_present": False, "b1_count": 5, "b1_amplitude": 3},
            )
            for i in range(3)
        )
        annotations = annotations_alice + annotations_bob

        outcome = calibration_round(
            annotations, raters=("alice", "bob"), round_name="calibration-1"
        )
        # Complete disagreement should fail the gate.
        assert outcome.gate_passed is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAnnotationIntegration:
    """End-to-end tests of the annotation workflow."""

    def test_full_workflow(self) -> None:
        """End-to-end: plan clips, assign raters, accept annotations, check calibration."""
        # Plan clips.
        clips_plan = clip_plan("session_1", 40.0, clip_seconds=8.0)
        assert len(clips_plan) == 5

        # Build assignments.
        raters = ("alice", "bob", "charlie")
        assignments = build_assignments(
            clips_plan,
            raters,
            behaviours=("B1", "B2"),
            round_name="calibration-1",
            raters_per_clip=2,
            seed=42,
        )
        assert len(assignments) == 5 * 2 * 2  # 5 clips * 2 behaviours * 2 raters

        # Accept some annotations.
        annotations = []
        for i, assignment in enumerate(assignments[:5]):
            annotation = accept_annotation(
                CODEBOOK_V1,
                clip_id=assignment.clip.clip_id,
                rater_id=assignment.rater_id,
                behaviour=assignment.behaviour,
                labels={
                    "b1_present": i % 2 == 0,
                    "b1_count": i,
                    "b1_amplitude": 1 + (i % 3),
                }
                if assignment.behaviour == "B1"
                else {
                    "b2_facing_proportion": 0.5,
                    "b2_head_torso_divergence": 1 + (i % 3),
                    "b2_dominant": "class",
                },
            )
            annotations.append(annotation)

        # Run calibration round.
        outcome = calibration_round(
            tuple(annotations), raters=raters, round_name="calibration-1"
        )
        assert outcome.n_clips > 0
        assert outcome.n_raters > 0
        assert outcome.agreement is not None
