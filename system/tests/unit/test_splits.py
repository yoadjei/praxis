# -*- coding: utf-8 -*-
"""Teacher-disjoint split generation.

R2 is the invariant a silent bug here would break, and the breakage would not look like a bug:
it would look like a good result. Every test in this file is therefore written to fail on the
permissive direction - a teacher crossing a partition, an S2 teacher who trained, a split that
moves between runs - rather than on the shape of the output.
"""
from __future__ import annotations

import pytest

from praxis.evaluation.shift import ShiftError, ShiftLevel, assert_no_leakage
from praxis.splits import (
    CLASSROOM,
    MICROTEACHING,
    SessionRecord,
    SplitError,
    plan_splits,
)

CONFIG_HASH = "b" * 64


def corpus(
    *,
    colleges: int = 3,
    teachers_per_college: int = 6,
    sessions_per_teacher: int = 3,
    classroom_teachers: tuple[str, ...] = (),
) -> list[SessionRecord]:
    """A synthetic corpus shaped like the real one: 60 to 100 sessions, a few colleges.

    Synthetic fixtures validate the partitioning contract. They say nothing about whether a
    model trained on this split performs well, which needs the corpus and is not what any test
    in this file claims.
    """
    sessions: list[SessionRecord] = []
    for college_index in range(colleges):
        college = f"C{college_index}"
        for teacher_index in range(teachers_per_college):
            teacher = f"C{college_index}T{teacher_index}"
            for session_index in range(sessions_per_teacher):
                sessions.append(SessionRecord(
                    session_id=f"{teacher}S{session_index}",
                    teacher_id=teacher,
                    college_id=college,
                    domain=MICROTEACHING,
                ))
    for teacher in classroom_teachers:
        college = teacher.split("T")[0]
        sessions.append(SessionRecord(
            session_id=f"{teacher}K0", teacher_id=teacher,
            college_id=college, domain=CLASSROOM))
    return sessions


def plan(sessions=None, **overrides):
    parameters = dict(config_sha256=CONFIG_HASH, seed=7,
                      val_fraction=0.15, test_fraction=0.20)
    parameters.update(overrides)
    return plan_splits(sessions if sessions is not None else corpus(), **parameters)


class TestDisjointness:
    """R2, from the generating side. The contract checks a manifest it is handed; these check
    that the generator never hands it a contaminated one in the first place."""

    def test_no_teacher_appears_in_two_partitions(self) -> None:
        manifest = plan().manifest
        train, val, test = (set(manifest.train_teachers), set(manifest.val_teachers),
                            set(manifest.test_teachers))
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_every_partition_is_populated(self) -> None:
        """An empty validation partition means early stopping monitors nothing, and an empty
        test partition means there is no S0 to compare S1 and S2 against."""
        manifest = plan().manifest
        assert manifest.train_teachers
        assert manifest.val_teachers
        assert manifest.test_teachers

    def test_a_corpus_too_small_to_partition_is_refused(self) -> None:
        small = corpus(colleges=1, teachers_per_college=2, sessions_per_teacher=2)
        with pytest.raises(SplitError, match="cannot fill three partitions"):
            plan(small)

    def test_fractions_that_starve_training_are_refused(self) -> None:
        """18 teachers at these fractions leave one to train on, which clears a floor of 1 and
        not a floor of 3. The floor is what decides, so the floor is what this varies."""
        assert plan(val_fraction=0.5, test_fraction=0.45).manifest.train_teachers

        with pytest.raises(SplitError, match=r"leave .* to train on"):
            plan(val_fraction=0.5, test_fraction=0.45, min_teachers_per_partition=3)

    def test_a_teacher_in_two_colleges_is_refused(self) -> None:
        """Not silently assigned to one of them. The split is stratified by college and S1 is
        defined by college, so this has no defensible answer."""
        sessions = corpus()
        sessions.append(SessionRecord(
            session_id="X", teacher_id="C0T0", college_id="C1", domain=MICROTEACHING))
        with pytest.raises(SplitError, match="more than one college"):
            plan(sessions)


class TestHeldOutCollege:
    """S1. A college withheld entirely, which is what institutional shift is measured on."""

    def test_heldout_college_teachers_are_in_no_partition(self) -> None:
        result = plan(heldout_college="C1")
        manifest = result.manifest

        assert result.heldout_college_teachers, "the fixture must actually hold someone out"
        for teacher in result.heldout_college_teachers:
            assert teacher not in manifest.train_teachers
            assert teacher not in manifest.val_teachers
            # Not test either. S0 is held-out *teachers* from the same institutions; putting
            # another college's teachers in it makes the baseline a shift condition.
            assert teacher not in manifest.test_teachers

    def test_the_heldout_college_is_recorded_on_the_manifest(self) -> None:
        assert plan(heldout_college="C1").manifest.heldout_college == "C1"

    def test_s1_sessions_are_the_heldout_colleges_microteaching(self) -> None:
        result = plan(heldout_college="C1")
        assert result.s1_sessions
        assert all(session.startswith("C1T") for session in result.s1_sessions)

    def test_a_heldout_college_with_no_teachers_is_refused(self) -> None:
        """An empty S1 reads as no institutional shift rather than as a missing condition."""
        with pytest.raises(SplitError, match="no teachers in this corpus"):
            plan(heldout_college="C99")

    def test_none_disables_s1_without_error(self) -> None:
        result = plan(heldout_college=None)
        assert result.manifest.heldout_college is None
        assert result.s1_sessions == ()


class TestClassroomTeachers:
    """S2. The thesis's primary finding rests on the model never having seen these teachers."""

    def test_classroom_teachers_never_train(self) -> None:
        sessions = corpus(classroom_teachers=("C0T0", "C1T1", "C2T2"))
        manifest = plan(sessions).manifest

        for teacher in ("C0T0", "C1T1", "C2T2"):
            assert teacher not in manifest.train_teachers
            assert teacher not in manifest.val_teachers

    def test_classroom_teachers_reach_the_test_partition_when_permitted(self) -> None:
        """Deliberate, and it is the stronger design: Phase 6 attributes the drop with a random
        intercept per teacher, which needs teachers observed at more than one level."""
        sessions = corpus(classroom_teachers=("C0T0",))
        manifest = plan(sessions, classroom_teachers_eligible_for_test=True).manifest
        assert "C0T0" in manifest.test_teachers

    def test_they_can_be_excluded_from_test_as_well(self) -> None:
        sessions = corpus(classroom_teachers=("C0T0",))
        manifest = plan(sessions, classroom_teachers_eligible_for_test=False).manifest
        assert "C0T0" not in manifest.test_teachers

    def test_every_classroom_session_is_declared_held_out(self) -> None:
        """An S2 session missing from shift_eval_sessions is one nothing was protecting."""
        sessions = corpus(classroom_teachers=("C0T0", "C2T3"))
        result = plan(sessions)
        classroom = {s.session_id for s in sessions if s.domain == CLASSROOM}
        assert set(result.manifest.shift_eval_sessions) == classroom

    def test_a_heldout_college_classroom_teacher_stays_out_of_every_partition(self) -> None:
        sessions = corpus(classroom_teachers=("C1T0",))
        manifest = plan(sessions, heldout_college="C1").manifest
        assert "C1T0" not in manifest.train_teachers
        assert "C1T0" not in manifest.val_teachers
        assert "C1T0" not in manifest.test_teachers


class TestLeakageGate:
    """The generator puts its own output through the gate Phase 6 runs before computing."""

    def test_the_generated_manifest_passes_the_phase_six_gate(self) -> None:
        sessions = corpus(classroom_teachers=("C0T0", "C1T1"))
        result = plan(sessions, heldout_college="C2")

        assert_no_leakage(
            result.manifest,
            {ShiftLevel.S1: result.s1_sessions, ShiftLevel.S2: result.s2_sessions},
            {s.session_id: s.teacher_id for s in sessions},
        )

    def test_the_gate_rejects_a_manifest_contaminated_by_hand(self) -> None:
        """The mutation the generator is protecting against, applied directly, so this file
        demonstrates the gate has teeth rather than assuming it."""
        sessions = corpus(classroom_teachers=("C0T0",))
        result = plan(sessions)
        contaminated = result.manifest.model_copy(
            update={"train_teachers": [*result.manifest.train_teachers, "C0T0"]})

        with pytest.raises(ShiftError, match="contribute S2 sessions"):
            assert_no_leakage(
                contaminated,
                {ShiftLevel.S2: result.s2_sessions},
                {s.session_id: s.teacher_id for s in sessions},
            )


class TestDeterminism:
    """R7. One config and one seed reproduce the partition."""

    def test_the_same_seed_gives_the_same_partition(self) -> None:
        sessions = corpus(classroom_teachers=("C0T0",))
        first = plan(sessions, seed=11, heldout_college="C2").manifest
        second = plan(sessions, seed=11, heldout_college="C2").manifest

        assert first.train_teachers == second.train_teachers
        assert first.val_teachers == second.val_teachers
        assert first.test_teachers == second.test_teachers
        assert first.shift_eval_sessions == second.shift_eval_sessions

    def test_a_different_seed_gives_a_different_partition(self) -> None:
        """Otherwise the seed is decorative and the previous test proves nothing."""
        sessions = corpus()
        assert (plan(sessions, seed=11).manifest.train_teachers
                != plan(sessions, seed=12).manifest.train_teachers)

    def test_the_order_sessions_arrive_in_does_not_change_the_split(self) -> None:
        """The corpus comes from a database query. An ORDER BY nobody wrote would otherwise
        change the partition, and with it every number measured on it."""
        sessions = corpus(classroom_teachers=("C0T0",))
        forwards = plan(sessions, seed=5).manifest
        backwards = plan(list(reversed(sessions)), seed=5).manifest

        assert forwards.train_teachers == backwards.train_teachers
        assert forwards.val_teachers == backwards.val_teachers
        assert forwards.test_teachers == backwards.test_teachers


class TestCollegeStratification:
    """S0 is the i.i.d. baseline, and a test partition drawn from one college is not i.i.d."""

    def test_every_college_is_represented_in_every_partition(self) -> None:
        sessions = corpus(colleges=3, teachers_per_college=6)
        manifest = plan(sessions, seed=3).manifest

        for name, teachers in (("train", manifest.train_teachers),
                               ("val", manifest.val_teachers),
                               ("test", manifest.test_teachers)):
            colleges = {teacher.split("T")[0] for teacher in teachers}
            assert len(colleges) > 1, (
                f"the {name} partition draws from {colleges} alone; a partition confined to "
                f"one college is institutional shift, which is S1, not the i.i.d. baseline")


def test_an_unknown_domain_is_refused_at_construction() -> None:
    """The domain decides the shift level, so one that is neither cannot be placed."""
    with pytest.raises(ValueError, match="expected one of"):
        SessionRecord(session_id="S", teacher_id="T", college_id="C", domain="staffroom")


def test_an_empty_corpus_is_refused() -> None:
    with pytest.raises(SplitError, match="no sessions"):
        plan([])


class TestWhenNoTeacherIsMicroteachingOnly:
    """D83. The strict rule of D73 bars any teacher with a classroom session from train and
    val, which assumes a pool of microteaching-only teachers. The real corpus is mostly
    classroom footage and has no such pool, so the assignable set empties and the planner
    refuses - correctly, because there would be nothing to train on that is not also what shift
    is measured against. `classroom_teachers_may_train` is the deliberate way out.
    """

    def every_teacher_also_teaches_a_class(self) -> list[SessionRecord]:
        teachers = tuple(f"C{c}T{t}" for c in range(3) for t in range(6))
        return corpus(classroom_teachers=teachers)

    def test_the_strict_rule_refuses_rather_than_training_on_nothing(self) -> None:
        with pytest.raises(SplitError, match="assignable teachers"):
            plan(self.every_teacher_also_teaches_a_class())

    def test_the_weaker_rule_produces_a_split(self) -> None:
        result = plan(self.every_teacher_also_teaches_a_class(),
                      classroom_teachers_may_train=True)
        manifest = result.manifest
        assert manifest.train_teachers and manifest.val_teachers and manifest.test_teachers

    def test_r2_still_holds_under_the_weaker_rule(self) -> None:
        """The relaxation is D73's extra protection, never R2 itself. A teacher in two
        partitions must stay impossible."""
        manifest = plan(self.every_teacher_also_teaches_a_class(),
                        classroom_teachers_may_train=True).manifest
        train, val, test = (set(manifest.train_teachers), set(manifest.val_teachers),
                            set(manifest.test_teachers))
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_the_manifest_records_which_rule_built_it(self) -> None:
        """Without this an S0-to-S2 drop measured under the weaker rule is indistinguishable
        from one measured under the strict rule, and they are different claims."""
        strict = plan(corpus(classroom_teachers=("C0T0",))).manifest
        weak = plan(self.every_teacher_also_teaches_a_class(),
                    classroom_teachers_may_train=True).manifest

        assert strict.classroom_teachers_trained is False
        assert weak.classroom_teachers_trained is True

    def test_the_leakage_gate_stands_down_only_for_a_manifest_that_declared_it(self) -> None:
        """The check reads the declaration off the manifest rather than taking a parameter, so
        the gate and the published artefact cannot disagree."""
        sessions = self.every_teacher_also_teaches_a_class()
        weak = plan(sessions, classroom_teachers_may_train=True).manifest

        s2 = [s.session_id for s in sessions if s.domain == CLASSROOM]
        teacher_of = {s.session_id: s.teacher_id for s in sessions}

        # the same manifest, with the declaration removed, must be refused
        strict_copy = weak.model_copy(update={"classroom_teachers_trained": False})
        with pytest.raises(ShiftError, match="contribute S2 sessions"):
            assert_no_leakage(strict_copy, {ShiftLevel.S2: s2}, teacher_of)

        assert_no_leakage(weak, {ShiftLevel.S2: s2}, teacher_of)

    def test_a_classroom_session_is_never_trained_on_even_under_the_weaker_rule(self) -> None:
        """The protection that never stands down. A teacher's microteaching may train; the
        classroom session held out for S2 may not, and that is a separate check."""
        sessions = self.every_teacher_also_teaches_a_class()
        weak = plan(sessions, classroom_teachers_may_train=True).manifest
        s2 = [s.session_id for s in sessions if s.domain == CLASSROOM]
        teacher_of = {s.session_id: s.teacher_id for s in sessions}

        with pytest.raises(ShiftError, match="never trained on"):
            assert_no_leakage(weak, {ShiftLevel.S2: s2}, teacher_of,
                              training_sessions=[s2[0]])

    def test_a_classroom_teacher_is_not_topped_into_test_on_top_of_a_train_place(self) -> None:
        """`classroom_teachers_eligible_for_test` adds S2 teachers to test. Once they can also
        train, adding one on top of a train place would put the same teacher in two partitions -
        R2 broken by the clause meant to use the corpus better."""
        sessions = corpus(classroom_teachers=tuple(f"C{c}T0" for c in range(3)))
        manifest = plan(sessions, classroom_teachers_may_train=True,
                        classroom_teachers_eligible_for_test=True).manifest

        assert set(manifest.train_teachers) & set(manifest.test_teachers) == set()
        assert set(manifest.val_teachers) & set(manifest.test_teachers) == set()
