# -*- coding: utf-8 -*-
"""Teacher-disjoint split generation. R2, enforced in code rather than by convention.

**The split key is `teacher_id`, never `session_id` and never `clip_id`.** A teacher records
several sessions, and two of them in different partitions is the same person at train and at
test time. The model learns the person - posture, gait, the particular way they gesture - and
reports it as behaviour recognition. BUILD-SPEC section 9 lists "random train/test split because
it is easier" as invalidating every result in the thesis, which is what this module exists to
make impossible rather than merely discouraged.

**Three levels have to come out of one partition.** A manifest that satisfies R2 across train,
val and test can still leak into S1 or S2, so this module does not stop at the contract's
pairwise check: it asks `assert_no_leakage` - the same gate Phase 6 runs before computing
anything - to accept its own output before returning it. The generator and the consumer
therefore cannot drift apart.

**Two ladders, named rather than guessed.** `shift_axis` decides what the levels hold out:

- `"domain"`: S1 is a held-out college's microteaching, S2 is all classroom footage. The
  original
  design, and what every manifest written before D98 is.
- `"site"`: S1 is a held-out basic school, S2 a held-out college. The only ladder available once
  the corpus is practicum throughout, because there is then no second domain to test in.

S2 picks out a different set of sessions on each, so the axis is written onto the manifest and
not inferred from the data. Inferring it would make the same config produce different level
definitions on different corpora, which R7 exists to prevent. D98.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from praxis.contracts.manifest import SplitManifest
from praxis.evaluation.shift import ShiftLevel, assert_no_leakage
from praxis.ids import new_ulid
from praxis.vocabulary import ShiftAxis

MICROTEACHING = "microteaching"
CLASSROOM = "classroom"
DOMAINS = (MICROTEACHING, CLASSROOM)


class SplitError(RuntimeError):
    """The corpus cannot be partitioned as asked, with the reason named."""


@dataclass(frozen=True)
class SessionRecord:
    """The facts about a session that decide which partition and which shift level it lands in.

    Deliberately not the `Session` contract. Splitting needs the split key, the institution, the
    site and the domain, and taking the full contract here would make the generator impossible
    to exercise without constructing a valid session in every other respect.

    `school_id` is None for microteaching, which happens on a college campus and has no basic
    school. On the site axis a classroom session without one cannot be placed on the ladder, and
    `plan_splits` refuses rather than letting it fall quietly into the training pool. D98.
    """

    session_id: str
    teacher_id: str
    college_id: str
    domain: str
    school_id: str | None = None

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(
                f"session {self.session_id} has domain {self.domain!r}; expected one of "
                f"{DOMAINS}. The domain decides shift level, so an unrecognised one cannot "
                f"be assigned a partition.")


@dataclass(frozen=True)
class SplitPlan:
    """A manifest with the reasoning that produced it kept alongside.

    The manifest alone says who is in which partition. It does not say who was excluded and
    why, and that is the question asked whenever a split looks wrong - typically "why is this
    teacher in neither" at the point where a result is being written up.
    """

    manifest: SplitManifest
    heldout_college_teachers: tuple[str, ...]
    heldout_school_teachers: tuple[str, ...]
    classroom_teachers: tuple[str, ...]
    s1_sessions: tuple[str, ...]
    s2_sessions: tuple[str, ...]

    @property
    def assignable_count(self) -> int:
        manifest = self.manifest
        return (len(manifest.train_teachers) + len(manifest.val_teachers)
                + len(manifest.test_teachers))


def _college_of(sessions: Sequence[SessionRecord]) -> dict[str, str]:
    """Each teacher's college, refusing a teacher who appears under two.

    A split is stratified by college and S1 is defined by college, so a teacher belonging to
    two of them has no well-defined partition. In a corpus this is a data-entry error rather
    than a real transfer, and it would otherwise surface as an unbalanced split nobody can
    explain.
    """
    colleges: dict[str, set[str]] = defaultdict(set)
    for session in sessions:
        colleges[session.teacher_id].add(session.college_id)

    conflicted = {teacher: sorted(found)
                  for teacher, found in colleges.items() if len(found) > 1}
    if conflicted:
        raise SplitError(
            f"teachers appear under more than one college: {conflicted}. The split is "
            f"stratified by college and a shift level is defined by it, so this has no answer.")
    return {teacher: found.pop() for teacher, found in colleges.items()}


def _levels_by_site(sessions: Sequence[SessionRecord], college_of: dict[str, str],
                    heldout_school: str | None,
                    heldout_college: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """S1 is the held-out school's sessions, S2 the held-out college's. D98.

    The two are built independently and a session can satisfy both - a teacher from the held-out
    college teaching at the held-out school. It is listed at both levels, which is truthful: it
    is held out for both reasons. What matters for leakage is that its teacher is in no
    partition, and appearing twice cannot weaken that.

    Not nested. Holding out a college does not hold out its schools, because one school hosts
    students from several colleges; nesting them would make S2 a superset of S1 by construction
    and the two rungs indistinguishable.
    """
    s1 = tuple(sorted(s.session_id for s in sessions
                      if heldout_school is not None and s.school_id == heldout_school))
    s2 = tuple(sorted(s.session_id for s in sessions
                      if heldout_college is not None
                      and college_of.get(s.teacher_id) == heldout_college))
    return s1, s2


def _levels_by_domain(sessions: Sequence[SessionRecord],
                      heldout_teachers: set[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """S1 is the held-out college's microteaching, S2 is every classroom session.

    The original ladder, kept unchanged so that the three stored manifests and the microteaching
    pilot still mean what they meant. `split_manifests` is append-only (D73), so a manifest
    built
    under this axis cannot be rewritten under the other.
    """
    s1 = tuple(sorted(s.session_id for s in sessions
                      if s.teacher_id in heldout_teachers and s.domain == MICROTEACHING))
    s2 = tuple(sorted(s.session_id for s in sessions if s.domain == CLASSROOM))
    return s1, s2


def _interleave(groups: Sequence[Sequence[str]]) -> list[str]:
    """Round-robin across colleges, so any contiguous slice is college-mixed.

    Why not shuffle the whole pool and slice it. S0 is defined as the *i.i.d.* microteaching
    test, and a uniform shuffle over a corpus of twenty-odd teachers can easily draw a test
    partition entirely from one college - which is S1, institutional shift, mislabelled as the
    baseline it is supposed to be compared against. Taking one teacher from each college in
    turn makes every partition proportionally representative by construction rather than in
    expectation.
    """
    ordered: list[str] = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                ordered.append(group[index])
    return ordered


def _partition_sizes(total: int, val_fraction: float, test_fraction: float,
                     minimum: int) -> tuple[int, int, int]:
    """How many teachers go to train, val and test.

    Rounded rather than truncated, then floored at `minimum`, because at the corpus size this
    project expects - 60 to 100 sessions, so roughly 15 to 25 teachers - truncation produces an
    empty validation partition and early stopping then monitors nothing.
    """
    needed = minimum * 3
    if total < needed:
        raise SplitError(
            f"{total} assignable teachers cannot fill three partitions of at least {minimum}. "
            f"Splitting fewer would produce a partition that is meaningless rather than small.")

    n_val = max(minimum, round(total * val_fraction))
    n_test = max(minimum, round(total * test_fraction))
    n_train = total - n_val - n_test
    if n_train < minimum:
        raise SplitError(
            f"val_fraction and test_fraction leave {n_train} teachers to train on, below the "
            f"floor of {minimum}. Lower the fractions or grow the corpus.")
    return n_train, n_val, n_test


def plan_splits(
    sessions: Sequence[SessionRecord],
    *,
    config_sha256: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    shift_axis: ShiftAxis = "domain",
    heldout_college: str | None = None,
    heldout_school: str | None = None,
    classroom_teachers_eligible_for_test: bool = True,
    classroom_teachers_may_train: bool = False,
    min_teachers_per_partition: int = 1,
) -> SplitPlan:
    """Partition the corpus by teacher, teacher-disjoint and leakage-checked.

    **Who is eligible for what.**

    A teacher in `heldout_college` is in no partition at all. Their microteaching sessions are
    the S1 condition, and putting them in the test partition would contaminate S0 - which is
    defined as held-out *teachers* from the same institutions, not a held-out institution.

    A teacher with classroom sessions is barred from train and val, because a model that has
    watched this candidate teaching will shift less than one that has not, and the S2 drop is
    the thesis's primary finding. They are placed in the test partition when they also have
    microteaching sessions and `classroom_teachers_eligible_for_test` is set, which makes the
    S0-to-S2 comparison within teacher wherever the corpus allows it. That is a stronger
    design, not a looser one: Phase 6 attributes the drop with a random intercept per teacher,
    and a teacher observed at both levels is what that model is able to use.

    Everyone else is shuffled, interleaved across colleges and allocated by fraction.

    **When the corpus has no microteaching-only teachers.** The rule above assumes a pool of
    teachers recorded only in microteaching, and a corpus of mostly classroom footage has none:
    the assignable pool empties and this refuses, correctly, because there is nothing to train
    on that is not also what shift is measured against. `classroom_teachers_may_train` lets
    those teachers into train and val. It is off by default and must be turned on deliberately,
    because it changes what the S0-to-S2 drop measures - from degradation in a model that never
    saw a classroom, to degradation across teacher-disjoint partitions where some training
    teachers were recorded in one. R2 still holds either way; D73's extra protection does not.
    The choice is written onto the manifest so no reader of the artefact can mistake one for
    the other. D83.

    **Determinism, R7.** Teachers are sorted before they are shuffled, so the result does not
    depend on the order sessions arrived in or on set iteration order. The same corpus, seed
    and fractions produce the same partition.

    Args:
        sessions: every session in the corpus, including classroom ones.
        config_sha256: the hash of the config that produced this split, recorded on the
            manifest so a result can be traced to the partition it was measured on.
        seed: derived from `run.seed`. Same seed, same split.
        val_fraction: share of assignable teachers for validation.
        test_fraction: share of assignable teachers for the S0 test.
        shift_axis: which ladder to build. "domain" gives the original levels; "site" gives the
            held-out school and held-out college ones. Written onto the manifest. D98.
        heldout_college: the college withheld entirely. S1 on the domain axis, S2 on the site
            axis. None disables that level.
        heldout_school: the basic school withheld entirely, which is S1 on the site axis. None
            disables S1. Refused on the domain axis, where it has no rung.
        classroom_teachers_eligible_for_test: whether S2 teachers may hold S0 test sessions.
        classroom_teachers_may_train: whether S2 teachers may also enter train and val. Off by
            default; see the paragraph above for what turning it on costs.
        min_teachers_per_partition: floor below which a partition is refused as meaningless.

    Returns:
        A SplitPlan carrying the manifest and the exclusions that produced it.

    Raises:
        SplitError: if the corpus cannot be partitioned as asked.
        ValueError: from SplitManifest, if R2 is somehow violated.
        ShiftError: from assert_no_leakage, if the result would leak into S1 or S2.
    """
    if not sessions:
        raise SplitError("no sessions to split")

    college_of = _college_of(sessions)
    sessions_of: dict[str, list[SessionRecord]] = defaultdict(list)
    for session in sessions:
        sessions_of[session.teacher_id].append(session)

    if heldout_college is not None and heldout_college not in set(college_of.values()):
        raise SplitError(
            f"heldout_college {heldout_college!r} has no teachers in this corpus. Its level "
            f"would be empty, and an empty shift level reads as no shift found rather than as "
            f"a condition nobody measured.")

    schools = {s.school_id for s in sessions if s.school_id is not None}
    if heldout_school is not None and heldout_school not in schools:
        raise SplitError(
            f"heldout_school {heldout_school!r} has no sessions in this corpus. S1 would be "
            f"empty, and an empty shift level reads as no site shift found rather than as a "
            f"condition nobody measured.")

    if shift_axis == "site":
        unplaced = sorted(s.session_id for s in sessions
                          if s.domain == CLASSROOM and s.school_id is None)
        if unplaced:
            raise SplitError(
                f"classroom sessions {unplaced} name no school, and on the site axis the "
                f"school "
                f"is what decides their level. Left unplaced they would fall into the training "
                f"pool, which is the leak the ladder exists to measure against. This mirrors "
                f"the "
                f"`a_classroom_session_names_its_school` constraint, which should have refused "
                f"them at ingest. D98.")

    heldout_teachers = {t for t, college in college_of.items() if college == heldout_college}
    school_teachers = {s.teacher_id for s in sessions
                       if heldout_school is not None and s.school_id == heldout_school}
    classroom_teachers = {s.teacher_id for s in sessions if s.domain == CLASSROOM}

    # Sorted at every step. A set would iterate in hash order and the seeded shuffle below would
    # then produce a different split between interpreter runs, which is R7 exactly.  On the site
    # axis the held-out school's teachers are barred too, and no flag relaxes it: a teacher who
    # trained is the thing that makes a site look familiar, so S1 measured on them measures
    # nothing. `classroom_teachers` is not barred there - every session is classroom, so barring
    # them would empty the pool, and what replaces that protection is the site rule.
    if shift_axis == "site":
        barred = heldout_teachers | school_teachers
    elif classroom_teachers_may_train:
        barred = heldout_teachers
    else:
        barred = heldout_teachers | classroom_teachers
    assignable = sorted(set(college_of) - barred)

    by_college: dict[str, list[str]] = defaultdict(list)
    for teacher in assignable:
        by_college[college_of[teacher]].append(teacher)

    rng = random.Random(seed)
    groups = []
    for college in sorted(by_college):
        members = by_college[college]
        rng.shuffle(members)
        groups.append(members)

    ordered = _interleave(groups)
    n_train, n_val, n_test = _partition_sizes(
        len(ordered), val_fraction, test_fraction, min_teachers_per_partition)

    train = sorted(ordered[:n_train])
    val = sorted(ordered[n_train:n_train + n_val])
    test = sorted(ordered[n_train + n_val:n_train + n_val + n_test])

    # The classroom teachers who can still contribute an S0 test session: under the strict rule
    # they are excluded from training either way, and excluding them from test as well would
    # discard microteaching data for no gain.
    #
    # `allocated` is what keeps this correct once `classroom_teachers_may_train` is on. Those
    # teachers are then in the assignable pool and may already hold a partition, and adding one
    # to test on top of a train place would put the same teacher in two partitions - R2 broken
    # by the very clause meant to use the corpus better. Under the strict rule the set is empty
    # of classroom teachers and this changes nothing.
    if classroom_teachers_eligible_for_test:
        allocated = set(train) | set(val) | set(test)
        also_microteaching = sorted(
            teacher
            for teacher in classroom_teachers - heldout_teachers - school_teachers - allocated
            if any(s.domain == MICROTEACHING for s in sessions_of[teacher]))
        test = sorted(set(test) | set(also_microteaching))

    if shift_axis == "site":
        s1_sessions, s2_sessions = _levels_by_site(
            sessions, college_of, heldout_school, heldout_college)
    else:
        s1_sessions, s2_sessions = _levels_by_domain(sessions, heldout_teachers)

    # Both levels are declared, not just S2. Until D98 `assert_no_leakage` checked the
    # declaration for S2 alone, so an S1 session could go unlisted with nothing recording that
    # it was held out. Listing a session twice is impossible here - it is one sorted set - and
    # would be harmless.
    manifest = SplitManifest(
        manifest_id=new_ulid(),
        created_at=datetime.now(UTC),
        config_sha256=config_sha256,
        train_teachers=train,
        val_teachers=val,
        test_teachers=test,
        shift_axis=shift_axis,
        heldout_college=heldout_college,
        heldout_school=heldout_school,
        shift_eval_sessions=sorted(set(s1_sessions) | set(s2_sessions)),
        classroom_teachers_trained=classroom_teachers_may_train,
    )

    # What the model actually sees. On the domain axis a train teacher's classroom sessions are
    # S2 and are not trained on, so the filter is right there. On the site axis every session is
    # classroom, so that same filter would make this empty and the "held-out session appears in
    # a training partition" check would silently never fire - the hole being closed, not opened.
    in_training = set(train) | set(val)
    training_sessions = tuple(
        s.session_id for s in sessions
        if s.teacher_id in in_training
        and (shift_axis == "site" or s.domain == MICROTEACHING))

    # The generator's output is put through the consumer's gate before it is returned. Phase 6
    # runs `assert_no_leakage` before computing anything; running it here means a contaminated
    # split fails where it was produced, naming the teacher, rather than weeks later in a
    # harness that can only say the manifest is unusable.
    assert_no_leakage(
        manifest,
        {ShiftLevel.S1: s1_sessions, ShiftLevel.S2: s2_sessions},
        {s.session_id: s.teacher_id for s in sessions},
        training_sessions=training_sessions,
    )

    return SplitPlan(
        manifest=manifest,
        heldout_college_teachers=tuple(sorted(heldout_teachers)),
        heldout_school_teachers=tuple(sorted(school_teachers)),
        classroom_teachers=tuple(sorted(classroom_teachers)),
        s1_sessions=s1_sessions,
        s2_sessions=s2_sessions,
    )


def plan_from_config(sessions: Sequence[SessionRecord], config, *,
                     config_sha256: str) -> SplitPlan:
    """The split described by a loaded config, so the parameters live in one place.

    `run.seed` is the master seed. The split derives from it rather than carrying its own, so
    that one number in one file reproduces the partition alongside everything else R7 covers.
    """
    splits = config.splits
    return plan_splits(
        sessions,
        config_sha256=config_sha256,
        seed=config.run.seed,
        val_fraction=splits.val_fraction,
        test_fraction=splits.test_fraction,
        shift_axis=splits.shift_axis,
        heldout_college=splits.heldout_college,
        heldout_school=splits.heldout_school,
        classroom_teachers_eligible_for_test=splits.classroom_teachers_eligible_for_test,
        classroom_teachers_may_train=splits.classroom_teachers_may_train,
        min_teachers_per_partition=splits.min_teachers_per_partition,
    )
