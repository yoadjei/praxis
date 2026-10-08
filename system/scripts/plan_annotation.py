# -*- coding: utf-8 -*-
"""Build annotation assignments from a corpus of clips.

This script bridges the gap between preprocessed sessions and raters. It cuts each
annotatable session into clips, samples them for a calibration round if requested,
builds a deterministic assignment roster, and records it in the database.

The roster is fully reproducible from this script's arguments and the config. This
script writes no audit event: praxis/audit/events.py has no event type for
assignment creation, adding one would widen a contract for something fully
reproducible from recorded inputs, and the labels themselves are audited as
annotation.created. The manifest file records every parameter that shaped the
sample, so the roster is derivable from it.

    python scripts/plan_annotation.py --raters R1,R2,R3 --round calibration-1
    python scripts/plan_annotation.py --raters R1,R2,R3 --round calibration-1 --dry-run
    python scripts/plan_annotation.py --raters R1,R2 --round production --all-clips
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.annotation.clips import clip_plan
from praxis.annotation.server import build_assignments
from praxis.annotation.store import record_assignments
from praxis.config import load_config
from praxis.db import build_engine, transaction
from praxis.db.schema import annotation_assignments, media_objects, sessions, teacher_tracks
from praxis.vocabulary import BEHAVIOUR_IDS


class PlanError(RuntimeError):
    """Assignment planning cannot proceed."""


class _DryRun(Exception):
    """Raised to abandon the transaction, which is how --dry-run writes nothing."""


def _load_annotatable_sessions(
    conn: Any,
) -> dict[str, dict[str, Any]]:
    """Load sessions that are ready for annotation.

    A session is annotatable when:
    - media_objects.is_blurred is true AND blurred_relative_path is not null:
      a clip nobody can be shown is not work. The original is gone and the only
      video that may be shown is the blurred one (D18).
    - sessions.quality_verdict is not "fail": the quality gate passed. The vocabulary is
      pass, warn and fail - QualityVerdict in praxis/contracts/session.py, with a CHECK
      constraint on the column enforcing it. A warn ingests and annotates normally; the
      warning is about the footage, not about whether it may be labelled.
    - sessions.excluded_at is null: a researcher has not decided against using it. An
      exclusion is a judgement about research use with a stated ground, kept apart from
      quality_verdict, which is what the gate measured about the file. Reported as excluded
      rather than as waiting, because "decided against" and "not got to yet" are different
      states and reporting them alike is how a decision comes to look like a backlog. D96.
    - teacher_tracks.confirmed_by is not null: a person has said which track is the teacher.
      docs/API.md section 4 states this as a rule - "confirmation is mandatory before a session
      enters annotation or inference" - and until now nothing enforced it anywhere.
      `confirmed_teacher_track`, whose docstring calls itself what every downstream phase must
      call, had no caller in the entire system outside its own tests.

      **Why it matters here specifically.** A clip is cut from a session and shown to a rater to
      be labelled for one teacher's behaviour. If nobody has identified which tracked person the
      teacher is, the clip can show somebody else, and the label is then attached to the wrong
      person's conduct. R1 says only the teacher is classified; an assignment planned before
      confirmation is how that is violated without any code appearing to break.

    Sessions still awaiting preprocessing or confirmation are reported as waiting, not silently
    dropped.

    Returns:
        A dict mapping session_id to a dict with:
        - teacher_id: the teacher
        - duration_s: the session length in seconds
        - media_sha256: the blurred media hash
    """
    query = select(
        sessions.c.session_id,
        sessions.c.teacher_id,
        media_objects.c.duration_s,
        media_objects.c.media_sha256,
        media_objects.c.is_blurred,
        media_objects.c.blurred_relative_path,
        sessions.c.quality_verdict,
        sessions.c.excluded_at,
        sessions.c.exclusion_reason,
        teacher_tracks.c.confirmed_by,
        teacher_tracks.c.track_id,
    ).join(
        media_objects,
        sessions.c.media_sha256 == media_objects.c.media_sha256,
    ).join(
        # Outer, so a session with no row at all is reported as waiting rather than vanishing
        # from the count. Every preprocessed session has one since 0009, proposal or not; a
        # session with none has not been preprocessed.
        teacher_tracks,
        teacher_tracks.c.session_id == sessions.c.session_id,
        isouter=True,
    )

    result = conn.execute(query)
    sessions_dict = {}
    problems = []
    decisions = []

    for row in result:
        if row.quality_verdict == "fail":
            continue

        # Reported, not dropped, and reported separately from the waiting list. A reader
        # counting annotatable sessions needs to know the difference between a session nobody
        # has reached and one somebody decided against, because only the first is work.
        if row.excluded_at is not None:
            decisions.append(
                f"session {row.session_id}: excluded from annotation on "
                f"{row.excluded_at:%Y-%m-%d} - {row.exclusion_reason}")
            continue

        if not row.is_blurred or not row.blurred_relative_path:
            problems.append(
                f"session {row.session_id}: awaiting preprocessing "
                f"(is_blurred={row.is_blurred}, "
                f"blurred_relative_path={row.blurred_relative_path})")
            continue

        if row.confirmed_by is None:
            waiting = ("the heuristic proposed nothing and nobody has identified a teacher"
                       if row.track_id is None else
                       f"the heuristic proposed track {row.track_id} and nobody has confirmed "
                       f"it")
            problems.append(
                f"session {row.session_id}: awaiting teacher confirmation ({waiting}). Clips "
                f"cut now could show somebody who is not the teacher, and the label would be "
                f"attached to them. R1.")
            continue

        sessions_dict[row.session_id] = {
            "teacher_id": row.teacher_id,
            "duration_s": row.duration_s,
            "media_sha256": row.media_sha256,
        }

    # Warn about waiting sessions but do not fail.
    for problem in problems:
        warnings.warn(problem, stacklevel=2)

    # Excluded sessions are surfaced the same way, because a plan that silently omitted them
    # would make the corpus look smaller than it is for no stated reason.
    for decision in decisions:
        warnings.warn(decision, stacklevel=2)

    return sessions_dict


def _cut_clips(
    sessions_dict: dict[str, dict[str, Any]],
    clip_seconds: float,
    min_tail_seconds: float = 4.0,
) -> dict[str, tuple[Any, ...]]:
    """Cut each session into clips using clip_plan.

    Returns:
        A dict mapping session_id to a tuple of ClipRef objects.
    """
    clips_by_session = {}
    for session_id, info in sessions_dict.items():
        clips = clip_plan(
            session_id,
            info["duration_s"],
            clip_seconds=clip_seconds,
            min_tail_seconds=min_tail_seconds,
        )
        if clips:
            clips_by_session[session_id] = clips

    return clips_by_session


def _sample_clips_for_calibration(
    clips_by_session: dict[str, tuple[Any, ...]],
    session_to_teacher: dict[str, str],
    target_clips: int,
    min_teachers: int,
    seed: int,
) -> tuple[Any, ...]:
    """Sample clips with round-robin spread across sessions.

    The sample must span at least `min_teachers` distinct teachers or the
    script refuses and says how many it found. Agreement measured on one
    teacher is not agreement about the codebook.

    The sample spreads clips round-robin across sessions in a seeded order,
    taking clips from each session in turn. This avoids clustering: clips
    from one session share that session's prevalence, so sampling all clips
    from session one before touching session two loses statistical power.

    Args:
        clips_by_session: dict mapping session_id to ClipRef tuples
        session_to_teacher: dict mapping session_id to teacher_id
        target_clips: number of clips to sample
        min_teachers: minimum distinct teachers the sample must cover
        seed: random seed for determinism

    Returns:
        A tuple of ClipRef objects, deterministic from the seed.

    Raises:
        PlanError if the sample cannot meet the teacher coverage requirement.
    """
    random.seed(seed)

    # Sort sessions by ID for determinism, then shuffle with seeded random.
    session_ids = sorted(clips_by_session.keys())
    random.shuffle(session_ids)

    # Collect clips in round-robin fashion.
    selected: list[Any] = []
    session_queues = {
        sid: list(clips_by_session[sid]) for sid in session_ids
    }
    session_order = list(session_ids)
    current_session_idx = 0

    while len(selected) < target_clips and any(
        session_queues.values()
    ):
        # Find next session with clips remaining.
        attempts = 0
        while (
            current_session_idx < len(session_order)
            and not session_queues[session_order[current_session_idx]]
        ):
            attempts += 1
            current_session_idx += 1
            if attempts > len(session_order):
                break

        if current_session_idx >= len(session_order):
            # Wrapped around; start again.
            current_session_idx = 0
            attempts = 0
            while (
                current_session_idx < len(session_order)
                and not session_queues[session_order[current_session_idx]]
            ):
                attempts += 1
                current_session_idx += 1
                if attempts > len(session_order):
                    break

        if current_session_idx >= len(session_order):
            # No more clips.
            break

        # Take one clip from the current session.
        session_id = session_order[current_session_idx]
        if session_queues[session_id]:
            selected.append(session_queues[session_id].pop(0))
            current_session_idx = (current_session_idx + 1) % len(
                session_order
            )

    # Verify teacher coverage.
    teacher_ids = {
        session_to_teacher[clip.session_id]
        for clip in selected
        if clip.session_id in session_to_teacher
    }

    if len(teacher_ids) < min_teachers:
        raise PlanError(
            f"sample spans only {len(teacher_ids)} distinct teachers, "
            f"but calibration requires at least {min_teachers} to measure "
            f"agreement about the codebook, not just one teacher"
        )

    return tuple(selected)


def _find_existing_assignments(
    conn: Any,
    rater_ids: tuple[str, ...],
    session_ids: set[str],
) -> dict[tuple[str, str, int, str], str]:
    """Query existing assignments to skip duplicates.

    Returns:
        A dict mapping (rater_id, session_id, clip_index, behaviour) to
        assignment_id.
    """
    query = select(
        annotation_assignments.c.rater_id,
        annotation_assignments.c.session_id,
        annotation_assignments.c.clip_index,
        annotation_assignments.c.behaviour,
        annotation_assignments.c.assignment_id,
    ).where(
        annotation_assignments.c.rater_id.in_(rater_ids),
        annotation_assignments.c.session_id.in_(session_ids),
    )

    result = conn.execute(query)
    existing = {}
    for row in result:
        key = (row.rater_id, row.session_id, row.clip_index, row.behaviour)
        existing[key] = row.assignment_id

    return existing


def _is_calibration_round(round_name: str) -> bool:
    """Check if a round name indicates a calibration round."""
    return round_name.lower().startswith("calibration")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--raters",
        required=True,
        help="comma-separated rater IDs (e.g., R1,R2,R3)",
    )
    parser.add_argument(
        "--round",
        required=True,
        help="round name (e.g., calibration-1, production)",
    )
    parser.add_argument(
        "--raters-per-clip",
        type=int,
        default=2,
        help="raters per clip per behaviour (default: 2)",
    )
    parser.add_argument(
        "--all-clips",
        action="store_true",
        help="take all clips; skip sampling for calibration rounds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be created and write nothing",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"error: could not load config: {exc}", file=sys.stderr)
        return 2

    rater_ids = tuple(r.strip() for r in args.raters.split(","))
    if not rater_ids or not all(rater_ids):
        print("error: --raters must be a non-empty comma-separated list",
              file=sys.stderr)
        return 2

    # Pad rater IDs to CHAR(26) to match database storage.
    rater_ids = tuple(rid.ljust(26) for rid in rater_ids)

    is_calibration = _is_calibration_round(args.round)
    clip_seconds = config.behaviour.clip.length_s
    # A partial tail is dropped rather than kept as a short clip. This was 4.0 with a TODO
    # asking for a config key; the key was never the fix. See `clips.plan_from_config`: Stage A
    # samples a fixed frame count across whatever span a clip has, so a short clip reaches the
    # model at a different frame rate from the one the config declares. D99.
    min_tail_seconds = clip_seconds

    engine = build_engine()
    already_exist = 0

    try:
        with transaction(engine) as conn:
            # Load annotatable sessions.
            sessions_dict = _load_annotatable_sessions(conn)
            if not sessions_dict:
                print(
                    "error: no annotatable sessions found "
                    "(need is_blurred=true, blurred_relative_path != null, "
                    "quality_verdict != 'fail', and a confirmed teacher track). "
                    "The warnings above say which condition each session is waiting on; "
                    "confirm a teacher with POST /api/v1/sessions/{id}/tracks/confirm.",
                    file=sys.stderr,
                )
                return 1

            # Cut clips.
            clips_by_session = _cut_clips(
                sessions_dict, clip_seconds, min_tail_seconds
            )
            if not clips_by_session:
                print("error: no clips generated from annotatable sessions",
                      file=sys.stderr)
                return 1

            # Flatten all clips for assignment building.
            all_clips = []
            for clips_tuple in clips_by_session.values():
                all_clips.extend(clips_tuple)

            # Sample if calibration and not --all-clips.
            if is_calibration and not args.all_clips:
                target_clips = config.annotation.calibration_clips_per_round
                min_teachers_required = config.annotation.calibration_min_teachers

                # Build session-to-teacher mapping.
                session_to_teacher = {
                    sid: info["teacher_id"]
                    for sid, info in sessions_dict.items()
                }

                try:
                    sampled_clips = _sample_clips_for_calibration(
                        clips_by_session,
                        session_to_teacher,
                        target_clips,
                        min_teachers_required,
                        config.run.seed,
                    )
                except PlanError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
            else:
                sampled_clips = tuple(all_clips)

            # Build assignments.
            try:
                assignments = build_assignments(
                    sampled_clips,
                    rater_ids,
                    behaviours=BEHAVIOUR_IDS,
                    round_name=args.round,
                    raters_per_clip=args.raters_per_clip,
                    seed=config.run.seed,
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            # Find what already exists and filter.
            session_ids = {clip.session_id for clip in sampled_clips}
            existing = _find_existing_assignments(conn, rater_ids, session_ids)

            new_assignments = []
            for assignment in assignments:
                key = (
                    assignment.rater_id,
                    assignment.clip.session_id,
                    assignment.clip.clip_index,
                    assignment.behaviour,
                )
                if key not in existing:
                    new_assignments.append(assignment)
                else:
                    already_exist += 1

            # Record new assignments.
            created = (
                record_assignments(conn, tuple(new_assignments))
                if new_assignments else 0
            )

            # Write manifest.
            manifest = {
                "round": args.round,
                "raters": [r.strip() for r in rater_ids],
                "raters_per_clip": args.raters_per_clip,
                "seed": config.run.seed,
                "clip_seconds": clip_seconds,
                "min_tail_seconds": min_tail_seconds,
                "behaviours": list(BEHAVIOUR_IDS),
                # Clips the corpus offered, per session, against the clips actually drawn. Both
                # are recorded because a calibration round samples: "243 clips existed in that
                # session" and "4 of them were chosen" answer different questions, and only the
                # second describes the round.
                "clips_available_per_session": {
                    sid: len(clips_by_session.get(sid, []))
                    for sid in sorted(clips_by_session.keys())
                },
                "clips_sampled_per_session": {
                    sid: sum(1 for clip in sampled_clips if clip.session_id == sid)
                    for sid in sorted(session_ids)
                },
                # The sample itself, by clip id, sorted. Without it the manifest names the seed
                # that produced a selection but never the selection, so reproducing the round
                # means re-running the sampler against a corpus that must not have changed in
                # the meantime - and the corpus grows every time a session finishes
                # preprocessing. The docstring claims the roster is derivable from this file;
                # these ids are what make that true.
                "sampled_clip_ids": sorted(clip.clip_id for clip in sampled_clips),
                "total_clips_available": sum(
                    len(clips) for clips in clips_by_session.values()
                ),
                "total_clips_sampled": len(sampled_clips),
                "assignments_created": created,
                "assignments_already_existed": already_exist,
                "created_at": datetime.now(UTC).isoformat(),
            }

            manifest_path = (
                config.paths.run_outputs
                / f"annotation-plan-{args.round}.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Print results.
            if args.dry_run:
                print(
                    f"dry run: would create {created} new assignments "
                    f"({already_exist} already existed)"
                )
                print(f"manifest would be written to: {manifest_path}")
                raise _DryRun
            else:
                print(
                    f"created {created} new assignments "
                    f"({already_exist} already existed)"
                )
                print(f"manifest: {manifest_path}")

    except _DryRun:
        # Dry run: transaction rolls back.
        pass
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
