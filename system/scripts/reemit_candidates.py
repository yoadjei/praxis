# -*- coding: utf-8 -*-
"""Recover the candidate rankings that migration 0009 could not write.

    python scripts/reemit_candidates.py                 # report what would change
    python scripts/reemit_candidates.py --write          # write it

Every `teacher_tracks` row that existed before 0009 carries a `candidates` default saying the
ranking was never recorded, and a reason telling the reader to re-emit it from the pose
artefact. Sessions whose heuristic declined have no row at all, because before 0009 `track_id`
was NOT NULL and a declined proposal had nowhere to go. Both states leave a reviewer unable to
confirm a teacher, which blocks annotation - so this is the step that unblocks the corpus, and
it writes nothing a reviewer could mistake for the original run's measurements.

Read `praxis/preprocess/reemit.py` before changing anything here: the division of what may and
may not be overwritten is argued there, and it is the whole point of the module.

Dry by default. `--write` is the only way a row changes, and each one is reported either way.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select

from praxis.config import load_config
from praxis.db.engine import build_engine
from praxis.db.schema import camera_setups, pose_artifacts, sessions, teacher_tracks
from praxis.preprocess import artifacts as artifacts_module
from praxis.preprocess.reemit import ReemitError, recover
from praxis.preprocess.zones import CameraSetup


def pending(connection) -> list[tuple[str, str, str, str | None]]:
    """Sessions with a pose artefact whose ranking is missing or was never recorded.

    Selected by the state of the data rather than by a list of ids, so a session preprocessed
    tomorrow by a build that still had the old code would be picked up the same way.
    """
    rows = connection.execute(
        select(sessions.c.session_id, pose_artifacts.c.relative_path,
               pose_artifacts.c.sha256, sessions.c.setup_id,
               teacher_tracks.c.candidates)
        .join(pose_artifacts, pose_artifacts.c.session_id == sessions.c.session_id)
        .outerjoin(teacher_tracks, teacher_tracks.c.session_id == sessions.c.session_id)
        .order_by(sessions.c.session_id)).all()

    out = []
    for session_id, relative_path, sha256, setup_id, candidates in rows:
        source = (candidates or {}).get("source")
        if source in (None, "unrecorded"):
            out.append((session_id, relative_path, sha256, setup_id))
    return out


def setup_for(connection, setup_id: str | None) -> CameraSetup | None:
    """The marked zones for a session, or None when it has none.

    None is not an error. Most of this corpus was recorded without a camera setup, and
    `propose` scales its floor for the share of the weight it could measure (see
    `praxis.preprocess.teacher`), so a session without zones is ranked on the two signals that
    exist rather than refused.
    """
    if setup_id is None:
        return None
    row = connection.execute(
        select(camera_setups).where(camera_setups.c.setup_id == setup_id)).first()
    if row is None:
        return None
    return CameraSetup.from_row(row._mapping)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="apply the changes; without it nothing is written")
    parser.add_argument("--actor", default=None,
                        help="user id recorded against the audit event")
    parser.add_argument("--role", default="researcher",
                        help="role recorded against the audit event")
    args = parser.parse_args(argv)

    config = load_config()
    root = Path(config.paths.pose_artifacts)
    teacher_config = config.preprocess.teacher_id
    engine = build_engine()

    with engine.begin() as connection:
        work = pending(connection)

    if not work:
        print("every preprocessed session already has a recorded ranking.")
        return 0

    print(f"{len(work)} session(s) with no recorded ranking:")
    recovered = failed = 0

    for session_id, relative_path, sha256, setup_id in work:
        path = root / relative_path
        try:
            pose = artifacts_module.load(path, sha256)
        except artifacts_module.ArtifactError as failure:
            print(f"  {session_id}  SKIPPED  {failure}")
            failed += 1
            continue

        # One transaction per session: a hash mismatch or an unreadable artefact halfway
        # through leaves the sessions already recovered recovered, rather than rolling back
        # work that was correct.
        try:
            with engine.begin() as connection:
                setup = setup_for(connection, setup_id)
                result = recover(connection, session_id=session_id, pose=pose,
                                 pose_sha256=sha256, teacher_config=teacher_config,
                                 setup=setup, max_candidates=
                                 teacher_config.max_candidates_recorded,
                                 actor_user_id=args.actor, actor_role=args.role)
                proposed = result.proposal.track_id
                score = result.proposal.score
                print(f"  {session_id}  "
                      f"{'row created' if result.row_created else 'ranking filled'}, "
                      f"{result.candidate_count} candidate(s), "
                      f"hull proposal "
                      f"{'none' if proposed is None else f'track {proposed}'}"
                      f"{'' if score is None else f' at {score:.3f}'}"
                      f"{'' if setup is not None else ', no marked zones'}")
                if not args.write:
                    raise _DryRun
                recovered += 1
        except _DryRun:
            pass
        except ReemitError as refused:
            print(f"  {session_id}  REFUSED  {refused}")
            failed += 1

    print()
    if args.write:
        print(f"{recovered} recovered, {failed} left alone.")
    else:
        print(f"nothing was written. {len(work) - failed} session(s) would be recovered; "
              f"pass --write to apply.")
    return 1 if failed and args.write else 0


class _DryRun(Exception):
    """Rolls the transaction back after the row has been built, so a dry run exercises the
    whole write path instead of a separate code path that predicts it. A prediction that does
    not run the insert is a prediction that cannot discover a constraint violation."""


if __name__ == "__main__":
    raise SystemExit(main())
