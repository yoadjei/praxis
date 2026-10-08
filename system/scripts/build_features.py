# -*- coding: utf-8 -*-
"""Stage A: build the feature cache for every annotatable clip.

    python scripts/build_features.py                 # what it would do, and why not
    python scripts/build_features.py --write         # build it
    python scripts/build_features.py --session <id> --write

Phase 4 cannot reach its training code until this has run: the behaviour model trains on cached
backbone activations, not on pixels, because D2 freezes the backbone and 64 frames of ResNet-50
do not fit in the headroom otherwise.

**Only sessions with a confirmed teacher track are built.** A clip cropped to the heuristic's
guess can show somebody who is not the teacher, and every feature cut from it would then be
attributed to them (R1). An excluded session is skipped and says so, rather than being absent
from the output with no explanation (D96).

Dry by default. `--write` is the only thing that puts a file on disk.

Slow by design on CPU: measured at roughly 28 minutes per 40-minute session. Above
`stage_a_kaggle_threshold_sessions` the configuration says to run this on Kaggle and import the
checksummed cache instead. See D13.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select

from praxis.annotation.clips import plan_from_config as clip_plan_from_config
from praxis.behaviour.backbone import build_backbone
from praxis.behaviour.dataset import policy_from_config
from praxis.behaviour.features import (
    FeatureCacheError,
    build_clip,
    cache_digest,
    is_cached,
    write_clip,
)
from praxis.config import load_config
from praxis.db.engine import build_engine
from praxis.db.schema import media_objects, pose_artifacts, sessions
from praxis.preprocess.artifacts import load as load_pose
from praxis.preprocess.store import confirmed_teacher_track
from praxis.tools import resolve


def buildable(connection, only: str | None) -> list[dict]:
    """Every session Stage A can legitimately build, with the reason for each one it cannot.

    The refusals are returned rather than filtered away. A session silently absent from a cache
    is the same failure D96 named for the annotation plan: nobody can tell a decision from a
    gap.
    """
    rows = connection.execute(
        select(sessions.c.session_id, sessions.c.excluded_at, sessions.c.exclusion_reason,
               media_objects.c.blurred_relative_path, media_objects.c.duration_s,
               pose_artifacts.c.relative_path.label("pose_path"),
               pose_artifacts.c.sha256.label("pose_sha256"))
        .join(media_objects, media_objects.c.media_sha256 == sessions.c.media_sha256)
        .outerjoin(pose_artifacts, pose_artifacts.c.session_id == sessions.c.session_id)
        .order_by(sessions.c.session_id)).all()

    out = []
    for row in rows:
        if only and row.session_id != only:
            continue
        reason = None
        if row.excluded_at is not None:
            reason = f"excluded from annotation: {row.exclusion_reason}"
        elif row.pose_path is None:
            reason = "not preprocessed, so there is no pose artefact to crop from"
        elif row.blurred_relative_path is None:
            reason = "no blurred video; the original is deleted and D18 forbids using it"
        else:
            track = confirmed_teacher_track(connection, row.session_id)
            if track is None:
                reason = ("no confirmed teacher track; a clip cut on the heuristic's guess "
                          "could show somebody who is not the teacher (R1)")
            else:
                out.append({"session_id": row.session_id, "track_id": track,
                            "video": row.blurred_relative_path, "pose": row.pose_path,
                            "pose_sha256": row.pose_sha256,
                            "duration_s": float(row.duration_s), "reason": None})
                continue
        out.append({"session_id": row.session_id, "reason": reason})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", help="build one session rather than all of them")
    parser.add_argument("--write", action="store_true", help="build it; without this, nothing")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild clips already cached, rather than skipping them")
    args = parser.parse_args(argv)

    config = load_config()
    media_root = config.require_media_root()
    cache_root = Path(config.paths.feature_cache)
    pose_root = Path(config.paths.pose_artifacts)
    clip = config.behaviour.clip

    # 1.4x around the box, as `behaviour.clip.crop_padding` states it, expressed as the fraction
    # each side extends by - which is what `box_at` takes. The thumbnail caller passes its own,
    # and the two are deliberately different numbers for different purposes.
    crop_pad = (clip.crop_padding - 1.0) / 2.0
    settings = cache_digest(config)
    variants = tuple(variant.name for variant in policy_from_config(config).variants)

    engine = build_engine()
    with engine.begin() as connection:
        work = buildable(connection, args.session)

    buildable_now = [row for row in work if row["reason"] is None]
    for row in work:
        if row["reason"] is not None:
            print(f"  skip   {row['session_id']}: {row['reason']}")

    if not buildable_now:
        print("\nnothing to build. Confirm a teacher track first: "
              "http://127.0.0.1:8011 shows every session awaiting one.")
        return 0

    backbone = build_backbone(config) if args.write else None
    ffmpeg = resolve("ffmpeg", config.tools.ffmpeg) if args.write else None
    built = skipped = refused = 0

    for row in buildable_now:
        pose = load_pose(pose_root / row["pose"], expected_sha256=row["pose_sha256"])
        plan = clip_plan_from_config(row["session_id"], row["duration_s"], config)
        print(f"\n  {row['session_id']}  track {row['track_id']}  "
              f"{len(plan)} clips of {clip.length_s:g}s")

        for reference in plan:
            if not args.rebuild and is_cached(cache_root, reference.clip_id):
                skipped += 1
                continue
            if not args.write:
                built += 1
                continue
            try:
                write_clip(cache_root, build_clip(
                    reference, pose=pose, video=media_root / row["video"],
                    backbone=backbone, ffmpeg=ffmpeg, track_id=row["track_id"],
                    frames=clip.frames, crop_size=clip.crop_size, crop_pad=crop_pad,
                    variants=variants, batch_frames=config.behaviour.backbone.
                    stage_a_batch_frames, dtype=config.behaviour.backbone.cache_dtype,
                    config_sha256=settings))
                built += 1
            except FeatureCacheError as declined:
                # One unusable clip does not end a session. A clip the teacher walked out of is
                # an ordinary thing and the next one is still worth building.
                print(f"    refused {reference.clip_id}: {declined}")
                refused += 1

    verb = "built" if args.write else "would build"
    print(f"\n{verb} {built} clip(s), skipped {skipped} already cached, refused {refused}.")
    if not args.write:
        print("nothing was written. Pass --write to build the cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
