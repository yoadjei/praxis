# -*- coding: utf-8 -*-
"""Ingest a downloaded corpus from a field-log manifest. One session per row.

    python scripts/ingest_corpus.py --manifest sessions.csv --media-dir A:/praxis-media/incoming
    python scripts/ingest_corpus.py --manifest sessions.csv --media-dir ... --dry-run
    python scripts/ingest_corpus.py --manifest sessions.csv --media-dir ... --create-missing

**Why a key file and not a column.** `teachers` carries a ULID, a college and a cohort, and
nothing else - no name, no staff number, no code. That is deliberate: R1 keeps identity out of
the system, and a `teacher_code` column would put the field log's re-identifying key inside the
database the thesis ships. So the mapping from the field log's codes to ULIDs lives in a
separate key file that the candidate holds, listed in `.gitignore`, and belongs on neither the
media volume nor the working volume alongside the data it would re-identify. See D76.

**Consent is resolved, never created.** A consent record is the database's account of a signed
paper form, and a script that minted one from a spreadsheet row would be manufacturing the legal
basis for processing the footage. The loader finds the active consent for that teacher whose
scope covers the session's domain, and refuses the row if there is not exactly one.

**Every row is validated before any file is touched.** A batch that ingests forty sessions and
then stops on a typo in the forty-first leaves the operator to work out what happened; this
reads the whole manifest, reports everything wrong with it, and only then starts writing.

**Idempotent by content.** A file whose SHA-256 is already in `media_objects` is skipped rather
than re-ingested, so an interrupted run is resumed by running it again.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.config import load_config, prepare_environment
from praxis.db import build_engine, transaction
from praxis.ingest.corpus import (
    ManifestError,
    group_parts,
    read_keymap,
    read_manifest,
    resolve_identifiers,
    run_batch,
    write_keymap,
)
from praxis.tools import resolve

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="the field log, one session per row")
    parser.add_argument("--media-dir", type=Path, required=True,
                        help="where the downloaded files are")
    parser.add_argument("--keymap", type=Path, default=Path("keys/pseudonyms.csv"),
                        help="field-log codes to ULIDs; keep this away from the video")
    parser.add_argument("--create-missing", action="store_true",
                        help="mint pseudonyms for teachers and colleges not yet known")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the manifest and resolve identifiers, ingest nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many sessions, for a first batch")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    prepare_environment(config)

    try:
        rows, problems = read_manifest(args.manifest, args.media_dir)
        mapping = read_keymap(args.keymap)
    except ManifestError as bad:
        print(f"manifest: {bad}", file=sys.stderr)
        return 2

    engine = build_engine()
    media_root = config.require_media_root()
    ffprobe = resolve("ffprobe", config.tools.ffprobe)
    ffmpeg = resolve("ffmpeg", config.tools.ffmpeg)

    with transaction(engine) as connection:
        problems.extend(resolve_identifiers(connection, rows, mapping,
                                            create_missing=args.create_missing))

    usable = [row for row in rows if row.consent_id]
    groups, grouping_problems = group_parts(usable)
    problems.extend(grouping_problems)

    joined = [g for g in groups if len(g) > 1]
    print(
        f"manifest   {len(rows)} rows, {len(usable)} resolved, {len(groups)} sessions, "
        f"{len(problems)} problems"
    )
    if joined:
        print(f"           {len(joined)} session(s) arrive in parts and will be rejoined:")
        for group in joined:
            print(f"             {group[0].part_group}: {len(group)} files")
    for problem in problems:
        print(f"  refused  {problem}")

    gaps = [row for row in usable if row.missing_covariates]
    if gaps:
        print(f"\ncovariates {len(gaps)} sessions are missing at least one. These ingest and "
              f"preprocess normally, and Phase 6's attribution cannot use them:")
        for row in gaps:
            print(f"  {row.filename}: missing {', '.join(row.missing_covariates)}")

    # written here, before the dry-run return, and not after the batch. `--create-missing` has
    # already committed any identifier it minted by this point, so a dry run that skipped this
    # would leave the manifest's codes with no recorded correspondence to the ULIDs now in the
    # database - and the next run, unable to match them, would mint a second teacher for the
    # same person. That is how a rehearsal silently doubles the corpus's teacher count, and a
    # split over doubled teachers puts one person's sessions in two partitions, which is R2
    # broken by a rehearsal nobody thought wrote anything.
    # every row, not just the usable ones. A row refused for want of consent still had its
    # teacher and college resolved, and on the bootstrap run - mint identifiers, import
    # consents, then ingest - every row is refused for exactly that reason. Passing `usable`
    # here wrote a header and nothing else, losing the six identifiers the same command had
    # just committed.
    write_keymap(args.keymap, mapping, rows)

    if args.dry_run:
        print(f"\ndry run: nothing was ingested. Key file {args.keymap} was written, because "
              f"the identifiers it names are already committed.")
        return 1 if problems else 0

    ingested, skipped, failed = run_batch(
        groups, engine=engine, config=config, ffprobe=ffprobe, ffmpeg=ffmpeg,
        media_root=media_root, limit=args.limit)
    print(f"\ningested {ingested}, skipped {skipped}, failed {failed}, "
          f"refused {len(problems)}")
    print(f"key file {args.keymap} - this is what re-identifies the corpus. Keep it off the "
          f"media volume and out of any backup the video is in.")
    return 0 if not (failed or problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
