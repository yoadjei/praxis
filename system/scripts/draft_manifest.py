# -*- coding: utf-8 -*-
"""Draft an ingest manifest from a directory of footage, filling in only what is derivable.

    python scripts/draft_manifest.py --media-dir "C:/.../videos" --out sessions.csv

The output is a draft for the candidate to complete, not a manifest to ingest. Everything this
can establish from the files themselves it fills in; everything that is a fact about the world
it leaves blank and counts. `scripts/ingest_corpus.py --dry-run` then reports exactly which
blanks block ingestion.

**It never invents a teacher.** `teacher_id` is the split key and R2 rests on it, so a file
whose teacher cannot be read off the directory structure gets an empty `teacher_code` rather
than a guess. Guessing would merge two candidates into one or split one across two, and the
first is a leakage bug that no downstream check can see.

**And it never makes a name an identifier.** Directories in a corpus assembled by hand are
named after people. Those names are the re-identifying information D76 keeps out of the
database, so each directory becomes a neutral code and the correspondence goes to the key
file. The `filename` column must keep the real path for the loader to resolve it, which is
why a drafted manifest is gitignored alongside the key file. D87.

**Duplicates and split recordings are resolved here, where they are visible.** Two names for
one recording would be silently deduplicated by content-addressed storage, and a recording a
transfer tool cut into parts would become several sessions carrying one teacher into several
partitions. Both are findings about the directory, so this reports them rather than leaving
them to surprise a batch. D82.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.config import load_config
from praxis.ingest.errors import IngestRefused
from praxis.ingest.probe import MediaMetadata, probe
from praxis.tools import ToolError, resolve

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v")

COLUMNS = ("filename", "teacher_code", "college_code", "domain", "recorded_on",
           "subject", "grade_level", "part_group", "camera_distance_m", "room_area_m2",
           "pupil_count", "ambient_noise_dba", "teacher_movement_range_m")

# A splitter appends ".1", ".2" to one stem. Requiring the dot is what keeps "teaching@4" and
# "teaching@5" - two different sessions that happen to end in consecutive digits - from being
# read as parts of one recording.
PART_PATTERN = re.compile(r"^(?P<stem>.+)\.(?P<index>\d+)$")

# Two parts of one recording are cut to the same length by a size cap. Half a second of slack
# covers container rounding; a genuine difference between sessions is far larger.
PART_DURATION_TOLERANCE_S = 0.5

# Dates in names written by phones and messaging apps: 20260925_1224, 2026-09-25, 2026_09_25.
NAME_DATE = re.compile(r"(?P<year>20\d{2})[-_]?(?P<month>\d{2})[-_]?(?P<day>\d{2})")


@dataclass
class Clip:
    path: Path
    relative: str
    digest: str
    metadata: MediaMetadata | None = None
    unreadable: str = ""
    part_group: str = ""
    duplicate_of: str = ""

    @property
    def folder(self) -> str:
        parts = Path(self.relative).parts
        return parts[0] if len(parts) > 1 else ""

    @property
    def recorded_on(self) -> str:
        """The date the file claims, or blank. Derived, and the operator confirms it."""
        match = NAME_DATE.search(self.path.name)
        if match:
            return f"{match['year']}-{match['month']}-{match['day']}"
        return ""


@dataclass
class Draft:
    rows: list[dict[str, str]] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)
    codes: dict[str, str] = field(default_factory=dict)


def digest_of(path: Path) -> str:
    running = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            running.update(block)
    return running.hexdigest()


def container_date(ffprobe, path: Path) -> str:
    """`creation_time` from the container, which a phone writes and a messaging app strips."""
    import subprocess

    done = subprocess.run(
        ffprobe.command("-v", "error", "-show_entries", "format_tags=creation_time",
                        "-of", "default=nw=1:nk=1", str(path)),
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    stamp = done.stdout.strip()
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ""


def read_clips(media_dir: Path, ffprobe) -> list[Clip]:
    clips: list[Clip] = []
    for path in sorted(p for p in media_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES):
        clip = Clip(path=path, relative=str(path.relative_to(media_dir)),
                    digest=digest_of(path))
        try:
            clip.metadata = probe(ffprobe, path)
        except IngestRefused as refused:
            clip.unreadable = str(refused)
        clips.append(clip)
    return clips


def mark_duplicates(clips: list[Clip]) -> None:
    """The second name for one recording is excluded, and the one in a named folder wins.

    Content-addressed storage would skip it at ingest with no explanation, so the choice is made
    here where it is a visible fact about the directory. The folder copy is preferred because it
    carries a teacher, which the loose copy does not.
    """
    by_digest: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        if clip.metadata is not None:
            by_digest[clip.digest].append(clip)

    for group in by_digest.values():
        if len(group) < 2:
            continue
        keeper = min(group, key=lambda c: (c.folder == "", c.relative))
        for clip in group:
            if clip is not keeper:
                clip.duplicate_of = keeper.relative


def mark_part_groups(clips: list[Clip]) -> list[str]:
    """Find recordings a splitter cut up, and report what was found and on what evidence."""
    candidates: dict[tuple[str, str], list[tuple[int, Clip]]] = defaultdict(list)
    for clip in clips:
        if clip.metadata is None or clip.duplicate_of:
            continue
        match = PART_PATTERN.match(Path(clip.relative).stem)
        if match:
            key = (str(Path(clip.relative).parent), match["stem"])
            candidates[key].append((int(match["index"]), clip))

    notes: list[str] = []
    for (_, stem), members in sorted(candidates.items()):
        if len(members) < 2:
            continue
        members.sort()
        indices = [index for index, _ in members]
        parts = [clip for _, clip in members]

        if indices != list(range(indices[0], indices[0] + len(indices))):
            notes.append(f"{stem!r}: indices {indices} are not consecutive; not grouped")
            continue

        durations = [c.metadata.duration_s for c in parts]
        geometries = {(c.metadata.width, c.metadata.height) for c in parts}
        if max(durations) - min(durations) > PART_DURATION_TOLERANCE_S:
            notes.append(
                f"{stem!r}: durations {min(durations):.1f}s to {max(durations):.1f}s differ by "
                f"more than {PART_DURATION_TOLERANCE_S}s; these look like separate sessions")
            continue
        if len(geometries) > 1:
            notes.append(f"{stem!r}: frame sizes {sorted(geometries)} differ; not grouped")
            continue

        group = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
        for clip in parts:
            clip.part_group = group
        notes.append(
            f"{stem!r}: {len(parts)} parts of {durations[0]:.1f}s each at "
            f"{parts[0].metadata.width}x{parts[0].metadata.height} -> one session {group!r}")
    return notes


def assign_codes(clips: list[Clip]) -> dict[str, str]:
    """A neutral code per directory. The directory names are people; the codes are not. D76."""
    folders = sorted({c.folder for c in clips if c.folder and not c.duplicate_of})
    return {folder: f"T-{index:02d}" for index, folder in enumerate(folders, start=1)}


def build(clips: list[Clip], *, domain: str, college: str, ffprobe) -> Draft:
    draft = Draft(codes=assign_codes(clips))
    seen_groups: set[str] = set()

    for clip in clips:
        if clip.metadata is None:
            draft.excluded.append((clip.relative, f"unreadable: {clip.unreadable}"))
            continue
        if clip.duplicate_of:
            draft.excluded.append(
                (clip.relative, f"byte-identical to {clip.duplicate_of}"))
            continue

        recorded = container_date(ffprobe, clip.path) or clip.recorded_on
        row = dict.fromkeys(COLUMNS, "")
        row.update(filename=clip.relative,
                   teacher_code=draft.codes.get(clip.folder, ""),
                   college_code=college,
                   domain=domain,
                   recorded_on=recorded,
                   part_group=clip.part_group)
        draft.rows.append(row)
        if clip.part_group:
            seen_groups.add(clip.part_group)

    return draft


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_codes(path: Path, codes: dict[str, str]) -> None:
    """Directory name to neutral code. This file re-identifies the corpus; keep it off the
    media volume and out of the repository. D76."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_directory", "teacher_code"])
        writer.writerows(sorted(codes.items()))


def report(draft: Draft, notes: list[str], out: Path, codes_at: Path) -> int:
    sessions = len({r["part_group"] or r["filename"] for r in draft.rows})
    print(f"\n=== drafted {sessions} session(s) from {len(draft.rows)} file(s) ===")

    if notes:
        print("\nsplit recordings:")
        for note in notes:
            print(f"  {note}")

    if draft.excluded:
        print(f"\nexcluded {len(draft.excluded)}:")
        for name, why in draft.excluded:
            print(f"  {name}\n      {why}")

    missing_teacher = [r["filename"] for r in draft.rows if not r["teacher_code"]]
    missing_date = [r["filename"] for r in draft.rows if not r["recorded_on"]]

    print(f"\nteacher_code      {len(draft.rows) - len(missing_teacher)}/{len(draft.rows)} "
          f"derived from directories")
    for name in missing_teacher:
        print(f"  BLANK  {name}")
    print(f"recorded_on       {len(draft.rows) - len(missing_date)}/{len(draft.rows)} "
          f"derived from the container or the filename")
    for name in missing_date:
        print(f"  BLANK  {name}")

    print(f"\nwrote {out}")
    print(f"wrote {codes_at}  <- re-identifies the corpus; keep it off the media volume")
    print("\nNothing here is a fact about the world that the files did not state. Fill the "
          "blanks,\nconfirm the dates, add the covariates, then run:\n"
          f"  python scripts/ingest_corpus.py --manifest {out} --media-dir <dir> --dry-run")
    return 1 if (missing_teacher or missing_date) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("sessions-draft.csv"))
    parser.add_argument("--codes", type=Path, default=Path("keys/source-directories.csv"))
    parser.add_argument("--domain", default="microteaching",
                        choices=("microteaching", "classroom"),
                        help="the shift level every drafted session belongs to")
    parser.add_argument("--college-code", default="",
                        help="written into every row; blank leaves it for the operator")
    args = parser.parse_args(argv)

    if not args.media_dir.is_dir():
        print(f"no such directory: {args.media_dir}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    try:
        ffprobe = resolve("ffprobe", config.tools.ffprobe)
    except ToolError as missing:
        print(f"tools: {missing}", file=sys.stderr)
        return 2

    print(f"reading {args.media_dir}", flush=True)
    clips = read_clips(args.media_dir, ffprobe)
    mark_duplicates(clips)
    notes = mark_part_groups(clips)
    draft = build(clips, domain=args.domain, college=args.college_code, ffprobe=ffprobe)

    write_manifest(args.out, draft.rows)
    write_codes(args.codes, draft.codes)
    return report(draft, notes, args.out, args.codes)


if __name__ == "__main__":
    raise SystemExit(main())
