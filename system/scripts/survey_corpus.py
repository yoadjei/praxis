# -*- coding: utf-8 -*-
"""Survey footage before ingesting it: what is there, what the gate will refuse, and why.

    python scripts/survey_corpus.py --media-dir "C:/.../videos"
    python scripts/survey_corpus.py --media-dir A:/praxis-media/incoming --detect

`scripts/ingest_corpus.py` needs a manifest with a row per session, and writing that manifest
means knowing which files are real sessions. This answers that, and nothing else: it touches no
database, writes no rows, and moves no files. Running it on a directory the candidate has just
downloaded is safe.

**It reuses the gate rather than restating it.** Every threshold comes from
`ingest.quality_gate` through `praxis.ingest.quality.measure`, the same function the ingest
service calls. A
survey with its own copy of the duration floor would drift from the gate and report a file as
ingestable that ingest then refuses - and the operator would trust the survey.

**Duplicates are found by content, not by name.** A corpus assembled from phone shares and
Drive folders contains the same recording under two names, and `media_objects` is content
addressed, so the second copy would be skipped at ingest with no explanation. Better to say so
here, where it is a fact about the directory rather than a surprise during a batch.

**`--detect` is the expensive option and the informative one.** Without it this reads
containers. With it, the vendored pose model runs on sampled frames and reports how many
distinct people are in them, which is what distinguishes a microteaching recording of one
person at a whiteboard from
a classroom session with pupils. The learner-facing behaviours - `b2_facing_proportion` against
the learner-region centroid, `b3_proximity` to the nearest learner - are not measurable on
footage with no learners in frame, and this is where that is discovered.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.config import load_config
from praxis.ingest.errors import IngestRefused
from praxis.ingest.probe import MediaMetadata
from praxis.ingest.quality import QualityReport, measure
from praxis.tools import ToolError, resolve

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
READ_BLOCK = 1 << 20


@dataclass
class Surveyed:
    """One file, as far as it could be read."""

    path: Path
    relative: str
    bytes: int
    digest: str
    metadata: MediaMetadata | None = None
    report: QualityReport | None = None
    refused: str = ""
    people: dict[str, float] = field(default_factory=dict)

    @property
    def readable(self) -> bool:
        return self.metadata is not None

    def as_dict(self) -> dict[str, object]:
        row: dict[str, object] = {"path": self.relative, "bytes": self.bytes,
                                  "sha256": self.digest}
        if self.metadata is not None:
            row |= {"duration_s": round(self.metadata.duration_s, 2),
                    "width": self.metadata.width, "height": self.metadata.height,
                    "fps": round(self.metadata.fps, 3),
                    "has_audio": self.metadata.has_audio,
                    "container": self.metadata.container}
        if self.report is not None:
            row |= {"verdict": self.report.verdict, "reason": self.report.reason(),
                    "checks": self.report.as_detail()}
        if self.refused:
            row["unreadable"] = self.refused
        if self.people:
            row["people"] = self.people
        return row


def digest_of(path: Path) -> str:
    """The same SHA-256 `media_objects` keys on, so a match here means a skip at ingest."""
    running = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK), b""):
            running.update(block)
    return running.hexdigest()


def videos_under(media_dir: Path) -> list[Path]:
    return sorted(p for p in media_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def survey_file(path: Path, media_dir: Path, gate, ffprobe, ffmpeg) -> Surveyed:
    row = Surveyed(path=path, relative=str(path.relative_to(media_dir)),
                   bytes=path.stat().st_size, digest=digest_of(path))
    try:
        row.metadata, row.report = measure(path, gate, ffprobe, ffmpeg)
    except IngestRefused as refused:
        # A container the prober cannot open is a finding, not a crash: a truncated download
        # looks exactly like this and the operator needs the list, not the first one.
        row.refused = str(refused)
    return row


def count_people(rows: list[Surveyed], config, sample_frames: int) -> None:
    """Fill in `people` for every readable row, using the vendored estimator."""
    import cv2
    import numpy as np

    from praxis.preprocess.pose import OnnxPoseEstimator

    pose = config.preprocess.pose
    estimator = OnnxPoseEstimator(
        weights_path=Path(config.paths.model_weights) / pose.weights_file,
        min_person_confidence=pose.min_person_confidence,
        min_keypoint_confidence=pose.min_keypoint_confidence,
        max_persons_per_frame=pose.max_persons_per_frame,
        nms_iou_threshold=pose.nms_iou_threshold)

    for row in rows:
        if not row.readable:
            continue
        capture = cv2.VideoCapture(str(row.path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            capture.release()
            continue
        counts = []
        for index in np.linspace(0, total - 1, min(sample_frames, total)).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                counts.append(len(estimator.detect(frame)))
        capture.release()
        if not counts:
            continue
        counts = np.array(counts)
        row.people = {
            "frames_sampled": int(counts.size),
            "mean": round(float(counts.mean()), 2),
            "max": int(counts.max()),
            "present_fraction": round(float((counts >= 1).mean()), 3),
            # the distinguishing figure: a microteaching recording of one person at a board
            # never reaches two, whatever its mean says
            "frames_with_two_plus": int((counts >= 2).sum()),
        }


def duplicate_groups(rows: list[Surveyed]) -> list[list[str]]:
    by_digest: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_digest[row.digest].append(row.relative)
    return [sorted(paths) for paths in by_digest.values() if len(paths) > 1]


def report_files(rows: list[Surveyed]) -> None:
    print(f"\n{'file':<52} {'MB':>8} {'dur s':>8} {'w x h':>11} {'fps':>6} aud  verdict")
    print("-" * 104)
    for row in rows:
        if not row.readable:
            print(f"{row.relative[:52]:<52} {row.bytes / 1e6:>8.1f} "
                  f"{'unreadable':>8} {'':>11} {'':>6} {'':>3}  UNREADABLE")
            continue
        meta, report = row.metadata, row.report
        print(f"{row.relative[:52]:<52} {row.bytes / 1e6:>8.1f} {meta.duration_s:>8.1f} "
              f"{meta.width:>5}x{meta.height:<5} {meta.fps:>6.2f} "
              f"{'y' if meta.has_audio else 'N':>3}  {report.verdict.upper()}")


def report_refusals(rows: list[Surveyed]) -> None:
    refused = [r for r in rows if not r.readable]
    if refused:
        print(f"\n=== unreadable: {len(refused)} ===")
        for row in refused:
            print(f"  {row.relative}\n      {row.refused}")

    failing = [r for r in rows if r.readable and r.report.verdict == "fail"]
    if failing:
        print(f"\n=== the gate refuses: {len(failing)} ===")
        for row in failing:
            print(f"  {row.relative}\n      {row.report.reason()}")

    abstaining = [r for r in rows if r.readable and r.report.abstentions]
    if abstaining:
        print(f"\n=== checks that abstained (D48: abstention is its own verdict): "
              f"{len(abstaining)} ===")
        for row in abstaining:
            names = ", ".join(c.name for c in row.report.abstentions)
            print(f"  {row.relative}: {names}")


def report_people(rows: list[Surveyed]) -> None:
    measured = [r for r in rows if r.people]
    if not measured:
        return
    print(f"\n=== people in frame ({len(measured)} files) ===")
    print(f"{'file':<52} {'mean':>6} {'max':>4} {'>=2 ppl':>8} {'present':>8}")
    print("-" * 82)
    for row in measured:
        p = row.people
        print(f"{row.relative[:52]:<52} {p['mean']:>6.2f} {p['max']:>4} "
              f"{p['frames_with_two_plus']:>4}/{p['frames_sampled']:<3} "
              f"{p['present_fraction']:>8.2f}")

    alone = [r for r in measured
             if r.people["frames_with_two_plus"] <= r.people["frames_sampled"] // 4]
    if alone:
        print(f"\n  {len(alone)} file(s) show one person in most sampled frames. The "
              f"learner-facing\n  behaviours are not measurable on these: b2_facing_proportion "
              f"is defined against the\n  learner-region centroid and b3_proximity against the "
              f"nearest learner, and neither\n  exists in footage without learners. This is a "
              f"codebook question, not a code one.")
        for row in alone:
            print(f"      {row.relative}")


def report_totals(rows: list[Surveyed], gate) -> None:
    readable = [r for r in rows if r.readable]
    passing = [r for r in readable if r.report.verdict == "pass"]
    warning = [r for r in readable if r.report.verdict == "warn"]
    duplicates = duplicate_groups(rows)

    if duplicates:
        print(f"\n=== the same recording under more than one name: "
              f"{len(duplicates)} group(s) ===")
        print("  content addressed, so ingest skips the second copy silently. "
              "Pick one per row.")
        for group in duplicates:
            print(f"  {group}")

    print("\n=== totals ===")
    print(f"  files found         {len(rows)}")
    print(f"  readable            {len(readable)}")
    print(f"  distinct content    {len({r.digest for r in rows})}")
    print(f"  gate passes         {len(passing)}")
    print(f"  passes with warns   {len(warning)}")
    print(f"  gate refuses        {len(readable) - len(passing) - len(warning)}")
    print(f"  total bytes         {sum(r.bytes for r in rows) / 1e9:.2f} GB")
    print(f"  total duration      {sum(r.metadata.duration_s for r in readable) / 60:.1f} min")
    print(f"\n  thresholds in force: duration >= {gate.min_duration_s}s, "
          f"{gate.min_frame_pixels} px of frame with a shorter edge of "
          f"{gate.min_shorter_edge}, fps >= {gate.min_fps}, "
          f"audio {'required' if gate.require_audio else 'optional'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--media-dir", type=Path, required=True,
                        help="directory to walk; searched recursively")
    parser.add_argument("--detect", action="store_true",
                        help="also count people per frame with the vendored pose model (slow)")
    parser.add_argument("--sample-frames", type=int, default=24,
                        help="frames per video for --detect")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the full survey here as JSON")
    args = parser.parse_args(argv)

    if not args.media_dir.is_dir():
        print(f"no such directory: {args.media_dir}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    gate = config.ingest.quality_gate
    try:
        ffprobe = resolve("ffprobe", config.tools.ffprobe)
        ffmpeg = resolve("ffmpeg", config.tools.ffmpeg)
    except ToolError as missing:
        print(f"tools: {missing}", file=sys.stderr)
        return 2

    paths = videos_under(args.media_dir)
    if not paths:
        print(f"no video files under {args.media_dir}")
        return 1

    print(f"surveying {len(paths)} file(s) under {args.media_dir}", flush=True)
    rows = [survey_file(path, args.media_dir, gate, ffprobe, ffmpeg) for path in paths]

    if args.detect:
        print("counting people on sampled frames", flush=True)
        count_people(rows, config, args.sample_frames)

    report_files(rows)
    report_refusals(rows)
    report_people(rows)
    report_totals(rows, gate)

    if args.json:
        args.json.write_text(
            json.dumps({"media_dir": str(args.media_dir),
                        "files": [r.as_dict() for r in rows]}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {args.json}")

    # Nothing here fails the run: a directory full of unusable footage is a finding the operator
    # needs reported, not an error that stops them reading the rest of it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
