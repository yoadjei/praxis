# -*- coding: utf-8 -*-
"""Phases 1, 2 and 3: getting the corpus in and prepared.

Phase 1 ingests the sessions a manifest names. Phase 2 reports inter-rater agreement over
whatever
annotations exist. Phase 3 extracts pose, tracks bodies, proposes the teacher, blurs faces and
writes the artefacts the modelling phases read.

These three are the phases that can do real work on the corpus as it stands today. Everything
downstream of them needs labels, and abstains until there are some.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select

from praxis.annotation.codebook import CodebookError, get_codebook
from praxis.annotation.irr import Annotation
from praxis.annotation.irr import report as agreement_report
from praxis.annotation.store import load_annotations
from praxis.config_schema import MediaRootUnavailable
from praxis.db import transaction
from praxis.db.schema import media_objects, pose_artifacts, sessions
from praxis.ingest.corpus import (
    group_parts,
    read_keymap,
    read_manifest,
    resolve_identifiers,
    run_batch,
)
from praxis.phases import PhaseContext, PhaseError, PhaseResult, abstain, register
from praxis.preprocess.pipeline import PreprocessError
from praxis.preprocess.pipeline import run as run_pipeline
from praxis.preprocess.pose import OnnxPoseEstimator, PoseError, WeightsMissing
from praxis.preprocess.store import PersistError
from praxis.preprocess.store import persist as persist_pipeline
from praxis.tools import resolve

NO_DATABASE = "DATABASE_URL or PRAXIS_TEST_DSN"


# ---------------------------------------------------------------------------
# Phase 1: ingest and quality gate
# ---------------------------------------------------------------------------

@register("1", "Ingest and quality gate")
def ingest(context: PhaseContext) -> PhaseResult:
    """Ingest every session the configured manifest names.

    The manifest, the media directory and the key file come from the config rather than from
    arguments, because R7 makes the corpus a property of the config: "which sessions went in" is
    the most consequential input this system has, and a run whose corpus came from a command
    line
    is not reproducible from the file that claims to describe it.
    """
    engine = context.engine()
    if engine is None:
        return abstain("phase 1 writes sessions to the database", NO_DATABASE)

    try:
        media_root = context.config.require_media_root()
    except MediaRootUnavailable as exc:
        return abstain(str(exc), "PRAXIS_MEDIA_ROOT")

    settings = context.config.ingest
    if settings.manifest is None:
        return abstain(
            "phase 1 needs a manifest naming the sessions to ingest. "
            "scripts/draft_manifest.py writes one from a directory of footage.",
            "ingest.manifest")
    manifest: Path = settings.manifest
    if not manifest.is_file():
        return abstain(f"the configured manifest {manifest} does not exist", str(manifest))

    media_dir: Path = settings.media_dir or manifest.parent
    rows, problems = read_manifest(manifest, media_dir)
    if problems:
        # The manifest is present and wrong, which is not an abstention: an abstention says an
        # input is missing and this one is here, with a typo in it that only a person can fix.
        raise PhaseError(
            f"{manifest} has {len(problems)} unusable rows: " + "; ".join(problems[:5])
            + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""))
    if not rows:
        return abstain(f"{manifest} names no sessions", "rows in the manifest")

    groups, grouping = group_parts(rows)
    if grouping:
        raise PhaseError(
            f"{manifest} has inconsistent part groups: " + "; ".join(grouping[:5]))

    if settings.keymap is None or not settings.keymap.is_file():
        return abstain(
            "phase 1 needs the key file mapping teacher and college codes to identifiers. "
            "scripts/ingest_corpus.py --create-missing writes one on a first run.",
            "ingest.keymap")

    mapping = read_keymap(settings.keymap)
    with transaction(engine) as connection:
        # create_missing is false here on purpose. Minting a teacher identity is a decision
        # about who the corpus contains, and R2 rests on getting it right; the script asks for
        # it explicitly and a scheduled phase must not make it silently.
        unresolved = resolve_identifiers(connection, rows, mapping, create_missing=False)
    if unresolved:
        return abstain(
            "codes in the manifest have no identifier in the key file: "
            + ", ".join(sorted(set(unresolved))[:8])
            + ". Run scripts/ingest_corpus.py --create-missing once to mint them.",
            "identifiers for " + ", ".join(sorted(set(unresolved))[:8]))

    ffprobe = resolve("ffprobe", context.config.tools.ffprobe)
    ffmpeg = resolve("ffmpeg", context.config.tools.ffmpeg)
    notes: list[str] = []
    ingested, skipped, failed = run_batch(
        groups, engine=engine, config=context.config, ffprobe=ffprobe, ffmpeg=ffmpeg,
        media_root=media_root, report=notes.append)

    report = context.write("ingest.json", {
        "manifest": str(manifest), "sessions": len(groups), "log": notes})
    summary = {
        "manifest": str(manifest),
        "sessions_in_manifest": len(groups),
        "ingested": ingested,
        "skipped_already_present": skipped,
        "refused": failed,
    }
    if failed:
        # Reported, not raised. A refusal is the gate doing its job - a corrupt file, a consent
        # that does not cover the domain - and a phase that failed on one bad row would make the
        # operator re-run forty good ones.
        summary["note"] = "refusals are in ingest.json; each names its reason"
    return PhaseResult(summary=summary, artefacts=(context.artefact("ingest_log", report),))


# ---------------------------------------------------------------------------
# Phase 2: annotation agreement baseline
# ---------------------------------------------------------------------------

@register("2", "Annotation agreement baseline", needs=("1",))
def agreement(context: PhaseContext) -> PhaseResult:
    """Inter-rater agreement over the annotations recorded so far, against the configured
    gate."""
    engine = context.engine()
    if engine is None:
        return abstain("phase 2 reads annotations from the database", NO_DATABASE)

    with engine.connect() as connection:
        rows = load_annotations(connection)
    if not rows:
        return abstain(
            "phase 2 needs recorded annotations; none exist yet, so there is no agreement to "
            "report and no gate to clear",
            "annotations from a calibration or production round")

    annotations = [
        Annotation(
            clip_id=row["clip_id"], rater_id=row["rater_id"], behaviour=row["behaviour"],
            codebook_version=row["codebook_version"], labels=row["labels"],
            is_nonscorable=row.get("is_nonscorable", False),
            rater_confidence=row.get("rater_confidence", "certain"),
            session_college_id=row.get("session_college_id"))
        for row in rows]

    raters = {annotation.rater_id for annotation in annotations}
    if len(raters) < 2:
        return abstain(
            f"agreement needs at least two raters and {len(raters)} has annotated",
            "a second rater")

    try:
        codebook = get_codebook(context.config.annotation.codebook_version)
    except CodebookError as exc:
        raise PhaseError(f"the configured codebook cannot be loaded: {exc}") from exc

    settings = context.config.annotation
    irr = agreement_report(
        annotations, codebook=codebook, bootstrap=settings.bootstrap_samples,
        ci_level=context.config.calibration.metrics.ci_level, seed=context.seed,
        exclude_guesses=settings.exclude_guesses_from_primary_irr)

    per_behaviour = {
        behaviour.behaviour: _weakest_field(behaviour)
        for behaviour in irr.behaviours}
    measured = {name: alpha for name, alpha in per_behaviour.items() if alpha is not None}
    gate = settings.alpha_gate
    report = context.write("agreement.json", irr.as_table())

    return PhaseResult(
        summary={
            "raters": len(raters),
            "annotations": len(annotations),
            "alpha_gate": gate,
            # The weakest field per behaviour, not the mean. The gate is a floor every field has
            # to clear, and averaging would let a field at 0.4 hide behind one at 0.9.
            "weakest_alpha_per_behaviour": measured,
            "gate_met": bool(measured) and all(a >= gate for a in measured.values()),
            "behaviours_without_a_measurable_alpha": sorted(
                name for name, alpha in per_behaviour.items() if alpha is None),
        },
        artefacts=(context.artefact("agreement", report),))


def _weakest_field(behaviour) -> float | None:
    """The lowest alpha across a behaviour's fields, or None when none could be estimated."""
    points = [field.alpha.point for field in behaviour.fields if field.alpha is not None]
    return min(points) if points else None


# ---------------------------------------------------------------------------
# Phase 3: preprocessing
# ---------------------------------------------------------------------------

@dataclass
class _Outcome:
    """What happened to one session, so the summary can report it rather than lose it."""

    done: list[str]
    refused: list[str]

    def as_json(self) -> dict[str, object]:
        return {"preprocessed": self.done, "refused": self.refused}


@register("3", "Preprocessing: pose, tracking, blur, zones, aggregates", needs=("1",))
def preprocess(context: PhaseContext) -> PhaseResult:
    """Pose, track, propose the teacher, blur faces, and write the artefacts.

    One transaction per session rather than one for the batch. Pose extraction over a thirty
    minute recording takes minutes, and a single transaction around the whole corpus would hold
    locks for the duration and lose every completed session to one bad file at the end.
    """
    engine = context.engine()
    if engine is None:
        return abstain("phase 3 reads ingested sessions from the database", NO_DATABASE)

    try:
        media_root = context.config.require_media_root()
    except MediaRootUnavailable as exc:
        return abstain(str(exc), "PRAXIS_MEDIA_ROOT")

    pose = context.config.preprocess.pose
    weights = Path(context.config.paths.model_weights) / pose.weights_file
    try:
        estimator = OnnxPoseEstimator(
            weights_path=weights,
            min_person_confidence=pose.min_person_confidence,
            min_keypoint_confidence=pose.min_keypoint_confidence,
            max_persons_per_frame=pose.max_persons_per_frame,
            nms_iou_threshold=pose.nms_iou_threshold)
    except WeightsMissing as exc:
        # R6: never fetched here. Vendoring is an operator step and the message names it.
        return abstain(str(exc), "scripts/vendor_weights.py")

    pending = _sessions_needing_preprocessing(engine)
    if not pending:
        with engine.connect() as connection:
            # Counted in the database, not read off `rowcount`. On a SELECT psycopg reports -1
            # until the rows are fetched, and -1 is not 0, so the empty-corpus branch below
            # would never be taken: a run against a corpus with nothing in it would report
            # that every session is already preprocessed. An empty corpus has to abstain, not
            # congratulate itself.
            total = connection.execute(
                select(func.count(sessions.c.session_id))).scalar() or 0
        if total == 0:
            return abstain("phase 3 has no ingested sessions to preprocess",
                           "sessions from phase 1")
        return PhaseResult(summary={
            "sessions_preprocessed": 0,
            "note": "every ingested session is already preprocessed"})

    outcome = _Outcome(done=[], refused=[])
    artefacts = []
    frames: dict[str, int] = {}
    teacher_config = context.config.preprocess.teacher_id

    for session_id, relative_path in pending:
        source = media_root / relative_path
        if not source.is_file():
            outcome.refused.append(f"{session_id}: no file at {relative_path}")
            continue

        blurred = f".blurred/{session_id}.mp4"
        destination = media_root / blurred
        pose_output = Path(context.config.paths.pose_artifacts) / f"{session_id}.npz"
        try:
            result = run_pipeline(
                session_id=session_id, source=source, blurred_destination=destination,
                pose_output=pose_output, estimator=estimator, config=context.config,
                setup=None)
            with transaction(engine) as connection:
                persist_pipeline(
                    connection, session_id=session_id, result=result,
                    relative_path=f"{session_id}.npz",
                    max_candidates=teacher_config.max_candidates_recorded,
                    blurred_relative_path=blurred, actor_role="system")
        except (PreprocessError, PoseError, PersistError, OSError) as exc:
            # Named exceptions, and recorded. A bare `except Exception: pass` here would turn a
            # systematically failing corpus into a phase that reports zero sessions and success.
            outcome.refused.append(f"{session_id}: {type(exc).__name__}: {exc}")
            continue

        outcome.done.append(session_id)
        frames[session_id] = result.frames_processed
        artefacts.append(context.artefact(f"pose_{session_id}", pose_output))

    report = context.write("preprocess.json", {**outcome.as_json(), "frames": frames})
    return PhaseResult(
        summary={
            "sessions_considered": len(pending),
            "sessions_preprocessed": len(outcome.done),
            "sessions_refused": len(outcome.refused),
            "frames_processed": sum(frames.values()),
            # R1: the heuristic proposes which track is the teacher and a person confirms it.
            # Nothing downstream may treat an unconfirmed proposal as a label.
            "teacher_tracks_awaiting_human_confirmation": len(outcome.done),
        },
        artefacts=(*artefacts, context.artefact("preprocess_log", report)))


def _sessions_needing_preprocessing(engine) -> list[tuple[str, str]]:
    """Ingested sessions with no pose artefact yet, oldest first, with their media path."""
    with engine.connect() as connection:
        return [
            (row.session_id, row.relative_path)
            for row in connection.execute(
                select(sessions.c.session_id, media_objects.c.relative_path)
                .join(media_objects,
                      media_objects.c.media_sha256 == sessions.c.media_sha256)
                .where(~sessions.c.session_id.in_(select(pose_artifacts.c.session_id)))
                .order_by(sessions.c.created_at))]
