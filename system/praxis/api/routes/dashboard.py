# -*- coding: utf-8 -*-
"""GET /api/v1/dashboard/* - monitoring dashboard endpoints.

The dashboard surfaces the system state: corpus counts by domain and quality,
preprocessing and annotation progress, model performance (when available), and
recent phase runs. Every endpoint works on an empty corpus, returning zeros and
empty lists rather than a 500.

Accuracy is always reported with calibration (R4), and confidence distributions
carry the full state (R3). No teacher identity or filename appears anywhere (R1
and D76).

Query status and phase outputs are read fresh each request. Transient states
like "phase is running now" are captured at query time, not cached.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import func, select

from praxis.db.schema import (
    annotation_assignments,
    annotations,
    audit_log,
    media_objects,
    pose_artifacts,
    sessions,
)

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


def _is_annotated(session_column):
    """Whether any annotation exists for a session, as a correlated EXISTS.

    Annotations carry a clip id rather than a session id, so the link runs through
    `annotation_assignments`. An EXISTS rather than a join because a session with forty
    annotations must appear once in the list, not forty times.
    """
    return (
        select(annotations.c.annotation_id)
        .join(annotation_assignments,
              annotation_assignments.c.assignment_id == annotations.c.assignment_id)
        .where(annotation_assignments.c.session_id == session_column)
        .exists())


def _read_run_manifest(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Read a run's manifest and result.json, or None if either is missing."""
    manifest_file = run_dir / "manifest.json"
    result_file = run_dir / "result.json"
    if not (manifest_file.is_file() and result_file.is_file()):
        return None
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        result = json.loads(result_file.read_text(encoding="utf-8"))
        return (manifest, result)
    except (OSError, json.JSONDecodeError):
        return None


@router.get("/dashboard/summary")
def get_summary(request: Request) -> dict[str, Any]:
    """The monitoring snapshot: corpus, preprocessing, annotations, model, audit trail.

    Returns corpus counts grouped by domain and quality verdict, preprocessing progress,
    annotation assignment and completion counts, audit chain head, and model performance.
    When a model exists, accuracy is paired with calibration (R4) and confidence
    distribution is reported (R3).

    Every field is present even when empty. No teacher identity or filename appears.
    """
    engine = request.app.state.engine

    summary: dict[str, Any] = {}

    # Corpus by domain and quality. Query fresh rather than from phase 9, so the
    # dashboard is always current. Phase 9 is for reproducibility snapshots.
    with engine.begin() as conn:
        domain_quality = conn.execute(
            select(
                sessions.c.domain,
                sessions.c.quality_verdict,
                func.count(sessions.c.session_id).label("count")
            ).group_by(sessions.c.domain, sessions.c.quality_verdict)
        ).all()

        corpus_by_domain: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        for domain, verdict, count in domain_quality:
            corpus_by_domain[domain][verdict] = count

        summary["corpus_by_domain"] = dict(corpus_by_domain)

        # Preprocessing progress: sessions with pose artifacts.
        preprocessed = conn.execute(
            select(func.count(pose_artifacts.c.session_id))
        ).scalar() or 0
        total_sessions = conn.execute(
            select(func.count(sessions.c.session_id))
        ).scalar() or 0
        summary["preprocessing"] = {
            "preprocessed_count": int(preprocessed),
            "total_sessions": int(total_sessions),
        }

        # Annotation progress. Assignments are counted the way `queue_for` serves them, which
        # means an excluded session's rows are left out of both numbers. The table is
        # append-only, so a round planned before an exclusion keeps its assignments; counting
        # them here would report work waiting that no rater will ever be offered. The withheld
        # count is reported beside the live one rather than folded away, because a reader who
        # remembers a larger figure needs to see where it went (D96).
        annotatable = (
            select(sessions.c.session_id)
            .where(sessions.c.session_id == annotation_assignments.c.session_id,
                   sessions.c.excluded_at.is_(None))
            .exists())
        live_assignments = conn.execute(
            select(func.count(annotation_assignments.c.assignment_id)).where(annotatable)
        ).scalar() or 0
        all_assignments = conn.execute(
            select(func.count(annotation_assignments.c.assignment_id))
        ).scalar() or 0
        # Annotations stay a plain count. Excluding a session does not undo labelling somebody
        # already did, and this number means work completed rather than work waiting.
        total_annotations = conn.execute(
            select(func.count(annotations.c.annotation_id))
        ).scalar() or 0
        summary["annotation"] = {
            "assignments": int(live_assignments),
            "annotations": int(total_annotations),
            "withheld_assignments": int(all_assignments - live_assignments),
        }

        # Audit log: latest entry as a timestamp.
        latest_audit = conn.execute(
            select(audit_log.c.occurred_at).order_by(
                audit_log.c.audit_id.desc()).limit(1)
        ).scalar()
        summary["last_audit_at"] = (latest_audit.isoformat()
                                     if latest_audit else None)

    # Model performance. When a model exists, returned with calibration and
    # confidence distribution. For now, both report absence.
    summary["model"] = {"status": "absent"}
    summary["confidence_distribution"] = {"status": "awaiting model"}

    return summary


@router.get("/dashboard/sessions")
def get_sessions(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(25, ge=1, le=100,
                       description="Maximum sessions to return"),
) -> dict[str, Any]:
    """Paginated session list.

    Returns session ID, domain, recorded_on date, quality verdict, duration in seconds,
    preprocessing status, annotation status, and whether a researcher has excluded the session
    from annotation and on what ground. No teacher identity or media filename is included
    (R1 and D76). Ordered by recorded_on descending (newest first).

    The exclusion is here because leaving it out is the failure D96 exists to prevent. Without
    it an excluded session looks, on this screen, exactly like one nobody has reached: the next
    reviewer opens it, tries to confirm a teacher track, and learns nothing about why it was set
    aside. The ground is a researcher's own words and carries no identity.

    Arguments:
        skip: Number of sessions to skip (default 0).
        limit: Maximum sessions per page (default 25, max 100).

    Returns:
        A dict with "sessions" (list of session objects) and "total_count" (int).
    """
    engine = request.app.state.engine

    with engine.begin() as conn:
        # Total count for pagination metadata.
        total_count = conn.execute(
            select(func.count(sessions.c.session_id))
        ).scalar() or 0

        # Outer joins so a session without a pose artefact still appears with preprocessed
        # false, rather than vanishing from a list whose job is to show what is outstanding.
        # Duration comes from media_objects and is not recomputed here: the probe read it off
        # the container at ingest and a second reading could disagree with the stored one.
        rows = conn.execute(
            select(
                sessions.c.session_id,
                sessions.c.domain,
                sessions.c.recorded_on,
                sessions.c.quality_verdict,
                sessions.c.excluded_at,
                sessions.c.exclusion_reason,
                media_objects.c.duration_s,
                pose_artifacts.c.session_id.label("has_pose"),
                _is_annotated(sessions.c.session_id).label("annotated"),
            )
            .join(media_objects, media_objects.c.media_sha256 == sessions.c.media_sha256)
            .outerjoin(pose_artifacts,
                       pose_artifacts.c.session_id == sessions.c.session_id)
            .order_by(sessions.c.recorded_on.desc(), sessions.c.session_id)
            .offset(skip).limit(limit)
        ).all()

        session_list = [
            {
                "session_id": row.session_id,
                "domain": row.domain,
                "recorded_on": row.recorded_on.isoformat() if row.recorded_on else None,
                "quality_verdict": row.quality_verdict,
                "duration_s": round(float(row.duration_s), 2),
                "preprocessed": row.has_pose is not None,
                "annotated": bool(row.annotated),
                "excluded_at": (row.excluded_at.isoformat()
                                if row.excluded_at is not None else None),
                "exclusion_reason": row.exclusion_reason,
            }
            for row in rows
        ]

    return {
        "sessions": session_list,
        "total_count": int(total_count),
        "skip": skip,
        "limit": limit,
    }


@router.get("/dashboard/sessions/{session_id}")
def get_session_detail(
    request: Request,
    session_id: str,
) -> dict[str, Any]:
    """One session's detail.

    Returns the same fields as /sessions plus preprocessing and annotation details.
    No teacher identity or media filename is included (R1 and D76).

    Arguments:
        session_id: The session ID (ULID).

    Returns:
        A dict with session metadata, preprocessing status, and annotation counts.

    Raises:
        404 if the session is not found.
    """
    engine = request.app.state.engine

    with engine.begin() as conn:
        row = conn.execute(
            select(
                sessions.c.session_id,
                sessions.c.domain,
                sessions.c.recorded_on,
                sessions.c.quality_verdict,
                sessions.c.media_sha256,
                sessions.c.subject,
                sessions.c.grade_level,
                sessions.c.created_at,
                media_objects.c.duration_s,
                pose_artifacts.c.session_id.label("has_pose"),
            )
            .join(media_objects, media_objects.c.media_sha256 == sessions.c.media_sha256)
            .outerjoin(pose_artifacts,
                       pose_artifacts.c.session_id == sessions.c.session_id)
            .where(sessions.c.session_id == session_id)
        ).first()

        if row is None:
            raise HTTPException(status_code=404, detail={
                "type": "/errors/not-found",
                "detail": f"session {session_id} not found",
            })

        # Counted through the assignment, not by matching the session id inside the clip id. A
        # substring match would depend on how clip ids happen to be formatted and would quietly
        # return zero the day that format changed.
        annotation_count = conn.execute(
            select(func.count(annotations.c.annotation_id))
            .join(annotation_assignments,
                  annotation_assignments.c.assignment_id == annotations.c.assignment_id)
            .where(annotation_assignments.c.session_id == session_id)
        ).scalar() or 0

        preprocessed = row.has_pose is not None
        duration_s = round(float(row.duration_s), 2)

    return {
        "session_id": row.session_id,
        "domain": row.domain,
        "recorded_on": row.recorded_on.isoformat() if row.recorded_on else None,
        "quality_verdict": row.quality_verdict,
        "subject": row.subject,
        "grade_level": row.grade_level,
        "duration_s": duration_s,
        "preprocessed": preprocessed,
        "annotation_count": int(annotation_count),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/dashboard/runs")
def get_runs(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Maximum runs to return"),
) -> dict[str, Any]:
    """Recent runs from the run_outputs directory.

    Returns a list of phase runs (phases 0-11), ordered by run_id descending (newest first).
    Each run includes run_id, phase, verdict (ran or abstained), timestamp, and abstention
    reason if applicable. Incomplete or corrupted run directories are skipped.

    Bounded, and the bound is the point. The run directory accumulates one entry per phase per
    build and had 326 in it the first time this was measured; unbounded, the endpoint opened and
    parsed two JSON files per entry on every dashboard load, and sent the lot to the browser.
    `total_runs` reports how many are there so the count stays honest about what was left out.

    Arguments:
        limit: Maximum runs to return, newest first (default 50, max 500).

    Returns:
        A dict with "runs" (list, newest first), "total_runs" and "limit".
    """
    config = request.app.state.config
    run_root = config.paths.run_outputs

    runs_list: list[dict[str, Any]] = []
    directories = sorted(
        (entry for entry in run_root.iterdir() if entry.is_dir()),
        reverse=True) if run_root.is_dir() else []

    for run_dir in directories:
        # Stop reading once the page is full. A ULID sorts in creation order, so everything
        # already taken is newer than everything remaining and there is nothing to gain by
        # opening the rest.
        if len(runs_list) >= limit:
            break

        pair = _read_run_manifest(run_dir)
        if pair is None:
            continue

        manifest, result = pair
        ran = bool(result.get("ran", False))
        run_record: dict[str, Any] = {
            "run_id": str(manifest.get("run_id", run_dir.name)),
            "phase": str(manifest.get("phase", "?")),
            "verdict": "ran" if ran else "abstained",
            "finished_at": manifest.get("finished_at"),
        }

        if not ran:
            abstained = result.get("abstained")
            if isinstance(abstained, dict):
                run_record["abstention_reason"] = abstained.get("reason")
                run_record["abstention_missing"] = abstained.get("missing", [])

        runs_list.append(run_record)

    # `run_directories`, not `total_runs`: this is how many directories are on disk, and a
    # corrupted or half-written one is not a run. Counting them as runs would overstate what the
    # system has actually done, which is the one direction this dashboard must never err in.
    return {"runs": runs_list, "run_directories": len(directories), "limit": limit}
