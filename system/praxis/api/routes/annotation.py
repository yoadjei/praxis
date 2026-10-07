"""GET /api/v1/codebook, /api/v1/queue, POST /api/v1/labels, GET disagreements, agreement.

The annotation workflow is two-round: calibration where raters label a shared set of clips,
then production where trusted raters label new clips. This endpoint serves the UI with
assignments, accepts labels, tracks disagreements for the calibration view, and reports
agreement statistics.

**Why routes are `def`, not `async def`:** reading the assignment queue and persisting a label
both query the database, which is blocking I/O. FastAPI runs a synchronous route in a worker
thread, so the event loop is not stalled. An `async def` route doing the same blocking call
would stall the loop for the duration of the query, starving other requests.

Status codes live in the domain layer (AnnotationRefused) rather than here, because whether a
duplicate is a 409 is a fact about the rule, not about HTTP. The route uses that status without
deciding it again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from praxis.annotation import (
    ACTIVE_CODEBOOK,
    AnnotationRefused,
    accept_annotation,
)
from praxis.annotation.irr import DEFAULT_ALPHA_GATE, Annotation, Rater, report
from praxis.annotation.server import disagreements as compute_disagreements
from praxis.annotation.store import (
    find_assignment,
    load_annotations,
    queue_for,
    record_annotation,
)
from praxis.ids import new_ulid

router = APIRouter(prefix="/api/v1/annotation", tags=["annotation"])


@dataclass(frozen=True)
class _AnnotationRecord:
    """An accepted annotation plus the identity it is stored under.

    `accept_annotation` returns the domain object, which carries what the rater said but no
    row identity: `annotation_id` is assigned at persistence time and `note` is not part of
    the agreement contract. This carries both alongside it so `record_annotation` receives one
    object with every column it writes.
    """

    annotation_id: str
    clip_id: str
    rater_id: str
    behaviour: str
    codebook_version: str
    labels: dict[str, Any]
    is_nonscorable: bool
    note: str | None
    rater_confidence: str
    session_college_id: str | None


class SubmitLabelRequest(BaseModel):
    """The request body for submitting an annotation."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    rater_id: str
    behaviour: str
    labels: dict[str, Any]
    is_nonscorable: bool = False
    note: str | None = None
    rater_confidence: str = "certain"
    session_college_id: str | None = None


@router.get("/codebook")
def get_codebook() -> dict[str, Any]:
    """The active codebook version and its label vocabulary, for the rater UI to render.

    This endpoint returns the codebook the UI is bound to. Every annotation submitted to
    /labels is validated against this version before it is persisted. A later revision of
    the borderline rulings cannot silently reinterpret a label made under an earlier version
    because each annotation records its codebook_version.
    """
    return {
        "version": ACTIVE_CODEBOOK.version,
        "behaviours": [
            {
                "behaviour": spec.behaviour,
                "name": spec.name,
                "fields": [
                    {
                        "name": field.name,
                        "scale": field.scale.value,
                        "levels": field.levels,
                        "minimum": field.minimum,
                        "maximum": field.maximum,
                        "description": field.description,
                    }
                    for field in spec.fields
                ],
                "nonscorable_when": spec.nonscorable_when,
            }
            for spec in ACTIVE_CODEBOOK.behaviours
        ],
        "confidence_levels": list(ACTIVE_CODEBOOK.confidence_levels),
        "excluded_from_primary_irr": list(ACTIVE_CODEBOOK.excluded_from_primary_irr),
    }


@router.get("/queue")
def get_queue(
    request: Request,
    rater_id: str = Query(..., description="The rater ID"),
    limit: int = Query(25, ge=1, le=100,
                       description="Maximum assignments to return"),
    round_name: str | None = Query(None,
                                    description="Limit to this round (calibration-1, etc.)"),
) -> dict[str, Any]:
    """The clips this rater is assigned to label, in clip order.

    The rater receives assignments that have not yet been submitted. An assignment is marked
    done when an annotation row exists with the same assignment_id and behaviour. The response
    is ordered by clip_index so the UI can walk them in sequence.

    Arguments:
        rater_id: The rater ID (typically a ULID).
        limit: Maximum number of assignments to return, default 25. Must be 1-100.
        round_name: If given, return only assignments from this round. If None, all rounds.

    Returns:
        A dict with an "assignments" key carrying a list of assignment objects, each with
        assignment_id, rater_id, clip (including clip_id, session_id, clip_index, times),
        behaviour, and round_name.
    """
    with request.app.state.engine.begin() as conn:
        assignments = queue_for(conn, rater_id, limit=limit, round_name=round_name)

    return {
        "assignments": [
            {
                "assignment_id": a.assignment_id,
                "rater_id": a.rater_id,
                "clip": {
                    "clip_id": a.clip.clip_id,
                    "session_id": a.clip.session_id,
                    "clip_index": a.clip.clip_index,
                    "start_seconds": a.clip.start_seconds,
                    "end_seconds": a.clip.end_seconds,
                },
                "behaviour": a.behaviour,
                "round_name": a.round_name,
            }
            for a in assignments
        ],
    }


@router.post("/labels", status_code=201)
def submit_label(
    request: Request,
    body: SubmitLabelRequest,
) -> dict[str, Any]:
    """Accept and persist one rater's annotation for one behaviour on one clip.

    **Codebook version enforcement.** The annotation is validated against ACTIVE_CODEBOOK,
    and the codebook version is stamped onto the stored row. A later revision of the codebook
    cannot silently reinterpret a label made under an earlier version (Phase 2 acceptance
    test).

    **Label validation.** Any label not defined in the active codebook version is refused by
    raising HTTPException(422) naming the offending key.

    **Duplicate detection.** If this rater has already labelled this clip and behaviour,
    a 409 Conflict is raised (handled by AnnotationRefused in the domain layer).

    Arguments:
        clip_id: The clip being annotated (e.g., "session_id:00001").
        rater_id: The rater ID.
        behaviour: The behaviour ID (B1 to B5).
        labels: The label dict (e.g., {"b1_present": True, "b1_count": 3}).
        is_nonscorable: Whether the rater marked this as nonscorable.
        note: Optional free-text note from the rater.
        rater_confidence: The rater's stated confidence (certain, probable, guess).
        session_college_id: The college ID of the session, for familiarity effect analysis.

    Returns:
        201 with {"annotation_id": <id>, "codebook_version": <version>}.

    Raises:
        422 if a label is not in the active codebook (unknown field or invalid value).
        409 if this rater has already labelled this clip and behaviour.
    """
    try:
        annotation = accept_annotation(
            ACTIVE_CODEBOOK,
            clip_id=body.clip_id,
            rater_id=body.rater_id,
            behaviour=body.behaviour,
            labels=body.labels,
            is_nonscorable=body.is_nonscorable,
            rater_confidence=body.rater_confidence,
            session_college_id=body.session_college_id,
        )
    except AnnotationRefused as refused:
        raise HTTPException(status_code=refused.http_status, detail={
            "type": "/errors/annotation-refused",
            "detail": refused.reason,
        }) from refused

    annotation_id = new_ulid()
    record = _AnnotationRecord(
        annotation_id=annotation_id,
        clip_id=annotation.clip_id,
        rater_id=annotation.rater_id,
        behaviour=annotation.behaviour,
        codebook_version=annotation.codebook_version,
        labels=annotation.labels,
        is_nonscorable=annotation.is_nonscorable,
        note=body.note,
        rater_confidence=annotation.rater_confidence,
        session_college_id=annotation.session_college_id,
    )

    with request.app.state.engine.begin() as conn:
        # Resolve the assignment this label answers and store the link. Without it the row
        # is invisible to every round-filtered read, so a label submitted here would never
        # reach the calibration view that exists to read it.
        try:
            assignment_id = find_assignment(
                conn,
                rater_id=record.rater_id,
                clip_id=record.clip_id,
                behaviour=record.behaviour,
            )
        except ValueError as malformed:
            raise HTTPException(status_code=422, detail={
                "type": "/errors/annotation-refused",
                "detail": str(malformed),
            }) from malformed

        try:
            record_annotation(conn, record, assignment_id=assignment_id)
        except AnnotationRefused as refused:
            raise HTTPException(status_code=refused.http_status, detail={
                "type": "/errors/annotation-refused",
                "detail": refused.reason,
            }) from refused

    return {
        "annotation_id": annotation_id,
        "codebook_version": annotation.codebook_version,
    }


@router.get("/calibration/{round_name}/disagreements")
def get_disagreements(
    request: Request,
    round_name: str,
) -> dict[str, Any]:
    """Surface per-clip, per-behaviour, per-field disagreements between raters in a round.

    This endpoint returns what each rater said for every (clip, behaviour, field) where
    disagreement exists (more than one distinct value). The calibration view uses this to
    show disagreements side by side, revealing where the codebook is ambiguous or where
    the raters are drifting.

    Arguments:
        round_name: The calibration or production round name (e.g., "calibration-1").

    Returns:
        A dict with a "disagreements" key carrying a list of disagreement objects, each with
        clip_id, behaviour, field, by_rater (dict of rater_id -> value), and distinct_values.
    """
    with request.app.state.engine.begin() as conn:
        annotations_rows = load_annotations(conn, round_name=round_name)

    annotations = tuple(
        Annotation(
            clip_id=row["clip_id"],
            rater_id=row["rater_id"],
            behaviour=row["behaviour"],
            codebook_version=row["codebook_version"],
            labels=row["labels"],
            is_nonscorable=row.get("is_nonscorable", False),
            rater_confidence=row.get("rater_confidence", "certain"),
            session_college_id=row.get("session_college_id"),
        )
        for row in annotations_rows
    )

    disagreement_list = compute_disagreements(annotations)

    return {
        "disagreements": [
            {
                "clip_id": d.clip_id,
                "behaviour": d.behaviour,
                "field": d.field,
                "by_rater": d.by_rater,
                "distinct_values": d.distinct_values,
            }
            for d in disagreement_list
        ],
    }


@router.get("/agreement")
def get_agreement(
    request: Request,
    round_name: str | None = Query(None, description="Limit to this round"),
) -> dict[str, Any]:
    """The per-behaviour agreement table for a round, with calibration and confidence.

    This endpoint returns inter-rater agreement statistics for every behaviour and field in
    the round. The statistics include Krippendorff's alpha and the per-field primary statistic
    (weighted kappa for ordinal, ICC(2,k) for continuous). Confidence intervals are bootstrap
    intervals at the alpha gate threshold.

    Arguments:
        round_name: If given, compute agreement only for this round. If None, for all data.

    Returns:
        A dict with an "agreement" key carrying the AgreementReport, which includes behaviours
        (each with fields), the gate value, and whether it was passed.
    """
    with request.app.state.engine.begin() as conn:
        annotations_rows = load_annotations(conn, round_name=round_name)

    annotations = tuple(
        Annotation(
            clip_id=row["clip_id"],
            rater_id=row["rater_id"],
            behaviour=row["behaviour"],
            codebook_version=row["codebook_version"],
            labels=row["labels"],
            is_nonscorable=row.get("is_nonscorable", False),
            rater_confidence=row.get("rater_confidence", "certain"),
            session_college_id=row.get("session_college_id"),
        )
        for row in annotations_rows
    )

    # sorted, not set order. The rater sequence reaches the bootstrap resampler, and a set
    # iterates in hash order, which varies between interpreter runs. R7 asks for a run to be
    # reproducible from one config; a confidence interval that moves because a set reordered
    # itself is not.
    rater_ids = sorted({a.rater_id for a in annotations})
    try:
        agreement_report = report(
            annotations,
            raters=tuple(Rater(rater_id) for rater_id in rater_ids),
        )
    except ValueError as refused:
        # `report` refuses to pool labels made under different codebook versions, because a
        # revision changes what a label means and pooling would read that change as rater
        # disagreement. That is a true fact about the stored round, not a server fault, and
        # CODEBOOK.md section 8 makes it reachable in production: the codebook is revised
        # between calibration rounds, so a round can genuinely hold two versions. Answering
        # 500 would report the project's own invariant as a crash and hide the reason from
        # the only person able to act on it.
        raise HTTPException(status_code=409, detail={
            "type": "/errors/agreement-unavailable",
            "detail": str(refused),
        }) from refused

    return {
        "agreement": {
            "behaviours": [
                {
                    "behaviour": b.behaviour,
                    "name": b.name,
                    "fields": [
                        {
                            "field": f.field,
                            "scale": f.scale.value,
                            "n_units": f.n_units,
                            "n_raters": f.n_raters,
                            "alpha": {
                                "point": f.alpha.point,
                                "lower": f.alpha.lower,
                                "upper": f.alpha.upper,
                            } if f.alpha else None,
                            "primary_statistic": f.primary_statistic,
                            "primary": {
                                "point": f.primary.point,
                                "lower": f.primary.lower,
                                "upper": f.primary.upper,
                            } if f.primary else None,
                            # Exactly the fields IccResult carries. The previous version asked
                            # for f_statistic, p_value, lower and upper, none of which exist on
                            # it, so this endpoint raised AttributeError on every call that
                            # reached a continuous field.
                            #
                            # `fully_crossed`, `units_dropped` and `warning` are not optional
                            # extras: BUILD-SPEC Phase 2 requires the non-fully-crossed warning
                            # to be reported alongside the coefficient, and an ICC(2,k) served
                            # without its design caveat is the number without the reason to
                            # doubt it. See D48 on an unavailable check abstaining.
                            "icc": {
                                "icc": f.icc.icc,
                                "n_units": f.icc.n_units,
                                "n_raters": f.icc.n_raters,
                                "fully_crossed": f.icc.fully_crossed,
                                "units_dropped": f.icc.units_dropped,
                                "warning": f.icc.warning,
                            } if f.icc else None,
                            "notes": f.notes,
                        }
                        for f in b.fields
                    ],
                }
                for b in agreement_report.behaviours
            ],
            # Derived here, not read off the report. `AgreementReport` carries no gate fields -
            # those live on the calibration-round object in `praxis.annotation.server`, and
            # reading them from this one raised AttributeError on every call.
            #
            # CODEBOOK.md section 8 step 5: production annotation begins when alpha is at or
            # above the gate on *every* categorical field. So a field whose alpha is undefined
            # cannot be counted as passing - `meets_gate` returns None there, and under D48 an
            # unavailable check abstains rather than quietly voting yes. `blocked_fields` names
            # the ones that actually fell short, which is what a reviewer needs in order to act
            # on section 8's instruction to redefine or drop them.
            "gate_value": DEFAULT_ALPHA_GATE,
            "gate_passed": all(
                f.meets_gate is True
                for b in agreement_report.behaviours for f in b.fields),
            "fields_below_gate": [
                {"behaviour": b.behaviour, "field": field}
                for b in agreement_report.behaviours for field in b.blocked_fields],
            "fields_without_alpha": [
                {"behaviour": b.behaviour, "field": f.field}
                for b in agreement_report.behaviours for f in b.fields
                if f.meets_gate is None],
        },
    }
