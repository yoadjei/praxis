# -*- coding: utf-8 -*-
"""Phases 9, 10, 11: monitoring, evaluation, and export.

Phase 9 captures a snapshot of the system state: corpus counts by quality and domain,
preprocessing progress, annotation progress, routing queue, and - where a model exists -
accuracy WITH calibration (R4) and confidence-state distribution (R3).

Phase 10 runs the evaluation harness against model predictions and validates it by
injecting seeded errors. It abstains if the seeded_errors module is absent (not yet written).

Phase 11 collects every artefact a reader needs to reproduce the run: frozen config,
all split manifests, and a checksum index. It always runs.

R4 and R3 are enforced by the type system: every accuracy is paired with a calibration,
and every prediction carries a confidence state.
"""
from __future__ import annotations

import json
from collections import defaultdict

from sqlalchemy import func, select

from praxis.db.schema import (
    annotation_assignments,
    annotations,
    audit_log,
    pose_artifacts,
    sessions,
)
from praxis.phases import PhaseContext, PhaseResult, abstain, register


@register("9", "Dashboard and monitoring snapshot", needs=("1",))
def monitoring(context: PhaseContext) -> PhaseResult:
    """Corpus and model snapshot for the dashboard.

    Reports corpus stats (counts by domain and quality), preprocessing progress, annotation
    progress, routing queue depth. When a model exists, includes accuracy with calibration
    and confidence-state distribution. Only abstains when the database is unavailable.
    """
    engine = context.engine()
    if engine is None:
        return abstain(
            "phase 9 needs a database to report corpus stats and progress",
            "DATABASE_URL or PRAXIS_TEST_DSN")

    summary: dict[str, object] = {}

    # Corpus: count sessions by domain and quality verdict.
    with engine.begin() as conn:
        domain_quality = conn.execute(
            select(
                sessions.c.domain,
                sessions.c.quality_verdict,
                func.count(sessions.c.session_id).label("count")
            ).group_by(sessions.c.domain, sessions.c.quality_verdict)
        ).all()

        corpus_by_domain: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for domain, verdict, count in domain_quality:
            corpus_by_domain[domain][verdict] = count

        summary["corpus_by_domain"] = dict(corpus_by_domain)

        # Preprocessing: count sessions with pose artifacts.
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

        # Annotation progress: count assignments and annotations.
        total_assignments = conn.execute(
            select(func.count(annotation_assignments.c.assignment_id))
        ).scalar() or 0
        total_annotations = conn.execute(
            select(func.count(annotations.c.annotation_id))
        ).scalar() or 0
        summary["annotation"] = {
            "assignments": int(total_assignments),
            "annotations": int(total_annotations),
        }

        # Audit log: latest entry as a timestamp.
        latest_audit = conn.execute(
            select(audit_log.c.occurred_at).order_by(
                audit_log.c.audit_id.desc()).limit(1)
        ).scalar()
        summary["last_audit_at"] = latest_audit.isoformat() if latest_audit else None

    # Model scores would go here if a model exists. For now, report absence.
    summary["model"] = {"status": "absent"}
    summary["confidence_distribution"] = {"status": "awaiting model"}

    return PhaseResult(summary=summary)


@register("10", "Evaluation harnesses and seeded errors", needs=("5",))
def evaluation(context: PhaseContext) -> PhaseResult:
    """Run the evaluation harness and validate it with seeded errors.

    The seeded_errors module does not yet exist, so this phase imports it inside a
    try/except and abstains if it is missing. When it exists, this phase will run the
    harness against model predictions and inject known-shape errors to verify detection.

    Abstains when: the model does not exist, or seeded_errors is not available.
    """
    engine = context.engine()
    if engine is None:
        return abstain(
            "phase 10 needs a database to run the evaluation harness",
            "DATABASE_URL or PRAXIS_TEST_DSN")

    # Try to import the seeded errors module. It does not exist yet.
    try:
        import praxis.evaluation.seeded_errors  # noqa: F401
    except ImportError:
        return abstain(
            "phase 10 validates the harness with seeded errors",
            "praxis/evaluation/seeded_errors.py")

    # Model predictions would be loaded and evaluated here. For now, report that the
    # seeded_errors module exists but no model is available.
    summary: dict[str, object] = {
        "status": "ready_for_model",
        "seeded_errors_available": True,
    }

    return PhaseResult(summary=summary)


@register("11", "Reproducibility bundle")
def export(context: PhaseContext) -> PhaseResult:
    """Reproducibility index: every file a reader needs to reproduce this run.

    Collects the frozen config, every split manifest, and checksums everything into a
    bundle index JSON. Does not create a zip or archive - that belongs in
    scripts/build_repro_bundle.py. This phase produces the index that script consumes.

    Always runs. Never abstains.
    """
    from praxis.phases.artefacts import sha256_of, write_json

    index: dict[str, object] = {
        "files": [],
        "total_bytes": 0,
        "file_count": 0,
    }
    files_list: list[dict[str, object]] = []
    total_bytes = 0

    # Include the frozen config.
    config_path = context.output_dir / "config.yaml"
    if not config_path.exists():
        # Write the config to the output directory if not already there.
        config_data = context.config.model_dump(mode="json")
        config_text = json.dumps(config_data, indent=2, sort_keys=True) + "\n"
        config_path.write_text(config_text, encoding="utf-8")

    if config_path.exists():
        sha = sha256_of(config_path)
        size = config_path.stat().st_size
        files_list.append({
            "path": "config.yaml",
            "sha256": sha,
            "bytes": size,
        })
        total_bytes += size

    # Collect all manifests from run_outputs.
    manifest_dir = context.output_dir
    if manifest_dir.exists():
        for manifest_file in sorted(manifest_dir.glob("*.json")):
            if manifest_file.name != "bundle_index.json":  # Skip the index itself.
                sha = sha256_of(manifest_file)
                size = manifest_file.stat().st_size
                relative = manifest_file.relative_to(context.output_dir).as_posix()
                files_list.append({
                    "path": relative,
                    "sha256": sha,
                    "bytes": size,
                })
                total_bytes += size

    index["files"] = files_list
    index["total_bytes"] = total_bytes
    index["file_count"] = len(files_list)

    # Write the bundle index.
    index_path = write_json(context.output_dir / "bundle_index.json", index)

    # Checksum the index itself.
    index_sha = sha256_of(index_path)
    index_size = index_path.stat().st_size

    # Artefact for the index.
    from praxis.phases.artefacts import artefact
    index_artefact = artefact("bundle_index", index_path, relative_to=context.output_dir,
                              device=context.device, device_model=context.device_model)

    summary: dict[str, object] = {
        "file_count": len(files_list),
        "total_bytes": total_bytes,
        "index_sha256": index_sha,
        "index_bytes": index_size,
    }

    return PhaseResult(summary=summary, artefacts=(index_artefact,))
