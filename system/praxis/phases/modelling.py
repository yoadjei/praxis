# -*- coding: utf-8 -*-
"""Phases 4 and 5: behaviour model training and calibration.

Phase 4 orchestrates the split, dataset construction, and training. Phase 5 fits
per-behaviour temperature scaling and reports calibration metrics before and after.
"""
from __future__ import annotations

import hashlib
import json

import torch
from sqlalchemy import select

from praxis.annotation.codebook import get_codebook
from praxis.annotation.store import load_annotations
from praxis.behaviour.dataset import dataset_from_config
from praxis.behaviour.heads import behaviour_heads
from praxis.behaviour.model import build_model
from praxis.behaviour.train import (
    plan_from_config as training_plan_from_config,
)
from praxis.behaviour.train import (
    train,
    weights_from_config,
)
from praxis.db.schema import sessions as sessions_table
from praxis.phases import PhaseContext, PhaseResult, abstain, register
from praxis.phases.artefacts import artefact
from praxis.splits.manifest import SessionRecord
from praxis.splits.manifest import plan_from_config as plan_splits_from_config
from praxis.splits.store import save_manifest


def _config_sha256(config) -> str:
    """Deterministic hash of the config for split reproducibility."""
    frozen_dict = json.dumps(config.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(frozen_dict.encode()).hexdigest()


def _load_session_records(engine) -> tuple[SessionRecord, ...]:
    """Query the database for all sessions, shaped for split planning."""
    with engine.begin() as conn:
        rows = conn.execute(select(sessions_table)).fetchall()
        return tuple(
            SessionRecord(
                session_id=row.session_id,
                teacher_id=row.teacher_id,
                college_id=row.college_id,
                domain=row.domain,
            )
            for row in rows
        )


def _construct_clip_records(
    annotations: tuple,
    engine,
    codebook,
    heads_spec: dict,
) -> list:
    """Build ClipRecord objects from annotations and database state.

    Raises when the feature cache does not exist - Phase 4 cannot run until
    Stage A has written cached features for every annotated clip.
    """
    # this phase orchestrates, not implements. the clip construction logic
    # reads from annotations, the feature cache, and the pose store.
    # until those are all wired up, this phase should abstain.
    raise NotImplementedError(
        "clip record construction requires the feature cache from Stage A, which is not "
        "yet implemented. once Stage A writes cached features, this phase will load them "
        "and wire them with annotations to build training examples."
    )


@register("4", "Behaviour model: splits, features and training", needs=("2", "3"))
def model(context: PhaseContext) -> PhaseResult:
    """Partition the corpus, construct datasets, and train the behaviour model."""
    engine = context.engine()
    if engine is None:
        return abstain(
            "phase 4 reads sessions and annotations from the database",
            "DATABASE_URL or PRAXIS_TEST_DSN",
        )

    # load session records and check for corpus
    sessions = _load_session_records(engine)
    if not sessions:
        return abstain(
            "phase 4 needs at least one ingested session to partition",
            "ingested sessions from phase 1",
        )

    # plan the split, catching the specific case where the corpus is too small
    config_sha = _config_sha256(context.config)
    try:
        split_plan = plan_splits_from_config(
            sessions, context.config, config_sha256=config_sha
        )
    except Exception as e:
        # SplitError includes useful messages like "X teachers cannot make 3 partitions"
        return abstain(str(e), "corpus with sufficient teachers for the split")

    # save the split manifest to the database
    with engine.begin() as conn:
        manifest_id = save_manifest(conn, split_plan.manifest)

    # load annotations - check that labels exist for training
    with engine.begin() as conn:
        annotations = load_annotations(conn)

    if not annotations:
        return abstain(
            "phase 4 needs human annotations to train on",
            "annotations from a calibration or production round",
        )

    # attempt to construct clip records from annotations and the feature cache
    try:
        codebook = get_codebook()
        heads_spec = {
            behaviour: behaviour_heads(codebook, behaviour)
            for behaviour in context.config.behaviour.heads.presence_behaviours
        }
        clip_records = _construct_clip_records(annotations, engine, codebook, heads_spec)
    except NotImplementedError as e:
        return abstain(str(e), "feature cache from Stage A")

    if not clip_records:
        return abstain(
            "phase 4 found no valid clips with both annotations and cached features",
            "annotated clips with features in the feature cache",
        )

    # partition clips by the split manifest
    train_records = [
        c for c in clip_records
        if c.teacher_id in split_plan.manifest.train_teachers
    ]
    val_records = [
        c for c in clip_records if c.teacher_id in split_plan.manifest.val_teachers
    ]

    if not train_records or not val_records:
        return abstain(
            "phase 4 partitioned the corpus but has no clips for train or val",
            "clips distributed across train and val partitions",
        )

    # construct datasets
    train_set = dataset_from_config(
        train_records,
        heads_spec,
        context.config,
        seed=context.seed,
        train=True,
    )
    val_set = dataset_from_config(
        val_records,
        heads_spec,
        context.config,
        seed=context.seed,
        train=False,
    )

    # build model and train
    model_obj = build_model(context.config, codebook)
    if context.device == "cuda":
        model_obj = model_obj.cuda()

    training_plan = training_plan_from_config(context.config)
    loss_weights = weights_from_config(context.config)

    training_run = train(
        model_obj,
        train_set,
        val_set,
        heads_spec,
        plan=training_plan,
        weights=loss_weights,
        seed=context.seed,
        deterministic=context.deterministic,
    )

    # save the trained weights as an artefact
    weights_path = context.output_dir / "model_weights.pt"
    torch.save(training_run.best_state, weights_path)
    weights_ref = artefact("model_weights", weights_path, relative_to=context.output_dir,
                          device=context.device, device_model=context.device_model)

    # summarise the run
    summary = {
        "manifest_id": manifest_id,
        "split_sizes": {
            "train_teachers": len(split_plan.manifest.train_teachers),
            "val_teachers": len(split_plan.manifest.val_teachers),
            "test_teachers": len(split_plan.manifest.test_teachers),
        },
        "training": {
            "epochs_run": int(training_run.epochs[-1].epoch + 1),
            "best_epoch": int(training_run.best_epoch),
            "best_val_macro_f1": float(training_run.best_score),
            "stopped_early": bool(training_run.stopped_early),
        },
    }

    return PhaseResult(summary=summary, artefacts=(weights_ref,))


@register("5", "Calibration: temperature scaling and reliability", needs=("4",))
def calibration(context: PhaseContext) -> PhaseResult:
    """Fit per-behaviour temperature on validation predictions and report calibration."""
    engine = context.engine()
    if engine is None:
        return abstain(
            "phase 5 reads the trained model and validation data",
            "DATABASE_URL or PRAXIS_TEST_DSN",
        )

    # phase 5 needs:
    # 1. the trained model weights from phase 4 (written to context.output_dir)
    # 2. the validation dataset (built from the split manifest and clip records)
    # 3. validation predictions (via validation_scores)
    # 4. fit per-behaviour temperatures
    # 5. report calibration metrics before and after

    # this requires reading the artefacts phase 4 wrote, which would be in a previous
    # run's output directory, not this phase's. that cross-phase artefact loading is
    # not yet wired up, so phase 5 abstains until phase 4 completes in this run.
    return abstain(
        "phase 5 requires the trained model and validation data from phase 4",
        "trained model weights and validation clip records from phase 4",
    )
