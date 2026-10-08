# -*- coding: utf-8 -*-
"""Phases 4 and 5: behaviour model training and calibration.

Phase 4 orchestrates the split, dataset construction, and training. Phase 5 fits
per-behaviour temperature scaling and reports calibration metrics before and after.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from sqlalchemy import select

from praxis.annotation.clips import parse_clip_id
from praxis.annotation.codebook import get_codebook
from praxis.annotation.store import load_annotations
from praxis.behaviour.backbone import declared_version
from praxis.behaviour.dataset import ClipRecord, dataset_from_config, policy_from_config
from praxis.behaviour.features import (
    CacheProvenance,
    FeatureCacheError,
    cache_digest,
    read_clip,
)
from praxis.behaviour.heads import behaviour_heads
from praxis.behaviour.model import build_model
from praxis.behaviour.train import (
    plan_from_config as training_plan_from_config,
)
from praxis.behaviour.train import (
    train,
    weights_from_config,
)
from praxis.db.schema import pose_artifacts
from praxis.db.schema import sessions as sessions_table
from praxis.phases import PhaseContext, PhaseResult, abstain, register
from praxis.phases.artefacts import artefact
from praxis.preprocess.store import confirmed_teacher_track
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


def _labels_by_clip(annotations: tuple) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """clip_id -> behaviour -> the rater's codebook fields, and the clips that cannot be used.

    **A clip labelled twice for one behaviour is refused, not resolved here.** The calibration
    round double-codes deliberately so that agreement can be measured, and nothing in this
    system adjudicates a disagreement into a single training label - there is no adjudications
    table and no resolution rule. Picking one rater, or a majority of two, would be inventing
    that rule silently at the point where it is least visible. Production-round clips carry one
    annotation per behaviour and are unaffected.
    """
    labels: dict[str, dict[str, dict]] = defaultdict(dict)
    contested: set[str] = set()

    for row in annotations:
        clip_id, behaviour = row["clip_id"], row["behaviour"]
        if behaviour in labels[clip_id]:
            contested.add(clip_id)
            continue
        labels[clip_id][behaviour] = dict(row["labels"])

    for clip_id in contested:
        labels.pop(clip_id, None)
    return dict(labels), sorted(contested)


def _construct_clip_records(
    annotations: tuple,
    engine,
    codebook,
    heads_spec: dict,
    *,
    config,
    report: Callable[[str], None] = lambda _message: None,
) -> list:
    """Join annotations to the Stage A feature cache, one ClipRecord per usable clip.

    A clip with no cached features is skipped and named rather than dropped silently: the usual
    reason is that Stage A has not been run for its session, and a training set that quietly
    shrank is the hardest kind of result to question later.
    """
    cache_root = Path(config.paths.feature_cache)
    variants = tuple(variant.name for variant in policy_from_config(config).variants)
    clip_config = config.behaviour.clip
    settings = cache_digest(config)
    wanted_behaviours = set(heads_spec)

    labels, contested = _labels_by_clip(annotations)
    if contested:
        report(f"{len(contested)} clip(s) carry more than one annotation for a behaviour and "
               f"are held out of training: nothing adjudicates a disagreement into one label.")

    teachers: dict[str, str] = {}
    tracks: dict[str, int | None] = {}
    poses: dict[str, str] = {}
    with engine.begin() as conn:
        for row in conn.execute(select(sessions_table.c.session_id,
                                       sessions_table.c.teacher_id)):
            teachers[row.session_id] = row.teacher_id
        for row in conn.execute(select(pose_artifacts.c.session_id, pose_artifacts.c.sha256)):
            poses[row.session_id] = row.sha256

    records = []
    uncached = 0
    for clip_id, per_behaviour in sorted(labels.items()):
        if not wanted_behaviours <= set(per_behaviour):
            # A clip labelled for some behaviours and not others would train the missing heads
            # on nothing while looking like a complete example.
            continue

        session_id, _ = parse_clip_id(clip_id)
        if session_id not in teachers:
            continue
        if session_id not in tracks:
            with engine.begin() as conn:
                tracks[session_id] = confirmed_teacher_track(conn, session_id)
        track_id = tracks[session_id]
        if track_id is None:
            continue

        expected = CacheProvenance(
            backbone_version=declared_version(config),
            pose_sha256=poses.get(session_id, ""),
            config_sha256=settings,
            feature_dim=config.behaviour.backbone.output_dim,
            frames=clip_config.frames,
            crop_size=clip_config.crop_size,
            crop_pad=(clip_config.crop_padding - 1.0) / 2.0,
            track_id=int(track_id),
            frames_located=0,
        )

        try:
            cached = read_clip(cache_root, clip_id, variants=variants, expect=expected)
        except FeatureCacheError as missing:
            uncached += 1
            if uncached <= 3:
                report(f"  no features for {clip_id}: {missing}")
            continue

        records.append(ClipRecord(
            clip_id=clip_id, session_id=session_id, teacher_id=teachers[session_id],
            features={name: torch.from_numpy(tensor.astype("float32"))
                      for name, tensor in cached.features.items()},
            keypoints=torch.from_numpy(cached.keypoints),
            labels=per_behaviour, variants=variants))

    if uncached:
        report(f"{uncached} annotated clip(s) have no cached features. Run "
               f"`python scripts/build_features.py --write` to build Stage A.")
    return records


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

    # Join the annotations to the Stage A cache. Every reason a clip drops out is reported
    # rather than counted, because a training set that quietly shrank is the hardest kind of
    # result to question months later.
    notes: list[str] = []
    codebook = get_codebook()
    heads_spec = {
        behaviour: behaviour_heads(codebook, behaviour)
        for behaviour in context.config.behaviour.heads.presence_behaviours
    }
    clip_records = _construct_clip_records(
        annotations, engine, codebook, heads_spec,
        config=context.config, report=notes.append)

    if not clip_records:
        detail = ("\n  " + "\n  ".join(notes)) if notes else (
            " Run `python scripts/build_features.py --write` to build Stage A.")
        return abstain(
            "phase 4 found no clips carrying both an annotation and cached features." + detail,
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
