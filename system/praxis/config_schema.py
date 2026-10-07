"""The shape of a run's configuration, section by section.

Declarations only. Loading, device selection and seeding live in `praxis/config.py`, which
BUILD-SPEC §2.1 names as the one place device selection may happen.

Every section sets ``extra="forbid"`` and ``frozen=True``. A typo is then a startup error
naming the key rather than a setting that silently does nothing, and a run cannot edit its own
configuration partway through, which would make the frozen copy a record of something that did
not happen.

Several validators refuse a value outright rather than warning. Each of those corresponds to an
entry in ``locked:`` at the foot of ``configs/default.yaml``: they are the settings where a
plausible-looking change invalidates the research rather than merely degrading it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from praxis.vocabulary import ShiftAxis

Device = Literal["cpu", "cuda"]
Behaviour = Literal["B1", "B2", "B3", "B4", "B5"]

# The fewest whole clips a session must be able to yield to be worth ingesting. Five rather than
# one: a single clip cannot show a behaviour changing, and the annotation round costs the same
# per session however short it is. `the_duration_floor_admits_enough_clips` turns this into the
# duration floor, so the two can never disagree. D81.
MIN_USEFUL_CLIPS = 5


class ConfigError(RuntimeError):
    """Raised when a configuration cannot be loaded, or describes an impossible run."""


class MediaRootUnavailable(ConfigError):
    """Raised when a phase needs the external media volume and it is not there.

    Separate from ConfigError because it is the expected, recoverable case: the SSD is
    unplugged. SCHEMA.md gives it a specific error for exactly this reason.
    """


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# run, compute, paths
# ---------------------------------------------------------------------------

class RunSection(_Section):
    name: str
    seed: int
    device: Device
    device_model: str | None = None
    deterministic: bool
    cuda_tolerance: float = Field(gt=0)
    num_workers: int = Field(ge=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class LocalMachine(_Section):
    role: str
    device: Device
    total_ram_gb: float
    typical_free_ram_gb: float
    cores: int
    resnet50_inference_fps: float


class KaggleMachine(_Section):
    role: str
    device: Device
    gpu: str
    vram_gb: int
    weekly_quota_h: float
    session_limit_h: float


class ColabMachine(_Section):
    role: str
    device: Device
    compute_units_remaining: float
    unit_burn_per_h: dict[str, float]
    prefer_gpu: str


class ComputeSection(_Section):
    local: LocalMachine
    kaggle: KaggleMachine
    colab: ColabMachine


class PathsSection(_Section):
    # None when PRAXIS_MEDIA_ROOT is unset. Loading still succeeds, because Phases 0, 2, 5
    # and 6 never touch video; the phases that do call require_media_root() and get a
    # specific error instead of a confusing one about a path named "${PRAXIS_MEDIA_ROOT}".
    media_root: Path | None
    feature_cache: Path
    pose_artifacts: Path
    model_weights: Path
    run_outputs: Path
    scratch: Path
    torch_home: Path
    min_free_gb_system_drive: float = Field(ge=0)



class ToolsSection(_Section):
    """Binaries resolved at run time. A string, not a Path: a bare command name is legal."""
    ffmpeg: str
    ffprobe: str

# ---------------------------------------------------------------------------
# Phase 1: ingest
# ---------------------------------------------------------------------------

class QualityGate(_Section):
    min_duration_s: float
    max_duration_s: float
    # Frame area and shorter edge rather than width and height, so portrait and landscape
    # footage are judged by the same question. See D81 and `quality.evaluate`.
    min_frame_pixels: int = Field(ge=1)
    min_shorter_edge: int = Field(ge=1)
    min_fps: float
    max_fps_jitter: float
    person_detection_sample_frames: int
    min_person_present_fraction: float = Field(ge=0, le=1)
    min_mean_luminance: int = Field(ge=0, le=255)
    max_mean_luminance: int = Field(ge=0, le=255)
    require_audio: bool
    fail_on_warn: bool

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> QualityGate:
        if self.min_duration_s >= self.max_duration_s:
            raise ValueError("min_duration_s must be below max_duration_s")
        if self.min_mean_luminance >= self.max_mean_luminance:
            raise ValueError("min_mean_luminance must be below max_mean_luminance")
        return self


class IngestSection(_Section):
    accepted_containers: list[str]
    max_upload_bytes: int
    hash_chunk_bytes: int
    quality_gate: QualityGate

    # Where the corpus comes from. Both default to None, because every phase except 1 ignores
    # them and a fresh checkout has neither.
    #
    # These are in the config rather than only on `ingest_corpus.py`'s command line because R7
    # says a run is reproducible from one config file. A manifest supplied as an argument makes
    # the corpus a property of whoever typed the command, and "which sessions went in" is the
    # single most consequential input the system has: get it wrong and every number downstream
    # is computed over a different corpus than the one the config describes.
    manifest: Path | None = None
    media_dir: Path | None = None
    keymap: Path | None = None


# ---------------------------------------------------------------------------
# Phase 3: preprocessing
# ---------------------------------------------------------------------------

class PoseSection(_Section):
    model: str
    export_format: Literal["onnx"]          # D10 retired coreml; onnx is the only value
    weights_file: str
    min_keypoint_confidence: float = Field(ge=0, le=1)
    min_person_confidence: float = Field(ge=0, le=1)
    # An IoU *similarity*, unlike `tracking.match_thresh`: two boxes overlapping by at least
    # this much are the same body and the weaker one is dropped. Bounded away from both ends
    # because the degenerate settings fail silently - see the validator.
    nms_iou_threshold: float = Field(gt=0, lt=1)
    max_persons_per_frame: int = Field(ge=1)

    @field_validator("nms_iou_threshold")
    @classmethod
    def suppression_is_neither_absent_nor_total(cls, value: float) -> float:
        # Not a style bound. At 1.0 nothing is ever suppressed and the estimator returns
        # `max_persons_per_frame` copies of one teacher; at 0.0 the first box removes every
        # other box that touches it and a full classroom collapses to a single person. Both
        # produce plausible-looking arrays, which is why they are refused here rather than
        # discovered in the learner aggregates. D77.
        if not 0.1 <= value <= 0.9:
            raise ValueError(
                f"nms_iou_threshold {value} is outside [0.1, 0.9]; near 1.0 no duplicate is "
                f"ever removed and near 0.0 adjacent pupils are merged into one body")
        return value


class TrackingSection(_Section):
    tracker: str
    track_high_thresh: float = Field(ge=0, le=1)
    track_low_thresh: float = Field(ge=0, le=1)
    new_track_thresh: float = Field(ge=0, le=1)
    track_buffer_frames: int = Field(ge=1)
    # An IoU *distance*, as in ByteTrack: a pair matches when `1 - IoU` is at most this. See
    # `ByteTrackConfig.min_iou`.
    match_thresh: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> TrackingSection:
        # The low band is what the second association works on. Inverted, it is empty and the
        # tracker silently degrades to plain IoU tracking while still being called bytetrack.
        if self.track_low_thresh >= self.track_high_thresh:
            raise ValueError(
                f"track_low_thresh ({self.track_low_thresh}) must be below track_high_thresh "
                f"({self.track_high_thresh}), or no detection is ever a low-confidence one")
        return self


class TeacherIdSection(_Section):
    weight_presence_duration: float
    weight_median_bbox_area: float
    weight_front_zone_time: float
    min_score_to_propose: float
    require_human_confirmation: bool
    max_candidates_recorded: int

    @field_validator("require_human_confirmation")
    @classmethod
    def confirmation_is_mandatory(cls, value: bool) -> bool:
        # The heuristic proposes; a human confirms. BUILD-SPEC Phase 3 and the locked list.
        if not value:
            raise ValueError("require_human_confirmation may never be false")
        return value

    @field_validator("max_candidates_recorded")
    @classmethod
    def a_reviewer_needs_at_least_one(cls, value: int) -> int:
        # Zero would store an empty ranking and leave the reviewer the same nothing the
        # discarded proposal left them. A 32-minute session forms hundreds of one-frame tracks,
        # so the cap is real; what it may not be is nothing.
        if value < 1:
            raise ValueError(
                f"max_candidates_recorded is {value}; a confirmation screen showing no "
                f"candidates cannot be confirmed against")
        return value


class BlurSection(_Section):
    method: str
    kernel_fraction: float
    dilate_face_box: float
    blur_all_faces: bool
    delete_original_after_blur: bool
    verify_deletion: bool

    @field_validator("delete_original_after_blur")
    @classmethod
    def original_must_be_deleted(cls, value: bool) -> bool:
        # R1. The unblurred file never outlives the processing window.
        if not value:
            raise ValueError("delete_original_after_blur may never be false")
        return value


class PreprocessSection(_Section):
    sample_fps: float
    pose: PoseSection
    tracking: TrackingSection
    teacher_id: TeacherIdSection
    blur: BlurSection


# ---------------------------------------------------------------------------
# Phase 4: behaviour model
# ---------------------------------------------------------------------------

class ClipSection(_Section):
    length_s: float
    fps: float
    frames: int
    overlap: float
    crop_size: int
    crop_padding: float

    @model_validator(mode="after")
    def frames_match_length_and_rate(self) -> ClipSection:
        expected = round(self.length_s * self.fps)
        if self.frames != expected:
            raise ValueError(f"frames={self.frames} but length_s*fps={expected}")
        return self


class BackboneSection(_Section):
    arch: str
    # The torchvision weights enum name. `DEFAULT` and `IMAGENET1K` are refused below.
    weights: str
    weights_file: str
    frozen: bool
    output_dim: int
    normalise_mean: tuple[float, float, float]
    normalise_std: tuple[float, float, float]
    interpolation: Literal["bilinear", "bicubic", "nearest"]
    cache_dtype: Literal["float16", "float32"]
    cache_features: bool
    stage_a_batch_frames: int = Field(ge=1)
    stage_a_kaggle_threshold_sessions: int = Field(ge=1)

    @field_validator("frozen")
    @classmethod
    def backbone_stays_frozen(cls, value: bool) -> bool:
        # D2. Backprop through ResNet-50 over 64 frames does not fit in the measured
        # 1.31 GB of headroom at any batch size.
        if not value:
            raise ValueError("the backbone is frozen on this estate; see D2 and D13")
        return value

    @field_validator("weights")
    @classmethod
    def weights_name_a_version(cls, value: str) -> str:
        # An alias is not a pin. `DEFAULT` resolves to whatever torchvision currently considers
        # best and has already moved once for this architecture, from the original recipe to
        # IMAGENET1K_V2. Accepting it would let a dependency upgrade change every cached
        # feature in the corpus while the config hash - the thing R7 rests on - stayed
        # identical. D72.
        if value.upper() in {"DEFAULT", "IMAGENET1K", "IMAGENET", "TRUE"}:
            raise ValueError(
                f"backbone.weights is {value!r}, which names no particular set of weights. "
                f"Spell the torchvision enum out, e.g. 'IMAGENET1K_V1'. See D72.")
        return value

    @field_validator("normalise_std")
    @classmethod
    def std_is_nonzero(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        # Division by a zero channel std yields inf, and the resulting features are silently
        # useless rather than loudly wrong.
        if any(channel <= 0 for channel in value):
            raise ValueError(f"normalise_std must be positive in every channel, got {value}")
        return value


class SplitsSection(_Section):
    """How the corpus is partitioned. By teacher, never by session or clip. R2."""

    val_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)
    # Which ladder to build. "domain" is the microteaching-trained, classroom-tested one; "site"
    # holds out a basic school for S1 and a college for S2, which is what a practicum-throughout
    # corpus needs. The axis is written onto the manifest, because S2 names different sessions
    # on each. D98.
    shift_axis: ShiftAxis = "domain"
    heldout_college: str | None = None
    heldout_school: str | None = None
    classroom_teachers_eligible_for_test: bool
    # Off unless the corpus leaves no other way to train. See D83 and `plan_splits`: it
    # changes what the S0-to-S2 drop measures, and the manifest records that it was used.
    classroom_teachers_may_train: bool = False
    min_teachers_per_partition: int = Field(ge=1)

    @model_validator(mode="after")
    def train_gets_a_share(self) -> SplitsSection:
        # Not a tidiness check. At a combined fraction of 1.0 the training partition is empty
        # and every downstream phase fails a long way from the cause.
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction {self.val_fraction} and test_fraction {self.test_fraction} "
                f"leave nothing to train on")
        return self


class KeypointBranchSection(_Section):
    input_keypoints: int
    hidden_dim: int
    output_dim: int
    normalise_by: Literal["bbox", "image", "torso_length"]


class TemporalSection(_Section):
    type: str
    layers: int
    kernel_size: int
    dilations: list[int]
    channels: int
    dropout: float = Field(ge=0, lt=1)
    residual: bool

    @model_validator(mode="after")
    def one_dilation_per_layer(self) -> TemporalSection:
        if len(self.dilations) != self.layers:
            raise ValueError(f"{self.layers} layers but {len(self.dilations)} dilations")
        return self


class PoolingSection(_Section):
    type: str
    attention_dim: int


class HeadsSection(_Section):
    presence_behaviours: list[Behaviour]
    intensity_behaviours: list[Behaviour]


class FlipSection(_Section):
    enabled: bool
    applies_to: list[Behaviour]
    probability: float = Field(ge=0, le=1)

    @field_validator("applies_to")
    @classmethod
    def never_flip_spatial_behaviours(cls, value: list[str]) -> list[str]:
        # A horizontal flip inverts orientation and position, so it destroys the label for
        # B2 and B3. Easy bug to ship, hard to find later, so it is refused here.
        forbidden = {"B2", "B3"} & set(value)
        if forbidden:
            raise ValueError(
                f"horizontal flip inverts orientation and position, so it may not apply to "
                f"{sorted(forbidden)}; see BUILD-SPEC Phase 4 and the locked list")
        return value


class ColourJitterSection(_Section):
    # False, and the code reads this rather than the values below. A colour jitter cannot be
    # applied to a cached 2048-dimensional activation, so under the Stage A feature cache these
    # settings describe an augmentation that does not happen. D75. They are kept rather than
    # deleted so the decision stays visible, and `praxis.behaviour.dataset.policy_from_config`
    # raises when this is true, because configured-but-unimplemented must fail rather than pass
    # quietly.
    applied: bool = False
    brightness: float
    contrast: float
    saturation: float
    applies_to: list[Behaviour]


class AugmentationSection(_Section):
    horizontal_flip: FlipSection
    colour_jitter: ColourJitterSection
    temporal_jitter_frames: int
    random_crop_scale: list[float]


class EarlyStoppingSection(_Section):
    monitor: str
    patience: int
    min_delta: float


class TrainSection(_Section):
    epochs: int
    batch_size: int = Field(ge=1)
    optimiser: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    warmup_epochs: int
    grad_clip_norm: float
    mixed_precision: bool
    # How many clips one session may contribute to a training epoch. An integer, null
    # for no cap, or "median" to cap at the median across sessions. Training only;
    # capping an evaluation set would report a figure on a subsample of the held-out
    # sessions while naming them in full. D84.
    max_clips_per_session: int | Literal["median"] | None = "median"
    early_stopping: EarlyStoppingSection

    @field_validator("max_clips_per_session")
    @classmethod
    def a_cap_admits_at_least_one_clip(cls, value):
        if isinstance(value, int) and value < 1:
            raise ValueError(
                "max_clips_per_session must be at least 1, null, or 'median'; a cap of "
                "zero drops every session from training without saying so")
        return value


class LossSection(_Section):
    presence: str
    focal_alpha: float
    focal_gamma: float
    intensity: str
    intensity_weight: float
    mask_nonscorable: bool


class BehaviourSection(_Section):
    clip: ClipSection
    backbone: BackboneSection
    keypoint_branch: KeypointBranchSection
    temporal: TemporalSection
    pooling: PoolingSection
    heads: HeadsSection
    augmentation: AugmentationSection
    train: TrainSection
    loss: LossSection
    baselines: list[str]


# ---------------------------------------------------------------------------
# Phase 5: calibration
# ---------------------------------------------------------------------------

class TemperatureSection(_Section):
    init_T: float
    max_iter: int
    lr: float
    fit_on: Literal["val"]


class FullNetworkCheckSection(_Section):
    enabled: bool
    size: int
    behaviour: Behaviour
    reduced_frames: int

    @field_validator("enabled")
    @classmethod
    def check_is_mandatory(cls, value: bool) -> bool:
        # D3. The shared backbone is a deviation from Ovadia et al.; this is the honesty
        # test on it, and disabling it would leave the deviation unexamined.
        if not value:
            raise ValueError("the full-network ensemble check may not be disabled; see D3")
        return value


class EnsembleSection(_Section):
    size: int = Field(ge=2)
    share_backbone: bool
    train_sequentially: bool
    vary: list[str]
    require_same_device_model: bool
    full_network_check: FullNetworkCheckSection

    @field_validator("require_same_device_model")
    @classmethod
    def members_share_a_device(cls, value: bool) -> bool:
        # D14. Members split across a T4 and an L4 would carry kernel variance into
        # ensemble disagreement, which is the epistemic signal Phase 6 rests on.
        if not value:
            raise ValueError("ensemble members must train on one device model; see D14")
        return value


class McDropoutSection(_Section):
    rate: float = Field(gt=0, lt=1)
    passes: int = Field(ge=2)


class CalibrationMetricsSection(_Section):
    ece_bins: int = Field(ge=2)
    compute: list[str]
    reliability_diagram: bool
    bootstrap_samples: int
    ci_level: float = Field(gt=0, lt=1)


class CalibrationSection(_Section):
    methods: list[Literal["temperature", "ensemble", "mc_dropout"]]
    primary: Literal["temperature", "ensemble", "mc_dropout"]
    temperature: TemperatureSection
    ensemble: EnsembleSection
    mc_dropout: McDropoutSection
    metrics: CalibrationMetricsSection

    @model_validator(mode="after")
    def primary_is_among_methods(self) -> CalibrationSection:
        if self.primary not in self.methods:
            raise ValueError(f"primary {self.primary!r} is not in methods {self.methods}")
        return self


# ---------------------------------------------------------------------------
# Phase 6: shift and out-of-distribution detection
# ---------------------------------------------------------------------------

class ShiftLevels(_Section):
    S0: str
    S1: str
    S2: str


class AttributionSection(_Section):
    model: str
    random_intercept: str
    covariates: list[str]
    report_ci: bool


class ShiftSection(_Section):
    levels: ShiftLevels
    attribution: AttributionSection


class MspSection(_Section):
    description: str


class MahalanobisSection(_Section):
    feature_layer: str
    class_conditional: bool
    shrinkage: float


class OodEvaluationSection(_Section):
    report: list[str]


class AbstentionSection(_Section):
    sweep_from: float
    sweep_to: float
    sweep_steps: int = Field(ge=2)
    selection_criterion: str
    max_acceptable_suppression_rate: float = Field(ge=0, le=1)
    chosen_threshold: float | None


class OodSection(_Section):
    methods: list[str]
    primary: str
    msp: MspSection
    mahalanobis: MahalanobisSection
    evaluation: OodEvaluationSection
    abstention: AbstentionSection


# ---------------------------------------------------------------------------
# Phase 7: explanation
# ---------------------------------------------------------------------------

class GradcamSection(_Section):
    target_layer: str
    normalise: bool


class FidelitySection(_Section):
    run_parameter_randomisation: bool
    run_data_randomisation: bool
    similarity_metrics: list[str]
    fail_threshold_rank_corr: float
    deletion_insertion_auc: bool
    deletion_steps: int


class ExplainSection(_Section):
    # guided_gradcam is absent on purpose: it fails Adebayo's parameter randomisation test,
    # and "looks nicer" is precisely the failure mode. D9.
    method: Literal["gradcam", "intrinsic"]
    fallback: Literal["intrinsic"]
    gradcam: GradcamSection
    fidelity: FidelitySection


# ---------------------------------------------------------------------------
# Phase 8: routing
# ---------------------------------------------------------------------------

class PresentWhenSection(_Section):
    min_calibrated_prob: float = Field(ge=0, le=1)
    ood_flag: bool


class EscalateWhenSection(_Section):
    min_epistemic: float = Field(ge=0)


class EvidenceBeforeSuggestionSection(_Section):
    enabled: bool
    require_reveal_action: bool


class RoutingSection(_Section):
    present_when: PresentWhenSection
    escalate_when: EscalateWhenSection
    suppress_removes_value_from_payload: bool
    evidence_before_suggestion: EvidenceBeforeSuggestionSection

    @field_validator("suppress_removes_value_from_payload")
    @classmethod
    def suppression_is_real(cls, value: bool) -> bool:
        # Greying a value out in the interface is not suppression; the value still reaches
        # the client. BUILD-SPEC Phase 8 and §6.
        if not value:
            raise ValueError("suppression must remove the value from the payload entirely")
        return value


# ---------------------------------------------------------------------------
# Phase 2: annotation, and Phase 10: the reliance study
# ---------------------------------------------------------------------------

class IrrSection(_Section):
    categorical: str
    ordinal: str
    continuous: str
    warn_when_not_fully_crossed: bool


class ArtefactChecksSection(_Section):
    restriction_of_range: bool
    rater_role_effects: bool
    familiarity_effects: bool


class AnnotationSection(_Section):
    codebook_version: str
    calibration_rounds: int
    calibration_clips_per_round: int
    calibration_min_teachers: int
    alpha_gate: float = Field(ge=0, le=1)
    exclude_guesses_from_primary_irr: bool
    bootstrap_samples: int
    irr: IrrSection
    artefact_checks: ArtefactChecksSection


class SeededErrorsSection(_Section):
    rate: float = Field(ge=0, le=1)
    corruption_kinds: list[str]
    seed_offset: int


class RelianceStudySection(_Section):
    seeded_errors: SeededErrorsSection
    arms: list[str]
    counterbalance: bool
    target_participants: int
    sessions_per_participant: int


# ---------------------------------------------------------------------------
# Dashboard, export, retention
# ---------------------------------------------------------------------------

class DashboardSection(_Section):
    min_cell_size: int = Field(ge=1)
    show_confidence_bands: bool
    mark_suppressed_sessions: bool


class ExportSection(_Section):
    require_adjudication: bool
    format: str
    include_evidence_timestamps: bool

    @field_validator("require_adjudication")
    @classmethod
    def nothing_unreviewed_leaves(cls, value: bool) -> bool:
        if not value:
            raise ValueError("no unadjudicated model output may leave the system")
        return value


class RetentionSection(_Section):
    blurred_media_years: int
    pose_artifacts_years: int
    audit_log: str
    run_nightly_at: str
    dry_run: bool


# ---------------------------------------------------------------------------
# The whole
# ---------------------------------------------------------------------------

class PraxisConfig(_Section):
    """The whole of a run's parameters. Frozen: a run may not edit its own configuration."""

    run: RunSection
    compute: ComputeSection
    paths: PathsSection
    tools: ToolsSection
    ingest: IngestSection
    preprocess: PreprocessSection
    splits: SplitsSection
    behaviour: BehaviourSection
    calibration: CalibrationSection
    shift: ShiftSection
    ood: OodSection
    explain: ExplainSection
    routing: RoutingSection
    annotation: AnnotationSection
    reliance_study: RelianceStudySection
    dashboard: DashboardSection
    export: ExportSection
    retention: RetentionSection
    locked: list[str]

    @model_validator(mode="after")
    def sampling_rates_agree(self) -> PraxisConfig:
        # The model must see the video at the rate the codebook defines a clip at. A mismatch
        # here would silently misalign every annotation against every prediction.
        if self.preprocess.sample_fps != self.behaviour.clip.fps:
            raise ValueError(
                f"preprocess.sample_fps={self.preprocess.sample_fps} must equal "
                f"behaviour.clip.fps={self.behaviour.clip.fps}")
        return self

    @model_validator(mode="after")
    def the_duration_floor_admits_enough_clips(self) -> PraxisConfig:
        """The gate's shortest session must still yield `MIN_USEFUL_CLIPS` whole clips.

        The floor used to be five minutes, which was an assumption about how long a practicum
        recording would be rather than a requirement of anything downstream. Derived instead
        from the clip length, it says what it is actually for: a recording too short to contain
        a few complete behaviour episodes cannot be annotated or scored, whatever its length in
        seconds happens to be. Tying the two together also stops the clip length being changed
        later while the duration floor silently keeps the old arithmetic. D81.
        """
        required = MIN_USEFUL_CLIPS * self.behaviour.clip.length_s
        if self.ingest.quality_gate.min_duration_s < required:
            raise ValueError(
                f"ingest.quality_gate.min_duration_s={self.ingest.quality_gate.min_duration_s} "
                f"admits a session shorter than {MIN_USEFUL_CLIPS} clips of "
                f"{self.behaviour.clip.length_s}s, which is {required}s. A session that yields "
                f"fewer whole clips than that cannot carry a behaviour episode.")
        return self

    def require_media_root(self) -> Path:
        """The external media volume, or a clean refusal naming what is missing."""
        if self.paths.media_root is None:
            raise MediaRootUnavailable(
                "PRAXIS_MEDIA_ROOT is unset. Video lives on the external SSD and has no "
                "fallback: see D17. Phases 0, 2, 5 and 6 do not need it.")
        if not self.paths.media_root.exists():
            raise MediaRootUnavailable(
                f"media volume {self.paths.media_root} is not reachable; it is probably "
                f"unmounted. Reconnect it or run a phase that does not need video.")
        return self.paths.media_root
