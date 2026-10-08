# -*- coding: utf-8 -*-
"""Stage A: turn an annotated clip into cached backbone activations, and read them back.

D2 freezes the backbone, so a clip's features are the same in every epoch and in every run. They
are therefore computed once and stored, which is the only arrangement that fits 64 frames of
ResNet-50 into the 1.3 GB this estate has. Stage B then trains the temporal head on tensors.

**Every pixel operation has to happen here**, because by Stage B the pixels are gone. That is
why `dataset.policy_from_config` refuses `colour_jitter.applied` and why the flip is a cached
variant rather than a transform: a 2048-dimensional activation cannot be flipped or recoloured
after the fact. `AugmentationPolicy.variants` is the list of what this module writes.

**A cache is invisible when it is stale.** It loads, it trains, the numbers come out, and they
describe a backbone or a pose artefact that is no longer the one in use. So every entry carries
the backbone version, the pose artefact's hash and the config digest, and `read_clip` refuses a
mismatch rather than warning about it. R7 rests on that refusal.

Nothing here touches the database. It takes a pose artefact, a video and a backbone, which is
what makes it testable without a corpus; `scripts/build_features.py` does the lookups.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from praxis.annotation.clips import ClipRef
from praxis.behaviour.dataset import HFLIP, IDENTITY
from praxis.preprocess.artifacts import LoadedPose
from praxis.preprocess.candidates import box_at
from praxis.preprocess.frames import rgb_at

FEATURES = "features"
KEYPOINTS = "keypoints.npy"
PROVENANCE = "provenance.json"

# A clip where the teacher is tracked in fewer than half its frames is refused. The crops for
# the untracked frames are carried from the nearest frame that did locate them, which is honest
# for a short gap - the person has not moved far in an eighth of a second - and becomes a
# fabrication once it is most of the clip. Half is a judgement, not a measurement; it is here
# rather than in the config because it is a property of what the crop means, and a config key
# would invite it to be lowered when a corpus came out short.
MIN_LOCATED_FRACTION = 0.5


class FeatureCacheError(RuntimeError):
    """The cache cannot be written or cannot be trusted, with the reason named."""


def cache_digest(config) -> str:
    """A hash of every setting that changes what comes out of Stage A.

    Narrower than the whole config on purpose. A run whose learning rate differs produced the
    same features, and hashing the whole file would invalidate the cache on every unrelated
    edit - which trains people to rebuild without reading why, and then to stop reading at all.

    `clip` and `backbone` are the two sections that decide the pixels and the activation.
    `crop_size`, `frames` and `crop_padding` appear as their own provenance fields as well,
    because a mismatch in one of those should say which rather than only that the hash moved.
    """
    import hashlib
    import json

    settings = {"clip": config.behaviour.clip.model_dump(),
                "backbone": config.behaviour.backbone.model_dump()}
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class CacheProvenance:
    """What a cached clip was built from. Compared on read, never assumed."""

    backbone_version: str
    pose_sha256: str
    config_sha256: str
    feature_dim: int
    frames: int
    crop_size: int
    crop_pad: float
    track_id: int
    frames_located: int

    @property
    def located_fraction(self) -> float:
        return self.frames_located / self.frames if self.frames else 0.0

    def disagreements(self, other: CacheProvenance) -> list[str]:
        """Every field that differs, so a stale cache is diagnosed in one read.

        `frames_located` is excluded: it is a measurement of the clip rather than a setting, so
        a caller has nothing to compare it against.
        """
        ignore = {"frames_located"}
        mine, theirs = asdict(self), asdict(other)
        return [f"{name}: cached {mine[name]!r}, expected {theirs[name]!r}"
                for name in sorted(mine)
                if name not in ignore and mine[name] != theirs[name]]


@dataclass(frozen=True)
class CachedClip:
    """One clip's cached tensors, as they come off disk."""

    clip_id: str
    features: dict[str, np.ndarray]          # variant -> (frames, feature_dim)
    keypoints: np.ndarray                    # (frames, 17, 3)
    provenance: CacheProvenance


def clip_directory(root: Path, clip_id: str) -> Path:
    """Where one clip's tensors live.

    Split on the colon so a session's clips sit together under one directory. A flat layout of
    `session:00001.npy` works until a filesystem is asked for ten thousand entries in one
    directory, and the session is the unit anything would ever clear.
    """
    session_id, _, index = clip_id.rpartition(":")
    if not session_id or not index:
        raise FeatureCacheError(f"{clip_id!r} is not a clip id")
    return root / session_id / index


def sample_times(clip: ClipRef, frames: int) -> np.ndarray:
    """The timestamps Stage A decodes, as segment midpoints rather than endpoints.

    D95: the last endpoint of a window sits exactly on its boundary, and a seek there decodes
    the next clip's first frame or nothing at all. Midpoints keep every sample inside the clip
    it is labelled as, which is also what makes the first and last frames symmetrical.
    """
    if frames < 1:
        raise FeatureCacheError(f"a clip of {frames} frames is not a clip")
    span = clip.end_seconds - clip.start_seconds
    if span <= 0:
        raise FeatureCacheError(
            f"{clip.clip_id} runs from {clip.start_seconds} to {clip.end_seconds}, which is "
            f"not a window any frame can be sampled from")
    return clip.start_seconds + (np.arange(frames) + 0.5) * span / frames


def pose_frames_at(times: np.ndarray, pose: LoadedPose) -> np.ndarray:
    """The artefact's frame index nearest each timestamp, clamped to what it holds.

    The pose artefact is sampled at its own rate, which is not the clip's. Clamping rather than
    refusing out-of-range: a clip ending within one sample of the end of the video is ordinary,
    and the alternative is dropping its last frame for an arithmetic reason nobody would guess.
    """
    sampled_fps = float(pose.provenance.get("sampled_fps", 0.0))
    if sampled_fps <= 0:
        raise FeatureCacheError(
            "the pose artefact records no sample rate, so no timestamp can be mapped to a "
            "frame of it")
    total = int(pose.track_ids.shape[0])
    return np.clip(np.rint(times * sampled_fps).astype(int), 0, max(total - 1, 0))


def located_boxes(pose: LoadedPose, track_id: int, frames: np.ndarray,
                  *, pad: float) -> tuple[list[tuple[int, int, int, int]], int]:
    """A crop rectangle for every frame, and how many of them the tracker actually produced.

    A frame where the track is absent takes the rectangle from the nearest frame that has one.
    The alternative - a black crop - puts a signal in the training data that the room never
    produced, and the model would learn it as whatever the label happened to be. Carrying a box
    assumes only that the person has not moved far, which over an eighth of a second holds.

    The count comes back so the caller can refuse a clip that is mostly carried, and so the
    number is recorded rather than inferred later from nothing.
    """
    boxes = [box_at(pose, track_id, int(frame), pad=pad) for frame in frames]
    present = [index for index, box in enumerate(boxes) if box is not None]
    if not present:
        raise FeatureCacheError(
            f"track {track_id} appears in none of the {len(boxes)} frames of this clip, so "
            f"there is no teacher in it to crop. A behaviour label describes a person, and a "
            f"clip with nobody in it has nothing for the label to be true of (R1).")

    filled = [box if box is not None
              else boxes[min(present, key=lambda p: abs(p - index))]
              for index, box in enumerate(boxes)]
    return filled, len(present)


def crops_for(ffmpeg, video: Path, times: np.ndarray,
              boxes: list[tuple[int, int, int, int]], *, size: int) -> np.ndarray:
    """`(frames, size, size, 3)` uint8, one teacher crop per timestamp.

    One decode per frame. A single pass with a filter graph would be faster and cannot express
    this: the rectangle moves between frames, so each needs its own crop.
    """
    if len(times) != len(boxes):
        raise FeatureCacheError(f"{len(times)} timestamps against {len(boxes)} boxes")
    return np.stack([
        rgb_at(ffmpeg, video, at_seconds=float(at), width=size, height=size, box=box)
        for at, box in zip(times, boxes, strict=True)])


def embed(backbone, crops: np.ndarray, *, batch_frames: int, dtype: str) -> np.ndarray:
    """`(frames, feature_dim)` activations, in batches the configured memory allows.

    `behaviour.backbone.stage_a_batch_frames` exists because inference holds the whole batch
    resident: measured at 429 MB RSS for 32 frames against 378 MB for 8, with throughput flat
    between them, so a small batch costs nothing and a whole clip at once costs the headroom.
    """
    import torch

    if batch_frames < 1:
        raise FeatureCacheError(f"a batch of {batch_frames} frames cannot be embedded")

    out = []
    for start in range(0, len(crops), batch_frames):
        chunk = crops[start:start + batch_frames]
        # (N, H, W, 3) uint8 to the (N, 3, H, W) float the backbone normalises itself.
        tensor = torch.from_numpy(np.ascontiguousarray(chunk)).permute(0, 3, 1, 2)
        with torch.no_grad():
            out.append(backbone.embed(tensor).cpu().numpy())
    return np.concatenate(out).astype(dtype)


def build_clip(clip: ClipRef, *, pose: LoadedPose, video: Path, backbone, ffmpeg,
               track_id: int, frames: int, crop_size: int, crop_pad: float,
               variants: tuple[str, ...], batch_frames: int, dtype: str,
               config_sha256: str,
               min_located: float = MIN_LOCATED_FRACTION) -> CachedClip:
    """Everything Stage A does to one clip, as one call.

    `variants` comes from `AugmentationPolicy.variants` and decides what is written. The
    identity crop is embedded once and the flip is embedded from the same pixels, so the two
    cannot drift apart through a second decode.
    """
    times = sample_times(clip, frames)
    boxes, found = located_boxes(pose, track_id, pose_frames_at(times, pose), pad=crop_pad)

    if found < min_located * frames:
        raise FeatureCacheError(
            f"{clip.clip_id}: track {track_id} was located in {found} of {frames} frames, "
            f"below the {min_located:.0%} floor. The rest would be carried from elsewhere in "
            f"the clip, and past this point that is a fabricated crop rather than a short gap.")

    pixels = crops_for(ffmpeg, video, times, boxes, size=crop_size)

    features: dict[str, np.ndarray] = {}
    for variant in variants:
        if variant == IDENTITY:
            features[variant] = embed(backbone, pixels, batch_frames=batch_frames, dtype=dtype)
        elif variant == HFLIP:
            flipped = np.ascontiguousarray(pixels[:, :, ::-1, :])
            features[variant] = embed(backbone, flipped, batch_frames=batch_frames, dtype=dtype)
        else:
            raise FeatureCacheError(
                f"{clip.clip_id}: no Stage A path builds the {variant!r} variant. "
                f"`AugmentationPolicy.variants` asked for it, so either the policy gained a "
                f"variant this module was not taught or the config enabled one that is "
                f"documented as unbuilt (see D75 on colour jitter).")

    keypoints = pose.track(track_id)[pose_frames_at(times, pose)].astype("float32")

    return CachedClip(
        clip_id=clip.clip_id,
        features=features,
        keypoints=keypoints,
        provenance=CacheProvenance(
            backbone_version=backbone.model_version,
            pose_sha256=str(pose.provenance.get("sha256", "")),
            config_sha256=config_sha256,
            feature_dim=int(next(iter(features.values())).shape[1]),
            frames=frames, crop_size=crop_size, crop_pad=crop_pad,
            track_id=int(track_id), frames_located=int(found)),
    )


def write_clip(root: Path, cached: CachedClip) -> Path:
    """One directory per clip: a tensor per variant, the keypoints, and the provenance."""
    directory = clip_directory(root, cached.clip_id)
    directory.mkdir(parents=True, exist_ok=True)

    for variant, tensor in cached.features.items():
        np.save(directory / f"{FEATURES}.{variant}.npy", tensor)
    np.save(directory / KEYPOINTS, cached.keypoints)

    # Written last. A reader that finds the provenance can rely on the tensors beside it being
    # complete; the reverse order would leave a half-written clip looking finished.
    (directory / PROVENANCE).write_text(
        json.dumps(asdict(cached.provenance), indent=2, sort_keys=True), encoding="utf-8")
    return directory


def is_cached(root: Path, clip_id: str) -> bool:
    """Whether a complete entry exists, which is the provenance being present."""
    return (clip_directory(root, clip_id) / PROVENANCE).is_file()


def read_clip(root: Path, clip_id: str, *, variants: tuple[str, ...],
              expect: CacheProvenance | None = None) -> CachedClip:
    """Read one clip back, refusing a cache that was built under different settings.

    `expect` is compared field by field and a difference raises. Training on a cache built by a
    different backbone produces a model whose features nothing downstream can reproduce, and it
    does so silently: the shapes agree, the loss falls, and the result describes a system that
    no longer exists.
    """
    directory = clip_directory(root, clip_id)
    provenance_file = directory / PROVENANCE
    if not provenance_file.is_file():
        raise FeatureCacheError(
            f"no cached features for {clip_id} under {root}. Run "
            f"`python scripts/build_features.py --write` to build Stage A.")

    provenance = CacheProvenance(**json.loads(provenance_file.read_text(encoding="utf-8")))
    if expect is not None:
        differs = provenance.disagreements(expect)
        if differs:
            raise FeatureCacheError(
                f"{clip_id} was cached under settings that no longer hold:\n  - "
                + "\n  - ".join(differs)
                + "\nRebuild the cache rather than training on it; the shapes would agree and "
                  "the numbers would describe the old settings.")

    features = {}
    for variant in variants:
        path = directory / f"{FEATURES}.{variant}.npy"
        if not path.is_file():
            raise FeatureCacheError(
                f"{clip_id} has no {variant!r} variant cached. The augmentation policy asks "
                f"for it, so the cache predates the policy and is incomplete rather than "
                f"merely old.")
        features[variant] = np.load(path)

    return CachedClip(clip_id=clip_id, features=features,
                      keypoints=np.load(directory / KEYPOINTS), provenance=provenance)
