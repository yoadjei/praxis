# -*- coding: utf-8 -*-
"""Fetch the model weights. Once, deliberately, by an operator. Never at runtime.

Two sets are vendored here: the YOLOv8 pose export used by Phase 3, and the frozen ResNet-50
backbone used by Phase 4.

R6 says no network access during inference. The usual way that invariant dies is not a
`requests` call somebody wrote; it is a library fetching a checkpoint on first use, inside a
function that looks local. This project has already found three. `ultralytics.trackers`'
`byte_tracker` resolves DNS and pip-installs its extras during the *import* (D57). The
environment variable the invariant test asserted turned out not to exist in ultralytics (D56).
And `torchvision.models.resnet50(weights=...)` downloads on a cache miss while honouring
neither `YOLO_OFFLINE` nor `HF_HUB_OFFLINE`, which `prepare_environment` sets - the two
variables that cover the other libraries do not cover this one (D72).

So the downloads live here, in a script that is never imported by `praxis`, and the runtime
paths (`OnnxPoseEstimator`, `FrozenBackbone`) only ever open a file that is already present. If
it is absent they raise `WeightsMissing` and name this script. They do not fetch it. D58.

    python scripts/vendor_weights.py                   # fetch, verify, install both
    python scripts/vendor_weights.py --only backbone   # just one of them
    python scripts/vendor_weights.py --verify-only     # check what is already vendored
    python scripts/vendor_weights.py --dry-run         # show what it would do

This needs a networked machine, with ultralytics installed for the pose export and torchvision
for the backbone. On an air-gapped machine, run it elsewhere and copy the resulting files and
their `.sha256` sidecars into the configured directory.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.config import load_config, prepare_environment
from praxis.preprocess.pose import (
    CPU_ONLY_PROVIDERS,
    KEYPOINT_COUNT,
    POSE_INPUT_SIZE,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DIGEST_SUFFIX = ".sha256"


def digest_of(path: Path) -> str:
    """Streaming, because a pose export is tens of megabytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(weights: Path) -> tuple[bool, str]:
    """Is what is on disk a usable 17-keypoint pose export, and does it match its digest?

    Loaded through onnxruntime with the providers pinned, which is the same check the runtime
    performs, so a file that passes here cannot fail there for a reason this script could have
    caught.
    """
    if not weights.is_file():
        return False, f"no file at {weights}"

    recorded = weights.with_suffix(weights.suffix + DIGEST_SUFFIX)
    if recorded.is_file():
        expected = recorded.read_text(encoding="utf-8").split()[0]
        actual = digest_of(weights)
        if actual != expected:
            return False, (
                f"{weights.name} hashes to {actual[:12]} and {recorded.name} records "
                f"{expected[:12]}; the vendored file is not the one that was exported")

    try:
        import onnxruntime as ort
    except ImportError:
        return False, "onnxruntime is not installed, so the export cannot be verified"

    try:
        session = ort.InferenceSession(str(weights), providers=list(CPU_ONLY_PROVIDERS))
    except Exception as exc:
        return False, f"onnxruntime refused {weights.name}: {exc}"

    shape = session.get_outputs()[0].shape
    expected_channels = 5 + KEYPOINT_COUNT * 3
    if len(shape) != 3 or (isinstance(shape[1], int) and shape[1] != expected_channels):
        return False, (
            f"{weights.name} outputs {shape}; a {KEYPOINT_COUNT}-keypoint pose model emits "
            f"{expected_channels} channels. This is the wrong export.")

    return True, f"{weights.name} is a valid {KEYPOINT_COUNT}-keypoint pose export"


def export(model_name: str, destination: Path, image_size: int) -> Path:
    """Download the checkpoint and export it to ONNX, in a scratch directory.

    Ultralytics is imported here and nowhere in `praxis`. It writes the checkpoint beside the
    working directory by default, which is why this runs somewhere disposable: a stray
    `yolov8m-pose.pt` next to the source tree is exactly the file that turns a later
    `YOLO("yolov8m-pose")` into a silent local load and hides a missing vendoring step.
    """
    from ultralytics import YOLO

    scratch = Path(tempfile.mkdtemp(prefix="praxis_vendor_"))
    previous = Path.cwd()
    try:
        os.chdir(scratch)
        model = YOLO(f"{model_name}.pt")
        produced = Path(model.export(format="onnx", imgsz=image_size, simplify=True))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, destination)
    finally:
        os.chdir(previous)
        shutil.rmtree(scratch, ignore_errors=True)

    recorded = destination.with_suffix(destination.suffix + DIGEST_SUFFIX)
    recorded.write_text(f"{digest_of(destination)}  {destination.name}\n", encoding="utf-8")
    return destination


def verify_backbone(weights: Path, arch: str, output_dim: int) -> tuple[bool, str]:
    """Is what is on disk the state dict the frozen backbone will be built from?

    The check that matters is the one the runtime performs: construct the architecture with
    `weights=None` and load this file into it strictly. A state dict that passes here cannot
    fail there, and a near-miss - the right architecture at a different depth, or a checkpoint
    saved from a fine-tuned model with the classifier replaced - fails on key names rather
    than surviving to produce quietly wrong features.
    """
    if not weights.is_file():
        return False, f"no file at {weights}"

    recorded = weights.with_suffix(weights.suffix + DIGEST_SUFFIX)
    if recorded.is_file():
        expected = recorded.read_text(encoding="utf-8").split()[0]
        actual = digest_of(weights)
        if actual != expected:
            return False, (
                f"{weights.name} hashes to {actual[:12]} and {recorded.name} records "
                f"{expected[:12]}; the vendored file is not the one that was fetched")

    try:
        import torch
        from torchvision.models import get_model
    except ImportError:
        return False, "torch and torchvision are not installed, so the file cannot be verified"

    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as exc:
        return False, f"torch refused {weights.name}: {exc}"

    try:
        # weights=None is the whole point: it builds the architecture without consulting the
        # network or the torch hub cache.
        model = get_model(arch, weights=None)
        model.load_state_dict(state)
    except Exception as exc:
        return False, f"{weights.name} does not load into {arch}: {exc}"

    features = getattr(model, "fc", None)
    if features is None or features.in_features != output_dim:
        got = None if features is None else features.in_features
        return False, (
            f"{arch} exposes a {got}-dimensional penultimate layer and the config declares "
            f"output_dim {output_dim}. The cached features would be the wrong width.")

    return True, f"{weights.name} loads into {arch} and is {output_dim}-dimensional"


def fetch_backbone(arch: str, weights_name: str, destination: Path) -> Path:
    """Download the pretrained classifier and save its state dict locally.

    torchvision's own `load_state_dict_from_url` verifies the SHA-256 prefix that upstream
    embeds in the filename, so the integrity check against the publisher happens during the
    download rather than being asserted afterwards against a hash this script made up.

    The enum member is looked up by name and never through `DEFAULT`. `DEFAULT` is an alias
    that has already moved once for ResNet-50, from the original recipe to IMAGENET1K_V2, and
    an alias resolved at download time would put weights on disk that the config does not
    identify. `BackboneSection.weights_name_a_version` refuses it on the config side; this is
    the same rule on the fetching side.
    """
    import torch
    from torchvision.models import get_model_weights

    available = get_model_weights(arch)
    try:
        weights = available[weights_name]
    except KeyError as exc:
        known = ", ".join(member.name for member in available)
        raise SystemExit(
            f"{arch} has no weights called {weights_name!r}. Available: {known}") from exc

    state = weights.get_state_dict(progress=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)

    recorded = destination.with_suffix(destination.suffix + DIGEST_SUFFIX)
    recorded.write_text(f"{digest_of(destination)}  {destination.name}\n", encoding="utf-8")
    return destination


def vendor_pose(config, *, verify_only: bool, dry_run: bool, force: bool) -> int:
    weights = Path(config.paths.model_weights) / config.preprocess.pose.weights_file
    model_name = config.preprocess.pose.model

    print(f"model      {model_name}")
    print(f"target     {weights}")
    print(f"imgsz      {POSE_INPUT_SIZE}   (must match the letterbox OnnxPoseEstimator uses)")

    ok, detail = verify(weights)
    print(f"vendored   {detail}")

    if verify_only:
        return 0 if ok else 1
    if ok and not force:
        print("nothing to do; pass --force to re-export")
        return 0
    if dry_run:
        print(f"would download {model_name}.pt and export it to {weights}")
        return 0

    try:
        export(model_name, weights, POSE_INPUT_SIZE)
    except ImportError:
        print("ultralytics is not installed. Install it on a networked machine, run this "
              "script there, and copy the .onnx and .sha256 files to the target above.",
              file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 2

    ok, detail = verify(weights)
    print(f"exported   {detail}")
    return 0 if ok else 1


def vendor_backbone(config, *, verify_only: bool, dry_run: bool, force: bool) -> int:
    backbone = config.behaviour.backbone
    weights = Path(config.paths.model_weights) / backbone.weights_file

    print(f"model      {backbone.arch} {backbone.weights}")
    print(f"target     {weights}")
    print(f"output_dim {backbone.output_dim}")

    ok, detail = verify_backbone(weights, backbone.arch, backbone.output_dim)
    print(f"vendored   {detail}")

    if verify_only:
        return 0 if ok else 1
    if ok and not force:
        print("nothing to do; pass --force to re-fetch")
        return 0
    if dry_run:
        print(f"would download {backbone.arch} {backbone.weights} and save it to {weights}")
        return 0

    try:
        fetch_backbone(backbone.arch, backbone.weights, weights)
    except ImportError:
        print("torchvision is not installed. Install it on a networked machine, run this "
              "script there, and copy the .pt and .sha256 files to the target above.",
              file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 2

    ok, detail = verify_backbone(weights, backbone.arch, backbone.output_dim)
    print(f"fetched    {detail}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", choices=("pose", "backbone"), default=None,
                        help="vendor just one of the two; default is both")
    parser.add_argument("--verify-only", action="store_true",
                        help="check the vendored files without fetching anything")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be fetched and where it would go")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even if a valid file is already vendored")
    args = parser.parse_args()

    config = load_config(args.config)
    # D17: nothing this project generates lands on the system volume. torch.hub writes its
    # checkpoint cache under TORCH_HOME, which defaults to ~/.cache/torch on C:, and it is the
    # download here rather than anything at runtime that puts a hundred megabytes there. This
    # script fetches, so this script redirects.
    prepare_environment(config)
    options = {"verify_only": args.verify_only, "dry_run": args.dry_run, "force": args.force}

    status = 0
    for name, vendor in (("pose", vendor_pose), ("backbone", vendor_backbone)):
        if args.only not in (None, name):
            continue
        print(f"--- {name} ---")
        # Both are attempted even when the first fails, so one run reports the whole picture
        # rather than making the operator discover the second problem after fixing the first.
        status = max(status, vendor(config, **options))
        print()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
