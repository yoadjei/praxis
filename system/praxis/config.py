# -*- coding: utf-8 -*-
"""Loading a configuration, and the three run-level decisions that must live in one place.

Rule R7 says a run is reproducible from one config file plus the data. That only holds if
nothing affecting a result is decided anywhere else, so three things live here and are
forbidden elsewhere:

* **Device selection.** BUILD-SPEC §2.1 puts it here "and nowhere else", so that a run moves
  between the laptop, Kaggle and Colab without touching a model, a loss, or a metric.
* **Seeding.** Python, NumPy, torch, and the DataLoader workers, from one master seed.
* **Environment guards.** ``TMPDIR`` and ``TORCH_HOME`` are redirected onto the working
  volume, and a run refuses to start when the system volume is too full to finish (D17).

The section models are declarations and live in `config_schema.py`. They are re-exported here
so that `from praxis.config import PraxisConfig` keeps working and callers never need to know
which of the two modules a name came from.

torch is imported inside the functions that need it. Importing it at module scope would make
every module that merely reads a path pay for it, and Phase 2 and Phase 5 are arithmetic over
NumPy that has no reason to load a deep learning framework.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from praxis.config_schema import (
    ConfigError,
    Device,
    MediaRootUnavailable,
    PraxisConfig,
)

__all__ = [
    "ConfigError",
    "Device",
    "MediaRootUnavailable",
    "PraxisConfig",
    "Settings",
    "config_sha256",
    "load_config",
    "prepare_environment",
    "resolve_device",
    "seed_everything",
    "worker_init_fn",
]

# ${VAR}, and nothing cleverer. A config that needs shell semantics is a config that has
# stopped being a record of what a run did.
_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_UNSET = "\x00unset\x00"


class Settings(BaseSettings):
    """Environment overrides. Only what genuinely varies between machines belongs here."""

    model_config = SettingsConfigDict(env_prefix="PRAXIS_", extra="ignore")

    device: Device | None = None
    media_root: Path | None = None
    config: Path = Path("configs/default.yaml")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def config_sha256(path: str | Path) -> str:
    """SHA-256 of the config file's bytes.

    Of the bytes, not of the parsed structure: the frozen copy in the run directory is what a
    reader will check, and it is a file.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expand(value: Any) -> Any:
    """Substitute ${VAR} from the environment, recursively. Unset becomes a sentinel."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), _UNSET), value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _drop_unset(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn every unresolved ${VAR} into None so the schema can mark the field optional.

    Walks the whole tree rather than just `paths`. It began as a rule about the media root and
    then `ingest.manifest` needed the same treatment; two spellings of one rule is how the
    second one silently keeps a literal "${PRAXIS_MANIFEST}" and reports that the file does not
    exist (L20). A field that is genuinely required still fails validation, and the message
    names it.
    """
    for key, value in list(raw.items()):
        if isinstance(value, dict):
            _drop_unset(value)
        elif isinstance(value, str) and _UNSET in value:
            raw[key] = None
    return raw


def load_config(path: str | Path | None = None,
                settings: Settings | None = None) -> PraxisConfig:
    """Read, expand, validate. The only way a configuration enters the system."""
    settings = settings or Settings()
    source = Path(path) if path is not None else settings.config
    if not source.exists():
        raise ConfigError(f"no configuration at {source}")

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} does not contain a mapping")

    raw = _drop_unset(_expand(raw))

    # Environment wins over the file, and only for what genuinely varies by machine.
    if settings.device is not None:
        raw["run"]["device"] = settings.device
    if settings.media_root is not None:
        raw["paths"]["media_root"] = str(settings.media_root)

    try:
        return PraxisConfig.model_validate(raw)
    except Exception as exc:                    # pydantic's message names the offending key
        raise ConfigError(f"{source} is not a valid PRAXIS configuration:\n{exc}") from exc


# ---------------------------------------------------------------------------
# device, seeding, environment. BUILD-SPEC §2.1 confines these to this module.
# ---------------------------------------------------------------------------

def resolve_device(requested: Device) -> tuple[Device, str | None]:
    """The device actually used, and the model of it.

    Phase 11 requires automatic CPU fallback where CUDA is absent, which is the normal case
    locally. The fallback is returned rather than hidden, so the run manifest records the
    device the numbers came from and not the one that was asked for. D14 makes the device
    *model* part of that record too, because a T4 and an L4 are not interchangeable within one
    experimental comparison.
    """
    import torch

    if requested == "cuda" and torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", _cpu_model()


def _cpu_model() -> str | None:
    import platform
    return platform.processor() or None


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every generator a result can depend on.

    R7 is bit-exact on CPU and tolerance-only on CUDA, so ``deterministic`` turns on the
    strict kernels rather than merely fixing the seeds. CUBLAS_WORKSPACE_CONFIG has to be set
    before the first CUDA context, which is why it is set here and not at the call site.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker from torch's per-worker seed.

    Windows spawns workers as fresh processes, so each one re-imports and re-seeds. Deriving
    from ``initial_seed`` keeps the workers distinct from each other and reproducible across
    runs, which seeding them all identically would not.
    """
    import numpy as np
    import torch

    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2 ** 32)


def prepare_environment(config: PraxisConfig) -> dict[str, str]:
    """Point the caches at the working volume and refuse to start on a full system drive.

    Returns what was changed, so the run manifest records it. D17: nothing this project
    generates belongs on the system volume, and a run that fills it fails halfway through
    rather than at the start, which is the expensive way to find out.
    """
    changed: dict[str, str] = {}
    for variable, target in (("TMPDIR", config.paths.scratch),
                             ("TEMP", config.paths.scratch),
                             ("TMP", config.paths.scratch),
                             ("TORCH_HOME", config.paths.torch_home)):
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
        changed[variable] = str(target)

    # Vendored weights only. Ultralytics otherwise fetches on first use, which R6 forbids.
    #
    # The variable it reads is YOLO_OFFLINE. `ULTRALYTICS_OFFLINE` appears nowhere in the
    # package and was setting nothing; ultralytics gates every network path on `ONLINE`, which
    # `is_online()` computes from a live DNS probe unless YOLO_OFFLINE is set. Verified both
    # ways against the installed version. D56.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    system_drive = Path(os.environ.get("SYSTEMDRIVE", "C:") + os.sep)
    free_gb = shutil.disk_usage(system_drive).free / 1024 ** 3
    if free_gb < config.paths.min_free_gb_system_drive:
        raise ConfigError(
            f"{system_drive} has {free_gb:.1f} GB free, below the "
            f"{config.paths.min_free_gb_system_drive} GB floor. Derived data belongs on the "
            f"working volume (D17); free space before starting a run.")
    return changed
