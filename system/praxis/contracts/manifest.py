# -*- coding: utf-8 -*-
"""Manifests: how a split is recorded, and how a run is made reproducible.

`SplitManifest` enforces R2. Teacher-disjointness is asserted in code and refused by a database
constraint, never left to convention, because a contaminated split invalidates every number
downstream of it and shows up in no metric.

`RunManifest` is what R7 rests on. It records the config hash, every seed, and — since the
estate now spans a laptop, Kaggle and Colab — the device *and the model of the device* a run
executed on. Artefact checksums are part of it because Kaggle and Colab sessions are ephemeral:
the machine that produced a weight file will not exist again, so the file has to carry proof of
what it is.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from praxis.ids import is_ulid
from praxis.vocabulary import ShiftAxis

Device = Literal["cpu", "cuda"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SplitManifest(BaseModel):
    """A teacher-disjoint partition of the corpus, frozen before any modelling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_id: str
    created_at: datetime
    config_sha256: str
    train_teachers: list[str]
    val_teachers: list[str]
    test_teachers: list[str]
    shift_axis: ShiftAxis = Field(
        default="domain",
        description=(
            "which ladder this manifest was built for. 'domain': S1 is a held-out college's "
            "microteaching and S2 is all classroom footage. 'site': S1 is a held-out basic "
            "school and S2 a held-out college, which is what a practicum-throughout corpus "
            "needs. Stored because S2 names a different set of sessions on each axis, so a "
            "manifest that does not say which it used cannot be read. Defaults to the older "
            "axis, so manifests written before this field read back as themselves. D98."))
    heldout_college: str | None = Field(
        default=None,
        description=(
            "the college withheld entirely, teachers and schools alike, which is S2. The name "
            "and the meaning are unchanged from the microteaching design; only the rung it "
            "labels moved, because a practicum-throughout corpus has no held-out domain to use "
            "as a level. None disables S2. D98."))
    heldout_school: str | None = Field(
        default=None,
        description=(
            "the basic school withheld entirely, its college still appearing in training, "
            "which is S1. Crossed with heldout_college rather than nested under it: one school "
            "hosts students from several colleges. None disables S1. D98."))
    shift_eval_sessions: list[str] = Field(
        default_factory=list, description="held out at a shift level, never trained on")
    classroom_teachers_trained: bool = Field(
        default=False,
        description=(
            "whether teachers with classroom sessions were eligible for train and val. False "
            "is the strict rule of D73, under which the S0-to-S2 drop is measured on a model "
            "that never saw a classroom teacher. True weakens that and the drop means "
            "something narrower; it is recorded here so a reader of the artefact cannot "
            "mistake one for the other. D83. **Vacuous for a manifest that names a held-out "
            "school or college:** there the shift is the site, every teacher of a held-out "
            "session is excluded from training outright, and this flag relaxes nothing. It is "
            "kept, unchanged in meaning, because three stored manifests carry it and "
            "split_manifests is append-only. D98."))

    @field_validator("config_sha256")
    @classmethod
    def hash_is_lowercase_hex(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("config_sha256 must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def teachers_are_disjoint(self) -> SplitManifest:
        """R2. Raise, never warn.

        The three pairwise overlaps are reported together rather than on first failure, so a
        contaminated manifest is diagnosed in one pass instead of three.
        """
        partitions = {
            "train": set(self.train_teachers),
            "val": set(self.val_teachers),
            "test": set(self.test_teachers),
        }
        problems: list[str] = []
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = partitions[left] & partitions[right]
            if shared:
                problems.append(f"{left} and {right} share {sorted(shared)}")
        if problems:
            raise ValueError(
                "R2 violated, teachers appear in more than one partition: "
                + "; ".join(problems))

        for name, teachers in partitions.items():
            if len(teachers) != len(getattr(self, f"{name}_teachers")):
                raise ValueError(f"{name}_teachers contains duplicates")
        return self

    @model_validator(mode="after")
    def the_axis_names_the_sites_it_holds_out(self) -> SplitManifest:
        """Mirrors `a_shift_axis_names_the_sites_it_holds_out`, the CHECK of the same rule.

        Both layers carry it for the reason R2 does: the contract catches it where the manifest
        is built, with the teacher named, and the database catches a writer that bypassed the
        contract.
        """
        if self.shift_axis == "site" and not (self.heldout_school or self.heldout_college):
            raise ValueError(
                "shift_axis 'site' names no held-out school and no held-out college, so S1 and "
                "S2 are both empty. An empty level reads as 'no shift was found' rather "
                "than as "
                "'no shift was measured', which is the stronger and truthful statement.")
        if self.shift_axis == "domain" and self.heldout_school is not None:
            raise ValueError(
                f"shift_axis 'domain' carries heldout_school {self.heldout_school!r}, which "
                f"has "
                f"no rung on that ladder - S1 there is a held-out college's microteaching. "
                f"Storing it would suggest a site was held out and measured when nothing did "
                f"either. Set shift_axis to 'site', or drop the school.")
        return self

    @property
    def all_teachers(self) -> set[str]:
        return set(self.train_teachers) | set(self.val_teachers) | set(self.test_teachers)


class ArtefactRef(BaseModel):
    """A file a later phase depends on, with proof of what it is.

    §2.1 Property 3: a Kaggle or Colab session has no persistent filesystem and the machine
    will not exist again, so anything a later phase needs is exported and checksummed before
    the session ends, and the checksum is recorded here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    relative_path: str
    sha256: str
    bytes: int = Field(ge=0)
    produced_on_device: Device
    produced_on_device_model: str | None = None

    @field_validator("sha256")
    @classmethod
    def hash_is_lowercase_hex(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value


class RunManifest(BaseModel):
    """Everything needed to say what a run was, and to tell two runs apart.

    Compared field by field by `test_run_is_deterministic`, which is why `run_id`,
    `started_at` and `finished_at` are the only fields expected to differ between two runs of
    the same config. Anything else differing is a determinism failure.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    phase: str
    config_path: str
    config_sha256: str

    seed: int
    deterministic: bool

    # R7 as extended by D14: the device AND its model, on every run, without exception.
    device_requested: Device
    device_used: Device
    device_model: str | None = None

    python_version: str
    torch_version: str
    platform: str
    num_workers: int

    environment_overrides: dict[str, str] = Field(default_factory=dict)
    artefacts: list[ArtefactRef] = Field(default_factory=list)

    started_at: datetime
    finished_at: datetime | None = None

    @field_validator("config_sha256")
    @classmethod
    def hash_is_lowercase_hex(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("config_sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("run_id")
    @classmethod
    def run_id_is_a_ulid(cls, value: str) -> str:
        if not is_ulid(value):
            raise ValueError(f"run_id must be a ULID, got {value!r}")
        return value

    @property
    def fell_back_to_cpu(self) -> bool:
        """Whether CUDA was asked for and CPU was used.

        Recorded rather than hidden. A run that silently fell back would report CUDA numbers
        that are CPU numbers, and R7 requires every figure to carry the device it came from.
        """
        return self.device_requested == "cuda" and self.device_used == "cpu"

    def comparable(self) -> dict[str, object]:
        """The fields two runs of the same config must agree on, bit for bit.

        `run_id` and the timestamps are excluded because they are expected to differ; that is
        the whole of the exception BUILD-SPEC Phase 0 allows.
        """
        return self.model_dump(exclude={"run_id", "started_at", "finished_at"}, mode="json")
