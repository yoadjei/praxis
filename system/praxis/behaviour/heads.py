# -*- coding: utf-8 -*-
"""The output layout, derived from the codebook rather than declared alongside it.

BUILD-SPEC sketches the head stack as "5 presence heads (sigmoid) + 5 intensity heads
(linear)". That is the shape of the idea, not the shape of the contract. `Detection` requires
`predicted` to carry **every** field the behaviour's codebook entry defines, validated through
`Codebook.validate_labels` - the same function a rater's label goes through, which is what makes
a model output and a human label comparable at all. B2 and B4 have no boolean field, B4 has two
categorical ones, and B3 has five fields. A fixed presence-plus-intensity pair could not express
any of them.

So the layout is read off `FieldSpec`, and the three head kinds fall out of the four scales:

    nominal, two levels    -> BINARY       one logit, sigmoid          (this is "presence")
    nominal, more levels   -> CATEGORICAL  one logit per level         (BUILD-SPEC omits these)
    ordinal/interval/ratio -> SCALAR       one linear output           (this is "intensity")

**Scalar targets are normalised to [0, 1] before the loss sees them, and this is not cosmetic.**
`b1_count` spans 0 to 20 and `b2_facing_proportion` spans 0 to 1. Summing raw squared errors
would weight the count roughly four hundred times more heavily than the proportion, so the model
would learn to count and ignore orientation - and nothing in the reported metrics would say so,
because each field is scored separately at evaluation time. Normalising by the declared domain
makes the loss weight the fields equally, which is the only defensible default when the codebook
states no preference.

**A change to the codebook changes the heads.** That is the point of deriving them: adding a
field to CODEBOOK.md and forgetting to widen the model is not a thing that can happen here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from praxis.annotation.codebook import Codebook, FieldSpec, ScaleType
from praxis.vocabulary import BehaviourId

# The decision threshold for a binary field, applied when a probability is turned back into the
# boolean the codebook declares. Deliberately not tunable here: the routing gate in Phase 8 is
# where a confidence threshold belongs, and a second one hidden in the decoder would be a
# threshold nobody could find. See D74.
BINARY_DECISION_THRESHOLD = 0.5

NONSCORABLE_SUFFIX = "_nonscorable"


class HeadKind(str, Enum):
    """What sort of output a field needs, and therefore which loss it is trained with."""

    BINARY = "binary"
    CATEGORICAL = "categorical"
    SCALAR = "scalar"


@dataclass(frozen=True)
class HeadSpec:
    """One codebook field's output: its width, its encoding, and its inverse."""

    field: str
    kind: HeadKind
    width: int
    scale: ScaleType
    levels: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None

    @property
    def is_nonscorable_flag(self) -> bool:
        """Whether this is the field by which the model declines to answer.

        It is excluded from the non-scorable mask, for a reason that is easy to miss: masking
        the clips a behaviour is non-scorable on would remove every positive example this head
        has. It could then never learn to raise the flag, and `routing.gate` would never see
        `model_abstained`. Every other field on the behaviour is masked; this one is what the
        mask is derived from.
        """
        return self.field.endswith(NONSCORABLE_SUFFIX)

    def encode(self, value: Any) -> float:
        """The codebook value as a training target.

        BINARY and SCALAR return one number. CATEGORICAL returns the class index, as a float so
        that one return type covers all three; the caller casts it for cross-entropy.
        """
        if self.kind is HeadKind.BINARY:
            return 1.0 if value else 0.0

        if self.kind is HeadKind.CATEGORICAL:
            try:
                return float(self.levels.index(value))
            except ValueError as unknown:
                raise ValueError(
                    f"{self.field}={value!r} is not one of {self.levels}") from unknown

        if self.scale is ScaleType.ORDINAL:
            # By position, not by face value. `b5_duration` runs 0 to 3 and `b1_amplitude` runs
            # 1 to 3; encoding the number itself would put the two on different scales and
            # place b5_duration's first band at 0 and b1_amplitude's at 1/3.
            try:
                index = self.levels.index(value)
            except ValueError as unknown:
                raise ValueError(
                    f"{self.field}={value!r} is not one of {self.levels}") from unknown
            return index / (len(self.levels) - 1) if len(self.levels) > 1 else 0.0

        span = self.maximum - self.minimum
        if span <= 0:
            return 0.0
        return (float(value) - self.minimum) / span

    def decode(self, raw: float | list[float]) -> Any:
        """A model output as the codebook value, guaranteed inside the declared domain.

        Clipped rather than trusted. A linear head can emit a count of -0.3 or 21.7, and
        `Detection` would refuse the resulting prediction outright - correctly, but as a crash
        at serialisation time rather than as the out-of-range estimate it actually is.
        """
        if self.kind is HeadKind.BINARY:
            return bool(float(raw) >= BINARY_DECISION_THRESHOLD)

        if self.kind is HeadKind.CATEGORICAL:
            scores = list(raw)
            if len(scores) != self.width:
                raise ValueError(
                    f"{self.field} expects {self.width} scores, got {len(scores)}")
            return self.levels[max(range(len(scores)), key=scores.__getitem__)]

        value = float(raw)
        if self.scale is ScaleType.ORDINAL:
            last = len(self.levels) - 1
            index = min(last, max(0, round(value * last)))
            return self.levels[index]

        restored = self.minimum + value * (self.maximum - self.minimum)
        restored = min(self.maximum, max(self.minimum, restored))
        # A count is a count. RATIO is the scale the codebook gives `b1_count` and
        # `b3_transitions`, both of which its table declares as integers, while INTERVAL covers
        # `b2_facing_proportion`, which is genuinely fractional.
        return round(restored) if self.scale is ScaleType.RATIO else restored


def _head_for(spec: FieldSpec) -> HeadSpec:
    if spec.scale is ScaleType.NOMINAL:
        levels = spec.levels or ()
        binary = len(levels) == 2 and all(isinstance(level, bool) for level in levels)
        return HeadSpec(
            field=spec.name,
            kind=HeadKind.BINARY if binary else HeadKind.CATEGORICAL,
            # A binary field is one logit, not two. Two would let the model hold beliefs that
            # do not sum to one and would double the parameters for nothing.
            width=1 if binary else len(levels),
            scale=spec.scale,
            levels=tuple(levels),
        )

    return HeadSpec(
        field=spec.name,
        kind=HeadKind.SCALAR,
        width=1,
        scale=spec.scale,
        levels=tuple(spec.levels) if spec.levels else None,
        minimum=spec.minimum,
        maximum=spec.maximum,
    )


def behaviour_heads(codebook: Codebook, behaviour: BehaviourId) -> tuple[HeadSpec, ...]:
    """The output layout for one behaviour, in the codebook's own field order.

    Order is the codebook's and is load-bearing: it fixes which slice of the output tensor
    belongs to which field, and a model trained under one order would read another's outputs as
    a different set of predictions entirely.
    """
    return tuple(_head_for(field) for field in codebook.behaviour(behaviour).fields)


def output_width(heads: tuple[HeadSpec, ...]) -> int:
    """How wide the output layer is for a behaviour."""
    return sum(head.width for head in heads)


def encode_labels(heads: tuple[HeadSpec, ...], labels: dict[str, Any]) -> dict[str, float]:
    """A rater's label dict as training targets, one per head.

    Raises:
        KeyError: if a field the behaviour defines is absent. A partial target would train the
        missing head on whatever a default happened to be, which is worse than not training it.
    """
    targets: dict[str, float] = {}
    for head in heads:
        if head.field not in labels:
            raise KeyError(
                f"label set omits {head.field}; every field the codebook defines must be "
                f"present, because a missing one trains its head on a default nobody chose")
        targets[head.field] = head.encode(labels[head.field])
    return targets


def decode_prediction(heads: tuple[HeadSpec, ...],
                      outputs: dict[str, float | list[float]]) -> dict[str, Any]:
    """Model outputs as a `Detection.predicted` dict, in the codebook's vocabulary.

    The result is exactly what `Codebook.validate_labels` accepts, which is what lets a
    prediction and a human label be compared at all.
    """
    return {head.field: head.decode(outputs[head.field]) for head in heads}
