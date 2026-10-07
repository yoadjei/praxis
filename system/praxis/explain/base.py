# -*- coding: utf-8 -*-
"""What an explanation is, and how the interface picks one.

BUILD-SPEC Phase 7 requires that "the UI reads explanation type from configuration; swapping
methods requires no code change", and that a method failing the Adebayo sanity checks is
**removed from the interface** rather than kept because it looks better. Both are properties of
the registry here: a method is selected by name from `explain.method`, and `select` refuses a
name that is not registered instead of falling through to something.

**Guided Grad-CAM is not registered and is not implemented.** Adebayo et al. (2018) found it
invariant to higher-layer parameters - it fails the parameter randomisation test - while
producing the most visually appealing maps of the methods they examined. Looking nicer is
precisely the failure mode, so there is nowhere in this package to put it.

**An explanation carries what it actually computed, and nothing it did not.** Grad-CAM produces
a spatial map and no keypoint attribution; the intrinsic method produces temporal attention and
keypoint gradients and no spatial map. Both are `None` where absent rather than zero-filled,
because a zero map renders as "the model looked nowhere" instead of "this method does not
answer that question".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ExplainError(RuntimeError):
    """An explanation could not be produced, with the reason named."""


@dataclass(frozen=True)
class ClipInput:
    """What an explainer is given. Not every method needs every part.

    `crops` are the pixels, which Grad-CAM needs because a spatial map is meaningless without
    them. `features` are Stage A's cached activations, which the intrinsic method works from.
    A method that needs a part it was not given raises rather than substituting.
    """

    keypoints: Any                      # (T, K, 3) tensor
    crops: Any | None = None            # (T, 3, H, W) tensor in [0, 1]
    features: Any | None = None         # (T, D) tensor

    def require_crops(self, method: str):
        if self.crops is None:
            raise ExplainError(
                f"{method} needs the clip's pixels and was given none. A spatial explanation "
                f"cannot be produced from cached activations; pass the crops or use the "
                f"intrinsic method.")
        return self.crops

    def require_features(self, method: str):
        if self.features is None:
            raise ExplainError(
                f"{method} needs the cached backbone features and was given none.")
        return self.features


@dataclass(frozen=True)
class Explanation:
    """One method's account of one prediction.

    `temporal` is the one part every method supplies, because every method can say *when* in
    the clip the evidence was, and it is what the fidelity checks compare when a method has no
    spatial map.
    """

    method: str
    behaviour: str
    field: str
    temporal: np.ndarray                      # (T,) attribution over frames
    spatial: np.ndarray | None = None         # (T, H, W) per-frame heatmap
    keypoint: np.ndarray | None = None        # (T, K) per-keypoint attribution

    def __post_init__(self) -> None:
        if self.temporal.ndim != 1:
            raise ExplainError(
                f"temporal attribution must be one value per frame, got "
                f"{self.temporal.shape}")
        if self.spatial is not None and self.spatial.shape[0] != self.temporal.shape[0]:
            raise ExplainError(
                f"the spatial map covers {self.spatial.shape[0]} frames and the temporal "
                f"attribution {self.temporal.shape[0]}; a heatmap shown against the wrong "
                f"frame is worse than no heatmap")

    @property
    def frames(self) -> int:
        return int(self.temporal.shape[0])

    def comparable_map(self) -> np.ndarray:
        """The array the fidelity checks correlate, flattened.

        The spatial map where there is one, the temporal attribution otherwise. Stated here
        rather than at each call site so that two checks cannot end up comparing different
        things and reporting one verdict.
        """
        return (self.spatial if self.spatial is not None else self.temporal).ravel()


class Explainer(Protocol):
    """Whatever turns a clip and a target field into an explanation."""

    name: str

    def explain(self, clip: ClipInput, behaviour: str, field: str) -> Explanation:
        ...


_REGISTRY: dict[str, Any] = {}

# Named so the refusal can say why, rather than "unknown method". Adebayo et al. (2018) §4:
# Guided BackProp and Guided Grad-CAM are invariant to higher-layer parameters.
FORBIDDEN = {
    "guided_gradcam": (
        "Guided Grad-CAM fails Adebayo et al.'s parameter randomisation test: its maps are "
        "invariant to the parameters of the layers above, so they cannot be describing the "
        "computation. It produces the most convincing pictures of the methods examined, which "
        "is the reason it is forbidden rather than merely unimplemented."),
    "guided_backprop": (
        "Guided BackProp fails the same test, for the same reason."),
}


def register(explainer: Any) -> Any:
    """Add a method to the registry. Refuses a forbidden one by name."""
    name = explainer.name
    if name in FORBIDDEN:
        raise ExplainError(f"{name} is forbidden: {FORBIDDEN[name]}")
    if name in _REGISTRY:
        raise ExplainError(f"{name} is already registered")
    _REGISTRY[name] = explainer
    return explainer


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def select(method: str, *, fallback: str | None = None) -> Any:
    """The explainer named in the config, or the fallback if it has been withdrawn.

    A method is withdrawn by being unregistered, which is what BUILD-SPEC's decision rule
    means in code: "if Grad-CAM fails either sanity check on this model and this data, it is
    removed from the user interface and the intrinsic explanation is used instead". The
    fallback is itself checked, so a config naming two unavailable methods fails loudly rather
    than serving no explanation while claiming one.
    """
    if method in FORBIDDEN:
        raise ExplainError(f"{method} is forbidden: {FORBIDDEN[method]}")
    if method in _REGISTRY:
        return _REGISTRY[method]
    if fallback is None:
        raise ExplainError(
            f"no explanation method called {method!r}; available: {list(available())}")
    if fallback not in _REGISTRY:
        raise ExplainError(
            f"neither {method!r} nor its fallback {fallback!r} is available; available: "
            f"{list(available())}. An interface with no explanation must say so rather than "
            f"show a detection as though one had been produced.")
    return _REGISTRY[fallback]


def normalise(attribution: np.ndarray) -> np.ndarray:
    """Scale an attribution map to [0, 1], flat where it is constant.

    A constant map is a real output - a broken explainer produces one, and Phase 7's acceptance
    test requires that such an explainer be caught - so dividing by a zero range must give
    zeros rather than NaN, which would propagate into the similarity scores and read as a
    missing result rather than as the failure it is.
    """
    lowest = float(np.min(attribution))
    highest = float(np.max(attribution))
    if not np.isfinite(lowest) or not np.isfinite(highest) or highest <= lowest:
        return np.zeros_like(attribution, dtype=float)
    return (attribution.astype(float) - lowest) / (highest - lowest)
