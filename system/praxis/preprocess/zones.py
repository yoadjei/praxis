# -*- coding: utf-8 -*-
"""Room zones as normalised polygons.

SCHEMA.md section 4 stores `zone_board`, `zone_front`, `zone_middle`, `zone_back` and
`learner_region` as JSONB, normalised (0,0) top-left to (1,1) bottom-right. Normalised so one
camera setup survives a change of recording resolution: the operator marks the room once per
setup, not once per session.

Nothing here infers a zone. The operator marks them, and a session whose setup is unmarked has
no front zone, which the teacher heuristic must be told rather than allowed to guess around.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float]
ZONE_NAMES = ("zone_board", "zone_front", "zone_middle", "zone_back", "learner_region")

# Resolution of the sampling grid `overlaps()` uses. 64 puts a cell at roughly 1.5% of the
# frame's width and height, which is finer than any overlap an operator could draw by accident
# and still under five thousand point-in-polygon tests per zone, run once per setup.
OVERLAP_GRID = 64


class ZoneError(ValueError):
    """A polygon is not usable as a region."""


@dataclass(frozen=True)
class Polygon:
    """A closed region in normalised coordinates."""

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ZoneError(f"a region needs at least three points, got {len(self.points)}")
        for x, y in self.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ZoneError(
                    f"({x}, {y}) is outside the normalised frame; zones are stored as "
                    f"fractions of width and height, not pixels")

    def contains(self, point: Point) -> bool:
        """Ray casting. A point exactly on an edge is inside, which keeps a person standing on
        the boundary of the front zone from flickering between zones frame to frame."""
        x, y = point
        inside = False
        count = len(self.points)
        for i in range(count):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % count]
            if _on_segment((x, y), (x1, y1), (x2, y2)):
                return True
            if (y1 > y) != (y2 > y):
                crossing = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
                if crossing > x:
                    inside = not inside
        return inside

    def as_json(self) -> list[list[float]]:
        return [[x, y] for x, y in self.points]

    @classmethod
    def from_json(cls, raw: Iterable[Sequence[float]]) -> Polygon:
        return cls(points=tuple((float(p[0]), float(p[1])) for p in raw))


def _on_segment(point: Point, start: Point, end: Point, tolerance: float = 1e-9) -> bool:
    (x, y), (x1, y1), (x2, y2) = point, start, end
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > tolerance:
        return False
    return (min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
            and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance)


@dataclass(frozen=True)
class CameraSetup:
    """The five regions an operator marks once per camera position."""

    setup_id: str
    zone_board: Polygon
    zone_front: Polygon
    zone_middle: Polygon
    zone_back: Polygon
    learner_region: Polygon

    def zone_of(self, point: Point) -> str | None:
        """Which of front, middle or back a normalised point falls in.

        Checked in that order and the first match wins. Overlapping polygons are an operator
        error rather than something to resolve by area, and `overlaps()` reports them at the
        point they are defined instead of letting the ambiguity reach the heuristic.
        """
        for name in ("zone_front", "zone_middle", "zone_back"):
            if getattr(self, name).contains(point):
                return name
        return None

    def overlaps(self) -> list[tuple[str, str]]:
        """Pairs of the three seating zones that share area rather than merely a boundary.

        Front, middle and back normally tile the room, so adjacent zones touch along an edge by
        design and `contains` counts an on-edge point as inside. Reporting that would make this
        check fire on every correct setup, which is worse than not having it. What is an
        operator error is a shared *region*: `zone_of` resolves it silently by check order, and
        a whole band of the room is then attributed to the wrong zone for the entire session.

        Measured by sampling cell centres because operator-drawn zones need not be convex, so
        polygon clipping would not be a two-line answer. Cell centres never land on a shared
        boundary, and an overlap thinner than one cell in both axes is not reported.
        """
        cells = [((i + 0.5) / OVERLAP_GRID, (j + 0.5) / OVERLAP_GRID)
                 for i in range(OVERLAP_GRID) for j in range(OVERLAP_GRID)]
        names = ("zone_front", "zone_middle", "zone_back")
        covered = {name: {c for c in cells if getattr(self, name).contains(c)}
                   for name in names}

        return [(first, second)
                for i, first in enumerate(names) for second in names[i + 1:]
                if covered[first] & covered[second]]

    def as_row(self) -> dict[str, Any]:
        return {name: getattr(self, name).as_json() for name in ZONE_NAMES}

    @classmethod
    def from_row(cls, setup_id: str, row: dict[str, Any]) -> CameraSetup:
        missing = [name for name in ZONE_NAMES if name not in row]
        if missing:
            raise ZoneError(f"camera setup {setup_id} is missing {', '.join(missing)}")
        return cls(setup_id=setup_id,
                   **{name: Polygon.from_json(row[name]) for name in ZONE_NAMES})


def normalise(point: tuple[float, float], width: int, height: int) -> Point:
    """Pixel coordinates into the frame fractions the zones are stored in."""
    if width <= 0 or height <= 0:
        raise ZoneError(f"cannot normalise against a {width}x{height} frame")
    return point[0] / width, point[1] / height
