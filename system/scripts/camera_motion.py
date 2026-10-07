# -*- coding: utf-8 -*-
"""Did the camera move, per session, measured from the blurred footage.

    python scripts/camera_motion.py
    python scripts/camera_motion.py --json
    python scripts/camera_motion.py --session 01M3VY8Q98YPPQCJ6TX87B7VF5

Zones are a fixed map from frame coordinates to room regions, so marking them on a session
whose camera moved records a room that is no longer where the polygon says. The codebook
already makes B3 non-scorable when "the camera moved", and nothing measured it: it was a
judgement a rater made per clip with no number behind it, and B3 is the behaviour the thesis
names as the primary domain-shift signal.

**The decision rule was fixed before any of these numbers existed**, and it is derived rather
than chosen: `zones.OVERLAP_GRID` is 64 and its docstring states that one cell of that grid is
the finest zone geometry this project will distinguish. Motion that keeps a room point inside
one cell cannot displace it further than the resolution zones are defined at. See
`praxis/preprocess/motion.py` for the derivation and `tests/unit/test_motion.py` for the sign
convention and the drift-versus-jitter choice.

Read-only. It opens the blurred media, decodes sampled frames, and writes nothing anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select

from praxis.config import load_config
from praxis.db import build_engine, keywords_to_url
from praxis.db.schema import media_objects, sessions
from praxis.ingest.errors import IngestRefused
from praxis.preprocess import frames as frame_module
from praxis.preprocess import motion as motion_module
from praxis.tools import resolve


def corpus(engine, session_id: str | None) -> list[dict[str, object]]:
    """Every session whose blurred copy is on record, or just the one asked for."""
    query = select(
        sessions.c.session_id,
        media_objects.c.blurred_relative_path,
        media_objects.c.duration_s,
        media_objects.c.width,
        media_objects.c.height,
    ).join(
        media_objects, media_objects.c.media_sha256 == sessions.c.media_sha256
    ).where(media_objects.c.is_blurred.is_(True),
            media_objects.c.blurred_relative_path.isnot(None)
            ).order_by(sessions.c.session_id)
    if session_id:
        query = query.where(sessions.c.session_id == session_id)

    with engine.connect() as connection:
        return [dict(row._mapping) for row in connection.execute(query)]


def measure_one(ffmpeg, media_root: Path, row: dict[str, object]) -> dict[str, object]:
    path = media_root / str(row["blurred_relative_path"])
    if not path.is_file():
        return {"session_id": row["session_id"], "error": f"no file at {path}"}

    width = motion_module.SAMPLE_WIDTH
    height = frame_module.scaled_height(int(row["width"]), int(row["height"]), width)
    count = motion_module.sample_count(float(row["duration_s"]))

    try:
        decoded = frame_module.gray_sequence(
            ffmpeg, path, count=count, duration_s=float(row["duration_s"]),
            width=width, height=height)
    except IngestRefused as refused:
        return {"session_id": row["session_id"], "error": refused.refusal.detail}

    result = motion_module.measure(decoded)
    return {"session_id": row["session_id"],
            "source": f"{row['width']}x{row['height']}",
            "sampled": f"{width}x{height}",
            "requested": count,
            **result.as_json()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", help="one session id instead of the whole corpus")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "default.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    keywords = os.environ.get("PRAXIS_TEST_DSN")
    engine = build_engine(keywords_to_url(keywords)) if keywords else build_engine()
    ffmpeg = resolve("ffmpeg", config.tools.ffmpeg)
    media_root = config.require_media_root()

    rows = corpus(engine, args.session)
    if not rows:
        print("no session has a blurred copy on record, so there is nothing to measure",
              file=sys.stderr)
        return 2

    results = [measure_one(ffmpeg, media_root, row) for row in rows]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"  tolerance {1 / motion_module.OVERLAP_GRID:.2%} of the frame diagonal, "
          f"one cell of the zone overlap grid")
    print(f"  {'session':>8} {'source':>11} {'samples':>8} {'max drift':>10} "
          f"{'p95 step':>9}  verdict")
    for result in results:
        if "error" in result:
            print(f"  {str(result['session_id'])[-6:]:>8} {result['error']}")
            continue
        verdict = "static, zones may be marked" if result["is_static"] else "moved, no zones"
        print(f"  {str(result['session_id'])[-6:]:>8} {result['source']:>11} "
              f"{result['samples']:>8} {result['max_drift']:>9.2%} "
              f"{result['p95_step']:>8.2%}  {verdict}")

    # Counted in three buckets and not two. A session whose frames would not decode is not a
    # still camera; subtracting the movers from the total called it one, which would have read
    # as "two sessions may be given zones" about two sessions nobody measured.
    measured = [r for r in results if "error" not in r]
    static = [r for r in measured if r["is_static"]]
    unmeasured = len(results) - len(measured)

    print()
    print(f"  {len(static)} of {len(measured)} measured sessions hold still enough for a "
          f"marked zone to mean what it says.")
    if unmeasured:
        print(f"  {unmeasured} could not be measured and are neither static nor moving here; "
              f"the rows above say why.")
    if len(static) < len(measured):
        print("  The rest keep B3 non-scorable by two clauses of the codebook's own rule:")
        print("  zones undefined for the setup, and the camera moved. That is a corpus")
        print("  limitation driven by unconstrained camera placement, which the thesis names")
        print("  as a cause of degradation, and it belongs in the write-up with its number")
        print("  rather than engineered around.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
