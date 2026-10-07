# -*- coding: utf-8 -*-
"""The closed vocabularies every other contract is written in.

Split out so that the dependency runs one way. `annotation/codebook.py` needs `BehaviourId` to
declare which behaviour a spec describes, and `contracts/detection.py` needs the codebook to
check that a prediction is a label the codebook actually defines. With the identifiers living
in `detection.py` those two imported each other.

It sits **above both packages** rather than inside `contracts/`, and that placement is the
whole point. Importing any submodule of a package runs that package's `__init__`, so
`contracts.vocabulary` would have pulled in `contracts.detection` and cycled again by a longer
route. From here the chain is

    praxis.vocabulary  <-  praxis.annotation.codebook  <-  praxis.contracts.detection

with `praxis/__init__.py` empty, and nothing is deferred into a function body to break a cycle.

**Every one of these is closed on purpose.** A `Literal` refuses an unrecognised value at
construction rather than carrying it to the database and failing a CHECK constraint, or worse,
passing one that was never added.
"""
from __future__ import annotations

from typing import Literal

BehaviourId = Literal["B1", "B2", "B3", "B4", "B5"]
GateOutcome = Literal["present", "suppress", "escalate"]
SuppressionReason = Literal["low_confidence", "out_of_distribution", "model_abstained"]

BEHAVIOUR_NAMES: dict[str, str] = {
    "B1": "Gesture production",
    "B2": "Body orientation to class",
    "B3": "Spatial position and mobility",
    "B4": "Postural stance",
    "B5": "Board and material use",
}

BEHAVIOUR_IDS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")

# Consent, whose two closed vocabularies are what `consent_records` CHECKs in migration 0002.
#
# They live here because they were previously declared in `scripts/import_consents.py` as well,
# and the two copies disagreed: the script accepted 'learner' and 'guardian', the table accepts
# 'guardian_class' and 'evaluator'. Only 'teacher' was in both. So a consent file saying
# 'learner' passed the script's own validation and then aborted the import on a CHECK violation
# instead of being reported as a bad row, and two values the table allows could not be imported
# at all. The module docstring above describes that outcome exactly.
#
# 'learner' is not merely a missing spelling of 'guardian_class'. There is no per-learner
# consent record because no learner is ever identified: R1 classifies only the teacher and D18
# blurs every face before the original is deleted. Consent for the pupils is held at class level
# through their guardians, which is what 'guardian_class' says and what 'learner' would
# misrepresent.
SubjectType = Literal["teacher", "guardian_class", "evaluator"]
ConsentScope = Literal["microteaching", "classroom", "both"]

SUBJECT_TYPES: tuple[str, ...] = ("teacher", "guardian_class", "evaluator")
CONSENT_SCOPES: tuple[str, ...] = ("microteaching", "classroom", "both")

# What a shift level holds out. 'domain' is the ladder the corpus was first designed around -
# train on microteaching, test on classroom practicum. 'site' holds out a basic school for S1
# and a college for S2, the only ladder available once the corpus is practicum throughout.
#
# It lives here rather than in either place that needs it. `config_schema` reads it off a YAML
# file and `contracts.manifest` writes it onto a published artefact; the two must agree on the
# spelling or a config would build a manifest the database refuses, and a Literal written twice
# is a vocabulary that drifts (L20). D98.
ShiftAxis = Literal["domain", "site"]
SHIFT_AXES: tuple[str, ...] = ("domain", "site")

# Where the track in `teacher_tracks.track_id` came from. The column has carried a default of
# 'heuristic' since 0004 and no CHECK, and for as long as a declined proposal stored no row at
# all, 'human' was representable and unreachable: the only writer was one hardcoded literal.
# 0009 adds the constraint and `confirm_teacher` writes the second value, for the case the
# confirmation step exists to serve - a reviewer naming a track the heuristic never proposed.
#
# 'human' describes the origin of the *track number*, not the act of confirming. A reviewer who
# agrees with a proposal leaves the origin 'heuristic'; the audit chain is what records who
# agreed and when. Reading this column as "was it reviewed" would be wrong in both directions,
# which is what `confirmed_by` is for.
TrackOrigin = Literal["heuristic", "human"]

TRACK_ORIGINS: tuple[str, ...] = ("heuristic", "human")
