# -*- coding: utf-8 -*-
"""The six research questions, and the code that is supposed to answer each one.

Declared here so the claim is checkable. `resolve` imports every evidence source named below and
reports whether it exists, so a question whose module is deleted, renamed or never written says
so instead of continuing to read as covered. `tests/unit/test_research_questions.py` holds the
wording against `thesis/02-refined-scope.md` section 6 in the same way, which is what stops the
system's account of a question drifting from the thesis's.

**A question with no evidence source is declared with none, not omitted.** RQ1, RQ4 and RQ6 have
no implementation anywhere in this system, and the honest report of that is a row saying so. D48
settled the shape of this for an unavailable check: it abstains, and the abstention is its own
verdict. Leaving the row out would be the one thing D48 forbids - silence that reads like
coverage.

`blocked_on` is prose, deliberately. What a question waits for is often a fact about the corpus
rather than about the code, and `scripts/rq_report.py` measures those against the live database
instead of trusting a sentence written here. A sentence that cannot be measured is still worth
carrying, because the alternative is a blank.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum


class EvidenceState(str, Enum):
    """Whether the named source is there. Not whether it has been run."""

    PRESENT = "present"
    MISSING = "missing"


@dataclass(frozen=True)
class Evidence:
    """One artefact that answers part of a question, and what produces it.

    `produced_by` is a dotted path to a module, or to a callable inside one. It is resolved by
    import rather than matched as a string, so a rename breaks the check rather than the claim.
    """

    what: str
    produced_by: str

    def state(self) -> EvidenceState:
        """Import the longest prefix that is a module, then walk the rest as attributes.

        Walked rather than split on the last dot, because a method is two levels deep -
        `praxis.routing.reveal.ReviewSession.adjudicate` is a module, a class and then a
        function - and a one-level resolver reports it missing while it is sitting there.
        """
        parts = self.produced_by.split(".")
        for cut in range(len(parts), 0, -1):
            try:
                found: object = importlib.import_module(".".join(parts[:cut]))
            except ImportError:
                continue
            for attribute in parts[cut:]:
                found = getattr(found, attribute, None)
                if found is None:
                    return EvidenceState.MISSING
            return EvidenceState.PRESENT
        return EvidenceState.MISSING


@dataclass(frozen=True)
class ResearchQuestion:
    """One question, its hypotheses, what answers it, and what it is waiting for."""

    id: str
    question: str
    hypotheses: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    blocked_on: tuple[str, ...]

    @property
    def is_implemented(self) -> bool:
        """Whether anything in this system claims to answer it at all."""
        return bool(self.evidence)

    def missing(self) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if e.state() is EvidenceState.MISSING)


# The wording is the thesis's, from 02-refined-scope.md section 6, trimmed of its own
# parenthetical amendments where they describe the editing of the question rather than the
# question. `test_research_questions.py` checks that what survives is still a substring of the
# thesis row, so trimming cannot become rewriting.
QUESTIONS: tuple[ResearchQuestion, ...] = (
    ResearchQuestion(
        id="RQ1",
        question=("What requirements must a convolutional teaching-assessment artifact satisfy "
                  "to be accepted as valid, usable, and governable by Ghanaian teacher "
                  "educators, and does an architecture meeting them pass expert design "
                  "validation?"),
        hypotheses=(),
        evidence=(),
        blocked_on=(
            "no requirements-elicitation or design-validation artefact exists in this system; "
            "O1 specifies a panel of at least eight teacher-education and computer-science "
            "specialists and the panel is Phase A of the research programme, which is not one "
            "of the twelve build phases and has no code",
            "the architecture it would validate exists as prose in BUILD-SPEC.md and "
            "docs/DECISIONS.md, which no code reads",
        ),
    ),
    ResearchQuestion(
        id="RQ2",
        question=("With what per-class accuracy, calibration, and reliability can a "
                  "teacher-independent spatiotemporal model detect the five observable "
                  "behaviours in teaching-practicum classrooms, relative to agreement among "
                  "trained human raters?"),
        hypotheses=("H1", "H2"),
        evidence=(
            Evidence("human agreement band per codebook field, with bootstrap intervals",
                     "praxis.annotation.irr.report"),
            Evidence("per-class accuracy, macro-F1, recall intervals and calibration together",
                     "praxis.evaluation.harness.evaluate_behaviour"),
            Evidence("the frozen Phase 4 comparison against the required baselines",
                     "praxis.behaviour.report.compare"),
        ),
        blocked_on=(
            "no labelled corpus: phase 2 abstains below two raters and phase 4 abstains "
            "without annotations",
            "no teacher track is confirmed, and annotation planning refuses an unconfirmed "
            "session because a clip cut on the heuristic's guess could show somebody who is "
            "not the teacher. The surface that unblocks this exists - the candidate ranking, "
            "the stills and the confirm endpoint - so this is waiting on a reviewer rather "
            "than on code",
            "R2 forbids a teacher appearing in more than one partition, so the number of "
            "distinct teachers is the binding constraint on teacher-disjoint splits rather "
            "than model capacity; the count is measured, not asserted here (L18)",
            "the comparator the question asks for is not implemented: nothing constructs a "
            "panel consensus or computes single-rater-to-panel agreement, and H2's "
            "non-inferiority margin appears in no configuration file",
            "the question asks about teaching-practicum classrooms and every session in the "
            "corpus is microteaching, so what exists is pilot footage from a different setting "
            "rather than development-domain data",
            "the band itself is still unmeasured: nothing feeds it until Phase 2 double-codes "
            "the pilot clips, so there is a comparator shape with no comparator in it. How the "
            "two numbers relate is settled (D97: reported side by side, no verdict, the "
            "coefficient named) and is no longer the open part",
        ),
    ),
    ResearchQuestion(
        id="RQ3",
        question=("How does model performance and calibration change across held-out practicum "
                  "schools and colleges, which measured classroom conditions are associated "
                  "with degradation, and can out-of-distribution detection flag unfamiliar "
                  "conditions?"),
        hypotheses=("H3", "H4", "H5"),
        evidence=(
            Evidence("the S0, S1 and S2 shift table with its leakage pre-flight",
                     "praxis.evaluation.shift.build_shift_table"),
            Evidence("per-session degradation regressed on named covariates",
                     "praxis.evaluation.attribution.attribute_degradation"),
            Evidence("out-of-distribution scores by three methods, one direction convention",
                     "praxis.confidence.ood.evaluate_ood"),
            Evidence("the per-domain flag-rate contrast H5 is stated as",
                     "praxis.confidence.ood.flag_rate_by_domain"),
        ),
        blocked_on=(
            "the capture protocol this question needs is not met and cannot be met by the "
            "footage in hand: S1 needs two schools under a college that also appears in "
            "training and S2 needs a college whose schools are all held out, so at least two "
            "colleges and three schools. The corpus has one college and no schools, so both "
            "levels are empty and only the S0 baseline is measurable. D98",
            "every session in the corpus is microteaching, so there is no practicum footage "
            "to measure variation across at all. The ladder that would measure it exists: D98 "
            "redefined S0, S1 and S2 to hold out a teacher, a basic school and a college, and "
            "split_manifests carries heldout_school and the axis the manifest was built on. "
            "What is missing is footage from two colleges and three schools",
            "the attribution model needs more complete sessions than the corpus has, and "
            "abstention is the expected path rather than the exceptional one",
            "phase 6 abstains on its own incompleteness: the analysis modules exist and "
            "nothing calls them end to end",
        ),
    ),
    ResearchQuestion(
        id="RQ4",
        question=("Where does the boundary fall between behaviours a calibrated model recovers "
                  "reliably and judgments that resist automation, and what does the resulting "
                  "placement imply for the National Teachers' Standards domains each behaviour "
                  "informs?"),
        hypotheses=(),
        evidence=(),
        blocked_on=(
            "no automatability placement exists in this system, and the revised question maps "
            "it to National Teachers' Standards domains rather than to criterion families, "
            "which appear nowhere in the code either",
            "the nearest artefact is the predicted agreement order registered before "
            "annotation, which ranks human reliability and carries no automatability meaning",
            "the one boundary the system does place is a single global abstention threshold, "
            "not a per-behaviour placement, and it depends on the same absent practicum data",
        ),
    ),
    ResearchQuestion(
        id="RQ5",
        question=("Does uncertainty-triggered routing to human adjudication produce "
                  "appropriate reliance, measured behaviourally through seeded-error probes, "
                  "and what is its effect on review time, score consistency, and workload?"),
        hypotheses=("H6", "H7"),
        evidence=(
            Evidence("the routing gate, with its stated precedence order",
                     "praxis.routing.gate.decide"),
            Evidence("the two-arm review session and its withheld pre-reveal payload",
                     "praxis.routing.reveal.ReviewSession"),
            Evidence("time on item, time before reveal, and evidence replays",
                     "praxis.routing.reveal.ReviewSession.adjudicate"),
        ),
        blocked_on=(
            "the seeded-error probes the question measures reliance through do not exist: "
            "praxis/evaluation/seeded_errors.py perturbs metric inputs to validate the "
            "harness and never injects a corrupted detection into a review session",
            "a probe outcome cannot be stored: adjudications has no was_seeded_error column "
            "and the seeded_errors table in docs/SCHEMA.md has no migration",
            "of the three outcomes asked about, only review time is instrumented; score "
            "consistency and workload appear in no module",
        ),
    ),
    ResearchQuestion(
        id="RQ6",
        question=("What transferable design principles for human-governed machine perception "
                  "in high-stakes educational assessment follow, and how do infrastructure and "
                  "governance constraints condition them?"),
        hypotheses=(),
        evidence=(),
        blocked_on=(
            "no design-principle register exists; docs/DECISIONS.md is a build-deviation log "
            "organised by decision number and does not claim to be one",
            "a synthesis question, and it depends on RQ1 to RQ5 having answers",
        ),
    ),
)


def question(question_id: str) -> ResearchQuestion:
    for item in QUESTIONS:
        if item.id == question_id:
            return item
    raise KeyError(f"{question_id} is not a declared research question; "
                   f"the six are {', '.join(q.id for q in QUESTIONS)}")


def resolve() -> list[dict[str, object]]:
    """Every question with the state of each evidence source it names.

    Reports no verdict about whether a question is answered. Whether the code exists and whether
    it has been run against data are different facts, and a function that conflated them would
    let an importable module stand in for a result - which is how phase 10 came to report
    `seeded_errors_available: True` because the wrong module imported cleanly.
    """
    return [
        {
            "id": item.id,
            "hypotheses": list(item.hypotheses),
            "implemented": item.is_implemented,
            "evidence": [{"what": e.what, "produced_by": e.produced_by,
                          "state": e.state().value} for e in item.evidence],
            "missing": [e.produced_by for e in item.missing()],
            "blocked_on": list(item.blocked_on),
        }
        for item in QUESTIONS
    ]
