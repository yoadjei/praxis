# -*- coding: utf-8 -*-
"""What the system claims to answer, and what it is still waiting for.

Every other package here is keyed to the invariants R1 to R7 and to the hypotheses H1 to H7.
Nothing was keyed to the research questions, which is why the only link between the code and the
thesis was a six-row table in the README naming an output per question and no module, metric or
script. A table like that cannot go stale loudly: it reads the same whether the module behind it
exists or not.
"""
from praxis.research.questions import (
    QUESTIONS,
    Evidence,
    EvidenceState,
    ResearchQuestion,
    question,
    resolve,
)

__all__ = ["QUESTIONS", "Evidence", "EvidenceState", "ResearchQuestion", "question", "resolve"]
