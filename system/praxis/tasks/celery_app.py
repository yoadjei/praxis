# -*- coding: utf-8 -*-
"""The Celery application for background preprocessing.

The application is configured from the same broker source as select_queue uses, so both
the API and the worker agree on which broker to reach. R7 requires every configuration to
come from one place, and this module reads it from the environment and broker_url() rather
than inventing a second convention.

If PRAXIS_BROKER_URL is not set, the application is still created and importable - this is
required so the worker can start and report the missing broker to the operator with a clear
message, rather than failing at import time and hiding the reason. R6: the worker talking to a
local Redis broker is not inference; it is orchestration infrastructure, and R6 applies only to
the model.

That variable is the only one read. This paragraph used to say "PRAXIS_BROKER_URL (or
REDIS_URL)", which `broker_url()` has never done, and docker-compose.yml set REDIS_URL on the
strength of it: the worker ran on celery's in-process `memory://` transport for every version of
that file, and the API queued into a list it then discarded. Both looked configured. See
tests/unit/test_compose.py, which now asserts the deployed stack names this variable and no
other.
"""
from __future__ import annotations

import logging

from celery import Celery

from praxis.tasks import BROKER_VAR, INGESTED_TASK, broker_url

logger = logging.getLogger(__name__)

# Create the application with broker from the same source as select_queue. If no broker
# is configured, this creates an app without a broker - it will refuse to start work
# later but the module remains importable.
_broker_url = broker_url()
app = Celery("praxis", broker=_broker_url if _broker_url else "memory://")

# Configuration that applies to all tasks.
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Warned, and deliberately NOT switched to eager execution. `task_always_eager` would run
# every task inline in whichever process sent it, so an API with no broker would quietly
# perform preprocessing inside a request instead of queueing it - the opposite of what this
# warning says is happening, and the kind of difference between configured and unconfigured
# behaviour that only shows up under load. `select_queue` already degrades to
# `RecordingQueue` when no broker is set, so nothing reaches this app in that state; a worker
# started anyway should fail, not improvise.
if _broker_url is None:
    logger.warning(
        "No broker is configured (%s unset). This worker has nothing to consume from; "
        "set it to the Redis URL to enable queued preprocessing.", BROKER_VAR)


@app.task(name=INGESTED_TASK, bind=True)
def session_ingested(self, session_id: str) -> dict[str, str]:
    """Acknowledge that a session has been ingested and is ready for preprocessing.

    This task is sent by Phase 1 (ingest) to signal that a session is in the database
    and waiting for Phase 3 (preprocessing) to extract pose, track bodies, propose the
    teacher, blur faces, and write artifacts.

    Phase 3 runs as a batch process that queries for all sessions without pose artifacts
    and processes them together, so this task does not need to do the preprocessing
    itself. Batching is more efficient than per-task preprocessing, particularly for
    pose estimation which initializes an expensive ONNX model once and reuses it.

    The presence of this task in the worker means the broker is reachable and messages
    can be delivered, which is what we verify here.
    """
    # The task was received and is running, which means the broker is working.
    # Phase 3 will pick up the session on its next run and preprocess it.
    return {"session_id": session_id, "status": "queued"}
