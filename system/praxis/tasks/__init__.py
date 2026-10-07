# -*- coding: utf-8 -*-
"""Handing work to the queue, and working correctly when there is no queue.

There is no Redis server on the development machine and one cannot honestly be installed: the
only Windows build available is the abandoned 3.0.504 against the 7.x in docker-compose.yml,
which is the version divergence D44 exists to prevent. Phase 1 therefore has to be complete and
testable with no broker, without a skip, and without the absence of a broker being invisible.

The shape that achieves it: ingest depends on a `Callable[[str], None]`, not on Celery. Three
implementations satisfy it and each is honest about what it is.

* `RecordingQueue` collects identifiers in memory. Used by tests, which then assert on exactly
  what was handed over rather than on a mock's call list.
* `CeleryQueue` sends the real task. Used whenever a broker is configured.
* `select_queue` picks between them from the configuration, and **fails rather than falling
  back** when a broker is configured and unreachable.

That last point is the whole design. A broker that is absent is a known state with a stated
consequence; a broker that is configured and down must not look like one that was never
configured, which is the same rule `tests/conftest.py::database_dsn` applies to the database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

BROKER_VAR = "PRAXIS_BROKER_URL"
INGESTED_TASK = "praxis.preprocess.session_ingested"


class BrokerUnavailable(RuntimeError):
    """A broker is configured and cannot be reached. Never downgraded to a local queue."""


@dataclass
class RecordingQueue:
    """Keeps what it was given. Not a stub: it is the queue Phase 1 runs with today.

    Nothing consumes `session.ingested` yet - the consumer is Phase 3 preprocessing - so the
    only thing a broker would add right now is a place for the message to sit. What matters at
    this phase is that ingest hands over the right identifier at the right moment, and that is
    what this records.
    """

    delivered: list[str] = field(default_factory=list)

    def __call__(self, session_id: str) -> None:
        self.delivered.append(session_id)


@dataclass
class CeleryQueue:
    """Sends `session.ingested` to the real broker, carrying an identifier and nothing else.

    The payload is deliberately just the session id. A message that duplicated the row would be
    a second copy of the session, able to disagree with the first by the time a worker read it.
    """

    broker_url: str
    task_name: str = INGESTED_TASK

    def __post_init__(self) -> None:
        from celery import Celery

        self._app = Celery("praxis", broker=self.broker_url)
        self._app.conf.update(task_serializer="json", accept_content=["json"],
                              timezone="UTC", enable_utc=True)

    def __call__(self, session_id: str) -> None:
        try:
            self._app.send_task(self.task_name, args=[session_id])
        except Exception as exc:
            raise BrokerUnavailable(
                f"{BROKER_VAR} is set to {self.broker_url!r} and the task could not be sent: "
                f"{exc}. The session is already committed; it has not been queued.") from exc


def broker_url() -> str | None:
    return os.environ.get(BROKER_VAR, "").strip() or None


def select_queue(url: str | None = None):
    """The queue this run should use. Absent broker degrades; broken broker raises."""
    url = url or broker_url()
    if url is None:
        return RecordingQueue()

    try:
        import celery  # noqa: F401
    except ImportError as exc:
        raise BrokerUnavailable(
            f"{BROKER_VAR} is set but celery is not installed") from exc
    return CeleryQueue(url)
