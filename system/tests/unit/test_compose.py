# -*- coding: utf-8 -*-
"""docker-compose.yml declares what the code reads, under the names the code reads.

Nothing checked this file before, and its own header said so: "NEVER EXECUTED". When it was
finally run, four of five services came up healthy and the worker did not, and the reason was
not the worker. The file set `REDIS_URL` for the api and the worker; `praxis.tasks.broker_url()`
reads `PRAXIS_BROKER_URL` and only that. So the broker was configured under a name nothing
consults, and both services degraded exactly as they are designed to when no broker is set:
`select_queue()` returned `RecordingQueue`, which appends a session id to a list in the API
process and drops it, and the worker started on celery's in-process `memory://` transport,
announced itself ready, and consumed from a broker no other process could reach. Redis was up
and healthy the whole time. The only symptom anywhere was the worker's own healthcheck, and that
failed for a third reason again - `celery inspect ping` builds its own `memory://` transport, so
the ping and the worker were in separate universes.

A variable name is exactly the kind of thing two files can disagree about while each looks
right on its own. See L20, and L27 for the same shape in a vocabulary.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from praxis.tasks import BROKER_VAR

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMPOSE = REPO_ROOT / "docker-compose.yml"

# The services that import `praxis.tasks` and so need a broker: one sends, the other consumes.
BROKER_SERVICES = ("api", "worker")


def services() -> dict[str, dict]:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]


def environment(service: dict) -> dict[str, str]:
    """A service's environment as a mapping, accepting either compose spelling."""
    declared = service.get("environment") or {}
    if isinstance(declared, list):
        pairs = (item.split("=", 1) for item in declared)
        return {key: value for key, value in pairs}
    return {key: str(value) for key, value in declared.items()}


def test_every_service_that_needs_a_broker_names_the_variable_the_code_reads() -> None:
    found = services()
    for name in BROKER_SERVICES:
        assert name in found, f"docker-compose.yml has no {name!r} service"
        env = environment(found[name])
        assert BROKER_VAR in env, (
            f"the {name} service does not set {BROKER_VAR}, which is the only variable "
            f"praxis.tasks.broker_url() reads; it will run as though no broker exists")
        assert env[BROKER_VAR].startswith("redis://"), (
            f"the {name} service sets {BROKER_VAR}={env[BROKER_VAR]!r}, which is not a redis "
            f"URL; the compose file runs a redis service and this should address it")


def test_no_service_configures_a_broker_under_a_name_nothing_reads() -> None:
    """The defect itself: a second name, set in good faith, consulted by nobody.

    Checked by shape rather than by blacklisting REDIS_URL, so the next variable invented for
    the same job fails here too.
    """
    for name, service in services().items():
        for key in environment(service):
            if key == BROKER_VAR:
                continue
            assert "BROKER" not in key.upper() and "REDIS" not in key.upper(), (
                f"the {name} service sets {key}, which reads like broker configuration and is "
                f"not {BROKER_VAR}; nothing in praxis consults any other name, so this would "
                f"be a broker that is declared and never found")


def test_the_media_volume_is_read_only_for_the_api_and_writable_for_the_worker() -> None:
    """D18 again, in the deployment: the API serves blurred media and never writes it, and
    phase 3 writes the blurred copy and deletes the original. A read-write mount on the api
    would put the deletion one bug away from a request handler."""
    found = services()
    api_mounts = [m for m in found["api"]["volumes"] if ":/media" in m]
    worker_mounts = [m for m in found["worker"]["volumes"] if ":/media" in m]

    assert api_mounts and all(m.endswith(":ro") for m in api_mounts), (
        f"the api's /media mount must be read-only, found {api_mounts}")
    assert worker_mounts and not any(m.endswith(":ro") for m in worker_mounts), (
        f"the worker writes blurred media and deletes originals, so its /media mount cannot "
        f"be read-only, found {worker_mounts}")


def test_the_vendored_weights_are_read_only_everywhere() -> None:
    """R6 and D58: weights are mounted, never downloaded, and never written."""
    for name, service in services().items():
        for mount in service.get("volumes") or []:
            if ":/app/weights" in mount:
                assert mount.endswith(":ro"), (
                    f"the {name} service mounts the weights writable: {mount}")
