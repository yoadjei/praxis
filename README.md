# PRAXIS

Vision-only measurement of teaching behaviour, with a model that reports when it does not know.

PRAXIS records a lesson, reports what the teacher's body did, and attaches a calibrated confidence
to every claim. It withholds output rather than guessing when a recording falls outside the
conditions it was validated on. The question behind it is whether a system's confidence degrades
faster than its accuracy when it moves from a controlled setting to a real classroom.

## What it observes, and what it refuses to

Five behaviours, all properties of the body and its position: gesture production, body orientation
to the class, spatial position and mobility, postural stance, and board and material use.

Three refusals are part of the design rather than limitations of it.

- **No speech.** Nothing it reports depends on the teacher's accent or language.
- **No facial expression, and no inference about a mental state.** Faces are blurred at ingest.
- **No learner is identified, tracked or scored.** Learners appear only as anonymous counts.

## The seven invariants

Each has a test in `system/tests/test_invariants.py` that runs in CI on every push.

| | Rule |
|---|---|
| R1 | Only the teacher is classified. Learner evidence exists only as anonymous aggregate counts. |
| R2 | No teacher appears in more than one data partition. Splits are enforced in code. |
| R3 | Every model output carries a confidence state. A bare label never leaves the inference layer. |
| R4 | Calibration is reported wherever accuracy is reported. |
| R5 | The audit trail is append-only. Corrections are new rows. |
| R6 | No network access at inference time. |
| R7 | Every run is reproducible from one config file. |

## Install

Python 3.11 or newer, PostgreSQL 16 or newer, and ffmpeg on `PATH`.

```bash
cd system
pip install -r requirements.txt
alembic upgrade head
```

Pose estimation needs ONNX weights, which are not in this repository because they are
licence-encumbered. Vendor them explicitly:

```bash
python scripts/vendor_weights.py
```

## Run

Phases run one at a time from a frozen config, and each writes a manifest recording the config
checksum, the device and the seed.

```bash
python scripts/run_phase.py --list
python scripts/run_phase.py --config configs/default.yaml --phase 1
```

A phase whose inputs are absent abstains and names what was missing, rather than failing or
producing a partial result.

## Test

```bash
python -m pytest -q
```

Tests needing a database read `PRAXIS_TEST_DSN`. Without it the suite still runs and the affected
checks abstain loudly, so a check that did not run never looks like one that passed. Setting it to
an unreachable server is an error rather than a skip, for the same reason.

## Frontend

```bash
cd system/frontend
npm install
npm run dev
```

## Licence

No licence is granted. This is the software artifact of doctoral research, published for
inspection rather than reuse.
