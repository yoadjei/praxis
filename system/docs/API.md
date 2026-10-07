# API Specification

**FastAPI.** All paths prefixed `/api/v1`. JSON in, JSON out. Auth by bearer token.

> **The rule this document exists to protect.** A detection whose `gate_outcome` is `suppress`
> must never have its predicted value appear in a response body. Not greyed out, not nulled at
> render time, not present-but-hidden. **Absent from the payload.** Every serialiser is written
> against this rule and `tests/integration/test_no_bare_predictions.py` asserts it against the
> raw JSON, not against Python objects.

---

## 1. Conventions

| Aspect | Rule |
|---|---|
| IDs | ULID strings, 26 chars |
| Timestamps | ISO 8601, UTC, `Z` suffix |
| Pagination | `?limit=` (default 50, max 500) and `?cursor=` (opaque). Never offset paging |
| Errors | RFC 7807 problem details: `type`, `title`, `status`, `detail`, `instance` |
| Idempotency | Every `POST` that creates something accepts `Idempotency-Key`. Replaying returns the original result with `200`, not a duplicate |
| Versioning | Path-based. A breaking change means `/api/v2`, never a silent shape change |

### Status codes actually used

`200` ok · `201` created · `202` accepted, job queued · `204` no content ·
`400` malformed · `401` unauthenticated · `403` role forbids · `404` not found or not visible
to this role · `409` conflict · `422` valid JSON, invalid content · `429` rate limited ·
`503` dependency unavailable, for example the media volume is unmounted

**`404` is deliberately overloaded with "not visible to your role."** A supervisor at College A
probing for a College B session gets `404`, not `403`, because `403` confirms the resource
exists.

---

## 2. Authentication and roles

```
POST   /auth/login              {email, password} -> {access_token, refresh_token, role}
POST   /auth/refresh            {refresh_token}   -> {access_token}
POST   /auth/logout             204
GET    /auth/me                 -> {user_id, display_name, role, college_id}
```

Access tokens are JWTs, 30-minute expiry, signed HS256 with a key from the environment.
Refresh tokens are opaque, stored hashed, 7-day expiry, single use.

**Role matrix.** Every endpoint below lists its permitted roles. Where a role is absent, the
request returns `403`, or `404` where existence itself is sensitive.

| Role | One-line summary |
|---|---|
| `rater` | Annotates clips. **Never sees model output.** |
| `supervisor` | Reviews and adjudicates sessions at their own college |
| `candidate` | Reads their own adjudicated reports only |
| `researcher` | Exports, splits, metrics. Never sees identities |
| `admin` | Everything, including audit. Cannot mutate append-only tables |

---

## 3. Ingest

```
POST   /sessions                                    [supervisor, researcher, admin]
```

Multipart. Fields: `file`, plus a JSON `metadata` part matching the `Session` contract.

**Order of operations, and it matters.** Consent is checked *before* a single byte is written
to disk. A refusal must leave the system byte-identical to its prior state.

```
422  consent_ref absent, unknown, withdrawn, expired, or scope mismatched
     -> {"type":"/errors/consent-invalid","detail":"consent scope 'microteaching'
         does not cover session domain 'classroom'"}
409  media_sha256 already present for this teacher and date
201  -> {session_id, media_sha256, quality: {verdict, checks:[...]}}
```

```
GET    /sessions                       ?domain=&college_id=&teacher_id=&cursor=
GET    /sessions/{session_id}
GET    /sessions/{session_id}/quality
POST   /sessions/{session_id}/reprocess              [researcher, admin]  -> 202
DELETE /sessions/{session_id}                        [admin]              -> 202
```

`DELETE` does not delete. It queues a retention job, writes `media.deleted` to the audit log,
and returns `202`. Hard deletion happens in the nightly retention run so it is always logged
before it is done.

### Media playback

```
GET    /media/{media_sha256}/stream     [rater, supervisor, candidate, admin]
```

Range-request streaming of the **blurred** media only. There is no endpoint that serves
unblurred media, because unblurred media does not persist. Returns `503` when
`media_objects.reachable` is false, with a detail naming the volume.

---

## 4. Preprocessing and teacher confirmation

```
GET    /sessions/{id}/pose                           [researcher, admin]
GET    /sessions/{id}/tracks/candidates              [supervisor, researcher, admin]
       ?thumbnails=1..12                             (stills located per candidate, default 5)
GET    /sessions/{id}/tracks/{track_id}/thumbnail    [supervisor, researcher, admin]
       ?frame=&width=64..1024                        -> image/jpeg
POST   /sessions/{id}/tracks/confirm                 [supervisor, researcher, admin]
       {track_id} -> 200
GET    /media/{id}/frame  ?at_seconds=&width=        -> image/jpeg
```

`/tracks/candidates` returns each candidate track with its heuristic score, the three signals
behind it, and a thumbnail strip. **Confirmation is mandatory before a session enters annotation
or inference.** Endpoints downstream return `409` with `detail: "teacher track not confirmed"`
until it happens; `scripts/plan_annotation.py` refuses to plan such a session and says so.

`proposal`, `confirmation` and `ranking` are three separate keys and are never merged into one
"teacher". The heuristic's pick is a guess and the reviewer's is a judgement, and no object in
this system may carry an unconfirmed identification.

**A thumbnail is addressed by frame, not by a position in the strip.** The strip is `thumbnails`
frames spread across a track's presence, so a position means different frames for different strip
lengths; `/tracks/candidates` bakes the frame into each `url`. The crop rectangle is computed
server-side from the pose artefact and is never accepted from the caller - a client-supplied
rectangle would let the screen show any region of the footage under a track number. D95.

**Three states are not a ranking, and each has its own slug.** `409 /errors/not-preprocessed`
(no pose artefact; run phase 3), `409 /errors/proposal-not-recorded` (preprocessed before 0009
gave a declined proposal a row; run `scripts/reemit_candidates.py`), and a `200` whose
`ranking.available` is `false` with the remedy in `ranking.detail` (the row predates the
candidates column). A single "not available" would send an operator after the wrong thing. D48.

`/media/{id}/frame` serves one still of the blurred video for any caller that needs to show a
person what a session looks like without downloading it. Both image endpoints answer `404`, not
`400`, for a timestamp past the end of the file: the frame is a resource, and asking for one that
is not there is not a server fault.

**The actor's user id must be a ULID.** `audit_log.actor_user_id` is `CHAR(26)` and blank-pads
anything shorter, which would make every later verification of the chain report a break, so a
token naming a non-ULID user is refused with `401` rather than written. A token with a role and
no user id is accepted where the endpoint permits an unknown actor; confirmation does not, because
a confirmation is attributed to a person. D93.

```
GET    /camera-setups                  ?college_id=
POST   /camera-setups                                [supervisor, researcher, admin]
PATCH  /camera-setups/{setup_id}                     [researcher, admin]
```

Zone polygons are normalised to `(0,0)`-`(1,1)`. A setup already referenced by a session with
annotations cannot be edited; create a new setup instead, and the API returns `409` saying so.

---

## 5. Annotation

Rater-facing, and the most privacy-sensitive surface in the system.

```
GET    /annotation/assignments                       [rater]
       -> next batch of clips assigned to this rater
GET    /annotation/clips/{clip_id}                   [rater]
POST   /annotation/clips/{clip_id}/labels            [rater]
       {behaviour, labels:{...}, is_nonscorable, rater_confidence, note, seconds_spent}
GET    /annotation/codebook/active                   [rater, researcher, admin]
```

**Three hard constraints on this surface:**

1. **The clip payload contains no model output.** No `detections` key, no confidence, nothing.
   Contaminating a rater with the prediction destroys the ground truth the prediction is
   evaluated against.
2. **A rater cannot see another rater's labels**, including their own from a prior codebook
   version, until the round is closed.
3. **`labels` is validated against the active codebook version.** Any key that version does not
   define returns `422` naming the offending key. This is how the codebook stays authoritative.

```
POST   /annotation/rounds/{round}/close              [researcher, admin]
GET    /annotation/irr                 ?round=&behaviour=          [researcher, admin]
```

`/annotation/irr` returns per-behaviour, per-field Krippendorff's alpha, ICC where continuous,
and quadratically weighted kappa where ordinal, each with bootstrap confidence intervals, plus
the rater-artefact checks. **It also returns `design_fully_crossed: false` and a warning when
the design is not fully crossed**, because ICC underestimates reliability in that case.

---

## 6. Detections and the routing gate

```
GET    /sessions/{id}/detections       ?behaviour=       [supervisor, researcher, admin]
```

**The serialisation contract, stated as the shapes actually returned.**

`gate_outcome = "present"`:

```json
{
  "detection_id": "01J...", "session_id": "01J...", "behaviour": "B1",
  "gate_outcome": "present",
  "t_start_s": 128.0, "t_end_s": 136.0,
  "predicted": {"b1_present": true, "b1_count": 3, "b1_amplitude": 2,
                "b1_nonscorable": false},
  "codebook_version": "v1.0-draft",
  "confidence": {
    "calibrated_prob": 0.91, "method": "ensemble", "epistemic": 0.04,
    "ood_flag": false, "validated_domain": "microteaching",
    "plain_language": "High confidence. This recording resembles what the model was checked on."
  },
  "evidence_ref": "/api/v1/evidence/01J...",
  "model_version": "b-resnet50-tcn-v3-ens5",
  "explanation_ref": "/api/v1/explanations/01J..."
}
```

**`predicted` is the behaviour's codebook fields, all of them.** It is not a single number and
it is not a subset: the shape is exactly the field set `docs/CODEBOOK.md` defines for that
behaviour under `codebook_version`, so B1 returns four fields, B3 returns six, and a partial
prediction is refused at construction rather than served as a complete one. `b1_nonscorable`
belongs in the list above and is easy to omit by eye; it is required. `codebook_version`
accompanies it so that a later revision never silently reinterprets an earlier detection, which
is the same rule `annotations.codebook_version` enforces for a rater's label.

`gate_outcome = "suppress"`: **no `predicted` key exists at all.**

```json
{
  "detection_id": "01J...", "session_id": "01J...", "behaviour": "B1",
  "gate_outcome": "suppress",
  "t_start_s": 128.0, "t_end_s": 136.0,
  "suppression_reason": "out_of_distribution",
  "confidence": {
    "ood_flag": true, "validated_domain": "unknown",
    "plain_language": "This recording is unlike those the model was checked on. No suggestion is offered."
  },
  "evidence_ref": "/api/v1/evidence/01J..."
}
```

`codebook_version`, `model_version` and `explanation_ref` are absent too: each of them says
something about a prediction that was withheld.

`suppression_reason` is one of `low_confidence`, `out_of_distribution`, `model_abstained`.
`model_abstained` is returned when the model itself marked the clip non-scorable for that
behaviour, `bN_nonscorable` true, which is the model declining to answer rather than
answering uncertainly.

`gate_outcome = "escalate"`: same shape as `present`, plus `"requires_second_reviewer": true`.

**The three outcomes are tested in a fixed order, and suppression wins every tie.** Abstention
first, then the OOD flag, then the confidence floor, then disagreement; anything remaining is
presented. A detection can satisfy several of these at once and exactly one outcome is
returned. Escalation puts the indication in front of a *second* reviewer, so it is never
reached by a detection the system has already flagged as outside its validated domain.

**`plain_language` is required, not decorative.** Teacher educators in this setting cannot be
assumed fluent with calibrated probability, so the numeric value is available and the sentence
is what the interface leads with.

### Evidence and explanation

```
GET    /evidence/{detection_id}          -> clip bounds, keypoint overlay, thumbnails
GET    /explanations/{detection_id}      -> {method, payload, fidelity_verdict}
```

`method` is `gradcam` or `intrinsic`. **`fidelity_verdict` is one of `passed_sanity_checks`,
`failed_sanity_checks`, `not_tested`.** If Grad-CAM failed, the API serves the intrinsic
explanation instead and says so, rather than serving a map the project has evidence not to
trust.

---

## 7. Review and adjudication

```
GET    /review/queue                   ?college_id=      [supervisor]
GET    /review/sessions/{id}                             [supervisor]
POST   /review/detections/{id}/reveal                    [supervisor]
POST   /review/detections/{id}/adjudicate                [supervisor]
       {action, edited_value?, rationale?, seconds_on_item, evidence_replays}
```

**`/reveal` implements the cognitive forcing function.** In the `dossier` arm the reviewer must
call it before the model's indication is returned; the initial payload carries evidence only.
In the `no_dossier` comparison arm the indication is included from the start. The arm is set
per participant by the study configuration, never by the client.

```
422  action in ('edit','reject') without a rationale of at least 10 characters
409  detection already adjudicated by this reviewer  -> returns the existing adjudication
403  reviewer's college does not match the session's
```

Adjudications are **append-only**. There is no `PATCH` or `DELETE`. To revise, `POST` again;
the API returns the latest and the history remains queryable at
`GET /review/detections/{id}/adjudications`.

---

## 8. Dashboard and export

```
GET    /candidates/{teacher_id}/timeline       [supervisor, candidate(self), admin]
GET    /colleges/{id}/cohort                   [supervisor, researcher, admin]
POST   /exports/candidate-report               [supervisor, admin] -> 202 {job_id}
GET    /exports/{job_id}                       -> 200 pdf | 202 pending
```

**Three constraints:**

1. Timeline points carry confidence bands, never bare point estimates, and sessions that were
   fully suppressed appear **marked as suppressed rather than omitted**. Omitting them would
   make the record look more complete than it is.
2. `/cohort` refuses any cell below `min_cell_size` (default 5) and returns `422` rather than a
   suppressed-but-inferable cell.
3. **`/exports/candidate-report` raises `422` if any included detection lacks an
   adjudication.** No unreviewed model output leaves the system, ever.

A `candidate` calling `/candidates/{teacher_id}/timeline` for a `teacher_id` other than their
own receives `404`.

---

## 9. Research endpoints

```
GET    /research/splits                        [researcher, admin]
POST   /research/splits                        [researcher, admin]
GET    /research/models                        [researcher, admin]
POST   /research/models/{version}/deploy       [admin]
GET    /research/evaluation      ?model_version=&shift_level=   [researcher, admin]
POST   /research/seeded-errors                 [researcher, admin]
GET    /research/export/analysis-dataset       [researcher, admin]
```

`POST /research/splits` returns `422` if teacher-disjointness fails, echoing the offending
teacher IDs. The database constraint would reject it anyway; the API check exists to give a
usable error rather than a constraint violation.

`GET /research/evaluation` returns accuracy **and** calibration together. **There is no
parameter that returns accuracy alone.** Rule R4 is enforced by the absence of the option.

`GET /research/export/analysis-dataset` returns the reliance-study dataset: per-item time,
reveal order, replays, edit distance between model output and final text, seeded-error catch
rate, and workload scores. It contains no email, no name, and no free-text note.

---

## 10. Operations

```
GET    /health          -> {status, db, redis, media_volume, device}
GET    /ready
GET    /audit           ?entity_type=&entity_id=&from=&to=     [admin]
GET    /audit/verify                                           [admin]
```

`/health` reports `media_volume: "unreachable"` when the external SSD is unmounted, and
`device` as `mps`, `cuda`, or `cpu`, so a run's device is discoverable without reading logs.

`/audit/verify` walks the hash chain and returns `{intact: bool, first_break: audit_id|null}`.

---

## 11. What the API deliberately does not offer

| Absent endpoint | Why |
|---|---|
| `POST /sessions/{id}/score` | The system does not score teaching. There is no score |
| `GET /detections?include_suppressed_values=true` | Would defeat the routing gate |
| `PATCH /adjudications/{id}` | Append-only. Corrections are new rows |
| `GET /learners/...` | No learner entity exists. Rule R1 |
| `GET /research/evaluation?metric=accuracy` | Rule R4. Calibration always accompanies accuracy |
| Any endpoint returning unblurred media | It is not persisted |
| Webhooks or outbound callbacks | Rule R6, offline operation |

An agent asked to add any of these should stop and report the conflict rather than implement it.
