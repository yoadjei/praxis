# Database Schema

**PostgreSQL 16.** Managed by Alembic. This document is the source of truth; migrations are
generated to match it, never the other way round.

> **Conventions.**
> - Primary keys are **ULIDs** stored as `CHAR(26)`. Sortable by creation time, no coordination
>   needed, and unlike a serial integer they leak no count.
> - All timestamps are `TIMESTAMPTZ`, stored UTC.
> - Money, none. Free text is `TEXT`, never `VARCHAR(n)` with a guessed limit.
> - Every table carries `created_at`. Only mutable tables carry `updated_at`.
> - Deletes are forbidden on `audit_log` and `adjudications`. See section 9.

---

## 1. Institutions and people

```sql
CREATE TABLE colleges (
    college_id      CHAR(26) PRIMARY KEY,
    code            TEXT NOT NULL UNIQUE,          -- short pseudonym, e.g. "COL-A"
    region          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A basic school hosting teaching practicum. Deliberately NOT a child of colleges: one school
-- hosts students from several Colleges of Education, and the reverse holds too, so the two keys
-- are crossed on the session instead. Nesting them would make a held-out college hold out its
-- schools as well, collapsing S2 into S1. 0011, D98.
CREATE TABLE schools (
    school_id       CHAR(26) PRIMARY KEY,
    code            TEXT NOT NULL UNIQUE,          -- short pseudonym, e.g. "SCH-A"
    district        TEXT,
    region          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Pre-service teachers. Pseudonymous by construction: no name, ever.
CREATE TABLE teachers (
    teacher_id      CHAR(26) PRIMARY KEY,
    college_id      CHAR(26) NOT NULL REFERENCES colleges(college_id),
    cohort          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_teachers_college ON teachers(college_id);

CREATE TABLE users (
    user_id         CHAR(26) PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    display_name    TEXT NOT NULL,
    role            TEXT NOT NULL
                    CHECK (role IN ('rater','supervisor','researcher','admin','candidate')),
    college_id      CHAR(26) REFERENCES colleges(college_id),
    teacher_id      CHAR(26) REFERENCES teachers(teacher_id),  -- set only for 'candidate'
    password_hash   TEXT NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT candidate_has_teacher
        CHECK (role <> 'candidate' OR teacher_id IS NOT NULL)
);
```

**The identity mapping lives outside this database.** Whatever links `teacher_id` to a real
person is held by the institution on paper or in a separate encrypted store the application
cannot reach. This schema must remain safe to dump for analysis.

---

## 2. Consent

Ingest refuses without a resolvable, active consent record. This is enforced in application
code **and** by the foreign key on `sessions`.

```sql
CREATE TABLE consent_records (
    consent_id      CHAR(26) PRIMARY KEY,
    subject_type    TEXT NOT NULL
                    CHECK (subject_type IN ('teacher','guardian_class','evaluator')),
    teacher_id      CHAR(26) REFERENCES teachers(teacher_id),
    college_id      CHAR(26) NOT NULL REFERENCES colleges(college_id),

    -- Act 843 s.27(1) requires purpose and recipients to be stated.
    purpose         TEXT NOT NULL,
    recipients      TEXT NOT NULL,
    scope           TEXT NOT NULL
                    CHECK (scope IN ('microteaching','classroom','both')),

    granted_on      DATE NOT NULL,
    expires_on      DATE,
    withdrawn_on    DATE,
    document_ref    TEXT NOT NULL,     -- where the signed paper form is filed
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW active_consents AS
SELECT * FROM consent_records
WHERE withdrawn_on IS NULL
  AND (expires_on IS NULL OR expires_on >= CURRENT_DATE);
```

**Guardian consent is per class, not per pupil**, because pupils are never individually
identified or modelled. One `guardian_class` record covers a recorded class, and the
off-camera opt-out seating zone handles non-consenting pupils physically rather than in
software.

**Withdrawal.** Setting `withdrawn_on` must cascade: a nightly job flags affected sessions,
suppresses them from all views, and queues media deletion. Derived annotations are retained in
de-identified form only if the consent text permitted it; otherwise they are deleted too.

---

## 3. Media and sessions

```sql
CREATE TABLE media_objects (
    media_sha256    CHAR(64) PRIMARY KEY,
    relative_path   TEXT NOT NULL,      -- media/<sha[:2]>/<sha>.mp4, under PRAXIS_MEDIA_ROOT
    bytes           BIGINT NOT NULL,
    duration_s      DOUBLE PRECISION NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    fps             DOUBLE PRECISION NOT NULL,
    has_audio       BOOLEAN NOT NULL,
    is_blurred      BOOLEAN NOT NULL DEFAULT false,
    reachable       BOOLEAN NOT NULL DEFAULT true,   -- external volume may be unmounted
    last_checked_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
    session_id      CHAR(26) PRIMARY KEY,
    teacher_id      CHAR(26) NOT NULL REFERENCES teachers(teacher_id),
    college_id      CHAR(26) NOT NULL REFERENCES colleges(college_id),
    consent_id      CHAR(26) NOT NULL REFERENCES consent_records(consent_id),
    media_sha256    CHAR(64) NOT NULL REFERENCES media_objects(media_sha256),

    domain          TEXT NOT NULL CHECK (domain IN ('microteaching','classroom')),
    school_id       CHAR(26) REFERENCES schools(school_id),
    subject         TEXT,
    grade_level     TEXT,
    recorded_on     DATE NOT NULL,

    -- Domain-shift covariates. Measured at capture, never imputed.
    camera_distance_m       DOUBLE PRECISION,
    room_area_m2            DOUBLE PRECISION,
    pupil_count             INTEGER,
    ambient_noise_dba       DOUBLE PRECISION,
    teacher_movement_range_m DOUBLE PRECISION,

    quality_verdict TEXT NOT NULL CHECK (quality_verdict IN ('pass','warn','fail')),
    quality_detail  JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- A researcher's decision not to annotate this session, with its author and its ground.
    -- Not on the Session contract: the contract is the upload request body, and an exclusion
    -- is decided long after ingest. 0010, D96.
    excluded_at     TIMESTAMPTZ,
    excluded_by     CHAR(26),
    exclusion_reason TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- All three or none. An exclusion with no author cannot be questioned later and one with
    -- no ground is indistinguishable from unfinished work.
    CONSTRAINT an_exclusion_names_a_person_a_time_and_a_ground CHECK (
        (excluded_at IS NULL AND excluded_by IS NULL AND exclusion_reason IS NULL) OR
        (excluded_at IS NOT NULL AND excluded_by IS NOT NULL AND exclusion_reason IS NOT NULL)
    ),
    CONSTRAINT an_exclusion_ground_is_not_blank CHECK (
        exclusion_reason IS NULL OR length(btrim(exclusion_reason)) > 0
    ),

    -- The school is what decides a practicum session's shift level. Nullable because
    -- microteaching happens on a college campus and has no basic school. 0011, D98.
    CONSTRAINT a_classroom_session_names_its_school CHECK (
        domain <> 'classroom' OR school_id IS NOT NULL
    )
);
CREATE INDEX idx_sessions_teacher ON sessions(teacher_id);
CREATE INDEX idx_sessions_domain  ON sessions(domain);
CREATE INDEX idx_sessions_college ON sessions(college_id);
```

`domain` is the most important column in the database. Every evaluation stratifies on it and
`teacher_id` is the split key, so **neither may ever be null**.

`school_id` is the second. On the site shift axis it is what places a session on the ladder, and a
classroom session without one would fall quietly into the training pool - which is the leak the
ladder exists to measure against. It is nullable only because a microteaching session genuinely has
no basic school, and `a_classroom_session_names_its_school` carries the rule that does apply.

---

## 4. Preprocessing artifacts

```sql
CREATE TABLE camera_setups (
    setup_id        CHAR(26) PRIMARY KEY,
    college_id      CHAR(26) NOT NULL REFERENCES colleges(college_id),
    label           TEXT NOT NULL,
    -- Normalised polygons, (0,0) top-left to (1,1) bottom-right.
    zone_board      JSONB NOT NULL,
    zone_front      JSONB NOT NULL,
    zone_middle     JSONB NOT NULL,
    zone_back       JSONB NOT NULL,
    learner_region  JSONB NOT NULL,     -- used by B2
    defined_by      CHAR(26) NOT NULL REFERENCES users(user_id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE sessions ADD COLUMN setup_id CHAR(26) REFERENCES camera_setups(setup_id);

-- Pose is bulk numeric data. Keep it out of Postgres; store a pointer.
CREATE TABLE pose_artifacts (
    session_id      CHAR(26) PRIMARY KEY REFERENCES sessions(session_id),
    relative_path   TEXT NOT NULL,      -- .npz: keypoints (F,P,17,3), track_ids (F,P)
    frame_count     INTEGER NOT NULL,
    sampled_fps     DOUBLE PRECISION NOT NULL,
    model_version   TEXT NOT NULL,
    sha256          CHAR(64) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE teacher_tracks (
    session_id      CHAR(26) PRIMARY KEY REFERENCES sessions(session_id),
    track_id        INTEGER,                        -- null: nobody has identified one yet
    proposed_by     TEXT NOT NULL DEFAULT 'heuristic'
                        CHECK (proposed_by IN ('heuristic','human')),
    heuristic_score DOUBLE PRECISION,               -- the best score, cleared or not
    confirmed_by    CHAR(26) REFERENCES users(user_id),
    confirmed_at    TIMESTAMPTZ,
    reason          TEXT NOT NULL,                  -- the heuristic's own explanation
    candidates      JSONB NOT NULL,                 -- the ranking it argued over
    CONSTRAINT must_be_confirmed_before_use
        CHECK (confirmed_by IS NULL OR confirmed_at IS NOT NULL),
    -- A person cannot confirm an absence. Without this the two nullables combine into a
    -- confirmation of nothing, and `confirmed_teacher_track` returns NULL for a session a
    -- reviewer signed off, which is R1 failing quietly.
    CONSTRAINT nothing_is_confirmed_without_a_track
        CHECK (confirmed_by IS NULL OR track_id IS NOT NULL)
);

-- Learner evidence: aggregate only. No per-learner row exists anywhere. Rule R1.
CREATE TABLE learner_aggregates (
    session_id      CHAR(26) NOT NULL REFERENCES sessions(session_id),
    t_start_s       DOUBLE PRECISION NOT NULL,
    hands_raised    INTEGER,
    gross_motion    DOUBLE PRECISION,
    person_count    INTEGER,
    PRIMARY KEY (session_id, t_start_s)
);
```

**There is deliberately no `learner_tracks` table.** `test_no_learner_tracks_persisted` asserts
that no table name matching `%learner%track%` exists and that `learner_aggregates` holds no
identifier column. R1 is enforced by the absence of a place to violate it.

**`teacher_tracks` holds one row per preprocessed session, proposal or not.** A null `track_id`
is the stored abstention: the heuristic cleared nothing above its floor and nobody has identified
a teacher yet. The row exists so that the abstention can be read, acted on, and confirmed against
- before 0009 it was expressed by the row's absence, which also put it out of reach of
`confirm_teacher`. D48, D90.

`candidates` is written once per preprocess run and is a snapshot of how the score was reached,
not a mirror of current state:

```json
{
  "source": "detection",
  "zones_available": false,
  "ranked": [
    {"track_id": 4, "presence_fraction": 0.454, "median_area_fraction": 0.151,
     "front_zone_fraction": 0.0, "score": 0.227}
  ],
  "truncated": 0
}
```

`source` is where the three fractions were measured from: `detection` for a live preprocess run,
`pose_artifact` for a run re-emitted from a stored `.npz` after the original video was deleted
under D18, and `unrecorded` for the four rows written before the column existed. The two are not
the same measurement - a detection box and a keypoint hull differ - so the source is recorded
rather than the difference being absorbed. `zones_available` is what was true at the time, so a
session that later acquires a camera setup does not retroactively make an old score look
comparable. `truncated` counts candidates dropped past
`preprocess.teacher_id.max_candidates_recorded`.

---

## 5. Clips and annotation

```sql
CREATE TABLE codebook_versions (
    version         TEXT PRIMARY KEY,       -- 'v1.0-draft', 'v2.0'
    document_sha256 CHAR(64) NOT NULL,      -- hash of CODEBOOK.md as loaded
    is_active       BOOLEAN NOT NULL DEFAULT false,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX one_active_codebook ON codebook_versions(is_active) WHERE is_active;

CREATE TABLE clips (
    clip_id         CHAR(26) PRIMARY KEY,
    session_id      CHAR(26) NOT NULL REFERENCES sessions(session_id),
    clip_index      INTEGER NOT NULL,
    t_start_s       DOUBLE PRECISION NOT NULL,
    t_end_s         DOUBLE PRECISION NOT NULL,
    is_calibration  BOOLEAN NOT NULL DEFAULT false,  -- excluded from train and eval
    UNIQUE (session_id, clip_index)
);
CREATE INDEX idx_clips_session ON clips(session_id);

CREATE TABLE annotations (
    annotation_id   CHAR(26) PRIMARY KEY,
    clip_id         CHAR(26) NOT NULL REFERENCES clips(clip_id),
    rater_id        CHAR(26) NOT NULL REFERENCES users(user_id),
    codebook_version TEXT NOT NULL REFERENCES codebook_versions(version),
    behaviour       TEXT NOT NULL CHECK (behaviour IN ('B1','B2','B3','B4','B5')),

    labels          JSONB NOT NULL,      -- validated against the codebook version's schema
    is_nonscorable  BOOLEAN NOT NULL DEFAULT false,
    rater_confidence TEXT NOT NULL
                    CHECK (rater_confidence IN ('certain','probable','guess')),
    note            TEXT,                -- qualitative only, never analysed quantitatively
    seconds_spent   DOUBLE PRECISION,

    round_label     TEXT,                -- 'calibration_1', 'calibration_2', 'production'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (clip_id, rater_id, behaviour, codebook_version)
);
CREATE INDEX idx_annotations_clip ON annotations(clip_id);
CREATE INDEX idx_annotations_rater ON annotations(rater_id);
```

`labels` is JSONB because the field set differs per behaviour and changes with codebook
versions. **The application validates it against the loaded codebook version before insert**,
and rejects any key the version does not define.

---

## 6. Splits

```sql
CREATE TABLE split_manifests (
    manifest_id     CHAR(26) PRIMARY KEY,
    config_sha256   CHAR(64) NOT NULL,
    train_teachers  TEXT[] NOT NULL,
    val_teachers    TEXT[] NOT NULL,
    test_teachers   TEXT[] NOT NULL,
    -- Which ladder built this manifest. S2 is all classroom footage on the domain axis and the
    -- held-out college on the site axis, so a manifest that does not say which it used cannot
    -- be read back. 0011, D98.
    shift_axis      TEXT NOT NULL DEFAULT 'domain',
    heldout_college CHAR(26) REFERENCES colleges(college_id),
    heldout_school  CHAR(26) REFERENCES schools(school_id),
    shift_sessions  TEXT[] NOT NULL DEFAULT '{}',   -- held out at a level, never trained on
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Rule R2, enforced by the database, not by convention.
    CONSTRAINT teachers_disjoint CHECK (
        NOT (train_teachers && val_teachers) AND
        NOT (train_teachers && test_teachers) AND
        NOT (val_teachers && test_teachers)
    ),
    CONSTRAINT a_shift_axis_is_domain_or_site CHECK (shift_axis IN ('domain','site')),

    -- On the site axis the levels are named sites, and naming neither declares a ladder with
    -- nothing on it. On the domain axis a held-out school has no rung to belong to.
    CONSTRAINT a_shift_axis_names_the_sites_it_holds_out CHECK (
        (shift_axis = 'domain' AND heldout_school IS NULL) OR
        (shift_axis = 'site' AND
         (heldout_school IS NOT NULL OR heldout_college IS NOT NULL))
    )
);
```

The `&&` array-overlap operator makes R2 a constraint the database refuses to violate. An
agent cannot ship a contaminated split even by accident.

`heldout_school` and `heldout_college` are **crossed, not nested**. One basic school hosts
teaching-practice students from several Colleges of Education, so holding out a college does not
hold out a school. Nesting them would make S2 a superset of S1 by construction and the two rungs
indistinguishable. D98.

---

## 7. Models, detections, confidence

```sql
CREATE TABLE model_versions (
    model_version   TEXT PRIMARY KEY,   -- 'b-resnet50-tcn-v3-ens5'
    manifest_id     CHAR(26) NOT NULL REFERENCES split_manifests(manifest_id),
    config_sha256   CHAR(64) NOT NULL,
    weights_sha256  CHAR(64) NOT NULL,
    device_trained  TEXT NOT NULL CHECK (device_trained IN ('mps','cuda','cpu')),
    ensemble_size   INTEGER NOT NULL DEFAULT 1,
    calibration_method TEXT CHECK (calibration_method IN ('none','temperature','ensemble','mc_dropout')),
    temperature     JSONB,              -- per-behaviour scalar T
    metrics         JSONB NOT NULL,     -- frozen evaluation report
    is_deployed     BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE detections (
    detection_id    CHAR(26) PRIMARY KEY,
    clip_id         CHAR(26) NOT NULL REFERENCES clips(clip_id),
    model_version   TEXT NOT NULL REFERENCES model_versions(model_version),
    behaviour       TEXT NOT NULL CHECK (behaviour IN ('B1','B2','B3','B4','B5')),
    predicted       JSONB NOT NULL,     -- same shape as annotations.labels
    codebook_version TEXT NOT NULL REFERENCES codebook_versions(version),

    -- ConfidenceState. Rule R3: these columns are NOT NULL for a reason.
    raw_prob            DOUBLE PRECISION NOT NULL,
    calibrated_prob     DOUBLE PRECISION NOT NULL,
    confidence_method   TEXT NOT NULL
                        CHECK (confidence_method IN ('temperature','ensemble','mc_dropout')),
    epistemic           DOUBLE PRECISION,
    ood_score           DOUBLE PRECISION NOT NULL,
    ood_flag            BOOLEAN NOT NULL,
    validated_domain    TEXT NOT NULL
                        CHECK (validated_domain IN ('microteaching','classroom','unknown')),

    -- Routing outcome, computed at write time so the API cannot leak a suppressed value.
    gate_outcome    TEXT NOT NULL CHECK (gate_outcome IN ('present','suppress','escalate')),
    evidence_ref    TEXT,
    explanation_ref TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (clip_id, model_version, behaviour)
);
CREATE INDEX idx_detections_clip ON detections(clip_id);
CREATE INDEX idx_detections_gate ON detections(gate_outcome);
```

**`codebook_version` is stored beside `predicted`, exactly as `annotations.codebook_version`
is stored beside a rater's label.** The two carry the same vocabulary and are compared against
each other, so a revision that adds a field must not retroactively reinterpret either. See D31.

**`gate_outcome` is stored, not computed at read time.** A suppressed detection's `predicted`
value must never be serialised to a client, and the serialiser keys off this column. Computing
it in the view layer would put the decision in the place most likely to be refactored wrongly.

---

## 8. Adjudication and the reliance study

```sql
CREATE TABLE adjudications (
    adjudication_id CHAR(26) PRIMARY KEY,
    detection_id    CHAR(26) NOT NULL REFERENCES detections(detection_id),
    reviewer_id     CHAR(26) NOT NULL REFERENCES users(user_id),
    action          TEXT NOT NULL CHECK (action IN ('confirm','edit','reject')),
    edited_value    JSONB,
    rationale       TEXT,

    -- Reliance-study instrumentation. Not product features; do not remove as unused.
    seconds_on_item DOUBLE PRECISION NOT NULL,
    evidence_replays INTEGER NOT NULL DEFAULT 0,
    revealed_indication BOOLEAN NOT NULL,   -- did the evidence-first gate get opened
    arm             TEXT CHECK (arm IN ('dossier','no_dossier')),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT rationale_required_for_change
        CHECK (action = 'confirm' OR (rationale IS NOT NULL AND length(rationale) >= 10))
);
CREATE INDEX idx_adjud_detection ON adjudications(detection_id);
CREATE INDEX idx_adjud_reviewer ON adjudications(reviewer_id);

-- Seeded errors for automation-bias probes. Reviewers are never shown this table.
CREATE TABLE seeded_errors (
    seed_id         CHAR(26) PRIMARY KEY,
    detection_id    CHAR(26) NOT NULL UNIQUE REFERENCES detections(detection_id),
    original_value  JSONB NOT NULL,
    corrupted_value JSONB NOT NULL,
    corruption_kind TEXT NOT NULL,
    session_run_id  CHAR(26) NOT NULL,
    rng_seed        BIGINT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
REVOKE SELECT ON seeded_errors FROM praxis_reviewer;
```

---

## 9. The audit trail, and how append-only is actually enforced

Rule R5 is not an application convention. It is a database guarantee.

```sql
CREATE TABLE audit_log (
    audit_id        BIGSERIAL PRIMARY KEY,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor_user_id   CHAR(26) REFERENCES users(user_id),
    actor_role      TEXT,
    event_type      TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    prev_hash       CHAR(64),
    row_hash        CHAR(64) NOT NULL
);
CREATE INDEX idx_audit_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_time   ON audit_log(occurred_at);

CREATE OR REPLACE FUNCTION forbid_mutation() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'append-only table: % not permitted on %', TG_OP, TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_log_no_update
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();

CREATE TRIGGER adjudications_no_update
    BEFORE UPDATE OR DELETE ON adjudications
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();

REVOKE UPDATE, DELETE, TRUNCATE ON audit_log, adjudications FROM praxis_app;
```

**Two layers on purpose.** The `REVOKE` stops the ordinary application role. The trigger stops
everything short of a superuser dropping it, including a migration that grants privileges back
by accident. `test_audit_trail_immutable` attempts both an `UPDATE` and a `DELETE` and asserts
the database raises.

**Hash chaining.** `prev_hash` is the previous row's `row_hash`, and

```
row_hash = sha256( frame(prev_hash) || frame(occurred_at) || frame(actor_user_id)
                || frame(actor_role) || frame(event_type) || frame(entity_type)
                || frame(entity_id) || frame(payload) )
```

This makes silent tampering by a superuser detectable: `scripts/verify_audit_chain.py` walks the
chain and reports the first break.

**The encoding is part of the formula.** Two implementations that disagree on any of the
following compute different hashes for the same row, and a false alarm is worse than no alarm
because it teaches people to ignore the real one. So:

* `frame(s)` is the 8-byte big-endian length of `s` in UTF-8, followed by those bytes. Fields
  are **framed, not concatenated**: plain concatenation cannot distinguish `("ab", "c")` from
  `("a", "bc")`, so a character could be moved across a field boundary with every hash still
  valid.
* `actor` in the original formula was ambiguous, since the table holds two actor columns. Both
  are framed, separately. A role escalated from `supervisor` to `admin` must not go unnoticed.
* `entity_type` is included although the original formula omitted it. Without it, editing which
  *kind* of thing an event concerned leaves `row_hash` valid.
* `occurred_at` is ISO 8601 normalised to UTC, microsecond precision, `Z` suffix, e.g.
  `2026-09-13T09:00:00.000000Z`. A naive timestamp is refused rather than assumed local.
* `payload` is compact JSON with sorted keys and no whitespace: `json.dumps(payload,
  sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
* For the genesis row `prev_hash` is SQL `NULL` in storage and framed as the **empty string**
  when hashing, not the text `"NULL"`, which is a plausible value of some other field.

`praxis/audit/chain.py` is the reference implementation and
`tests/unit/test_audit.py` asserts each of these points separately.

**A correction is a new row.** To revise an adjudication, insert a new one referencing the same
detection. The API returns the latest by `created_at`; the history stays.

### Events that must be logged

| Event type | When |
|---|---|
| `session.ingested` | Media accepted |
| `session.rejected` | Quality gate or consent refusal, with reason |
| `teacher_track.confirmed` | Human confirms the teacher identity |
| `annotation.created` | Any rater label |
| `codebook.activated` | A version becomes active |
| `model.deployed` | A model version goes live |
| `detection.served` | A detection is shown to a reviewer, with the gate outcome |
| `adjudication.created` | Confirm, edit, or reject |
| `export.generated` | A report leaves the system, with recipient |
| `consent.withdrawn` | Withdrawal recorded |
| `media.deleted` | Deletion executed, with reason |
| `config.changed` | A configuration value is changed |

`config.changed` was missing from this table while BUILD-SPEC Phase 8 required "every config
change" to be logged. Added, and recorded as D34.

**Each event's payload has a required field set**, declared in `praxis/audit/events.py` and
checked when the row is appended rather than when the trail is finally asked a question. The
acceptance test "replaying the audit log reconstructs the final state exactly" is only
meaningful if each payload carries what its own replay needs, and an event recorded without
those fields is a hole that shows up years later. `adjudication.created` therefore carries the
whole adjudication, including `seconds_on_item`, `evidence_replays` and
`revealed_indication`, rather than a reference to a row in another table: the trail is meant
to prove that table, not depend on it.

---

## 10. Roles and row-level access

```sql
CREATE ROLE praxis_app        LOGIN;   -- the API
CREATE ROLE praxis_reviewer   NOLOGIN; -- mapped by the API, not a DB login
CREATE ROLE praxis_researcher NOLOGIN;
CREATE ROLE praxis_admin      NOLOGIN;
```

| Role | Sees | Cannot |
|---|---|---|
| `rater` | Clips assigned for annotation, media playback | See detections, see other raters' labels, see model output |
| `supervisor` | Sessions at their college, detections subject to the gate, own adjudications | See `seeded_errors`, see other colleges |
| `candidate` | Own adjudicated reports only | See raw detections, see unadjudicated output, see other candidates |
| `researcher` | De-identified aggregates, splits, metrics, exports | See `users.email`, link a `teacher_id` to a person |
| `admin` | Everything including audit | Mutate audit or adjudications |

**Raters must not see model output.** If a rater sees a detection before labelling, the ground
truth is contaminated by the thing it is meant to evaluate. Enforce in the API layer and assert
in `tests/integration/test_rater_isolation.py`.

---

## 11. Retention and deletion

| Data | Retention | Deletion trigger |
|---|---|---|
| Unblurred media | **Never persisted.** Deleted in the same transaction that writes the blurred copy | Automatic, Phase 3 |
| Blurred media | Per the consent text, default 5 years | Consent withdrawal or expiry |
| Pose artifacts | Same as blurred media | Same |
| Annotations | Retained de-identified | Only on withdrawal where consent did not permit retention |
| Audit log | **Permanent** | Never |
| `users.email` | While active | Account closure |

`scripts/run_retention.py` runs nightly, is idempotent, and logs every action to `audit_log`.
It never deletes without writing the deletion event first.

---

## 12. Migration discipline

1. One logical change per migration.
2. Every migration has a tested `downgrade()`. An untested downgrade is not a downgrade.
3. Migrations that drop or rename a column touched by an existing model version require an
   entry in `DECISIONS.md` explaining what happens to prior detections.
4. **Never** write a migration that grants `UPDATE` or `DELETE` on `audit_log` or
   `adjudications`. CI greps for this and fails the build.
