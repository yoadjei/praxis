# Annotation Codebook

**Version:** v1.0-draft
**Status:** Literature-derived draft. **Not yet expert-validated.**
**Governs:** the annotation tool's label vocabulary, the model's output schema, and the
inter-rater agreement analysis.

> **Read this before annotating or implementing anything.**
>
> This codebook defines five **observable behaviours**. A behaviour is observable when two
> trained raters watching the same clip can agree that it *occurred* without having to agree on
> what it was *worth*. That is the whole test. If a proposed label requires a rater to judge
> quality, intent, or a mental state, it does not belong here.
>
> **Version discipline.** Every annotation records the codebook version it was made under. When
> this document changes, the version increments and prior annotations are not silently
> reinterpreted. The annotation tool refuses to emit a label absent from the loaded version.

---

## 0. What raters must never code

Raters do not judge, and the tool provides no way to record, any of the following:

- whether the teaching was **good**, effective, engaging, or appropriate
- the teacher's **emotion**, mood, confidence, enthusiasm, or warmth
- **learner** engagement, attention, understanding, or behaviour as individuals
- the teacher's **intent** ("she moved closer because she wanted to check on him")
- anything about a **named individual learner**

If a rater feels the need to record one of these, it goes in the free-text note, which is
excluded from all quantitative analysis and used only to revise this codebook.

**Why this matters and is not merely cautious.** The evidence this study is built on shows
machine inference tracking observable behaviour well and inferred constructs poorly: gesture
intensity at r = 0.84 against nonverbal immediacy at r = 0.44, where human raters themselves
reach only ICC 0.684 (Petković et al., 2024). Coding inferred constructs here would import that
unreliability into the ground truth and make every downstream result uninterpretable.

---

## 1. The unit of observation

| Property | Value |
|---|---|
| Clip length | **8 seconds**, non-overlapping |
| Sampling | Every clip in the session is labelled; no sub-sampling at annotation time |
| Frame rate presented | 8 fps, matching what the model sees |
| Behaviours per clip | **All five**, independently |
| Mutual exclusivity | **None.** Multiple behaviours may be present in one clip, and usually are |

**Non-scorable is per behaviour, not per clip.** A clip where the board is out of frame is
non-scorable for B5 and perfectly scorable for B1. Marking a whole clip unusable because one
behaviour cannot be judged discards good data.

**It is not per field, and that costs something.** Part of the corpus is footage of a student
teacher presenting with no learners in shot. B2 is non-scorable there because
`b2_facing_proportion` and `b2_dominant` are defined against the learner region, but
`b2_head_torso_divergence` compares the head with the torso and is perfectly measurable on the
same clip, and setting `b2_nonscorable` discards it along with the other two. The alternative is
a flag per field, which multiplies what a rater must answer for every clip in the corpus to
recover one field on a minority of them, and raises the annotation burden that §8's agreement
round has to clear. The behaviour-level flag stays, the loss is stated here rather than
absorbed quietly, and §2 of the thesis reports how many clips it affected. D85.

**Rater confidence.** Every label carries `rater_confidence` in `{certain, probable, guess}`.
Guesses are excluded from the primary agreement analysis and reported separately, because a
ground truth built partly from guesses is not a ground truth.

---

## 2. B1: Gesture production

**Construct.** Movement of the hands and arms that accompanies speech or depicts content.

**Literature anchor.** The strongest published result on this modality: Petković et al. (2024)
reached r = 0.84 for gesture intensity against human ratings, well above their r = 0.44 for the
inferred immediacy construct. B1 is expected to be the easiest of the five.

### The gesture unit

A **gesture unit** is a movement of one or both hands or forearms that either:

- departs from a rest position and returns toward one, **or**
- is held in a non-rest position for at least **1 second**.

**Rest positions** are: hands at the sides, clasped in front, clasped behind, in pockets, or
resting on a desk, podium, or chair back.

### Labels

| Field | Type | Range | Definition |
|---|---|---|---|
| `b1_present` | boolean | | At least one gesture unit occurred in the clip |
| `b1_count` | integer | 0-20 | Number of discrete gesture units |
| `b1_amplitude` | ordinal | 1-3 | The **largest** gesture in the clip: 1 = within torso width, 2 = to shoulder width, 3 = beyond shoulder width or above head |
| `b1_nonscorable` | boolean | | See conditions below |

### Positive evidence

Pointing at the board or at a learner region; tracing a shape or direction in the air; an
iconic gesture depicting content, for example indicating size or a container; beat gestures
that pulse with speech rhythm; an open-palm gesture addressing the class; counting on fingers;
a hand raised to request quiet.

### Negative evidence, do not code

Adjusting clothing, spectacles, or hair; touching the face; scratching; holding a pen or chalk
motionless; picking up or setting down an object with no communicative excursion, which is B5;
writing on the board, which is B5; wiping the board; walking with arms swinging naturally.

### Borderline rulings

| Situation | Ruling |
|---|---|
| Writing on the board with one hand, gesturing with the other | B1 present **and** B5 present. Behaviours are not exclusive |
| Pointing at the board using a piece of chalk | B1 present. B5 present only if also writing or otherwise using the material |
| Two identical beats in quick succession | Count as two units only if separated by at least 0.5 s of movement back toward rest. Otherwise one |
| A gesture starting before the clip and finishing inside it | Count it if the majority of its excursion falls inside the clip |
| Arms held wide and still for 3 seconds | One gesture unit, by the 1-second hold rule |
| Handing a book to a learner | B5, not B1. The movement is transport, not communication |

### Non-scorable when

The teacher is out of frame for more than 50 per cent of the clip; both hands are occluded for
more than 50 per cent; motion blur prevents locating the hands.

---

## 3. B2: Body orientation to class

**Construct.** The direction the teacher's head and torso face, relative to the region the
learners occupy.

**Why not eye contact.** The earlier proposal targeted eye contact. Gaze estimation at
classroom distance on a wide-angle camera is unreliable, and a measure that cannot be recovered
reliably cannot serve as ground truth. Head and torso pose is recoverable. This is a deliberate
substitution of a measurable proxy for an unmeasurable target, and it is stated rather than
hidden.

### Labels

| Field | Type | Range | Definition |
|---|---|---|---|
| `b2_facing_proportion` | continuous | 0.0-1.0 | Proportion of the clip in which the **torso** is within ±45° of the learner-region centroid |
| `b2_head_torso_divergence` | ordinal | 1-3 | Median absolute difference between head and torso direction: 1 = aligned, under 30°; 2 = 30 to 60°; 3 = over 60° |
| `b2_dominant` | categorical | see below | What the teacher predominantly faces across the clip |
| `b2_nonscorable` | boolean | | |

`b2_dominant` takes one of: `class`, `board`, `materials`, `away`.

- **`class`**: facing the learner region, including facing a single learner within it
- **`board`**: facing the writing surface
- **`materials`**: facing a desk, book, device, or projection the learners are not in line with
- **`away`**: facing a wall, door, window, or out of the room

**Raters estimate proportions in quarters**, that is 0, 0.25, 0.5, 0.75, or 1.0. Finer
judgment is not reliable by eye and pretending otherwise inflates apparent precision.

### Borderline rulings

| Situation | Ruling |
|---|---|
| Writing on the board, head turned to speak to the class | `b2_dominant = board`, divergence 3. Both fields carry information; that is why there are two |
| Circulating and facing one learner closely | `b2_dominant = class`. An individual learner is inside the learner region |
| Facing a projector screen the class also faces | `b2_dominant = materials` |
| Turning continuously, no dominant direction | Choose the direction held longest; if genuinely tied, `b2_dominant = class` and mark confidence `guess` |
| Teacher at the back of the room facing forward, learners ahead | `class`. The learner region is defined by learners, not by the front of the room |

### Non-scorable when

Fewer than five upper-body keypoints are trackable for more than 50 per cent of the clip; the
learner region was not defined for this camera setup; **no learner is in frame for more than
50 per cent of the clip**; the camera moved mid-clip.

The learner condition is not a quality rule. `b2_facing_proportion` and `b2_dominant` are both
defined against the region the learners occupy, and a recording of a student teacher
presenting to a camera has no such region - the measure has no referent rather than a hard-to-
see value. A rater who guesses a facing direction there is inventing the construct. Note that
`b2_head_torso_divergence` *is* measurable without learners, and the per-behaviour flag cannot
express that; §2 records the consequence. D85.

---

## 4. B3: Spatial position and mobility

**Construct.** Where the teacher is in the room and how they move through it.

**Why this one matters most.** B3 carries the largest expected difference between campus
microteaching and an authentic classroom. In a 10-minute peer-taught segment a candidate
largely stands still; in a real classroom they circulate. **H3 predicts degradation will be
largest for B3**, so its definitions must be tight enough that the prediction is testable.

### Zones

Zones are defined **once per camera setup**, not per session, as normalised polygons on the
image plane in coordinates where `(0,0)` is top-left and `(1,1)` is bottom-right of the frame.

| Zone | Definition |
|---|---|
| `board` | Within roughly one arm's length of the writing surface |
| `front` | The teaching area between the board and the first row of learners |
| `middle` | Among the learners, forward half |
| `back` | Among the learners, rear half, and behind them |

The teacher's position is the midpoint between their ankle keypoints, or the hip midpoint when
ankles are occluded, which is common.

### Labels

| Field | Type | Range | Definition |
|---|---|---|---|
| `b3_zone` | categorical | board / front / middle / back | Zone occupied longest in the clip |
| `b3_zone_changed` | boolean | | The teacher crossed a zone boundary during the clip |
| `b3_transitions` | integer | 0-6 | Number of boundary crossings |
| `b3_movement` | ordinal | 1-4 | 1 = stationary, under half a body-width; 2 = shifting in place; 3 = walking within a zone; 4 = traversing zones |
| `b3_proximity` | ordinal | 1-4 | Distance to the nearest learner: 1 = adjacent, within arm's reach; 2 = near, 1-2 m; 3 = mid, 2-4 m; 4 = far, over 4 m |
| `b3_nonscorable` | boolean | | |

**Proximity is coarse on purpose.** Precise depth from a single wide-angle camera is not
reliable, so four bands are the most that can be defended.

### Borderline rulings

| Situation | Ruling |
|---|---|
| Standing exactly on a zone boundary | Assign to the zone containing the ankle midpoint. If ambiguous, the zone occupied at the clip's midpoint |
| Pacing back and forth across one boundary | Count every crossing in `b3_transitions`; `b3_movement = 4` |
| Seated at a desk for the whole clip | `b3_movement = 1`; zone as occupied. B4 records the sitting |
| Teacher leaves the frame and returns | Score the visible portion if at least 70 per cent is visible, otherwise non-scorable |

### Non-scorable when

The teacher is out of frame for more than 30 per cent of the clip, a stricter threshold than B1
because position is the measure itself; zones are undefined for the setup; **no learner is in
frame for more than 30 per cent of the clip**; the camera moved.

`b3_proximity` is defined as the distance to the nearest learner. With no learner in frame
there is no nearest learner, and the band a rater would choose is a statement about the room
they imagine rather than the one recorded. D85.

---

## 5. B4: Postural stance

**Construct.** The configuration of the teacher's body.

**Expected to be the weakest.** H1 predicts B4 will show the lowest human agreement of the
five. That prediction is only meaningful if the definitions here are as tight as they can be
made, so that low agreement reflects the construct rather than a sloppy codebook.

### Labels

| Field | Type | Range | Definition |
|---|---|---|---|
| `b4_posture` | categorical | see below | Dominant posture across the clip |
| `b4_lean` | ordinal | 1-3 | Torso deviation from vertical: 1 = upright, under 10°; 2 = 10 to 30°; 3 = over 30° |
| `b4_arms` | categorical | see below | Dominant arm configuration |
| `b4_nonscorable` | boolean | | |

`b4_posture`: `standing_upright`, `standing_supported`, `seated`, `crouching`, `other`.
`b4_arms`: `crossed`, `at_sides`, `hands_clasped`, `one_raised`, `both_raised`,
`holding_object`, `indeterminate`.

**A deliberate omission.** An earlier draft included an "openness" scale coding crossed arms as
closed and open arms as open. That has been removed. Openness is an interpretation of what a
posture signifies, not a description of it, and it belongs to the class of inferred constructs
section 0 excludes. `b4_arms` records the configuration; nobody codes what it means.

### Borderline rulings

| Situation | Ruling |
|---|---|
| Leaning on a desk while on their feet | `standing_supported` |
| Perched on the edge of a desk | `seated` |
| Bending to a learner's eye level at their desk | `crouching`, even if the knees are straight |
| Arms crossed while holding a book | `holding_object` takes precedence |
| Posture changes halfway through the clip | The one held longer; if equal, mark confidence `probable` |

### Non-scorable when

The torso is occluded for more than 50 per cent of the clip; the teacher is out of frame for
more than 50 per cent; only the head and shoulders are visible.

---

## 6. B5: Board and material use

**Construct.** Engagement with instructional surfaces and artifacts.

**Why this replaces facial expressiveness.** The earlier proposal listed facial expressiveness
among its five targets while also committing that the system would not infer mental state from
facial appearance, and its own primary source reports that automated facial expression analysis
does not predict evaluation (Shirasaka et al., 2026). B5 is observable, privacy-safe, and
anchored to the Professional Practice domain of the National Teachers' Standards.

### Labels

| Field | Type | Range | Definition |
|---|---|---|---|
| `b5_board` | boolean | | Writing, drawing, erasing, or pointing at the board or a wall chart |
| `b5_material` | boolean | | Handling an instructional artifact |
| `b5_type` | categorical | see below | Dominant activity |
| `b5_duration` | ordinal | 0-3 | Seconds engaged: 0 = none, 1 = under 2 s, 2 = 2 to 5 s, 3 = over 5 s |
| `b5_nonscorable` | boolean | | |

`b5_type`: `writing`, `pointing_at_board`, `erasing`, `displaying_object`,
`demonstrating_with_object`, `distributing`, `none`.

**Instructional artifact** means a textbook, exercise book, chart, poster, model, specimen,
apparatus, calculator, phone or tablet used for teaching, or any object held up to illustrate a
point.

### Negative evidence, do not code

Personal items: a phone used privately, a bag, a water bottle, keys, a handkerchief. Moving
furniture. Handling the register or attendance book, which is administrative and specifically
excluded.

### Borderline rulings

| Situation | Ruling |
|---|---|
| Pointing at the board | B5 `pointing_at_board` **and** B1, since pointing is a gesture |
| Holding a textbook while talking, not referring to it | `b5_material = true`, `b5_type = none`. Holding is not use |
| Reading aloud from a textbook | `b5_material = true`, `b5_type = displaying_object` |
| Writing on a learner's exercise book at their desk | `b5_material = true`, `b5_type = writing`. Also B3 proximity 1 |
| Using a phone, unclear whether for teaching | Code `false` and mark confidence `guess`. Do not infer purpose |

### Non-scorable when

The board is outside the frame and the clip's activity appears board-directed; the teacher's
hands are occluded for more than 50 per cent.

---

## 7. Expected agreement, stated in advance

Predictions registered before annotation begins, so that H1 is a test rather than a description.

| Behaviour | Expected agreement | Reasoning |
|---|---|---|
| B1 Gesture | **Highest** | Discrete, visually salient, strong literature precedent at r = 0.84 |
| B3 Spatial | High | Position is geometric; zone boundaries are fixed in advance |
| B5 Board and material | Moderate to high | Clear positives, though `b5_type` boundaries will blur |
| B2 Orientation | Moderate | Proportion estimation by eye is coarse even in quarters |
| B4 Posture | **Lowest** | Categorical boundaries are genuinely fuzzy, and `b4_lean` is estimated without a reference line |

If B4 does not come out lowest, that is a finding and it is reported as one.

---

## 8. Rater calibration procedure

1. **Round 1.** All raters independently label the same 100-clip rater-calibration set drawn from at
   least four teachers and both domains.
2. Compute Krippendorff's alpha per behaviour per field. Surface every disagreement side by
   side in the tool.
3. **Codebook revision.** Add borderline rulings for the disagreements found. Increment the
   version. Record what changed and why in `DECISIONS.md`.
4. **Round 2.** New 100-clip set, same procedure.
5. **Gate.** Production annotation begins when alpha is at or above **0.667** on every
   categorical field. If two rounds do not reach it for a given field, **the field is either
   redefined or dropped, and the failure is reported.** It is not carried forward with a
   footnote.
6. Rater-calibration clips are excluded from all training and evaluation partitions.

---

## 9. Open items for expert validation

This draft is derived from the literature and from the nine supervisor attention points
recorded by Martin and Atteh (2021). It has **not** been through the expert panel required by
Phase A of the research programme. The following need panel resolution and are flagged in the
tool as provisional:

1. **Is the 8-second clip the right unit?** Shorter units raise boundary artefacts; longer ones
   hide within-clip variation.
2. **Zone definitions assume a conventional classroom layout.** Ghanaian basic-school rooms vary
   in size, seating, and whether a usable board exists. The panel should review whether four
   zones fit the range.
3. **`b3_proximity` band widths** are set by judgment, not by evidence.
4. **`b5_type` category list** is likely incomplete for the setting; the panel should add or
   merge categories.
5. **Whether B4 is worth retaining at all**, if rater calibration confirms it cannot reach the alpha
   gate.

Panel decisions replace the corresponding sections here and increment the version to v2.0.
