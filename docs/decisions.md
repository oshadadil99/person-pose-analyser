# Design decisions

A running log of the main design choices, what else I considered, and why I
went the way I did. When a decision changes, the old one stays here marked
**superseded**, with the reason for the change.

---

## D1. Process the video in two passes, caching perception

**Date:** 2026-10-02 | **Status:** active

Pass 1 runs YOLO over the video and saves every detection to
`perception.jsonl`. Pass 2 (features, rules, state machine, agent, alerts)
only reads that file.

- **Alternative:** one streaming pass doing everything frame by frame.
- **Why:** YOLO is the only slow step (minutes); everything after it takes
  seconds. Caching means thresholds can be tuned and re-run instantly, and
  runs are reproducible. It also fits the agent: since we analyse a recorded
  video, looking *forward* in time is allowed.

## D2. Sample at 5 fps, keep exact timestamps

**Date:** 2026-10-02 | **Status:** active

- **Alternative:** process every frame (25 to 30 fps).
- **Why:** getting out of bed takes seconds, so 5 fps loses nothing that
  matters and is 5 to 6 times cheaper. Timestamps come from
  `frame_idx / native_fps`, not a counter, so durations stay correct for any
  source frame rate (tested with 24 fps, where the step is 4.8 frames).

## D3. Store all coordinates in original video pixels

**Date:** 2026-10-02 | **Status:** active

Inference runs on a frame shrunk to at most 960 px wide; results are scaled
back before saving.

- **Why:** the bed and seat regions are defined on the full frame. If the two
  used different scales, "are the hips in the bed" would be silently wrong.

## D4. Pose model: YOLO11n-pose + ByteTrack

**Date:** 2026-10-02 | **Status:** active

- **Alternatives:** two-stage (top-down) pose such as a detector + HRNet;
  MediaPipe (single person only).
- **Why:** one network pass gives boxes and 17 keypoints for every person,
  which we need because a caregiver can be in the room. Nano is fast enough
  on a 4 GB GTX 1650 Ti (~21 frames/s measured). ByteTrack is built into
  Ultralytics and keeps IDs through short occlusions because it also matches
  low-confidence detections. Switching to `yolo11s-pose` is a config change
  if keypoints turn out noisy.

## D5. Config is one YAML file loaded as a plain dict

**Date:** 2026-10-02 | **Status:** active

- **Alternative:** typed dataclasses per config section.
- **Why:** simpler to read and explain, nothing to keep in sync. Downside:
  a typo in a key only fails at runtime, so the loader checks that all
  sections exist at startup.

## D6. Bed region: manual polygon drawn once per camera

**Date:** 2026-10-02 | **Status:** superseded by D7

The original plan was a small OpenCV tool where I click the bed corners on
the first frame and save the polygon as JSON.

## D7. Detect the bed and seats automatically with YOLO11s-seg

**Date:** 2026-10-03 | **Status:** active (replaces D6)

Sample about 20 frames across the video, run a COCO segmentation model on
each, merge the masks per object across frames, convert each mask to a
polygon, and save `scene.json` with one bed and a list of seats, plus a
preview image to check it.

**Why the change from D6:**
1. The person doesn't only use the bed. They also sit on chairs, stools or a
   couch, and "sitting on a chair" is one of the difficult cases. A manual
   bed polygon says nothing about seats, so chair sitting could only be
   inferred from the thigh angle, which breaks when the camera faces the
   person head-on.
2. The system should run on a new video without manual setup.

**Why a separate model at all:** YOLO11-pose was trained on COCO-keypoints,
where only people are labelled. Its head outputs a single "person" class, so
it cannot report beds or chairs. The COCO detection/segmentation models share
the same backbone and neck but have a different head and weights trained on
80 classes, including bed and chair. There are no official weights doing pose
and furniture together, and training one would need a labelled dataset.

**Why segmentation and not just boxes:** seen from an angle, a bed's box also
covers floor beside it, so someone standing next to the bed would count as
"in bed". A mask follows the bed's real outline. For chairs a box would be
fine, but one model handles both.

**Why only ~20 frames:** furniture is static and the camera is fixed, so
there is nothing new to learn from every frame. Merging over many frames
also handles the person covering part of the bed in some of them, and drops
objects seen in only one or two frames (false detections).

**Why the small (s) model and not nano:** it runs ~20 times per video, not
thousands, so accuracy matters more than speed.

| Model | Params | COCO mask mAP |
|---|---|---|
| yolo11n-seg | 2.9M | 32.0 |
| yolo11s-seg (chosen) | 10.1M | 37.8 |
| yolo11m-seg | 22.4M | 41.5 |

**Classes kept:** bed as the bed; chair, couch, bench and toilet as seats
(a bedside commode is often detected as a toilet). Configurable.

**Alternatives considered:**
- *SAM (Segment Anything):* excellent masks but no labels; it needs a prompt
  (clicks or boxes from another detector). Heavier, and a new dependency.
- *YOLOE (open-vocabulary YOLO):* finds objects by text, e.g. "stool",
  "wheelchair". Solves the stool gap but adds a text-encoder package and
  another architecture. Kept as a fallback.
- *Gemini detection:* already in the stack, but it would make a core input
  depend on an online API, and the offline mode (`--vlm none`) must work.

**Known limits:**
- COCO has no "stool" class. Stools are often detected as "chair" but not
  always. If that fails on the real video, add YOLOE or a one-time cached
  Gemini query for extra seats. (Confirmed in testing, see D16.)
- Furniture is assumed static. A chair moved mid-video is missed;
  re-scanning periodically is future work.
- If no bed is found (e.g. very dark footage) the run warns, and the manual
  polygon tool from D6 is kept as an optional override.

## D8. Seats feed into SITTING_OUTSIDE_BED, no new state

**Date:** 2026-10-03 | **Status:** active

Upright posture with hips on a detected seat is classified as
`SITTING_OUTSIDE_BED`, even if the thigh angle is ambiguous.

- **Alternative:** a new `SITTING_ON_CHAIR` state.
- **Why:** the assignment's state list already covers it, and it asks not to
  add states without need.

## D9. Patient selection moved from Phase 2 to Phase 3

**Date:** 2026-10-02 | **Status:** active

The patient is the track that spends the most time in or near the bed.
New track IDs that appear close to the patient's last position within a few
seconds are re-linked to the patient (ByteTrack can issue a new ID after an
occlusion or a return to view).

- **Why moved:** it needs the bed region, which only exists from Phase 3.
  Perception stays a pure "what YOLO saw" step.

## D10. Hidden hips fall back to the box centre

**Date:** 2026-10-03 | **Status:** active

- **Why:** under a blanket the hip keypoints are often missing or low
  confidence. The box centre is a weaker but still useful guess for "is this
  person on the bed". Features record which one was used.

## D11. Gemini models: 2.5 Flash, with 2.5 Flash-Lite as fallback

**Date:** 2026-10-02 | **Status:** active

Vertex AI, location `global`, service-account key read from the environment.

- **Why:** a smoke test on my project showed both models answer image
  questions, return valid JSON, and do function calling, in both `global`
  and `us-central1`. `gemini-3-flash-preview` and `gemini-flash-latest`
  only worked in `global`, and a preview model can change under me, so it is
  only an optional comparison. Flash-Lite is cheaper and faster, which suits
  a fallback when the main model is rate-limited.

## D12. UNKNOWN time counts as out of bed, but is also reported separately

**Date:** 2026-10-02 | **Status:** active

- **Alternative:** a third bucket so in-bed + out-of-bed + unknown = total.
- **Why:** the assignment's summary only has in-bed and out-of-bed totals.
  Reporting `unknown_sec` separately means nothing is hidden.

## D13. Added `lap` to requirements

**Date:** 2026-10-02 | **Status:** active

- **Why:** ByteTrack uses it for the matching step. Ultralytics installed it
  by itself during the first run; listing it makes installs reproducible.
  It is part of the tracker already chosen, not a new tool.

## D14. Patient re-linking also accepts a "lone returner"

**Date:** 2026-10-03 | **Status:** active

Besides the short-gap rule (new track within 3 s and close to where the
patient was last seen), a new track also joins the patient chain if it is the
only person in view when it appears.

- **Why:** when the patient leaves the camera view and comes back minutes
  later, ByteTrack gives a new ID and the person reappears at the door, far
  from where they were. The short-gap rule can't link that, and "temporarily
  leaving camera view" is one of the assignment's test cases.
- **Trade-off:** a caregiver who walks in alone while the patient is out of
  view would be taken for the patient. Fixing that properly needs appearance
  re-identification (future work). The rule can be switched off in config
  (`patient.relink_lone_returner`).

## D15. Scene merging rules

**Date:** 2026-10-03 | **Status:** active

- **Bed needs only 2 sightings, seats need 30% of frames.** The person lying
  on the bed often stops the model detecting it, so a strict presence rule
  could throw the bed away. There is only one bed, so "the bed seen most
  often" is safe. Seats are smaller and false detections are more common, so
  they must show up consistently.
- **Convex hull of the voted mask.** A person or blanket cuts notches out of
  the bed mask in some frames. Beds and chairs are convex from almost any
  view, so the hull fills those notches back in.
- **One detection per object per frame.** Found during testing: two chairs
  side by side overlapped enough (box IoU above 0.3) to be merged into one
  object, which then reported being "seen in 195% of frames". Each cluster now
  takes at most one detection per frame, highest confidence first.
- **A "couch" on top of the bed is dropped,** since it is the bed mis-labelled.

## D16. Stool gap confirmed, fallback not added yet

**Date:** 2026-10-03 | **Status:** open

Smoke test on a public COCO bedroom photo: the bed was found (0.95) with an
outline that follows its slanted side, but a small upholstered bench/stool
under the window was not detected at all, even at confidence 0.05. On a
living-room photo, all four dining chairs were found.

- **What this means:** ordinary chairs are fine, stools and benches without a
  back are not reliable with COCO classes. This is the gap predicted in D7.
- **Decision:** wait for the real test video. If the patient sits on a stool
  there, add a fallback (YOLOE text prompt "stool", or a one-time cached
  Gemini query). Until then, sitting on an undetected stool still works
  through the thigh-angle rule; the seat region is extra evidence, not the
  only evidence.

## D17. Module layout differs from the original plan

**Date:** 2026-10-03 | **Status:** active

The plan had one `bed_region.py`. It became three small modules:
`geometry.py` (polygon/box maths, no knowledge of beds),
`scene.py` (furniture detection) and `patient.py` (who is the patient).

- **Why:** after D7 the region code covers seats too, and patient selection
  was moved here (D9). Keeping them apart keeps each file short and each one
  testable on its own.

## D18. Thigh angle gets an "unsure band"; legs out of frame give UNKNOWN

**Date:** 2026-10-03 | **Status:** active (refines D8)

First version: thigh angle >= 55 deg = sitting, otherwise standing. A first
run on my test video (test1.mp4) showed two failure patterns:

| Time | Truth | First version said | Why |
|---|---|---|---|
| 12-19 s | sitting on bed edge, facing the camera | STANDING, conf 0.86 | thighs point at the lens, measured 28-37 deg instead of ~90 |
| 21-28 s, 46 s | standing close to the camera | SITTING_ON_BED | legs cut off by the image bottom, hips overlap the bed in 2D, and "legs hidden + in bed" meant sitting |

Measured on the same video: clearly standing thighs were 0-10 deg.

**Change:**
- Thigh angle has three zones: >= 55 deg clearly bent (sitting), <= 20 deg
  clearly straight (standing/walking), and an unsure band in between. In the
  band, moving fast means walking; hips on the bed or a detected seat means
  sitting (people don't stand on beds); otherwise standing. Confidence is
  halved, so these frames are later picked up by the agent.
- New feature `box_truncated`: the box reaches the bottom of the image. If the
  legs can't be measured and the box is truncated, the legs are out of frame,
  not under a blanket, so the answer is UNKNOWN (or WALKING if clearly
  moving). The old "upright + legs hidden + in bed = sitting" rule now only
  applies when the box is not truncated, which is the blanket case it was
  meant for.

**Result on test1.mp4:** edge sitting 12-19 s now SITTING_ON_BED (low
confidence), standing near the camera now mostly UNKNOWN/WALKING instead of
a confident wrong SITTING_ON_BED. Lying and chair sitting unchanged.

- **Alternatives considered:** lowering the sitting threshold to ~25 deg
  (would turn walking strides into sitting); thigh-length/torso-length ratio
  (measured 0.61 both for edge sitting and standing on this camera, so it
  doesn't separate them).
- **Why it's acceptable to say UNKNOWN here:** from a single 2D frame with no
  legs visible, sitting vs standing genuinely can't be told. The assignment
  asks for UNKNOWN rather than a forced guess, and these are exactly the
  moments the agent (look back/forward, ask the VLM) is for.
- **Overfitting note:** the 20 deg value comes from one video. It will be
  re-checked on the full test recording.

## D19. States live in one enum (`src/states.py`)

**Date:** 2026-10-03 | **Status:** active

- **Why:** frame rules, state machine, events, alerts and evaluation all use
  the state names. One `str` enum avoids typos like "SITTING_ON_BEd" silently
  becoming a new state, and serialises to plain strings in JSON.

## D20. How the timeline is smoothed

**Date:** 2026-10-03 | **Status:** active

Order: blanket hold -> confidence-weighted vote -> minimum dwell with
back-dating -> OUT_OF_BED marking -> segments.

- **Confidence-weighted vote (1.5 s window, centred).** A plain majority vote
  lets a run of shaky 0.15 guesses outvote a clear 0.8 one. Weighting by
  confidence fixes that. UNKNOWN has confidence 0 by definition, so it gets a
  small fixed weight (0.2): a real guess beats it, but a long unknown stretch
  stays unknown. The window is centred (uses a little of the future), which
  is fine because we analyse a recorded video.
- **Minimum dwell, back-dated.** A new state must last its dwell time before
  it is accepted. The accepted change is moved back to the first frame of the
  new state. Without back-dating every segment would start 1-3 s late and the
  duration errors would add up.
- **Segments tile the range.** Each sampled frame covers [its time, next
  frame's time), the last runs to the end of the range. Durations therefore
  always add up to the analysed duration (tested).

**Alternatives considered:** an HMM / Viterbi decoder (principled, but needs
transition probabilities I'd have to invent without training data, and is
harder to explain and tune); a median filter on state IDs (states aren't
ordered numbers, so a median is meaningless).

## D21. Blanket hold: keep LYING_IN_BED for up to 30 s when the patient becomes unreadable

**Date:** 2026-10-03 | **Status:** active

If the last real state was LYING_IN_BED and the frame is UNKNOWN because the
patient isn't detected, or the pose is unreadable while the box is still on
the bed, keep LYING_IN_BED (confidence 0.3) for up to `occlusion_hold_sec`.

- **Why:** a duvet pulled over the head makes YOLO lose the person, but
  nobody leaves a bed without first being seen sitting up and standing.
  "Partially hidden by blankets" is one of the assignment's test cases.
- **Why a time limit:** after 30 s of seeing nothing we really don't know any
  more, and the assignment prefers UNKNOWN to a guess. Long UNKNOWN then
  becomes a MONITOR decision (Phase 7).

## D22. OUT_OF_BED is decided per unknown stretch, not per frame

**Date:** 2026-10-03 | **Status:** active

An UNKNOWN stretch becomes OUT_OF_BED if the last known state was standing,
walking or sitting outside the bed, and at least half of its frames have no
patient detected.

- **Why not per frame:** the first version did it per frame, and on test1 a
  single partly visible frame (a hand at the image edge) split an
  out-of-bed stretch, leaving a 0-second UNKNOWN sliver in the timeline.
- **Phase 6 note:** "last known state was out of bed" will be replaced by
  "after a confirmed bed exit", which is what the assignment defines.

## D23. `--start` / `--end` options on the main command

**Date:** 2026-10-03 | **Status:** active

- **Why:** in test1.mp4 the last ~11 s are the camera being picked up, which
  breaks the fixed-camera assumption. Trimming in the command is simpler and
  more honest than editing the video, and the evaluation can state exactly
  which range was analysed (`analysed_range_sec` in summary.json).

## D24. Walking at 20-27 s on test1 came out UNKNOWN: three fixes

**Date:** 2026-10-03 | **Status:** active (changes part of D18 and D20)

On test1 I stand up from the bed edge and pace slowly between the bed and the
chair, close to the camera, legs below the image. The timeline said UNKNOWN
for 20-27 s. Three causes, three changes:

1. **Legs out of frame -> UNKNOWN was too strict (changes D18).** When the
   box is cut off by the image bottom and there is no seat under the hips,
   sitting isn't realistic, so the rule now says STANDING (or WALKING if
   moving) with low confidence. The 2D bed overlap is ignored in that case:
   a person that close to the camera is in front of the bed, not on it.
   Over a detected seat it stays UNKNOWN (standing in front of a chair looks
   the same in 2D).
2. **Walking threshold 0.25 -> 0.15 box-heights/s.** Measured on test1:
   standing still 0.02-0.10, slow pacing/turning 0.05-0.26 (net displacement
   over 1 s, so turning partly cancels). 0.15 sits above the standing noise.
   Elderly people walk slowly, so erring low suits the real use case.
3. **UNKNOWN vote weight 0.2 -> 0.05 (changes D20).** The legs-hidden guesses
   have confidence ~0.10-0.20, so a 0.2 weight for UNKNOWN let unreadable
   frames outvote real guesses, the opposite of what D20 intended. At 0.05 a
   real guess wins, while one stray guess inside a long unknown stretch
   still doesn't (both cases tested).

**Result on test1 (0-47 s):** 20-23 STANDING, 23-32 WALKING, unknown time
0 s (was 10 s). Lying, edge sitting and chair sitting unchanged; the few edge
sitting frames that now read WALKING are smoothed away.

**Risk:** all three values come from one 59 s clip. Re-check on the full
test recording before trusting them.

## D25. Bed events are detected on a coarse in-bed / out-of-bed layer

**Date:** 2026-10-03 | **Status:** active (completes D22)

States are mapped to a bed status: IN_BED (lying/sitting on bed), OUT
(standing, walking, chair, lying outside, out of view) or UNKNOWN. Events are
transitions between IN_BED and OUT stretches; UNKNOWN in between doesn't
break a sequence but lowers the event's confidence (x0.7).

**Bed exit** (in bed -> up -> moving away):
- `start_time` = first moment out of bed (standing up).
- `confirmed_time` = "moving away" (walking, chair, lying outside, out of
  view) has lasted 3 s; or, if the person only stands there, 10 s after
  standing up.
- Back in bed before `confirmed_time` = not an exit. That is what filters
  "stand briefly and sit back down" and "two steps and back". Sitting up and
  edge sitting never leave IN_BED, so they can't start an exit.

**Return to bed** (out -> sits on bed -> lies down):
- `start_time` = sits (or lies) on the bed.
- `confirmed_time` = lies down, or 10 s of sitting on the bed, whichever is
  first. Sitting on the bed for a few seconds and leaving again is not a
  return.

**"Official" location.** Found by tests: if events are just raw IN_BED <-> OUT
switches, a 4 s sit on the bed followed by walking off produced a fake exit.
The detector keeps where the patient officially is, which only changes when
an event is confirmed. Exits can only start from officially in bed, returns
only from officially out.

**OUT_OF_BED state.** Now strictly "not visible after a confirmed bed exit"
(assignment definition). If the person disappears without a confirmed exit
(e.g. the video starts with them walking), that time is UNKNOWN.

- **Alternative considered:** detecting exits per frame from "hip distance
  from bed edge" crossing a threshold. On test1 a person standing close to
  the camera overlaps the bed in 2D, so distance is unreliable; the segment
  sequence (which already went through smoothing and dwell times) is more
  robust and easier to explain.
- **Result on test1 (0-47 s):** one bed exit, start 00:00:20 (stands up from
  the edge), confirmed 00:00:26 (walking), sitting_on_bed -> walking,
  confidence 0.17 (low because the legs were out of frame while walking;
  a candidate for the agent to verify).

## D26. Alert engine: rules produce time windows; decision = highest active window

**Date:** 2026-10-03 | **Status:** active

Six rules (see docs/alert_rules.md). Each produces a window that starts when
the condition has lasted long enough and ends when the condition ends. The
level at any time is the highest active window. Outputs: overall decision,
every fired rule with its reason, and a decision timeline.

- **Why windows and not one decision per video:** the assignment talks about
  whether "an event requires monitoring or an alert". A single verdict for a
  20-minute video hides when it happened; a list of fired rules with times is
  what a carer would act on, and the overall level is still one value.
- **Why rules and not the LLM:** safety decisions must be predictable and
  testable. Every alert can be traced to a rule and a duration.
- **"Edge sitting" covers all sitting on the bed.** From 2D I can't reliably
  separate edge sitting from sitting up in the middle of the bed (on test1
  the hips during edge sitting were well inside the detected bed outline).
  Sitting up awake for 3+ minutes is worth a look either way.
- **Absence is measured from a confirmed bed exit**, not from any
  "not in bed" time. Otherwise a long UNKNOWN in bed (e.g. under a duvet past
  the hold time) would count as absence. Long UNKNOWN has its own rule.
- **Caregiver downgrade only on absence rules**, never on possible_fall.
- **Possible fall is geometry-only for now.** CLAUDE.md asks for the agent/VLM
  to confirm LYING_OUTSIDE_BED. That comes in Phase 8/9; until then the rule
  relies on the state machine's 3 s dwell plus its own 20 s limit.
- **Result on test1 (0-47 s):** overall MONITOR, from the confirmed bed exit
  at 00:26 to the end. NORMAL before.

## D27. Gemini works with an API key (AI Studio) or Vertex AI, chosen in config

**Date:** 2026-10-04 | **Status:** active (extends D11)

I test on Vertex AI using GCP credits, but the project is meant to be run
with a plain Gemini API key from AI Studio. Both go through the same
`google-genai` client; only how the client is created differs:

    vlm.provider: aistudio  ->  genai.Client(api_key=GEMINI_API_KEY)
    vlm.provider: vertex    ->  genai.Client(vertexai=True, project=..., location=...)
    vlm.provider: none      ->  no Gemini calls, offline rule-based agent

- **Same model names on both** (`gemini-2.5-flash`, fallback
  `gemini-2.5-flash-lite`), so switching provider changes nothing else.
- **Default for delivery: `aistudio`.** Whoever runs it only needs to put
  `GEMINI_API_KEY` in `.env`. `--vlm vertex` / `--vlm none` override it.
- **Rate limits:** the AI Studio free tier allows far fewer requests per
  minute than Vertex. The disk cache, minimum delay between calls, retry
  with backoff and the Flash-Lite fallback are there for that, and the agent
  only calls Gemini for ambiguous moments.
- **Keys:** read from the environment only (`.env`, gitignored); never in
  code, config or committed files.

## D28. Investigator agent: plain-Python bounded loop, rule policy first

**Date:** 2026-10-04 | **Status:** active

Triggers pick the ambiguous moments (possible fall, low-confidence bed event,
UNKNOWN >= 5 s, a second person, flicker). For each, a loop asks a policy for
the next tool call or a conclusion, at most 4 tool calls. Tools are plain
functions over the processed video (look_back, look_forward,
check_bed_overlap, get_pose_summary, count_people, ask_vlm). Conclusions can
fill an UNKNOWN stretch, turn a false "lying outside the bed" into lying in
bed, confirm/reject a bed event, or just note what was found. Events are
then re-detected on the corrected timeline and the alert rules run last.

- **No agent framework.** The loop is ~20 lines (`investigate`). Easy to
  explain, test and debug; nothing hidden.
- **Policy is swappable.** `policy_rules.py` is deterministic and offline;
  Phase 9 adds an LLM policy that chooses tools by function calling, with
  the same tools and the same budget. Comparing the two is part of the
  evaluation.
- **The VLM is the last tool.** Every plan tries the cheap timeline tools
  first and only asks the VLM when they leave the question open.
- **Safety asymmetry.** The agent can only remove a fall alert with positive
  evidence that the person is on the bed (hips at the bed edge, or the VLM
  says so). "Can't tell" keeps the alert. Missing a fall is worse than a
  false alarm.
- **Unresolved changes nothing.** The state stays as it was; the alert rules
  treat it as usual (long UNKNOWN still gives MONITOR).
- **Why look_forward is allowed:** we analyse a recorded video. In a live
  system the equivalent is waiting a few seconds before concluding, which is
  what `confirmed_time` already models.
- **Result on test1 (0-47 s):** one trigger (the bed exit, confidence 0.17
  because the legs were out of frame). look_back: lying then sitting on the
  bed; look_forward: standing, walking. Confirmed, confidence 0.95, 2 tool
  calls. On the untrimmed video it also noted a flicker zone and a second
  "person" (a hand while the camera was being picked up).

## D29. ask_vlm: cached, retried, validated, never fatal

**Date:** 2026-10-04 | **Status:** active

The VLM gets 2-4 frames (downscaled to 512 px JPEG, patient marked with a
green box so a caregiver isn't mistaken for them) and a question, and must
answer with JSON matching a schema: state (one of our states), confidence,
answer, reason.

- **Cache** on disk keyed by hash(model + prompt + frame bytes). Reruns and
  ablations are free and reproducible. Failures are not cached.
- **Retries** on 429/5xx with backoff 2-4-8-16 s, then the Flash-Lite
  fallback (2 retries). Other errors (e.g. 400) skip straight to the
  fallback. Minimum 1 s between calls.
- **Validation:** unknown state, non-numeric confidence or broken JSON count
  as a failed attempt. Confidence is clamped to 0-1.
- **Never fatal:** if everything fails the tool returns UNKNOWN / 0.0 /
  vlm_unavailable and the agent treats the question as unanswered.
- **Why mark the patient instead of drawing the bed:** the box tells the
  model who to look at. Drawing the detected bed outline could bias it
  toward our own (possibly wrong) bed region, which is what it's supposed to
  double-check.
- **Smoke test on test1 (Vertex, gemini-2.5-flash):** edge sitting at 15 s
  -> SITTING_ON_BED, chair at 40 s -> SITTING_OUTSIDE_BED, lying at 5 s ->
  LYING_IN_BED, all confidence 1.0, 2.5-6.6 s per call. Repeat from cache:
  0.24 s, no API call.

## D30. LLM policy: Gemini picks the tools, code keeps the safety rules

**Date:** 2026-10-04 | **Status:** active

Same loop, tools and 4-call budget as the rule policy. Each turn Gemini gets
the task plus all previous tool calls and results, and must answer with one
function call (mode ANY): a tool or `conclude`. When the budget is used up
only `conclude` is allowed. Thinking is turned off (budget 0): faster, and
the "thought" argument on every call already gives the reasoning for the
trace.

- **Stateless turns.** The conversation is rebuilt from the steps each turn,
  so the cache key is simply hash(model + task + steps), and a failed call
  can fall back to the rule policy cleanly.
- **Code checks the conclusion (`sanitize`).** Outcome must fit the trigger
  type; state changes only where the rule policy could make them; a possible
  fall can only be dismissed with evidence in the tool results (hips at the
  bed, or the VLM seeing the person on the bed). The LLM never decides an
  alert.
- **Finding from the first real run:** with a short task description,
  Gemini confirmed my bed exit after looking back only 0.5 s and never
  looking forward (confidence 0.8, reasoning "stood up with hips inside the
  bed"). That's a weak investigation the rule policy wouldn't make. Adding
  what counts as evidence ("a real exit needs in bed before AND moving away
  after, look back ~10 s and forward ~15 s; one moment is never enough")
  fixed it: look_back 10 s, look_forward 15 s, confirmed 0.9. Lesson: the
  LLM is only as careful as the task it's given, and its conclusions need
  checking by code.
- **Limit seen on the untrimmed video:** while the camera was being handled,
  a "patient" detection with keypoint confidence 0.16 (actually a hand) was
  reported by Gemini as "patient sitting outside the bed". The tools return
  numbers like keypoint confidence, but the model didn't weigh them.
- **Rule vs LLM policy** is part of the evaluation (Phase 10).

## D31. Lying = horizontal OR lined up with the bed's long axis

**Date:** 2026-10-04 | **Status:** active (refines the lying rule)

First run on test2 (13 min): from 0:09 to 1:02 I was lying in bed and the
system said STANDING, then flickered between lying and sitting until 3:08.
The bed points diagonally toward the camera, so a person lying along it
measures only 49-55 deg from vertical in 2D (threshold 60), and with straight
legs the rules chose STANDING. This is the perspective problem flagged in
the plan.

Measured on test2: bed long axis 64 deg (min-area rectangle around the bed
polygon). Lying along it: 49-55 deg (other lying positions 56-84 deg).
Sitting, standing, walking and chair sitting: at most ~19 deg.

**Rule added:** hips in bed AND torso >= 35 deg AND torso within 20 deg of the
bed's long axis -> lying. Plain "torso > 60 deg" still works as before.

- **Why it's general, not tuned to one clip:** it uses the bed's own geometry,
  whatever its angle in the image, instead of a fixed number. Upright states
  are far below 35 deg, so the margin is large.
- **Result:** 0:06-5:14 is now one LYING_IN_BED segment (labels: 0:07-5:09,
  including 1:36 under the blanket). test1 unchanged.

## D32. Camera moves are detected and the scene is re-detected per position

**Date:** 2026-10-04 | **Status:** active (fixes a D7 limitation)

In test2 the phone was nudged at about 8:10 (~45 px sideways), so after that
the bed and chair outlines didn't line up. Example: sitting on the chair at
11:39-12:23 was read as sitting on the bed, which also produced a fake
return and exit.

Every 2 s a small grey frame is compared with a reference by phase
correlation (global shift; the room dominates over a moving person). A shift
over 20 px for 2 checks in a row starts a new camera position, and the bed
and seats are detected again for each position. scene.json now stores one
scene per camera position (old single-scene files still load).

- **Measured:** test2 move found at 8:12, test1's phone pick-up at 0:52; before
  the moves the shift stayed at <= 2 px. Costs ~1 min per 13 min of video, cached.
- **Result:** chair sitting 11:39-12:24 now SITTING_OUTSIDE_BED; the fake
  return/exit around it is gone.
- **Limits:** translation only (no rotation/zoom model); a very slow drift
  would be missed; moves during darkness are only seen once it's light again.

## D33. test2 labels and an honest note on evaluation

**Date:** 2026-10-04 | **Status:** active

Labels in data/ground_truth/test2_*.csv come from my own script of what I did,
checked against frames. Decisions: "unknown" in my script = out of camera
view (OUT_OF_BED); the dark 12:23-12:52 and the camera set-up 0:00-0:07 are
IGNORE; the fall at 11:13 is LYING_OUTSIDE_BED but not labelled as a bed exit;
11:22-11:28, which my script called unknown, is labelled as what the frames
show (getting up and standing beside the bed).

**Caveat:** D31 and D32 were developed while looking at test2, and the
earlier thresholds on test1. So scores on these two videos are optimistic.
A fair test needs a third recording that I don't tune on. If there's time,
I'll record one; otherwise the evaluation says so plainly.

## D34. Evaluation design and what it showed

**Date:** 2026-10-04 | **Status:** active

- **Per-second scoring** (middle of each second) instead of per sampled frame:
  independent of the sampling rate, and durations fall out directly.
- **IGNORE label** for camera set-up and complete darkness: no fair label
  exists there, and scoring it would reward lucky guesses.
- **Event matching:** same kind, start within 5 s, one-to-one, closest first.
  Events inside IGNORE ranges aren't scored (the "return" at 0:06 is me
  getting into bed while setting up).
- **Ablation runs share one perception file**, so differences come only from
  the agent / Gemini layer. `tools/run_ablation.py` reproduces everything.

Results on test2 (739 s scored): no agent 80%, rule agent 89% (+10 points
in/out of bed), rule agent + vision 89% (vision never needed), LLM policy
81%. The LLM policy lost 59 s by answering UNKNOWN for the blanket stretch
while its own reasoning said "remained in bed". I did not tune the prompt on
this video.

**Biggest lesson:** asking Gemini about the three failure moments afterwards
(standing on the bed, the fall behind the bed, darkness) gave the right
answer every time. The gap is the trigger, not the model: the agent only
looks at low-confidence moments, and these were confidently wrong. Next
step would be to send every bed event to the agent and add an
"on the bed / on the floor" field to the VLM answer.

## D35. Models per provider, and Gemini 3 changes (tested with an AI Studio key)

**Date:** 2026-10-04 | **Status:** active (updates D11, D27, D30)

Testing the delivered set-up with a fresh AI Studio API key showed three things:

1. **`gemini-2.5-flash` and `-lite` are "no longer available to new users"**
   on AI Studio (404), while my Vertex project still has them. So the config
   now has a model pair per provider: AI Studio `gemini-3.5-flash` /
   `gemini-3.5-flash-lite`, Vertex `gemini-2.5-flash` / `-lite`.
   `gemini-3.8-flash` works but its free tier is 20 requests per day, used up
   by a single run, so it's not the default.
2. **Gemini 3 rejects `thinking_budget=0`** (400 INVALID_ARGUMENT). The LLM
   policy no longer sets a thinking config; the `thought` argument on every
   tool call still gives the reasoning for the trace.
3. **Gemini 3 returns a "thought signature" with each function call** that
   must be sent back with that call in the next turn. The policy rebuilds the
   history from the steps each turn, so it now stores each call's signature
   and replays it (also kept in the LLM cache).

Also: a 429 that names a **per-day** quota now skips straight to the fallback
model; retrying with backoff can't help with a daily limit.

**Verified:** AI Studio key, gemini-3.5-flash: VLM answers correct (standing on
the bed, chair sitting) in 4-5 s; LLM policy ran a 4-call investigation and
confirmed the bed exit. Vertex with 2.5 still works without the thinking
config. 173 tests pass.
