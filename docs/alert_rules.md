# Alert rules

The system gives one of three decisions: **NORMAL**, **MONITOR** or **ALERT**.
The decision is made by fixed rules (`src/alerts.py`), not by the LLM. That
keeps every alert predictable, testable and easy to justify: each decision in
`decisions.json` names the rule that fired and why. All thresholds are in
`configs/default.yaml` under `alerts`.

## How a rule works

Each rule watches for a condition and becomes active once the condition has
lasted long enough. It stays active until the condition ends. The decision at
any moment is the most serious active rule, NORMAL if none.

Example: sitting on the bed from 10:00 to 14:00 with a 3-minute limit gives
MONITOR from 13:00 to 14:00, and NORMAL before and after.

## The rules

| Level | Rule | Fires when | Default |
|---|---|---|---|
| MONITOR | `prolonged_sitting_on_bed` | sitting on the bed longer than the limit | 180 s |
| MONITOR | `prolonged_unknown` | activity unknown longer than the limit | 60 s |
| MONITOR | `bed_exit` | a bed exit is confirmed; stays on until the person is back in bed | on |
| MONITOR | `out_of_bed_long` | out of bed (after an exit) longer than the limit | 600 s |
| ALERT | `possible_fall` | lying outside the bed longer than the limit | 20 s |
| ALERT | `prolonged_absence` | out of bed (after an exit) longer than the limit | 1200 s |

**NORMAL** is everything else: lying, sitting, standing or walking without
any of the conditions above.

## Why each rule exists

**Prolonged sitting on the bed (MONITOR, 3 min).** Sitting on the edge of the
bed for a long time often comes before an attempt to get up alone, which is
when falls happen. It also shows the person is awake and restless. From a 2D
camera I can't reliably tell sitting on the edge from sitting up in the
middle of the bed, so the rule covers both. Three minutes is long enough to
ignore someone sitting up to drink water or rearrange the pillow.

**Prolonged unknown (MONITOR, 60 s).** If the system can't tell what the
person is doing for a minute, someone should look. Short unknown stretches
(a turn, a blanket pulled up, a few blurred frames) are common and harmless,
so they're ignored. A person hidden under a duvet in bed is held as "lying in
bed" for up to 30 s before it counts as unknown, so a blanket alone doesn't
trigger this.

**Bed exit (MONITOR).** For an elderly person every unassisted bed exit is a
fall-risk moment. The assignment's own example event has decision MONITOR.
It stays active until the person is back in bed, so a carer looking at the
current status sees "the patient is up". It can be switched off in config
(`monitor_every_bed_exit`) for a patient who is steady on their feet.

**Out of bed for a long time (MONITOR, 10 min).** A trip to the bathroom
takes a few minutes. After ten minutes it's worth checking. This only
matters when `monitor_every_bed_exit` is off; otherwise the exit already
gives MONITOR, and this rule just records the reason it's still on.

**Possible fall (ALERT, 20 s).** Lying outside the bed is the clearest sign
of a fall. 20 seconds filters out a misread pose for a few frames, or
someone bending down to pick something up, but is still fast enough to
matter. A caregiver being in the room does **not** lower this alert.

**Prolonged absence (ALERT, 20 min).** Twenty minutes away from bed at night
is unusual and can mean the person fell somewhere out of view, got confused,
or left the room. This is the assignment's example of an ALERT condition.

## Caregiver present

If a second person is in view for at least 5 seconds while the patient is
out of bed, the two absence rules (`out_of_bed_long`, `prolonged_absence`)
are lowered by one level: someone is already with the patient. The bed-exit
MONITOR itself stays, and `possible_fall` is never lowered, because a fall is
just as serious with a caregiver in the room (and the caregiver may be the
one who needs the alert to call for help).

## What the rules look at

The rules only see the final timeline and the bed events, never raw frames.
So their quality depends on the earlier stages: a missed bed exit means no
`bed_exit` MONITOR, and a wrong `LYING_OUTSIDE_BED` would give a false fall
alert. The investigator agent (later phase) re-checks ambiguous moments such
as "lying outside the bed" before the rules see them.

## Tuning

All limits are in `configs/default.yaml`. For example, to react faster to a
possible fall, lower `alerts.floor_lying_alert_sec`; to stop every bed exit
raising MONITOR, set `alerts.monitor_every_bed_exit: false`. The tests in
`tests/test_alerts.py` cover each rule, the caregiver downgrade, and that a
caregiver never lowers the fall alert.
