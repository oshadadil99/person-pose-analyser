# Labelling a test video

The evaluation compares the system's output with what really happened. That
"ground truth" is two small CSV files per video, written by hand while
watching it. Use a player that shows seconds (VLC: View > Status bar), and
pause at every change of activity.

## 1. `data/ground_truth/<name>_segments.csv`

What the person is doing, as back-to-back time ranges covering the video:

```
start_sec,end_sec,state
0,7,IGNORE
7,191,LYING_IN_BED
191,309,LYING_IN_BED
309,405,SITTING_ON_BED
405,426,STANDING
460,488,WALKING
488,496,OUT_OF_BED
...
```

States: `LYING_IN_BED`, `SITTING_ON_BED` (also on the edge), `SITTING_OUTSIDE_BED`
(chair, stool...), `STANDING`, `WALKING`, `OUT_OF_BED` (not in the picture
after leaving the bed), `LYING_OUTSIDE_BED` (on the floor), `UNKNOWN`.

`IGNORE` excludes a range from scoring, e.g. setting up the camera, or a
stretch so dark that nobody could tell what is happening.

Rules of thumb:
- Label what really happened, not what the camera shows well: under a
  blanket in bed is still `LYING_IN_BED`.
- Seconds are enough; 1-2 s accuracy at the boundaries is fine.
- Consecutive rows with the same state are fine (they're merged when scoring).

## 2. `data/ground_truth/<name>_events.csv`

When each bed event starts:

```
event,time_sec
bed_exit,460
return_to_bed,600
```

- `bed_exit`: the moment the person stands up from the bed and then moves
  away. Sitting up, edge sitting, or standing up and sitting straight back
  down are not exits.
- `return_to_bed`: the moment they sit or lie down on the bed after being
  out of it.

A predicted event counts as correct if it starts within 5 s of a labelled
one (`evaluation.event_tolerance_sec`).
