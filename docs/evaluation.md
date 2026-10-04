# Evaluation

## What was tested

**Video:** `test 2.mp4`, 13 min, phone camera in my bedroom. I acted out most of the
assignment's difficult cases: sleeping, a long stretch fully under a blanket,
lying at the bed edge, sitting up, standing *on* the bed, edge sitting, a bed
exit, chair sitting, leaving the camera view three times, walking back in low
light, a return to bed, turning over, a fall off the bed, and a fully dark
stretch. The phone was nudged at about 8:10, which turned out to be a useful
test as well.

**Labels:** I wrote down what I did while recording and turned it into
`data/ground_truth/test2_segments.csv` and `test2_events.csv`, checked against
the frames (see `tools/label_helper.md`). Camera set-up (0:00-0:07), the
end (12:55-13:04) and the completely dark 12:23-12:52 are excluded from
scoring. 739 seconds are scored.

**Method:** states compared once per second; a bed event counts as found if
it starts within 5 s of the labelled one; durations compared over the scored
seconds. Full tables and confusion matrices: [evaluation_results.md](evaluation_results.md).
Reproduce with:

```bash
python tools/run_ablation.py --video "data/videos/test 2.mp4" --base outputs/test2 \
    --gt data/ground_truth/test2_segments.csv --gt-events data/ground_truth/test2_events.csv \
    --vlm vertex --out docs/evaluation_results.md
```

**Caveat, stated up front:** I developed two fixes (lying along a diagonal bed,
camera-move detection; decisions D31, D32) while looking at this video, and
earlier thresholds on a shorter first clip. These numbers are therefore
optimistic. A recording I never tuned on would be the fair test.

## Results

| Run | State accuracy | Macro F1 | In/out of bed | Exit precision | Exit recall | Return precision | Return recall | Gemini calls |
|---|---|---|---|---|---|---|---|---|
| no agent | 80% | 0.58 | 88% | 33% | 100% | 50% | 100% | 0 |
| agent, rule policy | **89%** | **0.68** | **98%** | 33% | 100% | 50% | 100% | 0 |
| agent, rule policy + Gemini vision | 89% | 0.68 | 98% | 33% | 100% | 50% | 100% | 0 |
| agent, Gemini picks the tools | 81% | 0.58 | 90% | 33% | 100% | 50% | 100% | 69 |

Duration errors for the best run (agent, rule policy):

| State | Labelled | Predicted | Error |
|---|---|---|---|
| Lying in bed | 6:15 | 6:18 | 3 s |
| Sitting on bed | 2:10 | 2:11 | 1 s |
| Sitting outside bed | 1:37 | 1:20 | 17 s |
| Standing | 0:27 | 0:51 | 24 s |
| Walking | 1:02 | 0:48 | 14 s |
| Out of bed (out of view) | 0:39 | 0:51 | 12 s |
| Lying outside bed (fall) | 0:09 | 0:00 | 9 s |

## What the numbers say

**The agent is what makes the timeline usable.** Without it, the 59 s I spent
completely under the blanket (3:45-4:44) and about 10 s at the bed edge
(11:00) are UNKNOWN: YOLO sees nobody. The rule-policy agent looks back and forward,
finds "lying in bed before and after, nobody seen getting up", and fills them
in. That is +9 points accuracy and +10 points on in/out of bed, using only
cheap timeline tools.

**Long states are measured well, short upright ones less so.** Lying and
sitting on the bed are within 3 s over 6 and 2 minutes. Standing vs walking
is the weakest pair (F1 ~0.6): slow walking and turning in a small room sits
right at the speed threshold, and the transitions in and out of the chair are
read as standing for a few seconds each. The biggest single confusion is
walking predicted as out of bed (18 s): mostly 9:49-10:00, walking back to
bed in low light, when YOLO stopped detecting me and the timeline treated it
as "gone from view".

**Bed events: every real one found, but false ones too.** The labelled exit
(7:40) was found 2 s late and the return (10:00) 4 s late. The two false exits
and one false return all come from moments that aren't real bed exits:
standing *on* the bed (6:48) counts as "up", and the fall off the bed (11:13)
counts as an exit because I'd labelled it as a fall, not an exit.

**The Gemini-driven agent was worse than the rules (81% vs 89%).** It gathered
the same evidence for the blanket stretch, and even wrote "this suggests the
patient remained in bed", but answered with the state UNKNOWN. Its reasoning
and its answer disagreed. The code-side check (`sanitize`) kept this harmless
(UNKNOWN changes nothing) but the correct conclusion was lost. I didn't tune the
prompt to fix it, because that would be tuning on the test video.

**Gemini vision was never called.** Neither policy needed `ask_vlm` on this
video: the cheap tools either settled the question or the moment wasn't
flagged. That is also the most important lesson, see below.

## Where vision would have helped

The agent only wakes up for moments the geometry is unsure about. The worst
errors are where geometry is confidently wrong. Asking Gemini directly about
those moments afterwards:

| Moment | Gemini (3 frames) | System |
|---|---|---|
| 6:55 standing on the bed | "standing on the bed itself, feet clearly on the mattress" (1.0) | out of bed, false exit |
| 11:15 after the fall | "sitting on the floor next to the bed" (1.0) | lying in bed, fall missed |
| 12:25 darkness | UNKNOWN (0.1), "almost completely dark" | out of bed |

So the VLM has the information that would fix the three failure cases; the
gap is in *when* the agent asks. With more time I would (1) send every bed
event to the agent, not only low-confidence ones (there are only a few per
night), (2) add an "on the bed / on the floor" field to the VLM answer so
"standing on the bed" can be expressed, and (3) wake the agent when the
patient's box suddenly drops low next to the bed (possible fall hidden by the
bed). Details in [failure_cases.md](failure_cases.md).

## Not covered

- Only one labelled video, the same one used during development (see caveat).
- No real elderly patient, no second person entering in this recording (the
  caregiver logic is only covered by unit tests), no real night-time infrared
  footage.
- `test1.mp4` (59 s) was used for development but has no labels.
