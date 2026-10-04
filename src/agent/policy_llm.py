"""LLM agent policy: Gemini chooses the next tool by function calling.

Same loop, same tools and same 4-call budget as the rule policy; only the
"which tool next?" decision moves to the model. Each turn we send:
    the task (trigger description, allowed conclusions, safety rule)
    + every previous tool call and its result
and force the model to answer with exactly one function call: a tool, or
`conclude`. We run the tool ourselves (no automatic function calling), so the
loop stays visible plain Python.

The model never gets the last word on safety. Its conclusion is checked by
code (`sanitize`) before it is used:
- only the outcomes that make sense for the trigger type are accepted;
- it can only change the timeline where the rule policy could (fill an
  UNKNOWN stretch, or turn a false possible fall into lying in bed);
- it can only dismiss a possible fall with evidence in its own tool results
  (hips at the bed, or the VLM seeing the person on the bed).
If Gemini fails, the investigation is redone with the rule policy.

Answers are cached by hash(model + task + previous steps), so reruns are free
and reproducible, like the VLM cache.
"""

import hashlib
import json
import logging
from pathlib import Path

from src.agent.policy_rules import Action, Conclusion
from src.agent.vlm import GeminiCaller
from src.states import State
from src.triggers import Trigger

log = logging.getLogger(__name__)

STATE_NAMES = [s.value for s in State]
ALLOWED = {
    "possible_fall": ["confirmed", "rejected", "unresolved"],
    "bed_event_check": ["confirmed", "rejected", "unresolved"],
    "unknown_stretch": ["resolved", "unresolved"],
    "multiple_people": ["noted"],
    "flicker": ["noted"],
}
GUIDE = {
    "possible_fall": "Decide if the patient is lying on the floor (confirmed) or actually on the bed (rejected). "
                     "Check where the hips are relative to the bed and what happened just before.",
    "bed_event_check": "Decide if this bed event really happened (confirmed) or not (rejected). "
                       "A real bed exit needs BOTH: in bed before (look back ~10 s) AND out of bed and moving "
                       "away after (look forward ~15 s). A real return needs out of bed before and in bed after. "
                       "A single moment is never enough.",
    "unknown_stretch": "Work out what the patient was doing during the unknown stretch (resolved, give the state). "
                       "Check whether the patient is visible in it, and the states just before and after.",
    "multiple_people": "Find out who is in the room and what the patient is doing; just note it.",
    "flicker": "Find out what the patient was really doing while the per-frame rules kept changing; just note it.",
}


class PolicyError(Exception):
    pass


def _num(desc: str) -> dict:
    return {"type": "NUMBER", "description": desc}


THOUGHT = {"type": "STRING", "description": "one short sentence: why you are making this call"}
TOOL_DECLS = [
    {"name": "look_back", "description": "States and summary of the timeline in [t - seconds, t].",
     "parameters": {"type": "OBJECT", "properties": {"t": _num("time in seconds"), "seconds": _num("how far back"),
                                                     "thought": THOUGHT}, "required": ["t", "seconds", "thought"]}},
    {"name": "look_forward", "description": "States and summary of the timeline in [t, t + seconds].",
     "parameters": {"type": "OBJECT", "properties": {"t": _num("time in seconds"), "seconds": _num("how far ahead"),
                                                     "thought": THOUGHT}, "required": ["t", "seconds", "thought"]}},
    {"name": "check_bed_overlap", "description": "Where the patient's hips are relative to the bed at time t.",
     "parameters": {"type": "OBJECT", "properties": {"t": _num("time in seconds"), "thought": THOUGHT},
                    "required": ["t", "thought"]}},
    {"name": "get_pose_summary", "description": "Torso/thigh angle, speed, keypoint confidence at t, "
                                                "or averaged over [t, t + seconds] if seconds > 0.",
     "parameters": {"type": "OBJECT", "properties": {"t": _num("time in seconds"), "seconds": _num("0 for one frame"),
                                                     "thought": THOUGHT}, "required": ["t", "thought"]}},
    {"name": "count_people", "description": "How many people are in view in [t, t + seconds], and is the patient one.",
     "parameters": {"type": "OBJECT", "properties": {"t": _num("time in seconds"), "seconds": _num("duration"),
                                                     "thought": THOUGHT}, "required": ["t", "thought"]}},
    {"name": "ask_vlm", "description": "Show 2-4 video frames to a vision model and ask a question. Expensive: "
                                       "only use it if the other tools leave the question open.",
     "parameters": {"type": "OBJECT", "properties": {
         "times": {"type": "ARRAY", "items": {"type": "NUMBER"}, "description": "2 to 4 times in seconds"},
         "question": {"type": "STRING"}, "thought": THOUGHT}, "required": ["times", "question", "thought"]}},
    {"name": "conclude", "description": "Finish the investigation.",
     "parameters": {"type": "OBJECT", "properties": {
         "outcome": {"type": "STRING", "enum": sorted({o for v in ALLOWED.values() for o in v})},
         "state": {"type": "STRING", "enum": STATE_NAMES + ["NONE"],
                   "description": "for 'resolved' or a rejected fall: the patient's real state, else NONE"},
         "confidence": _num("0 to 1"), "reason": {"type": "STRING"}, "thought": THOUGHT},
         "required": ["outcome", "state", "confidence", "reason", "thought"]}},
]


def task_prompt(trigger: Trigger, cfg: dict) -> str:
    return f"""You are investigating one ambiguous moment in a recorded video from a fixed camera in the \
bedroom of an elderly patient. Times are seconds from the start; you may look forward in time.

Trigger ({trigger.kind}): {trigger.describe()}
Window: {trigger.t_start:.1f}s to {trigger.t_end:.1f}s.
Task: {GUIDE[trigger.kind]}
Allowed outcomes: {", ".join(ALLOWED[trigger.kind])} (use "unresolved"/"noted" if you can't tell).

Rules:
- At most {cfg['agent']['max_tool_calls']} tool calls, then call conclude.
- Prefer the cheap timeline tools; call ask_vlm only if they leave the question open.
- Never dismiss a possible fall unless the hips are at the bed or the vision model clearly sees the \
patient on the bed. Missing a fall is worse than a false alarm.
- Answer with exactly one function call each turn."""


def sanitize(trigger: Trigger, args: dict, steps: list, cfg: dict) -> Conclusion:
    """Turn the model's `conclude` call into a Conclusion the rest of the code can trust."""
    outcome = args.get("outcome")
    reason = str(args.get("reason", ""))
    try:
        conf = round(min(max(float(args.get("confidence", 0)), 0.0), 1.0), 2)
    except (TypeError, ValueError):
        conf = 0.0
    if outcome not in ALLOWED[trigger.kind]:
        fallback = "noted" if trigger.kind in ("multiple_people", "flicker") else "unresolved"
        return Conclusion(fallback, f"model gave outcome {outcome!r}, not allowed here; {reason}")

    state = None
    if outcome == "resolved":
        state = State(args["state"]) if args.get("state") in STATE_NAMES else None
        if state is None:
            return Conclusion("unresolved", f"model resolved without a valid state; {reason}")
    if trigger.kind == "possible_fall" and outcome == "rejected":
        if not _bed_evidence(steps, cfg):
            return Conclusion("unresolved", f"model wanted to dismiss the fall without evidence; kept. {reason}", 0.5)
        state = State.LYING_IN_BED
    rng = (trigger.t_start, trigger.t_end) if state else None
    return Conclusion(outcome, f"LLM: {reason}", conf, state, rng)


def _bed_evidence(steps: list, cfg: dict) -> bool:
    for s in steps:
        r = s.result
        if s.tool == "check_bed_overlap" and r.get("bed_edge_dist") is not None \
                and r["bed_edge_dist"] >= -cfg["patient"]["bed_margin"]:
            return True
        if s.tool == "ask_vlm" and r.get("state") == "LYING_IN_BED" \
                and r.get("confidence", 0) >= cfg["agent"]["min_confidence"]:
            return True
    return False


class LLMPolicy:
    name = "llm"

    def __init__(self, client, cfg: dict):
        self.caller = GeminiCaller(client, cfg)
        self.cfg = cfg
        self.cache_dir = Path(cfg["vlm"]["llm_cache_dir"])

    def next_action(self, trigger: Trigger, steps: list, cfg: dict) -> Action:
        from google.genai import types
        prompt = task_prompt(trigger, cfg)
        budget_left = len(steps) < cfg["agent"]["max_tool_calls"]
        history = [{"tool": s.tool, "args": s.args, "result": s.result} for s in steps]

        key = hashlib.sha256(json.dumps([cfg["vlm"]["model"], prompt, history, budget_left],
                                        sort_keys=True, default=str).encode()).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            self.caller.stats["cache_hits"] += 1
            call = json.loads(cached.read_text(encoding="utf-8"))
        else:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            for s in steps:
                contents.append(types.Content(role="model", parts=[types.Part.from_function_call(
                    name=s.tool, args={**s.args, "thought": s.thought})]))
                contents.append(types.Content(role="user", parts=[types.Part.from_function_response(
                    name=s.tool, response={"result": s.result})]))
            config = types.GenerateContentConfig(
                temperature=cfg["vlm"]["temperature"],
                tools=[types.Tool(function_declarations=TOOL_DECLS)],
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=None if budget_left else ["conclude"])),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_budget=0))

            def parse(resp):
                calls = resp.function_calls or []
                return {"name": calls[0].name, "args": dict(calls[0].args or {})} if calls else None

            call, _ = self.caller.call(contents, config, parse)
            if call is None:
                raise PolicyError("Gemini didn't return a function call")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(call, indent=2, default=str), encoding="utf-8")

        args = dict(call["args"])
        thought = str(args.pop("thought", ""))
        if call["name"] == "conclude":
            return Action(thought, conclusion=sanitize(trigger, args, steps, cfg))
        if call["name"] not in {d["name"] for d in TOOL_DECLS}:
            raise PolicyError(f"unknown tool {call['name']!r}")
        return Action(thought, call["name"], args)
