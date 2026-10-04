"""The investigator agent: a bounded Reason -> Act -> Observe loop.

For each trigger:
    observation = what fired the trigger
    repeat (at most agent.max_tool_calls times):
        policy looks at the trigger + findings so far and either
          - picks a tool and arguments (Reason -> Act), we run it (Observe), or
          - concludes
    if the budget runs out without a conclusion -> "unresolved" (change nothing)

Plain Python on purpose, no agent framework: the whole loop is the
`investigate` function below. The policy is swappable: policy_rules.py
(deterministic, offline) now, an LLM choosing the tools later.

Every investigation is kept as a trace that reads like
Observation -> Thought -> Action -> Finding -> ... -> Conclusion.
"""

import logging
from dataclasses import dataclass, field

from src.agent import policy_rules
from src.agent.policy_rules import Conclusion
from src.agent.tools import TOOLS, AgentContext
from src.events import Event
from src.state_machine import Segment, merge_segments
from src.summary import format_clock
from src.triggers import Trigger

log = logging.getLogger(__name__)


@dataclass
class Step:
    thought: str
    tool: str
    args: dict
    result: dict

    @property
    def finding(self) -> str:
        return self.result.get("summary", "")


@dataclass
class AgentTrace:
    trace_id: str
    trigger: Trigger
    steps: list[Step] = field(default_factory=list)
    conclusion: Conclusion | None = None
    final_thought: str = ""
    decision: str | None = None      # NORMAL / MONITOR / ALERT at that moment, filled after the alert engine

    def to_dict(self) -> dict:
        c = self.conclusion
        return {
            "trace_id": self.trace_id,
            "trigger": {"type": self.trigger.kind, "start": format_clock(self.trigger.t_start),
                        "end": format_clock(self.trigger.t_end), "observation": self.trigger.describe()},
            "steps": [{"thought": s.thought, "action": _call(s.tool, s.args), "finding": s.finding,
                       "result": {k: v for k, v in s.result.items() if k != "summary"}} for s in self.steps],
            "conclusion": {"outcome": c.outcome, "state": c.state.value.lower() if c.state else None,
                           "confidence": c.confidence, "reason": c.reason},
            "decision": self.decision,
        }

    def to_text(self) -> str:
        c = self.conclusion
        lines = [f"{self.trace_id}  [{self.trigger.kind}]",
                 f"  Observation: {self.trigger.describe()}"]
        for s in self.steps:
            lines += [f"  Thought:     {s.thought}",
                      f"  Action:      {_call(s.tool, s.args)}",
                      f"  Finding:     {s.finding}"]
        if self.final_thought:
            lines.append(f"  Thought:     {self.final_thought}")
        state = f" -> {c.state.value}" if c.state else ""
        lines.append(f"  Conclusion:  {c.outcome.upper()}{state} (confidence {c.confidence}): {c.reason}")
        if self.decision:
            lines.append(f"  Decision:    {self.decision}")
        return "\n".join(lines)


def _call(tool: str, args: dict) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.1f}"
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if isinstance(v, str):
            return f'"{v}"'
        return str(v)
    return f"{tool}(" + ", ".join(f"{k}={fmt(v)}" for k, v in args.items()) + ")"


def investigate(trigger: Trigger, ctx: AgentContext, trace_id: str) -> AgentTrace:
    trace = AgentTrace(trace_id, trigger)
    max_calls = ctx.cfg["agent"]["max_tool_calls"]
    while True:
        action = policy_rules.next_action(trigger, trace.steps, ctx.cfg)
        if action.tool is None:
            trace.conclusion, trace.final_thought = action.conclusion, action.thought
            break
        if len(trace.steps) >= max_calls:
            trace.conclusion = Conclusion("unresolved", f"tool budget ({max_calls}) used up without a clear answer")
            break
        result = TOOLS[action.tool](ctx, **action.args)
        trace.steps.append(Step(action.thought, action.tool, action.args, result))
    return trace


def run_agent(triggers: list[Trigger], ctx: AgentContext) -> list[AgentTrace]:
    traces = [investigate(t, ctx, f"trace_{i + 1:04d}") for i, t in enumerate(triggers)]
    if traces:
        outcomes = {}
        for tr in traces:
            outcomes[tr.conclusion.outcome] = outcomes.get(tr.conclusion.outcome, 0) + 1
        log.info("Agent: %d investigations, %d tool calls, outcomes %s",
                 len(traces), sum(len(t.steps) for t in traces), outcomes)
    return traces


def apply_to_segments(segments: list[Segment], traces: list[AgentTrace]) -> list[Segment]:
    """Write each conclusion's state over its time range, then re-merge neighbours."""
    out = list(segments)
    for tr in traces:
        c = tr.conclusion
        if c.state is None or c.apply_range is None:
            continue
        a, b = c.apply_range
        new = []
        for s in out:
            if s.end_sec <= a or s.start_sec >= b:
                new.append(s)
                continue
            if s.start_sec < a:
                new.append(Segment(s.start_sec, a, s.state, s.confidence))
            new.append(Segment(max(s.start_sec, a), min(s.end_sec, b), c.state, c.confidence))
            if s.end_sec > b:
                new.append(Segment(b, s.end_sec, s.state, s.confidence))
        out = new
    return merge_segments(out)


def apply_to_events(events: list[Event], traces: list[AgentTrace], tolerance_sec: float) -> list[Event]:
    """Link bed events to their investigation: new confidence if confirmed, dropped if rejected."""
    kept = []
    for e in events:
        tr = next((t for t in traces if t.trigger.kind == "bed_event_check"
                   and t.trigger.evidence["event"] == e.event
                   and abs(t.trigger.evidence["start_sec"] - e.start_sec) <= tolerance_sec), None)
        if tr is not None:
            e.agent_trace_id = tr.trace_id
            if tr.conclusion.outcome == "rejected":
                log.info("Agent rejected %s at %.1fs: %s", e.event, e.start_sec, tr.conclusion.reason)
                continue
            if tr.conclusion.outcome == "confirmed":
                e.confidence = tr.conclusion.confidence
                e.note = (e.note + "; " if e.note else "") + "confirmed by agent"
        kept.append(e)
    return kept
