"""OFFLINE TRACE INGESTION - the replay adapter.

This is the bridge that lets LoopGuard analyze *someone else's* agent without being
plugged into their runtime. They export a trace as JSON; this module maps each step
onto the same Monitor.observe() calls the live runner makes, so every existing detector
works unchanged.

The whole point: the detectors are agent-agnostic already (they only read Events). The
only thing missing for an external user was an input port. This is that port - the
lowest-effort integration, needing no SDK and no access to their production system.

External trace shape we accept (forgiving about field names)::

    {
      "run_id": "abc",
      "steps": [
        {"step": 1, "agent": "research_agent", "tool": "web_search",
         "args": {"query": "pricing page"}, "output": "...", "caller": "supervisor"}
      ]
    }

A file may hold a single run (object) or many runs (list of objects).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detectors import Alert, Detector, LoopDetector, StallDetector
from .monitor import Monitor
from .runner import tool_signature


# Each external field can arrive under several names depending on whose agent emitted
# the trace. The adapter's job is to absorb that variation, so we try aliases in order.
_TOOL_KEYS = ("tool", "tool_name", "name", "function")
_ARGS_KEYS = ("args", "arguments", "input", "parameters", "params")
_OUTPUT_KEYS = ("output", "result", "observation", "response", "content")
_AGENT_KEYS = ("agent", "node", "actor", "role")
_CALLER_KEYS = ("caller", "parent", "parent_agent", "invoked_by")


def _first(d: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    """Return the first present, non-None value among the candidate keys."""
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


@dataclass
class TraceReport:
    """The verdict for one external run after replaying it through the detectors."""

    run_id: str
    steps: int                                   # external steps in the trace
    events: int                                  # internal Events produced (>= steps)
    alerts: list[Alert] = field(default_factory=list)

    @property
    def looped(self) -> bool:
        """Did any fatal detector (a real loop) fire on this run?"""
        return any(a.fatal for a in self.alerts)

    def summary(self) -> str:
        head = f"run {self.run_id}: {self.steps} steps -> {self.events} events"
        if not self.alerts:
            return f"  [clean] {head} - no loops or stalls detected"
        lines = [f"  [FLAGGED] {head}"]
        for a in self.alerts:
            tag = "LOOP" if a.fatal else "warn"
            lines.append(f"      - {tag} ({a.detector}): {a.message}")
        return "\n".join(lines)


def _steps_to_observations(steps: list[dict]) -> list[tuple[str, str, str, dict]]:
    """Map external steps onto (node, action, signature, payload) observations.

    Each step with a tool call yields up to TWO observations - the decision and the
    result - mirroring how the live runner interprets a real agent step, so that
    LoopDetector (reads 'tool:' events) and StallDetector (reads 'result:' events) both
    have the events they need.
    """
    obs: list[tuple[str, str, str, dict]] = []
    for raw in steps:
        agent = str(_first(raw, _AGENT_KEYS, "agent"))
        tool = _first(raw, _TOOL_KEYS)
        args = _first(raw, _ARGS_KEYS, {}) or {}
        if not isinstance(args, dict):
            args = {"value": args}

        if tool is not None:
            sig = tool_signature(str(tool), args)
            payload = {"last_tool": str(tool), "last_args": args}
            caller = _first(raw, _CALLER_KEYS)
            if caller is not None:
                payload["caller"] = caller  # carried for a future handoff-loop detector
            obs.append((agent, f"decided: {tool}({args})", sig, payload))

        output = _first(raw, _OUTPUT_KEYS)
        if output is not None:
            text = str(output).strip().replace("\n", " ")[:160]
            obs.append((agent, f"observed: {text}", f"result:{text}", {"last_result": text}))
    return obs


def analyze_trace(trace: dict, detectors: list[Detector] | None = None) -> TraceReport:
    """Replay one external trace through a fresh Monitor and return its report."""
    detectors = detectors if detectors is not None else [LoopDetector(), StallDetector()]
    for d in detectors:
        d.reset()

    monitor = Monitor(detectors=detectors)
    steps = trace.get("steps", [])
    observations = _steps_to_observations(steps)
    for node, action, sig, payload in observations:
        monitor.observe(node, action, sig, **payload)

    return TraceReport(
        run_id=str(trace.get("run_id", "unknown")),
        steps=len(steps),
        events=len(observations),
        alerts=list(monitor.alerts),
    )


def analyze_file(path: str | Path, detectors: list[Detector] | None = None) -> list[TraceReport]:
    """Load a JSON file holding one run (object) or many (list) and analyze each."""
    data = json.loads(Path(path).read_text())
    runs = data if isinstance(data, list) else [data]
    return [analyze_trace(run, detectors) for run in runs]


if __name__ == "__main__":
    # CLI:  python -m loopguard.ingest <trace.json>
    # Uses offline detectors only (no API key needed). Add SemanticLoopDetector yourself
    # when you want paraphrase-loop catching and have OPENAI_API_KEY set.
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m loopguard.ingest <trace.json>")
        raise SystemExit(1)

    reports = analyze_file(sys.argv[1])
    flagged = sum(r.looped for r in reports)
    print(f"Analyzed {len(reports)} run(s) from {sys.argv[1]}:\n")
    for r in reports:
        print(r.summary())
    print(f"\n{flagged}/{len(reports)} run(s) flagged as looping.")
